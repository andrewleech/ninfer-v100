#pragma once

// W8G32 RowSplit x BF16 GEMM via int8 __dp4a on Volta (sm_70), wide-T prefill.
//
// The W8 sibling of q4_volta_dp4a_gemm.cuh -- same MMQ-style structure that ncu showed makes
// llama's int8 dp4a prefill COMPUTE-bound where ninfer's fp16 mma.sync W8 kernel is L1/shared-bound.
// The design (verbatim from the Q4 route): WARP-COOPERATIVE tiling -- within a warp, threadIdx.x
// (lane) selects the output ROW and threadIdx.y (warp) selects the output TOKEN, so the activation
// y[token][k] is a warp SHARED BROADCAST (one transaction feeds all 32 lanes) while the weight
// x[row][k] is distinct per lane on an odd (9-word) row stride for bank-conflict-free int32 reads.
//
// W8-vs-Q4 differences: the weight is a FULL signed int8 byte stored directly (W8RowSplitStorage
// packs 32 signed int8 per 32-K group, one fp16 scale/group) -- NO nibble unpack, so decoding a
// group-word is just reading 4 consecutive code bytes as one int32 of dp4a-ready packed int8. The
// int8 weight is already symmetric-signed, matching the symmetric int8 activation, so the dp4a dot
// needs no sum-correction term. kGroupK is 32 (Q4's is 64), so kWordsK is 8.
//
// Two kernels, mirroring the Q4 route (quantize_mmq_q8_1 + mul_mat_q):
//   1. w8_dp4a_quantize_x_kernel -- quantise the whole activation matrix to int8 ONCE into scratch.
//   2. w8_volta_dp4a_gemm_kernel -- the warp-cooperative dp4a GEMM.

#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct W8VoltaDp4aSchedule {
    static constexpr int kGroupK  = W8RowSplitStorage::kGroupK; // 32 (one weight group per K-step)
    static constexpr int kWarp    = 32;
    static constexpr int kNWarps  = 8;
    static constexpr int kThreads = kWarp * kNWarps; // 256
    static constexpr int kTileN   = 64;              // output rows per CTA (lane axis, stride 32)
    static constexpr int kTileT   = 128;             // tokens per CTA (warp axis, stride kNWarps)
    static constexpr int kRowsPerThread = kTileN / kWarp;   // 2
    static constexpr int kToksPerThread = kTileT / kNWarps; // 16
    static constexpr int kWordsK        = kGroupK / 4;      // 8 int32 words of int8 per group-row
    static constexpr int kStrideW       = kWordsK + 1;      // 9: odd stride -> conflict-free per-
                                                            // lane weight rows (int32 reads)
    static constexpr int kStrideX       = kWordsK;          // 8: 32B rows -> uint4-aligned; the
                                                            // activation read is a warp broadcast,
                                                            // so it is bank-conflict-immune anyway

    static constexpr int kQuantThreads = 256;
};

// Quantise activations x[k, t] (token-major, stride k) to int8 xq[t][k] + per-(token,group) fp16
// scale xs[t][groups]. One warp per (token, group): 32 bf16 -> amax -> int8 + scale. Group is 32,
// so one lane handles one element (Q4's group-64 gave two per lane).
__global__ __launch_bounds__(W8VoltaDp4aSchedule::kQuantThreads) void w8_dp4a_quantize_x_kernel(
    const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ xq,
    std::uint16_t* __restrict__ xs, int k, int t, int groups) {
    constexpr int kGroupK = W8VoltaDp4aSchedule::kGroupK;
    const int warp  = static_cast<int>(blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5));
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    if (warp >= t * groups) { return; }
    const int token = warp / groups;
    const int g     = warp % groups;

    const std::int64_t base = static_cast<std::int64_t>(token) * k + g * kGroupK;
    const float v0 = __bfloat162float(x[base + lane]);
    float amax     = fabsf(v0);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
    const float s   = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / s;
    xq[base + lane] = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v0 * inv))));
    if (lane == 0) { xs[static_cast<std::int64_t>(token) * groups + g] = __half_as_ushort(__float2half(s)); }
}

// out[N, T] = W[N, K] (W8) * xq[T, K] (int8). Grid = (ceil(N/kTileN), ceil(T/kTileT)).
__global__ __launch_bounds__(W8VoltaDp4aSchedule::kThreads) void w8_volta_dp4a_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const std::int8_t* __restrict__ xq, const std::uint16_t* __restrict__ xs,
    __nv_bfloat16* __restrict__ out, int out_ld, int n, int k, int t, int padded_groups,
    int groups, bool add_residual) {
    using S = W8VoltaDp4aSchedule;
    constexpr int kGroupK = S::kGroupK;
    constexpr int kCodeB  = W8RowSplitStorage::kCodeBytesPerGroup; // 32 signed int8
    constexpr int kW  = S::kStrideW;
    constexpr int kXW = S::kStrideX;

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n0   = static_cast<int>(blockIdx.x) * S::kTileN;
    const int t0   = static_cast<int>(blockIdx.y) * S::kTileT;

    // int32-word shared tiles of packed int8. w_sh is padded to kStrideW (9) for conflict-free
    // per-lane weight rows; x_sh is kStrideX (8) so its rows are 32B/uint4-aligned.
    __shared__ __align__(16) std::int32_t w_sh[S::kTileN][kW];
    __shared__ __align__(16) std::int32_t x_sh[S::kTileT][kXW];
    __shared__ float w_scale[S::kTileN];
    __shared__ float x_scale[S::kTileT];

    // Persistent fp32 accumulators: this thread owns rows {lane, lane+32} x tokens {warp + tt*8}.
    float acc[S::kRowsPerThread][S::kToksPerThread];
#pragma unroll
    for (int r = 0; r < S::kRowsPerThread; ++r)
#pragma unroll
        for (int c = 0; c < S::kToksPerThread; ++c) { acc[r][c] = 0.0f; }

    for (int g = 0; g < groups; ++g) {
        __syncthreads();

        // decode weights: kTileN rows x kWordsK int32 words. The W8 code bytes ARE signed int8, so
        // 4 consecutive bytes read as one int32 is already dp4a-ready packed int8 -- no unpack.
        for (int u = tid; u < S::kTileN * S::kWordsK; u += S::kThreads) {
            const int row  = u / S::kWordsK;
            const int word = u % S::kWordsK;
            const int gr   = n0 + row;
            std::int32_t v = 0;
            if (gr < n) {
                const std::uint8_t* crow =
                    codes + (static_cast<std::int64_t>(gr) * padded_groups + g) * kCodeB;
                v = reinterpret_cast<const std::int32_t*>(crow)[word]; // 4 signed int8, K=[4word..+3]
            }
            w_sh[row][word] = v;
        }
        for (int r = tid; r < S::kTileN; r += S::kThreads) {
            const int r2 = n0 + r;
            w_scale[r] =
                (r2 < n) ? __half2float(__ushort_as_half(reinterpret_cast<const std::uint16_t*>(
                               scales + static_cast<std::int64_t>(r2) * padded_groups *
                                            W8RowSplitStorage::kScaleBytesPerGroup)[g]))
                         : 0.0f;
        }

        // copy pre-quantised int8 activations: kTileT tokens x kWordsK int32 (contiguous int8).
        for (int u = tid; u < S::kTileT * S::kWordsK; u += S::kThreads) {
            const int tok  = u / S::kWordsK;
            const int word = u % S::kWordsK;
            const int col  = t0 + tok;
            std::int32_t v = 0;
            if (col < t) {
                v = *reinterpret_cast<const std::int32_t*>(
                    &xq[static_cast<std::int64_t>(col) * k + g * kGroupK + word * 4]);
            }
            x_sh[tok][word] = v;
        }
        for (int c = tid; c < S::kTileT; c += S::kThreads) {
            const int c2 = t0 + c;
            x_scale[c] = (c2 < t) ? __half2float(__ushort_as_half(
                                        xs[static_cast<std::int64_t>(c2) * groups + g]))
                                  : 0.0f;
        }

        __syncthreads();

        std::int32_t iacc[S::kRowsPerThread][S::kToksPerThread];
#pragma unroll
        for (int r = 0; r < S::kRowsPerThread; ++r)
#pragma unroll
            for (int c = 0; c < S::kToksPerThread; ++c) { iacc[r][c] = 0; }

        // The activation read is the bulk of shared traffic and is a warp broadcast, so it is
        // vectorised to uint4 (one 16-byte load carries 4 int32 words = 16 K). Weights are few per
        // thread and stay on the conflict-free odd stride, read as int32.
        static_assert(S::kWordsK % 4 == 0, "uint4 K-vectorisation needs kWordsK % 4 == 0");
#pragma unroll
        for (int wq = 0; wq < S::kWordsK; wq += 4) {
            std::int32_t wv[S::kRowsPerThread][4];
#pragma unroll
            for (int r = 0; r < S::kRowsPerThread; ++r)
#pragma unroll
                for (int s = 0; s < 4; ++s) { wv[r][s] = w_sh[lane + r * S::kWarp][wq + s]; }
#pragma unroll
            for (int c = 0; c < S::kToksPerThread; ++c) {
                const uint4 xc =
                    *reinterpret_cast<const uint4*>(&x_sh[warp + c * S::kNWarps][wq]);
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const std::int32_t xw = reinterpret_cast<const std::int32_t*>(&xc)[s];
#pragma unroll
                    for (int r = 0; r < S::kRowsPerThread; ++r) {
                        iacc[r][c] = __dp4a(wv[r][s], xw, iacc[r][c]);
                    }
                }
            }
        }

#pragma unroll
        for (int r = 0; r < S::kRowsPerThread; ++r) {
            const float ws = w_scale[lane + r * S::kWarp];
#pragma unroll
            for (int c = 0; c < S::kToksPerThread; ++c) {
                acc[r][c] += static_cast<float>(iacc[r][c]) * ws * x_scale[warp + c * S::kNWarps];
            }
        }
    }

#pragma unroll
    for (int r = 0; r < S::kRowsPerThread; ++r) {
        const int row = n0 + lane + r * S::kWarp;
        if (row >= n) { continue; }
#pragma unroll
        for (int c = 0; c < S::kToksPerThread; ++c) {
            const int col = t0 + warp + c * S::kNWarps;
            if (col >= t) { continue; }
            const std::int64_t idx = static_cast<std::int64_t>(col) * out_ld + row;
            // linear_add fold: out already holds the residual, add W*x in place (matches
            // w8_linear_add_gemm_simt.cu: residual[i] = f32(residual[i]) + acc). Plain GEMM else.
            const float base = add_residual ? __bfloat162float(out[idx]) : 0.0f;
            out[idx] = __float2bfloat16(base + acc[r][c]);
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
