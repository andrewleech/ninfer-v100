#pragma once

// Q4G64 RowSplit x BF16 GEMM via int8 __dp4a on Volta (sm_70), wide-T prefill.
//
// Phase-1 prototype of the "different algorithm" the fair-comparison against llama.cpp pointed to.
// llama's titan build forces MMQ, so its prefill GEMM is int8 dp4a: activations are quantised to
// int8 (Q8_1-style, per-block fp16 scale) ONCE, Q4 weights are used as int8 (each nibble is
// already a signed [-8,7] code), the inner product runs on __dp4a (4 int8 MACs/instr) with an
// int32 accumulator, and the group scales are applied in the epilogue. ncu on the V100 showed
// llama's dp4a kernel is COMPUTE-bound (~75% FMA/int pipe, L1/TEX ~30%), whereas this port's
// fp16 mma.sync kernel (q4_volta_mma_gemm.cuh) is L1/shared-bound (84.6%, compute 46%). The point
// of this kernel is to move the bottleneck off L1 onto the integer pipe the same way.
//
// Two kernels, mirroring llama's (quantize_mmq_q8_1 + mul_mat_q):
//   1. q4_dp4a_quantize_x_kernel -- quantise the whole activation matrix to int8 ONCE into a
//      scratch buffer (int8 xq[t][k] + fp16 xs[t][groups]). The first prototype re-quantised
//      inside every N-tile (n/64 times redundant); hoisting this is the dominant speedup.
//   2. q4_volta_dp4a_gemm_kernel -- the dp4a GEMM proper, reading pre-quantised int8 activations.
//
// Weight layout (Q4RowSplitStorage): row-major, K/64 groups/row, each group = 32 code bytes
// (byte b holds K=2b low nibble, K=2b+1 high nibble) + one fp16 scale. int8 weight = ((nibble ^ 8)
// - 8) in [-8,7], no zero-point -- weight and activation are both symmetric, so the dp4a dot needs
// no sum-correction:  w.x = scale_w * scale_x * sum_k( q_w[k] * q_x[k] ).

#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Q4VoltaDp4aSchedule {
    static constexpr int kGroupK  = Q4RowSplitStorage::kGroupK; // 64 (one weight group per K-step)
    static constexpr int kTileN   = 128;                        // output rows per CTA
    static constexpr int kTileT   = 128;                        // tokens per CTA
    static constexpr int kThreads = 256;                        // 8 warps
    // 128x128 = 16384 outputs / 256 threads = 64 outputs/thread, arranged 8 rows x 8 tokens.
    // Each 4-k dp4a step loads 8 weight + 8 activation int32 (16 shared reads) for 64 dp4a (4:1
    // reuse). Measured sweep at n=16384 k=5120: 4x4 (2:1) L1-bound 92% = 11 TF/s; 8x4 (2.67:1) =
    // 10; 8x8 (4:1) L1 66%, occupancy 12.5% (1 block, 256 regs) = 17. Reuse dominates occupancy
    // here, so the widest tile that still builds wins.
    static constexpr int kMicroN     = 8;
    static constexpr int kMicroT     = 8;
    static constexpr int kRowThreads = kTileN / kMicroN; // 16
    static constexpr int kColThreads = kTileT / kMicroT; // 16
    static_assert(kRowThreads * kColThreads == kThreads);

    static constexpr int kQuantThreads = 256; // activation-quantise kernel block
};

// Quantise the activation matrix x[k, t] (token-major, stride k) to int8 xq[t][k] + per-(token,
// group) fp16 scale xs[t][groups]. One warp per (token, group): 64 bf16 -> amax -> int8 + scale.
__global__ __launch_bounds__(Q4VoltaDp4aSchedule::kQuantThreads) void q4_dp4a_quantize_x_kernel(
    const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ xq,
    std::uint16_t* __restrict__ xs, int k, int t, int groups) {
    constexpr int kGroupK = Q4VoltaDp4aSchedule::kGroupK;
    const int warp   = static_cast<int>(blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5));
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int total  = t * groups;
    if (warp >= total) { return; }
    const int token = warp / groups;
    const int g     = warp % groups;

    const std::int64_t base = static_cast<std::int64_t>(token) * k + g * kGroupK;
    // 32 lanes, 64 values -> 2 per lane (lane, lane+32).
    const float v0 = __bfloat162float(x[base + lane]);
    const float v1 = __bfloat162float(x[base + lane + 32]);
    float amax     = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const float s   = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / s;
    std::int8_t* dst = xq + base;
    dst[lane]      = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v0 * inv))));
    dst[lane + 32] = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v1 * inv))));
    if (lane == 0) {
        xs[static_cast<std::int64_t>(token) * groups + g] =
            __half_as_ushort(__float2half(s));
    }
}

// out[N, T] = W[N, K] (Q4) * xq[T, K] (int8, pre-quantised). One CTA owns a kTileN x kTileT tile;
// grid = (ceil(N/64), ceil(T/64)). Weights and int8 activations are staged in shared per group.
__global__ __launch_bounds__(Q4VoltaDp4aSchedule::kThreads) void q4_volta_dp4a_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const std::int8_t* __restrict__ xq, const std::uint16_t* __restrict__ xs,
    __nv_bfloat16* __restrict__ out, int out_ld, int n, int k, int t, int padded_groups,
    int groups) {
    using S = Q4VoltaDp4aSchedule;
    constexpr int kGroupK = S::kGroupK;
    constexpr int kCodeB  = Q4RowSplitStorage::kCodeBytesPerGroup; // 32

    const int tid = static_cast<int>(threadIdx.x);
    const int n0  = static_cast<int>(blockIdx.x) * S::kTileN;
    const int t0  = static_cast<int>(blockIdx.y) * S::kTileT;

    __shared__ __align__(16) std::int8_t w_sh[S::kTileN][kGroupK];
    __shared__ __align__(16) std::int8_t x_sh[S::kTileT][kGroupK];
    __shared__ float w_scale[S::kTileN];
    __shared__ float x_scale[S::kTileT];

    const int row_base = (tid % S::kRowThreads) * S::kMicroN;
    const int col_base = (tid / S::kRowThreads) * S::kMicroT;

    float acc[S::kMicroN][S::kMicroT];
#pragma unroll
    for (int i = 0; i < S::kMicroN; ++i)
#pragma unroll
        for (int j = 0; j < S::kMicroT; ++j) { acc[i][j] = 0.0f; }

    // Generic staging strides (kGroupK == 64 -> 8 uint32 of codes/row, 4 uint4 of int8/token).
    constexpr int kCodeU32 = kGroupK / 8;  // 8 uint32 of code bytes decode 64 nibbles -> wait: 32
    constexpr int kCodeVecs = kCodeB / 4;  // 8 uint32 code words per row-group (32 bytes)
    constexpr int kXVecs    = kGroupK / 16; // 4 uint4 of int8 activations per token-group (64 int8)
    (void)kCodeU32;

    for (int g = 0; g < groups; ++g) {
        __syncthreads();

        // decode weights: kTileN rows x kCodeVecs uint32 code words each. Grid-stride over
        // (row, word); each word (4 code bytes) decodes to 8 int8 nibbles.
        for (int u = tid; u < S::kTileN * kCodeVecs; u += S::kThreads) {
            const int row  = u / kCodeVecs;
            const int word = u % kCodeVecs;
            const int gr   = n0 + row;
            std::uint32_t packed = 0;
            if (gr < n) {
                const std::uint8_t* crow =
                    codes + (static_cast<std::int64_t>(gr) * padded_groups + g) * kCodeB;
                packed = reinterpret_cast<const std::uint32_t*>(crow)[word];
            }
            // byte b of this word -> K = word*8 + b*2 (lo), +1 (hi).
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const std::uint8_t code = static_cast<std::uint8_t>(packed >> (8 * b));
                const int lo = (static_cast<int>(code & 0x0fu) ^ 0x08) - 0x08;
                const int hi = (static_cast<int>(code >> 4) ^ 0x08) - 0x08;
                w_sh[row][word * 8 + b * 2]     = static_cast<std::int8_t>(lo);
                w_sh[row][word * 8 + b * 2 + 1] = static_cast<std::int8_t>(hi);
            }
        }
        for (int r = tid; r < S::kTileN; r += S::kThreads) {
            const int r2 = n0 + r;
            w_scale[r] =
                (r2 < n) ? __half2float(__ushort_as_half(reinterpret_cast<const std::uint16_t*>(
                               scales + static_cast<std::int64_t>(r2) * padded_groups *
                                            Q4RowSplitStorage::kScaleBytesPerGroup)[g]))
                         : 0.0f;
        }

        // copy pre-quantised int8 activations: kTileT tokens x kXVecs uint4 (16 int8) each.
        for (int u = tid; u < S::kTileT * kXVecs; u += S::kThreads) {
            const int tok = u / kXVecs;
            const int vec = u % kXVecs;
            const int col = t0 + tok;
            if (col < t) {
                const std::int64_t sbase =
                    static_cast<std::int64_t>(col) * k + g * kGroupK + vec * 16;
                *reinterpret_cast<uint4*>(&x_sh[tok][vec * 16]) =
                    *reinterpret_cast<const uint4*>(&xq[sbase]);
            } else {
                *reinterpret_cast<uint4*>(&x_sh[tok][vec * 16]) = uint4{0, 0, 0, 0};
            }
        }
        for (int c = tid; c < S::kTileT; c += S::kThreads) {
            const int c2 = t0 + c;
            x_scale[c] = (c2 < t) ? __half2float(__ushort_as_half(
                                        xs[static_cast<std::int64_t>(c2) * groups + g]))
                                  : 0.0f;
        }

        __syncthreads();

        std::int32_t iacc[S::kMicroN][S::kMicroT];
#pragma unroll
        for (int i = 0; i < S::kMicroN; ++i)
#pragma unroll
            for (int j = 0; j < S::kMicroT; ++j) { iacc[i][j] = 0; }

#pragma unroll
        for (int kk = 0; kk < kGroupK; kk += 4) {
            std::int32_t wv[S::kMicroN];
            std::int32_t xv[S::kMicroT];
#pragma unroll
            for (int i = 0; i < S::kMicroN; ++i) {
                wv[i] = *reinterpret_cast<const std::int32_t*>(&w_sh[row_base + i][kk]);
            }
#pragma unroll
            for (int j = 0; j < S::kMicroT; ++j) {
                xv[j] = *reinterpret_cast<const std::int32_t*>(&x_sh[col_base + j][kk]);
            }
#pragma unroll
            for (int i = 0; i < S::kMicroN; ++i)
#pragma unroll
                for (int j = 0; j < S::kMicroT; ++j) {
                    iacc[i][j] = __dp4a(wv[i], xv[j], iacc[i][j]);
                }
        }

#pragma unroll
        for (int i = 0; i < S::kMicroN; ++i) {
            const float ws = w_scale[row_base + i];
#pragma unroll
            for (int j = 0; j < S::kMicroT; ++j) {
                acc[i][j] += static_cast<float>(iacc[i][j]) * ws * x_scale[col_base + j];
            }
        }
    }

#pragma unroll
    for (int i = 0; i < S::kMicroN; ++i) {
        const int row = n0 + row_base + i;
        if (row >= n) { continue; }
#pragma unroll
        for (int j = 0; j < S::kMicroT; ++j) {
            const int col = t0 + col_base + j;
            if (col >= t) { continue; }
            out[static_cast<std::int64_t>(col) * out_ld + row] = __float2bfloat16(acc[i][j]);
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
