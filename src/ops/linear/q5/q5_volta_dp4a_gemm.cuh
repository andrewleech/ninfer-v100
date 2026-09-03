#pragma once

// Q5G64 RowSplit x BF16 GEMM via int8 __dp4a on Volta (sm_70), wide-T prefill. The Q5 sibling of
// q4_volta_dp4a_gemm.cuh -- identical warp-cooperative tiling (lane=row, warp=token, activation a
// shared broadcast), just a 5-bit weight decode. Q5G64: 32 code bytes (low nibble per weight) + an
// 8-byte high-bit plane (1 bit per weight) + one fp16 scale per 64-K group. int8 weight =
// ((nibble | (high_bit<<4)) ^ 0x10) - 0x10 in [-16,15], symmetric, so the dp4a dot needs no
// sum-correction. The high bit for weight K is (high[K/8] >> (K%8)) & 1.

#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Q5VoltaDp4aSchedule {
    static constexpr int kGroupK  = Q5RowSplitStorage::kGroupK; // 64
    static constexpr int kWarp    = 32;
    static constexpr int kNWarps  = 8;
    static constexpr int kThreads = kWarp * kNWarps; // 256
    static constexpr int kTileN   = 64;
    static constexpr int kTileT   = 128;
    static constexpr int kRowsPerThread = kTileN / kWarp;   // 2
    static constexpr int kToksPerThread = kTileT / kNWarps; // 16
    static constexpr int kWordsK        = kGroupK / 4;      // 16
    static constexpr int kStrideW       = kWordsK + 1;      // 17: conflict-free weight rows
    static constexpr int kStrideX       = kWordsK;          // 16: uint4-aligned broadcast rows

    static constexpr int kQuantThreads = 256;
};

// Quantise activations x[k, t] to int8 xq[t][k] + per-(token,group) fp16 scale xs[t][groups].
__global__ __launch_bounds__(Q5VoltaDp4aSchedule::kQuantThreads) void q5_dp4a_quantize_x_kernel(
    const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ xq,
    std::uint16_t* __restrict__ xs, int k, int t, int groups) {
    constexpr int kGroupK = Q5VoltaDp4aSchedule::kGroupK;
    const int warp = static_cast<int>(blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5));
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (warp >= t * groups) { return; }
    const int token = warp / groups;
    const int g     = warp % groups;
    const std::int64_t base = static_cast<std::int64_t>(token) * k + g * kGroupK;
    const float v0 = __bfloat162float(x[base + lane]);
    const float v1 = __bfloat162float(x[base + lane + 32]);
    float amax     = fmaxf(fabsf(v0), fabsf(v1));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o)); }
    const float s   = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / s;
    std::int8_t* dst = xq + base;
    dst[lane]      = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v0 * inv))));
    dst[lane + 32] = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v1 * inv))));
    if (lane == 0) { xs[static_cast<std::int64_t>(token) * groups + g] = __half_as_ushort(__float2half(s)); }
}

// out[N, T] (+= if kAddResidual) = W[N, K] (Q5) * xq[T, K] (int8).
template <bool kAddResidual>
__global__ __launch_bounds__(Q5VoltaDp4aSchedule::kThreads) void q5_volta_dp4a_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ high,
    const std::uint8_t* __restrict__ scales, const std::int8_t* __restrict__ xq,
    const std::uint16_t* __restrict__ xs, __nv_bfloat16* __restrict__ out, int out_ld, int n, int k,
    int t, int padded_groups, int groups) {
    using S = Q5VoltaDp4aSchedule;
    constexpr int kGroupK = S::kGroupK;
    constexpr int kCodeB  = Q5RowSplitStorage::kCodeBytesPerGroup; // 32
    constexpr int kHighB  = Q5RowSplitStorage::kHighBytesPerGroup; // 8
    constexpr int kW  = S::kStrideW;
    constexpr int kXW = S::kStrideX;

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n0   = static_cast<int>(blockIdx.x) * S::kTileN;
    const int t0   = static_cast<int>(blockIdx.y) * S::kTileT;

    __shared__ __align__(16) std::int32_t w_sh[S::kTileN][kW];
    __shared__ __align__(16) std::int32_t x_sh[S::kTileT][kXW];
    __shared__ float w_scale[S::kTileN];
    __shared__ float x_scale[S::kTileT];

    float acc[S::kRowsPerThread][S::kToksPerThread];
#pragma unroll
    for (int r = 0; r < S::kRowsPerThread; ++r)
#pragma unroll
        for (int c = 0; c < S::kToksPerThread; ++c) { acc[r][c] = 0.0f; }

    for (int g = 0; g < groups; ++g) {
        __syncthreads();

        // decode weights: kTileN rows x kWordsK int32 words. Each word = 4 K: two code bytes (low
        // nibbles) + a high-bit nibble from the 8-byte high plane.
        for (int u = tid; u < S::kTileN * S::kWordsK; u += S::kThreads) {
            const int row  = u / S::kWordsK;
            const int word = u % S::kWordsK;
            const int gr   = n0 + row;
            std::uint32_t two  = 0; // two code bytes -> low nibbles for K=[4word..4word+3]
            std::uint32_t hbit = 0; // 4 high bits for the same K, in bits 0..3
            if (gr < n) {
                const std::int64_t grp = static_cast<std::int64_t>(gr) * padded_groups + g;
                two = reinterpret_cast<const std::uint16_t*>(codes + grp * kCodeB)[word];
                // high byte K/8 = (4word)/8 = word/2; the 4 bits start at 4*(word&1).
                const std::uint8_t hbyte = (high + grp * kHighB)[word >> 1];
                hbit = (static_cast<std::uint32_t>(hbyte) >> (4 * (word & 1))) & 0x0fu;
            }
            int q[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int lo = static_cast<int>((two >> (4 * i)) & 0x0fu);
                const int hi = static_cast<int>((hbit >> i) & 0x1u);
                q[i] = ((lo | (hi << 4)) ^ 0x10) - 0x10; // [-16,15]
            }
            w_sh[row][word] = (q[0] & 0xff) | ((q[1] & 0xff) << 8) | ((q[2] & 0xff) << 16) |
                              ((q[3] & 0xff) << 24);
        }
        for (int r = tid; r < S::kTileN; r += S::kThreads) {
            const int r2 = n0 + r;
            w_scale[r] =
                (r2 < n) ? __half2float(__ushort_as_half(reinterpret_cast<const std::uint16_t*>(
                               scales + static_cast<std::int64_t>(r2) * padded_groups *
                                            Q5RowSplitStorage::kScaleBytesPerGroup)[g]))
                         : 0.0f;
        }

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
            const std::int64_t o = static_cast<std::int64_t>(col) * out_ld + row;
            if constexpr (kAddResidual) {
                out[o] = __float2bfloat16(__bfloat162float(out[o]) + acc[r][c]);
            } else {
                out[o] = __float2bfloat16(acc[r][c]);
            }
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
