#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

using W8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

#ifdef NINFER_VOLTA_BUILD
// Fused-dequant tensor-core route (w8_volta_mma_gemm.cuh). Unlike its Q4/Q5 siblings this one
// fits the W8Launch signature, because the shapes it is selected for supply far more CTAs than
// the machine holds resident and so need no split-K, and therefore no workspace.
void launch_w8_volta_mma(const Tensor&, const Weight&, Tensor&, cudaStream_t);
[[nodiscard]] bool w8_volta_mma_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept;

// Quadpair-split-N form (w8_volta_qpn_gemm.cuh), for the narrow verify widths the 32x8 route
// pads its A rows away on. Also workspace-free: the CTA's warps split K and reduce in shared.
void launch_w8_volta_qpn(const Tensor&, const Weight&, Tensor&, cudaStream_t);
[[nodiscard]] bool w8_volta_qpn_supported(std::int32_t n, std::int32_t k,
                                          std::int32_t t) noexcept;

// int8/dp4a route (w8_volta_dp4a_gemm.cuh) -- the MMQ-style prefill GEMM that takes the W8
// projections compute-bound (see the ncu note in q4_volta_dp4a_gemm.cuh; W8's mma route is the
// same L1/shared-bound shape). NEEDS an int8-activation scratch, so unlike the mma/qpn routes it
// carries a WorkspaceArena& and cannot go in the plain W8Launch table. `weight_row_offset` selects
// a contiguous row band of a parent weight (attn_input_proj splits one parent qkv weight).
// `add_residual` folds `out += W*x` in the epilogue (out must already hold the residual), for the
// linear_add / o_proj call site; false does a plain store.
void launch_w8_volta_dp4a(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                          cudaStream_t stream, std::int32_t weight_row_offset = 0,
                          bool add_residual = false);
// Quantise-once variant for a fused parent (e.g. qkv): quantise the activation ONCE into scratch,
// then run one gemm_only per row band over the SAME int8 activation -- avoids re-quantising per band
// (which flattened the qkv win in the first cut). Caller holds ws.scope() across all band GEMMs.
struct W8Dp4aActivation {
    const std::int8_t* xq;    // int8 activation [t, k]
    const std::uint16_t* xs;  // fp16 per-(token,group) scales [t, groups]
    std::int32_t k;
    std::int32_t t;
};
W8Dp4aActivation w8_dp4a_quantize(const Tensor& x, WorkspaceArena& ws, cudaStream_t stream);
void launch_w8_volta_dp4a_gemm_only(const W8Dp4aActivation& act, const Weight& w, Tensor& out,
                                    cudaStream_t stream, std::int32_t weight_row_offset = 0,
                                    bool add_residual = false);
[[nodiscard]] bool w8_volta_dp4a_supported(std::int32_t n, std::int32_t k,
                                           std::int32_t t) noexcept;
[[nodiscard]] std::size_t w8_volta_dp4a_workspace_bytes(std::int32_t n, std::int32_t k,
                                                        std::int32_t t) noexcept;
// Token width at/above which prefill prefers the int8/dp4a route over fp16 mma: dp4a's 128-token
// tile wants a wide T; decode/verify widths stay on mma/qpn. Prefill runs at chunk 2048.
constexpr std::int32_t kW8Dp4aMinT = 128;
#endif

void launch_w8_decode_r4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_small_t(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_t_splitk(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_t_composite(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_dflash_medium(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_medium_splitk_c144(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_simt_r8_c4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_simt_r8_c8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_simt_r4_c16(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_mma_r32_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r32_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r32_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c112(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c112(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r96_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r128_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r128_c80(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64x16_c48_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_exact_mma_r32_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r32_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r48_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r48_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r64_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r96_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r128_c80(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
