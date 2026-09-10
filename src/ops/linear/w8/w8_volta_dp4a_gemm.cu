#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear/w8/w8_volta_dp4a_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

// Scratch bytes for the int8 activation buffer + per-(token,group) fp16 scales.
static std::size_t w8_dp4a_scratch_bytes(std::int32_t k, std::int32_t t) noexcept {
    const std::size_t groups = static_cast<std::size_t>(k) / W8RowSplitStorage::kGroupK;
    const std::size_t xq     = static_cast<std::size_t>(t) * k;            // int8
    const std::size_t xs     = static_cast<std::size_t>(t) * groups * 2;   // fp16
    return ((xq + 15) & ~std::size_t{15}) + xs;
}

// Pass 1 only: quantise activations x[k,t] to int8 once into caller-owned scratch (ws is held by
// the caller across all subsequent GEMMs). Returns the int8/scale views. Used by the fused qkv
// parent to quantise ONCE and feed 4 band GEMMs, instead of re-quantising per band.
W8Dp4aActivation w8_dp4a_quantize(const Tensor& x, WorkspaceArena& ws, cudaStream_t stream) {
    using S = W8VoltaDp4aSchedule;
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t groups = k / W8RowSplitStorage::kGroupK;

    const DeviceSpan sbuf  = ws.alloc_bytes(w8_dp4a_scratch_bytes(k, t));
    auto* xq               = static_cast<std::int8_t*>(sbuf.data);
    const std::size_t xq_b = (static_cast<std::size_t>(t) * k + 15) & ~std::size_t{15};
    auto* xs = reinterpret_cast<std::uint16_t*>(static_cast<std::uint8_t*>(sbuf.data) + xq_b);

    constexpr int kQWarps = S::kQuantThreads / 32;
    const int quant_blocks =
        (static_cast<int>(static_cast<std::int64_t>(t) * groups) + kQWarps - 1) / kQWarps;
    w8_dp4a_quantize_x_kernel<<<quant_blocks, S::kQuantThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), xq, xs, k, t, groups);
    CUDA_CHECK(cudaGetLastError());
    return W8Dp4aActivation{xq, xs, k, t};
}

// Pass 2 only: the dp4a GEMM reading pre-quantised activations. `weight_row_offset` selects a
// contiguous row band of a parent weight (pointer arithmetic only) -- the fused qkv parent calls
// this once per band over the SAME activation.
void launch_w8_volta_dp4a_gemm_only(const W8Dp4aActivation& act, const Weight& w, Tensor& out,
                                    cudaStream_t stream, std::int32_t weight_row_offset,
                                    bool add_residual) {
    using S = W8VoltaDp4aSchedule;
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = act.k;
    const std::int32_t t      = act.t;
    const std::int32_t groups = k / W8RowSplitStorage::kGroupK;
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / W8RowSplitStorage::kGroupK;

    const std::int64_t roff = static_cast<std::int64_t>(weight_row_offset) * padded_groups;
    const auto* codes =
        static_cast<const std::uint8_t*>(w.qdata) + roff * W8RowSplitStorage::kCodeBytesPerGroup;
    const auto* scales =
        static_cast<const std::uint8_t*>(w.scales) + roff * W8RowSplitStorage::kScaleBytesPerGroup;

    const dim3 grid(static_cast<unsigned>((n + S::kTileN - 1) / S::kTileN),
                    static_cast<unsigned>((t + S::kTileT - 1) / S::kTileT));
    w8_volta_dp4a_gemm_kernel<<<grid, S::kThreads, 0, stream>>>(
        codes, scales, act.xq, act.xs, static_cast<__nv_bfloat16*>(out.data), out_ld, n, k, t,
        padded_groups, groups, add_residual);
    CUDA_CHECK(cudaGetLastError());
}

// Single-shot quantise + GEMM (o_proj, generic w8_dispatch). Scopes its own scratch.
void launch_w8_volta_dp4a(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                          cudaStream_t stream, std::int32_t weight_row_offset, bool add_residual) {
    auto scope = ws.scope();
    const W8Dp4aActivation act = w8_dp4a_quantize(x, ws, stream);
    launch_w8_volta_dp4a_gemm_only(act, w, out, stream, weight_row_offset, add_residual);
}

bool w8_volta_dp4a_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    return n > 0 && t > 0 && k > 0 && k % W8RowSplitStorage::kGroupK == 0;
}

std::size_t w8_volta_dp4a_workspace_bytes(std::int32_t /*n*/, std::int32_t k,
                                          std::int32_t t) noexcept {
    return w8_dp4a_scratch_bytes(k, t);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
