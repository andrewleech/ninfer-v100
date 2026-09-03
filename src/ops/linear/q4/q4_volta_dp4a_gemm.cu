#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_volta_dp4a_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

// Scratch bytes for the int8 activation buffer + per-(token,group) fp16 scales.
static std::size_t q4_dp4a_scratch_bytes(std::int32_t k, std::int32_t t) noexcept {
    const std::size_t groups = static_cast<std::size_t>(k) / Q4RowSplitStorage::kGroupK;
    const std::size_t xq     = static_cast<std::size_t>(t) * k;            // int8
    const std::size_t xs     = static_cast<std::size_t>(t) * groups * 2;   // fp16
    return ((xq + 15) & ~std::size_t{15}) + xs;
}

// int8/dp4a Q4 x BF16 GEMM (see q4_volta_dp4a_gemm.cuh). Two passes: quantise activations to int8
// once, then the dp4a GEMM. `weight_row_offset` selects a contiguous row band of a parent weight,
// mirroring launch_q4_volta_mma (pointer arithmetic only).
void launch_q4_volta_dp4a(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                          cudaStream_t stream, std::int32_t weight_row_offset) {
    using S = Q4VoltaDp4aSchedule;
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t groups = k / Q4RowSplitStorage::kGroupK;
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / Q4RowSplitStorage::kGroupK;

    const std::int64_t roff = static_cast<std::int64_t>(weight_row_offset) * padded_groups;
    const auto* codes =
        static_cast<const std::uint8_t*>(w.qdata) + roff * Q4RowSplitStorage::kCodeBytesPerGroup;
    const auto* scales =
        static_cast<const std::uint8_t*>(w.scales) + roff * Q4RowSplitStorage::kScaleBytesPerGroup;

    auto scope             = ws.scope();
    const DeviceSpan sbuf  = ws.alloc_bytes(q4_dp4a_scratch_bytes(k, t));
    auto* xq               = static_cast<std::int8_t*>(sbuf.data);
    const std::size_t xq_b = (static_cast<std::size_t>(t) * k + 15) & ~std::size_t{15};
    auto* xs = reinterpret_cast<std::uint16_t*>(static_cast<std::uint8_t*>(sbuf.data) + xq_b);

    // Pass 1: quantise the whole activation matrix to int8 (once), like llama's quantize_mmq_q8_1.
    constexpr int kQWarps = S::kQuantThreads / 32;
    const int quant_blocks =
        (static_cast<int>(static_cast<std::int64_t>(t) * groups) + kQWarps - 1) / kQWarps;
    q4_dp4a_quantize_x_kernel<<<quant_blocks, S::kQuantThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), xq, xs, k, t, groups);
    CUDA_CHECK(cudaGetLastError());

    // Pass 2: the dp4a GEMM.
    const dim3 grid(static_cast<unsigned>((n + S::kTileN - 1) / S::kTileN),
                    static_cast<unsigned>((t + S::kTileT - 1) / S::kTileT));
    q4_volta_dp4a_gemm_kernel<<<grid, S::kThreads, 0, stream>>>(
        codes, scales, xq, xs, static_cast<__nv_bfloat16*>(out.data), out_ld, n, k, t,
        padded_groups, groups);
    CUDA_CHECK(cudaGetLastError());
}

bool q4_volta_dp4a_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    return n > 0 && t > 0 && k > 0 && k % Q4RowSplitStorage::kGroupK == 0;
}

std::size_t q4_volta_dp4a_workspace_bytes(std::int32_t /*n*/, std::int32_t k,
                                          std::int32_t t) noexcept {
    return q4_dp4a_scratch_bytes(k, t);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
