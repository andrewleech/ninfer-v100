#include "core/arena.h"
#include "core/device.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/linear/q5/q5_volta_dp4a_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

static std::size_t q5_dp4a_scratch_bytes(std::int32_t k, std::int32_t t) noexcept {
    const std::size_t groups = static_cast<std::size_t>(k) / Q5RowSplitStorage::kGroupK;
    const std::size_t xq     = static_cast<std::size_t>(t) * k;
    const std::size_t xs     = static_cast<std::size_t>(t) * groups * 2;
    return ((xq + 15) & ~std::size_t{15}) + xs;
}

// int8/dp4a Q5 x BF16 GEMM (see q5_volta_dp4a_gemm.cuh). `add_residual` folds the result into `out`
// (for o_proj-style adds); the MLP-down shard uses add_residual=false and reduces separately.
void launch_q5_volta_dp4a(const Tensor& x, const Weight& w, Tensor& out, bool add_residual,
                          std::int32_t weight_row_offset, WorkspaceArena& ws, cudaStream_t stream) {
    using S = Q5VoltaDp4aSchedule;
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t groups = k / Q5RowSplitStorage::kGroupK;
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / Q5RowSplitStorage::kGroupK;

    const std::int64_t roff = static_cast<std::int64_t>(weight_row_offset) * padded_groups;
    const auto* codes =
        static_cast<const std::uint8_t*>(w.qdata) + roff * Q5RowSplitStorage::kCodeBytesPerGroup;
    const auto* high =
        static_cast<const std::uint8_t*>(w.qhigh) + roff * Q5RowSplitStorage::kHighBytesPerGroup;
    const auto* scales =
        static_cast<const std::uint8_t*>(w.scales) + roff * Q5RowSplitStorage::kScaleBytesPerGroup;

    auto scope             = ws.scope();
    const DeviceSpan sbuf  = ws.alloc_bytes(q5_dp4a_scratch_bytes(k, t));
    auto* xq               = static_cast<std::int8_t*>(sbuf.data);
    const std::size_t xq_b = (static_cast<std::size_t>(t) * k + 15) & ~std::size_t{15};
    auto* xs = reinterpret_cast<std::uint16_t*>(static_cast<std::uint8_t*>(sbuf.data) + xq_b);

    q5_dp4a_quantize_x_kernel<<<(static_cast<int>(static_cast<std::int64_t>(t) * groups) +
                                 S::kQuantThreads / 32 - 1) /
                                    (S::kQuantThreads / 32),
                                S::kQuantThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), xq, xs, k, t, groups);
    CUDA_CHECK(cudaGetLastError());

    const dim3 grid(static_cast<unsigned>((n + S::kTileN - 1) / S::kTileN),
                    static_cast<unsigned>((t + S::kTileT - 1) / S::kTileT));
    auto* out_data = static_cast<__nv_bfloat16*>(out.data);
    if (add_residual) {
        q5_volta_dp4a_gemm_kernel<true><<<grid, S::kThreads, 0, stream>>>(
            codes, high, scales, xq, xs, out_data, out_ld, n, k, t, padded_groups, groups);
    } else {
        q5_volta_dp4a_gemm_kernel<false><<<grid, S::kThreads, 0, stream>>>(
            codes, high, scales, xq, xs, out_data, out_ld, n, k, t, padded_groups, groups);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool q5_volta_dp4a_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    return n > 0 && t > 0 && k > 0 && k % Q5RowSplitStorage::kGroupK == 0;
}

std::size_t q5_volta_dp4a_workspace_bytes(std::int32_t /*n*/, std::int32_t k,
                                          std::int32_t t) noexcept {
    return q5_dp4a_scratch_bytes(k, t);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
