#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/q5/q5_launch.h"
#endif

#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {

// Dual-device MLP down projection for one card's compact column shard, split-K over the 17408
// intermediate (8192 primary / 9216 secondary). Produces a plain (non-residual) partial [5120,T];
// the two cards' partials are summed with the residual by residual_add_two. Unlike the single-card
// schedules in q5_linear_add_plan.cpp, this admits k=8192/9216 rather than the exact k in {6144,
// 17408} -- both are whole numbers of Q5 groups, so launch_q5_volta_mma admits them directly and
// runs the split-K + narrow pass internally (never the sm_86 mma_r64 kernels, which trap on Volta).
std::size_t q5_linear_graph_partial_workspace_bytes(std::int32_t output_rows,
                                                    std::int32_t input_rows, std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q5 linear graph partial: invalid column interval");
    }
    if (output_rows <= 0) {
        throw std::invalid_argument("q5 linear graph partial: output_rows must be positive");
    }
#ifdef NINFER_VOLTA_BUILD
    // launch_q5_volta_mma's fp32 accumulator is output_rows*T*4 when the shape multi-splits and 0
    // when it resolves to a single split. Splits is non-increasing in T, so if the largest-split
    // end (min_tokens) already resolves to a single split, every T in the interval does too and no
    // accumulator is ever allocated; otherwise reserve the largest possible extent
    // (output_rows*max_tokens*4), which dominates output_rows*T*4 for every T<=max_tokens.
    if (q5_volta_mma_splits(output_rows, input_rows, min_tokens) <= 1) { return 0; }
    return static_cast<std::size_t>(output_rows) * static_cast<std::size_t>(max_tokens) *
           sizeof(float);
#else
    (void)input_rows;
    throw std::logic_error("q5 linear graph partial is Volta-only");
#endif
}

void q5_linear_graph_partial_launch(const Tensor& x, const Weight& w, Tensor& out,
                                    WorkspaceArena& ws, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    // Non-residual partial: add_residual=false stores W@x straight out; weight_row_offset=0 because
    // the shard weight is already this card's compact column slice. q5_volta_mma_supported holds for
    // every 27B down shard (n=5120, k in {8192,9216} are whole Q5 groups, splits<=2 so k/splits>>
    // kKStep); guard + SIMT fallback anyway, mirroring the swiglu sibling, so a future smaller n
    // can't silently launch a malformed config.
    // Route by token width like the non-shard q5 dispatch does for the down projection (n=5120): the
    // Q5 GEMV is hardcoded to the k=5120 QKV shapes, so the down proj (k = shard intermediate) uses
    // the SIMT route at small T -- T=1 (decode) via r8_c4, larger small T via r8_c8 -- and only wide T
    // (prefill) amortises the fused Volta MMA. Small T on the MMA path was ~20% of peak (the decode
    // bottleneck; MTP verify runs the main model at draft+1, still small T).
    const std::int32_t t = x.ne[1];
    if (t == 1) {
        launch_q5_simt_r8_c4(x, w, out, stream);
    } else if (t >= 16 && q5_volta_mma_supported(out.ne[0], x.ne[0], t)) {
        launch_q5_volta_mma(x, w, out, /*add_residual=*/false, /*weight_row_offset=*/0, ws, stream);
    } else {
        launch_q5_simt_r8_c8(x, w, out, stream);
    }
#else
    (void)x;
    (void)w;
    (void)out;
    (void)ws;
    (void)stream;
    throw std::logic_error("q5 linear graph partial is Volta-only");
#endif
}

} // namespace ninfer::ops::detail
