#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ninfer/ops/silu_mul.h"
#include "core/layout.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/q4/q4_launch.h"
#endif

#include <stdexcept>

namespace ninfer::ops::detail {

// Dual-device MLP gate/up for one card's compact row shard of the packed gate_up weight. Unlike the
// single-card schedules in q4_linear_swiglu_plan.cpp, this admits any even row count rather than the
// exact [34816,5120] registration: the two 27B shards are [16384,5120] (out 8192) and [18432,5120]
// (out 9216). It materializes gate_up through the runtime-shaped Volta tensor-core GEMM (never the
// sm_86 split_half_pair mma, which traps on Volta) and applies SwiGLU, matching the Volta compose
// the exact plan uses for its Materialized / CutlassSm70 routes.
std::size_t q4_linear_swiglu_graph_shard_workspace_bytes(std::int32_t gate_up_rows,
                                                         std::int32_t input_rows,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q4 linear_swiglu graph shard: invalid column interval");
    }
    if (gate_up_rows <= 0 || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument(
            "q4 linear_swiglu graph shard: gate_up_rows must be positive and even");
    }
#ifdef NINFER_VOLTA_BUILD
    (void)input_rows;
    // The launcher materializes gate_up [gate_up_rows, T] into the arena, then launch_q4_volta_mma
    // stacks its own fp32 split-K accumulator above it. The gate_up buffer is monotonic in T. The
    // accumulator is gate_up_rows*T*4 when the shape multi-splits and 0 when it resolves to a single
    // split -- not monotonic -- so reserve its largest possible extent (gate_up_rows*max_tokens*4)
    // rather than probing q4_volta_mma_workspace_bytes at max_tokens, which could sit at a
    // single-split 0 while an interior T multi-splits.
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gate_up_rows, max_tokens});
    (void)layout.alloc_bytes(static_cast<std::size_t>(gate_up_rows) *
                             static_cast<std::size_t>(max_tokens) * sizeof(float));
    return layout.peak_bytes();
#else
    (void)input_rows;
    throw std::logic_error("q4 linear_swiglu graph shard is Volta-only");
#endif
}

void q4_linear_swiglu_graph_shard_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         WorkspaceArena& ws, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    const std::int32_t gate_up_rows = w.n;       // 2 * shard intermediate
    const std::int32_t output_rows  = out.ne[0]; // shard intermediate
    const std::int32_t k            = x.ne[0];
    const std::int32_t t            = x.ne[1];
    if (gate_up_rows != 2 * output_rows) {
        throw std::invalid_argument(
            "q4 linear_swiglu graph shard: gate_up rows must be 2*out rows");
    }
    auto scope     = ws.scope();
    Tensor gate_up = ws.alloc(DType::BF16, {gate_up_rows, t});
    // gate_up = W @ x. Both shard geometries ([16384,5120], [18432,5120]) satisfy
    // q4_volta_mma_supported (k=5120 is a whole number of Q4 groups); fall back to the general SIMT
    // route for any shape it declines.
    if (q4_volta_mma_supported(gate_up_rows, k, t)) {
        launch_q4_volta_mma(x, w, gate_up, ws, stream);
    } else {
        launch_q4_simt_r8_c8(x, w, gate_up, stream);
    }
    silu_mul(gate_up.slice(0, 0, output_rows), gate_up.slice(0, output_rows, output_rows), out,
             stream);
#else
    (void)x;
    (void)w;
    (void)out;
    (void)ws;
    (void)stream;
    throw std::logic_error("q4 linear_swiglu graph shard is Volta-only");
#endif
}

} // namespace ninfer::ops::detail
