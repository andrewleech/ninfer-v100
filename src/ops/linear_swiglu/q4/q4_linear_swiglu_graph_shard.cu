#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ninfer/ops/silu_mul.h"
#include "core/layout.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/q4/q4_launch.h"
#endif

#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD
// A/B toggle: NINFER_NO_DP4A=1 forces the fp16 mma route for the wide-T prefill GEMMs, for
// measuring the int8/dp4a speedup end-to-end. Read once.
static bool q4_dp4a_prefill_enabled() {
    static const bool disabled = [] {
        const char* e = std::getenv("NINFER_NO_DP4A");
        return e != nullptr && e[0] == '1';
    }();
    return !disabled;
}
#endif

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
    // The launcher materializes gate_up [gate_up_rows, T] into the arena, then launch_q4_volta_mma
    // stacks its own fp32 split-K accumulator above it. The gate_up buffer is monotonic in T. The
    // accumulator is gate_up_rows*T*4 when the shape multi-splits and 0 when it resolves to a single
    // split. Splits is non-increasing in T, so if the largest-split end (min_tokens) already
    // resolves to a single split -- which every 27B shard does, gate_up_rows>=16384 puts row_ctas
    // over the CTA budget at all T -- the accumulator is never allocated; otherwise reserve its
    // largest possible extent (gate_up_rows*max_tokens*4), which dominates gate_up_rows*T*4 for
    // every T<=max_tokens.
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gate_up_rows, max_tokens});
    // Above gate_up, whichever wide-T route runs stacks its own scratch: mma its fp32 split-K
    // accumulator (0 when the shape single-splits, as every 27B shard does), dp4a its int8
    // activation buffer. Reserve the larger so either route fits.
    std::size_t mma_extra = 0;
    if (q4_volta_mma_splits(gate_up_rows, input_rows, min_tokens) > 1) {
        mma_extra = static_cast<std::size_t>(gate_up_rows) *
                    static_cast<std::size_t>(max_tokens) * sizeof(float);
    }
    const std::size_t dp4a_extra =
        q4_volta_dp4a_workspace_bytes(gate_up_rows, input_rows, max_tokens);
    (void)layout.alloc_bytes(mma_extra > dp4a_extra ? mma_extra : dp4a_extra);
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
    // gate_up = W @ x. Route by token width the same way the non-shard q4 dispatch does (the split
    // shapes can't key into its shape table): T=1 (decode) is a matrix-vector -- use the dedicated
    // GEMV instead of wasting the tensor-core tile; small T (MTP verify runs the main model at
    // draft+1) uses the SIMT route; only wide T (prefill) amortises the fused Volta MMA. This was the
    // decode bottleneck -- the MLP is ~67% of decode and T=1 on the MMA path ran at ~20% of peak.
    if (t == 1) {
        // r4_w1 (4 rows/CTA, 1 warp/row) measured faster than r1_w8 on the SHARD gate/up shapes
        // (n=2*shard_intermediate, k=5120) -- the non-shard dispatch's r1_w8 was tuned for the full
        // n=34816. MLP phase -13%, decode +7% at n=16384.
        launch_q4_gemv_r4_w1_direct(x, w, gate_up, stream);
    } else if (t <= 4) {
        launch_q4_simt_r8_c4(x, w, gate_up, stream); // MTP dt2/dt3 verify widths (T=3,4)
    } else if (t <= 8 && q4_volta_qpn_supported(gate_up_rows, k, t)) {
        launch_q4_volta_qpn(x, w, gate_up, stream); // QPN's band (T 5-8): MTP dt4/dt5 verify widths
    } else if (q4_dp4a_prefill_enabled() && t >= kQ4Dp4aMinT &&
               q4_volta_dp4a_supported(gate_up_rows, k, t)) {
        // Wide-T prefill: the int8/dp4a route is compute-bound (~72% FMA pipe) where the fp16
        // mma route is L1/shared-bound, so it runs the gate/up GEMM ~1.4x faster on the V100.
        launch_q4_volta_dp4a(x, w, gate_up, ws, stream);
    } else if (t >= 16 && q4_volta_mma_supported(gate_up_rows, k, t)) {
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
