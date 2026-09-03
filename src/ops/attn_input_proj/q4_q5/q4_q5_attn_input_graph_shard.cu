#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q5/q5_launch.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD
static bool attn_q4_dp4a_prefill_enabled() {
    static const bool disabled = [] {
        const char* e = std::getenv("NINFER_NO_DP4A");
        return e != nullptr && e[0] == '1';
    }();
    return !disabled;
}
#endif

// One card's tensor-parallel attention projection. The 24q/4kv layer splits 12q/2kv per card, so the
// packed Q4 query_key parent [7168,5120] (q 6144 | k 1024) becomes a [3584,5120] shard (q 3072 | k
// 512) and the Q5 gate_value parent likewise. Each shard is a plain row band of its parent, so a
// runtime-shaped Volta GEMM computes it directly (never the exact q4_q5 fused schedules): qk_out =
// query_key_shard @ x, gv_out = gate_value_shard @ x. The caller slices q/k and gate/v out of them.
std::size_t attn_input_proj_graph_shard_workspace_bytes(std::int32_t qk_rows, std::int32_t gv_rows,
                                                        std::int32_t input_rows,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("attn_input_proj graph shard: invalid column interval");
    }
    if (qk_rows <= 0 || gv_rows <= 0) {
        throw std::invalid_argument("attn_input_proj graph shard: shard rows must be positive");
    }
#ifdef NINFER_VOLTA_BUILD
    // The two GEMMs run sequentially, each scoped, so the arena only needs the larger of the two
    // launchers' fp32 split-K accumulators. Each accumulator is rows*T*4 when the shape multi-splits
    // and 0 when it resolves to a single split; splits is non-increasing in T, so gate the reserve on
    // the largest-split end (min_tokens) and size it to max_tokens.
    std::size_t bytes = 0;
    if (q4_volta_mma_splits(qk_rows, input_rows, min_tokens) > 1) {
        bytes = std::max(bytes, static_cast<std::size_t>(qk_rows) *
                                    static_cast<std::size_t>(max_tokens) * sizeof(float));
    }
    // The wide-T query_key GEMM may take the dp4a route; its int8 activation scratch is scoped to
    // that GEMM, so reserve the larger of it and the mma accumulators.
    bytes = std::max(bytes, q4_volta_dp4a_workspace_bytes(qk_rows, input_rows, max_tokens));
    bytes = std::max(bytes, q5_volta_dp4a_workspace_bytes(gv_rows, input_rows, max_tokens));
    if (q5_volta_mma_splits(gv_rows, input_rows, min_tokens) > 1) {
        bytes = std::max(bytes, static_cast<std::size_t>(gv_rows) *
                                    static_cast<std::size_t>(max_tokens) * sizeof(float));
    }
    return bytes;
#else
    (void)input_rows;
    throw std::logic_error("attn_input_proj graph shard is Volta-only");
#endif
}

void attn_input_proj_graph_shard_launch(const Tensor& x, const Weight& query_key_shard,
                                        const Weight& gate_value_shard, Tensor& qk_out,
                                        Tensor& gv_out, WorkspaceArena& ws, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    // qk_out = query_key_shard @ x. Both cards' shards satisfy q4/q5_volta_mma_supported (k=5120 is a
    // whole number of groups); fall back to the general SIMT route for any shape they decline. Each
    // GEMM is scoped so its accumulator is reclaimed before the next -- safe because both launch on
    // the same stream (serialized).
    // Route by token width like the MLP shards: T=1 (decode) and small T (MTP verify = draft+1) are
    // matrix-vector-ish, so the tensor-core MMA wastes its tile -- use the GEMV / SIMT routes; only
    // wide T (prefill) amortises the MMA. The q4 GEMV (launch_gemv) is shape-generic; the q5 GEMV is
    // hardcoded to the k=5120 full QKV shapes (n in {6144,7168}), so the n=3584 gate/value shard takes
    // the SIMT route at small T.
    {
        auto scope = ws.scope();
        if (t == 1) {
            launch_q4_gemv_r1_w8_direct(x, query_key_shard, qk_out, stream);
        } else if (t <= 7) {
            launch_q4_simt_r8_c4(x, query_key_shard, qk_out, stream);
        } else if (attn_q4_dp4a_prefill_enabled() && t >= kQ4Dp4aMinT &&
                   q4_volta_dp4a_supported(qk_out.ne[0], k, t)) {
            launch_q4_volta_dp4a(x, query_key_shard, qk_out, ws, stream); // compute-bound prefill
        } else if (t >= 16 && q4_volta_mma_supported(qk_out.ne[0], k, t)) {
            launch_q4_volta_mma(x, query_key_shard, qk_out, ws, stream);
        } else {
            launch_q4_simt_r8_c8(x, query_key_shard, qk_out, stream);
        }
    }
    {
        auto scope = ws.scope();
        if (t <= 7) {
            launch_q5_simt_r8_c4(x, gate_value_shard, gv_out, stream);
        } else if (attn_q4_dp4a_prefill_enabled() && t >= kQ4Dp4aMinT &&
                   q5_volta_dp4a_supported(gv_out.ne[0], k, t)) {
            launch_q5_volta_dp4a(x, gate_value_shard, gv_out, /*add_residual=*/false, 0, ws, stream);
        } else if (t >= 16 && q5_volta_mma_supported(gv_out.ne[0], k, t)) {
            launch_q5_volta_mma(x, gate_value_shard, gv_out, /*add_residual=*/false,
                                /*weight_row_offset=*/0, ws, stream);
        } else {
            launch_q5_simt_r8_c8(x, gate_value_shard, gv_out, stream);
        }
    }
#else
    (void)x;
    (void)query_key_shard;
    (void)gate_value_shard;
    (void)qk_out;
    (void)gv_out;
    (void)ws;
    (void)stream;
    throw std::logic_error("attn_input_proj graph shard is Volta-only");
#endif
}

} // namespace ninfer::ops::detail
