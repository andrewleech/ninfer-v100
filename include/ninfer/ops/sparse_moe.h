#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct SparseMoeWeights {
    Weight router_shared_gate;
    Weight routed_gate_up;
    Weight routed_down;
    Weight shared_gate_up;
    Weight shared_down;
};

enum class SparseMoeEpilogue : std::uint8_t {
    AddResidual,
    // Expert-parallel partial: overwrite destination with this card's weighted partial sum (no
    // residual read). Used by sparse_moe_partial; the residual is added once by the orchestrator.
    WritePartial,
};

// Dual-card expert-parallel descriptor. A card owns global experts [expert_lo, expert_hi); the
// router+top-8 selection is still computed globally over all 256 experts, but only the owned subset
// is accumulated. The shared expert is computed once (primary only, include_shared=true).
struct SparseMoeShard {
    int expert_lo       = 0;    // first global expert this card owns
    int expert_hi       = 256;  // one past the last owned global expert
    bool include_shared = true; // compute the always-on shared expert here (PRIMARY ONLY)
    bool add_residual   = true; // epilogue seeds from destination (single-card) vs 0 (EP partial)
};

/**
 * Returns the transient capacity required by SparseMoe for every T in the inclusive
 * [min_tokens,max_tokens] interval. The routed QTypes are the fixed implementation profile.
 * Invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t sparse_moe_workspace_capacity_bytes(QType routed_gate_up,
                                                              QType routed_down,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Closed sparse-MoE Op for the exact future 35B-A3B geometry.
 *
 * For contiguous BF16 x [2048,T] and destination [2048,T] with T>0, 256 routed experts, top-8
 * selection, and one always-on shared expert, this Op owns router projection and selection,
 * selected routed and shared SwiGLU projections, down projections, their merge, and the
 * AddResidual epilogue independently for every token column. At an exact top-8 boundary tie the
 * lower expert id wins. destination is the only observable mutation: its incoming value is the
 * residual and its outgoing value is the BF16 sparse-MoE result plus that residual.
 *
 * The complete mathematical oracle starts from represented BF16 inputs, exact stored-weight
 * decode, and evaluates the logical formula naively in FP32/FP64. Scores, route weights, expert
 * activations, workspace representation, reduction association, and scale placement are private
 * execution choices rather than semantic rounding boundaries.
 *
 * The five weights have the exact registered shapes: BF16 router/shared gate [257,2048], routed
 * gate/up [256*1024,2048], routed down [256*2048,512], shared gate/up [1024,2048], and shared down
 * [2048,512]. Admitted codec profiles are Q4+Q5, Q4+Q6, and W8+W8 for the two routed banks; both
 * shared banks are W8. Expert e directly selects its stored row spans; no selected-weight gather
 * or repack occurs.
 *
 * Every positive T is supported.
 *
 * x, destination, all weight planes, and live workspace must be pairwise non-overlapping.
 * Execution is enqueued on stream without host synchronization. Workspace is caller-owned,
 * graph-stable transient storage and carries no state beyond the call.
 */
void sparse_moe(const Tensor& x, const SparseMoeWeights& weights, SparseMoeEpilogue epilogue,
                Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Dual-card expert-parallel partial of sparse_moe. Runs the global router + top-8 selection over all
 * 256 experts (from weights.router_shared_gate, replicated on both cards), then accumulates only the
 * selected experts whose global id falls in [shard.expert_lo, shard.expert_hi). weights.routed_gate_up
 * / routed_down are the COMPACTED shard for that expert band (local row 0 == expert_lo), so their row
 * counts are (expert_hi-expert_lo)*1024 and *2048. When shard.include_shared the always-on shared
 * expert is added (primary card only). destination is overwritten with this card's weighted partial
 * (shard.add_residual must be false for EP); the residual is reduced in once by the orchestrator.
 *
 * Only the decode (T==1) and small-T (2<=T<=46) kernel paths run; larger T is chunked through the
 * small-T kernels (the grouped prefill path is out of scope for expert-parallel in P1). Otherwise the
 * geometry, codec profiles, and determinism guarantees match sparse_moe.
 */
void sparse_moe_partial(const Tensor& x, const SparseMoeWeights& weights, const SparseMoeShard& shard,
                        Tensor& destination, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Transient capacity required by sparse_moe_partial for every T in [min_tokens,max_tokens]. A safe
 * upper bound (the partial path never exceeds the full sparse_moe workspace for the same interval).
 */
[[nodiscard]] std::size_t sparse_moe_partial_workspace_capacity_bytes(QType routed_gate_up,
                                                                      QType routed_down,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens);

} // namespace ninfer::ops
