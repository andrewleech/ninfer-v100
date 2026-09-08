#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// RTX 5090 codec frontiers balance trace-like and independent expert distributions. The public
// workspace query starts at the earliest codec-specific prefill route.
inline constexpr std::int32_t kSparseMoePrefillWorkspaceMin = 20;
inline constexpr std::int32_t kSparseMoePrefillQ4Q5Min      = 47;
inline constexpr std::int32_t kSparseMoePrefillQ4Q6Min      = 47;
inline constexpr std::int32_t kSparseMoePrefillW8W8Min      = 20;
inline constexpr std::int32_t kSparseMoePrefillWideMin      = 768;
inline constexpr std::int32_t kSparseMoePrefillSliceMax     = 4096;
inline constexpr std::int32_t kSparseMoeRouteTileTokens     = 8;
// 257 logits padded to a 16-byte-aligned per-token stride.
inline constexpr std::int32_t kSparseMoeRouterScoreRows = 260;

struct SparseMoePrefillPlan {
    std::int32_t tokens         = 0;
    std::int32_t slice_tokens   = 0;
    std::size_t workspace_bytes = 0;
};

struct SparseMoePrefillWorkspace {
    Tensor token_ids;
    Tensor token_alpha;
    // Selection writes one rank local to a routing tile. Gather must keep this source separate
    // from the final inverse map because all threads in an assignment block consume the rank while
    // thread 0 publishes the packed column.
    Tensor local_rank;
    Tensor shared_scale;
    // Scan is the final consumer of tile_counts. The inverse map aliases its dead prefix for the
    // remaining gather/reduce lifetime; 256 * ceil(T / 8) is at least 8 * T elements.
    Tensor tile_counts;
    Tensor packed_index;
    Tensor tile_bases;
    Tensor expert_offsets;
    Tensor route_job_experts;
    Tensor route_job_columns;
    // A negative count selects the token-oriented adaptive route; its magnitude is the unused
    // grouped-route job count. Nonnegative values select the normal grouped route.
    Tensor route_job_count;

    // The three large allocations are lifetime unions:
    //   router scores FP32 <-> shared SwiGLU BF16
    //   gathered X BF16    <-> routed down output BF16
    //   routed SwiGLU BF16 <-> routed FP32 token reduction
    Tensor score_storage;
    Tensor shared_activation;
    Tensor grouped_io;
    Tensor routed_storage;
    Tensor routed_sum;

    // Volta int8/dp4a grouped-prefill scratch. grouped_i8 (the int8 gathered activation) aliases the
    // grouped_io region — it is half the size and dead before the down kernel reuses that region for
    // its bf16 output — so only the small scale planes and the int8 SwiGLU add arena growth, and only
    // on the Volta build.
    Tensor grouped_xs; // fp16 per-(slot, kHidden/64) activation scales for gate/up
    Tensor swiglu_i8;  // int8 SwiGLU activation (down input)
    Tensor swiglu_xs;  // fp16 per-(slot, kIntermediate/64) activation scales for down
};

// The int8/dp4a grouped path (Volta, Q4 gate/up + Q5/Q6 down) reserves extra scratch; nothing else
// does. Gating the reservation on the codec keeps it off the tight W8/W8 (MTP) leaf and non-Volta.
[[nodiscard]] inline bool sparse_moe_prefill_wants_dp4a_scratch(QType routed_gate_up,
                                                                QType routed_down) noexcept {
#ifdef NINFER_VOLTA_BUILD
    return routed_gate_up == QType::Q4G64_F16S &&
           (routed_down == QType::Q5G64_F16S || routed_down == QType::Q6G64_F16S);
#else
    (void)routed_gate_up;
    (void)routed_down;
    return false;
#endif
}

template <class Arena>
SparseMoePrefillWorkspace allocate_sparse_moe_prefill_workspace(Arena& arena,
                                                                std::int32_t capacity_tokens,
                                                                bool dp4a_scratch = false) {
    SparseMoePrefillWorkspace out;
    const std::int32_t assignments = 8 * capacity_tokens;
    const std::int32_t route_tiles =
        (capacity_tokens + kSparseMoeRouteTileTokens - 1) / kSparseMoeRouteTileTokens;

    out.token_ids      = arena.alloc(DType::I32, {assignments}, 256);
    out.token_alpha    = arena.alloc(DType::FP32, {assignments}, 256);
    out.local_rank     = arena.alloc(DType::I32, {assignments}, 256);
    out.shared_scale   = arena.alloc(DType::FP32, {capacity_tokens}, 256);
    out.tile_counts    = arena.alloc(DType::I32, {256, route_tiles}, 256);
    out.packed_index   = Tensor(out.tile_counts.data, DType::I32, {assignments});
    out.tile_bases     = arena.alloc(DType::I32, {256, route_tiles}, 256);
    out.expert_offsets = arena.alloc(DType::I32, {257}, 256);
    // A route job is one nonempty expert column tile. The bound is for the
    // narrowest prefill tile (32 assignments) and includes every expert tail.
    const std::int32_t max_route_jobs = assignments / 32 + 256;
    out.route_job_experts             = arena.alloc(DType::I32, {max_route_jobs}, 256);
    out.route_job_columns             = arena.alloc(DType::I32, {max_route_jobs}, 256);
    out.route_job_count               = arena.alloc(DType::I32, {1}, 256);

    out.score_storage = arena.alloc(DType::FP32, {kSparseMoeRouterScoreRows, capacity_tokens}, 256);
    out.shared_activation = Tensor(out.score_storage.data, DType::BF16, {512, capacity_tokens});

    out.grouped_io = arena.alloc(DType::BF16, {2048, assignments}, 256);

    out.routed_storage = arena.alloc(DType::BF16, {512, assignments}, 256);
    out.routed_sum     = Tensor(out.routed_storage.data, DType::FP32, {2048, capacity_tokens});

    // grouped_i8 aliases grouped_io (int8 view of the same region); only the scale planes and the
    // int8 SwiGLU are fresh. Reserved only for the dp4a-eligible profile so the W8/W8 (MTP) leaf and
    // non-Volta builds keep their exact footprint.
    if (dp4a_scratch) {
        out.grouped_xs = arena.alloc(DType::FP16, {assignments, 2048 / 64}, 256);
        out.swiglu_i8  = arena.alloc(DType::I8, {assignments, 512}, 256);
        out.swiglu_xs  = arena.alloc(DType::FP16, {assignments, 512 / 64}, 256);
    }
    return out;
}

[[nodiscard]] bool sparse_moe_uses_prefill(std::int32_t tokens, QType routed_gate_up,
                                           QType routed_down) noexcept;
[[nodiscard]] std::size_t sparse_moe_prefill_workspace_bytes(std::int32_t max_tokens,
                                                             bool dp4a_scratch = false);
[[nodiscard]] SparseMoePrefillPlan
resolve_sparse_moe_prefill_plan(std::int32_t tokens, QType routed_gate_up, QType routed_down);

// shard defaults to the whole 256-expert set with the shared expert and the residual add (a strict
// specialization of the single-card path). A subset shard [expert_lo,expert_hi) with add_residual =
// false runs the dual-card expert-parallel partial: the routed banks are the compacted shard, the
// shared expert + residual are the owning (primary) card's, and the destination receives this card's
// weighted partial for the orchestrator to reduce over NVLink.
void sparse_moe_prefill_launch(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoePrefillPlan& plan,
                               const SparseMoePrefillWorkspace& workspace, cudaStream_t stream,
                               SparseMoeShard shard = {});

} // namespace ninfer::ops::detail
