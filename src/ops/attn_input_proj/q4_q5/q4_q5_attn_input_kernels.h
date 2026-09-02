#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Dual-device (NVLink tensor-parallel) attention projection for one card's head shard. Unlike the
// exact q4_q5 schedules above, this admits any per-card [q_rows+k_rows] Q4 and [gate_rows+v_rows] Q5
// shard: it runs two runtime-shaped Volta tensor-core GEMMs into caller-owned qk_out / gv_out (the
// caller slices q/k from qk_out and gate/v from gv_out). Mirrors the MLP q4_linear_swiglu_graph_shard
// composition. Volta-only. See docs/DUAL-V100-PORT-PLAN.md (Phase 7).
std::size_t attn_input_proj_graph_shard_workspace_bytes(std::int32_t qk_rows, std::int32_t gv_rows,
                                                        std::int32_t input_rows,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens);
void attn_input_proj_graph_shard_launch(const Tensor& x, const Weight& query_key_shard,
                                        const Weight& gate_value_shard, Tensor& qk_out,
                                        Tensor& gv_out, WorkspaceArena& ws, cudaStream_t stream);

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

void q4_q5_attn_input_grouped_mma_r16_c64_s3_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream);

void q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(const Tensor& x, const Weight& query_key_weight,
                                                    const Weight& gate_value_weight, Tensor& q,
                                                    Tensor& gate, Tensor& k, Tensor& v,
                                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
