#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);

// Dual-device MLP gate/up for one card's compact row shard of the packed gate_up weight (see
// q4_linear_swiglu_graph_shard.cu). Admits any even w.n rather than the exact [34816,5120]
// registration; out is [w.n/2, T].
void q4_linear_swiglu_graph_shard_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         WorkspaceArena& ws, cudaStream_t stream);
std::size_t q4_linear_swiglu_graph_shard_workspace_bytes(std::int32_t gate_up_rows,
                                                         std::int32_t input_rows,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens);
void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& w,
                                                          Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_small_t_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);

} // namespace ninfer::ops::detail
