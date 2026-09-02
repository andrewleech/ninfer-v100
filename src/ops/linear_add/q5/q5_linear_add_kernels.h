#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void q5_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream);

// Dual-device MLP down projection for one card's compact column shard: a plain (non-residual)
// partial [5120,T] (see q5_linear_graph_partial.cu). Admits k=8192/9216 rather than the exact
// single-card k in {6144,17408}.
void q5_linear_graph_partial_launch(const Tensor& x, const Weight& w, Tensor& out,
                                    WorkspaceArena& ws, cudaStream_t stream);
std::size_t q5_linear_graph_partial_workspace_bytes(std::int32_t output_rows,
                                                    std::int32_t input_rows, std::int32_t min_tokens,
                                                    std::int32_t max_tokens);
void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q5_linear_add_simt_wide_t_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c24_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);

} // namespace ninfer::ops::detail
