#pragma once

// ninfer::ops - fused residual += W @ x.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the A16-only transient capacity required by LinearAdd for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Op: linear_add
 *
 * Math / indexing:
 *   ideal[:,t] = residual[:,t] + Linear(x,w)[:,t].
 *
 * Logical shapes:
 *   Contiguous BF16 x [K,T] and residual [N,T]. Registered weights are Q5G64_F16S RowSplit
 *   [5120,17408] or [5120,6144], W8G32_F16S RowSplit [2048,4096] or [2048,6144], NVFP4
 *   BlockScaleK16M128x4 [5120,6144] or [5120,17408], row-scaled
 *   FP8_E4M3FN_ROW_BF16S [5120,6144] or [5120,17408], or BF16_CTRL Contiguous [5120,6144]. T may
 *   be any positive value.
 *
 * Numeric:
 *   The oracle reads a registered BF16 weight directly or exact-decodes a registered packed
 *   weight, then evaluates `ideal` naively in FP64 from the represented inputs. The updated BF16
 *   residual is promoted and compared directly with that result; output storage rounding belongs
 *   to LinearAdd's selected A16, A8, or A4 criterion, not the oracle. Production routes may fuse
 *   or materialize the projection and may choose their natural accumulator, activation
 *   quantization, staging, and workspace precision; those private choices are not semantic
 *   rounding boundaries.
 *
 * Compute policy:
 *   Q5, W8, and BF16_CTRL admit only A16Only. NVFP4 admits A16Only and AllowA4. Row-scaled FP8
 *   admits A16Only and AllowA8. Its two semantic registrations own independent production plans:
 *   [5120,6144] resolves T<22 to A16 and T>=22 to A8, while [5120,17408] resolves T<25 to A16 and
 *   T>=25 to A8. A permissive policy allows the private resolver to select either qualified
 *   arithmetic profile; it does not itself prescribe a kernel.
 *
 * Effects:
 *   Updates the full residual tensor in place; x/weight must not alias residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_add_workspace_capacity_bytes(), scoped to
 *   the call. A16 routes require no storage; quantized-activation routes use the reported capacity.
 *   There is no persistent state side effect.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream);

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream);

/**
 * Transient capacity for linear_partial over every T in [min_tokens,max_tokens]. Volta-only.
 */
[[nodiscard]] std::size_t linear_partial_workspace_bytes(std::int32_t output_rows,
                                                         std::int32_t input_rows,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens);

/**
 * Op: linear_partial (dual-device MLP down projection for one card's column shard)
 *
 * Math / indexing:
 *   out[:,t] = Linear(x, w)[:,t]   (plain projection, NO residual add).
 *
 * The split-K counterpart to linear_add: each card computes a non-residual partial over its slice
 * of the 17408 intermediate (k=8192 primary / 9216 secondary), and the two partials are summed with
 * the residual by residual_add_two. Admits any Q5G64_F16S RowSplit weight `[N,K]` whose K is a whole
 * number of Q5 groups (the single-card linear_add restricts to K in {6144,17408}). Runtime-shaped
 * Volta tensor-core route, so Volta-only. Contiguous BF16 x `[K,T]` and out `[N,T]`; x/weight and
 * out must not alias. Caller-owned transient storage reported by linear_partial_workspace_bytes().
 */
void linear_partial(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                    cudaStream_t stream);

} // namespace ninfer::ops
