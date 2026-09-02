#pragma once

// ninfer::ops::detail - private launch prototype for residual_add. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/op-development.md §2.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void residual_add_launch(const Tensor& y, Tensor& x, cudaStream_t stream);

// Split-K reduce for the dual-device MLP down projection: x += a + b in place.
void residual_add_two_launch(const Tensor& a, const Tensor& b, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops::detail
