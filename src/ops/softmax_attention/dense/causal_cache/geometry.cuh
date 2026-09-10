#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale    = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits = 85 * SmallTSplitScale;
};

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2>;
// Per-card geometry for NVLink tensor-parallel attention: the 24q/4kv layer is split 12q/2kv across
// the two V100s (GQA group of 6 preserved). Same kv_heads as the 16/2 case, so scale 2. See
// docs/DUAL-V100-PORT-PLAN.md (Phase 7).
using CausalD256H12Kv2 = CausalAttentionGeometry<12, 2, 2>;
// 35B-A3B tensor-parallel attention: one card owns 8 Q heads and 1 KV head.
using CausalD256H8Kv1 = CausalAttentionGeometry<8, 1, 2>;

} // namespace ninfer::ops
