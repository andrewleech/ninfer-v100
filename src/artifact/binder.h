#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

enum class TensorPlacement : std::uint8_t {
    Device,
    ValidateOnly,
};

struct ObjectHandle {
    std::size_t index = 0;
};

struct DeviceMaterialization {
    ObjectHandle object;
    std::uint64_t offset         = 0;
    std::uint64_t bytes          = 0;
    std::uint64_t alignment      = 0;
    std::uint64_t primary_offset = 0;
    std::uint64_t primary_bytes  = 0;
};

enum class RowSplitShardAxis : std::uint8_t {
    PairedRows,
    Columns,
    // Tensor-parallel attention projection: two unequal row bands (Q of `split` rows, then K/V of the
    // remainder) each halved by head across the two cards, giving equal [rows/2, cols] shards. `split`
    // is the Q-band boundary (q_size), not a tunable point -- the head split is always 50/50.
    QkvHeadHalf,
};

struct RowSplitShardMaterialization {
    ObjectHandle object;
    RowSplitShardAxis axis          = RowSplitShardAxis::Columns;
    std::uint64_t split             = 0;
    std::uint64_t secondary_offset = 0;
    std::uint64_t secondary_bytes  = 0;
};

struct HostMaterialization {
    ObjectHandle object;
};

struct MaterializationPlan {
    std::size_t object_count                     = 0;
    std::uint64_t device_capacity_bytes          = 0;
    std::uint64_t source_device_capacity_bytes    = 0;
    std::uint64_t secondary_device_capacity_bytes = 0;
    std::vector<DeviceMaterialization> device_objects;
    std::vector<RowSplitShardMaterialization> row_split_shards;
    std::vector<HostMaterialization> host_objects;
};

class Binder {
public:
    explicit Binder(const Reader& reader);

    ObjectHandle require_tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    void materialize_on_device(ObjectHandle handle);
    void shard_row_split_across_devices(ObjectHandle handle, RowSplitShardAxis axis,
                                        std::uint64_t split);
    void retain_on_host(ObjectHandle handle);
    void validate_only(ObjectHandle handle);
    MaterializationPlan finish();

private:
    ObjectHandle find_unconsumed(std::string_view name);

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    MaterializationPlan materialization_;
};

} // namespace ninfer::artifact
