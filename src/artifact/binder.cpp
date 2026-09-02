#include "artifact/binder.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace ninfer::artifact {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    return (value + mask) & ~mask;
}

} // namespace

Binder::Binder(const Reader& reader)
    : reader_(reader), consumed_(reader.objects().size(), false),
      planned_(reader.objects().size(), false) {
    materialization_.object_count = reader.objects().size();
}

ObjectHandle Binder::find_unconsumed(std::string_view name) {
    const auto& objects            = reader_.objects();
    const ObjectDescriptor* object = reader_.find(name);
    if (object == nullptr) {
        throw ArtifactError("required artifact object is missing: " + std::string(name));
    }
    const auto index = static_cast<std::size_t>(object - objects.data());
    if (consumed_[index]) {
        throw ArtifactError("artifact object was bound more than once: " + std::string(name));
    }
    consumed_[index] = true;
    return ObjectHandle{index};
}

ObjectHandle Binder::require_tensor(std::string_view name, NumericFormat format,
                                    StorageLayout layout, std::span<const std::uint64_t> shape) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* tensor        = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("required tensor is a resource: " + std::string(name));
    }
    if (tensor->format != format || tensor->layout != layout ||
        !std::equal(tensor->shape.begin(), tensor->shape.end(), shape.begin(), shape.end())) {
        throw ArtifactError("tensor descriptor does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

ObjectHandle Binder::require_resource(std::string_view name, ResourceEncoding encoding) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* resource      = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("required resource is a tensor: " + std::string(name));
    }
    if (resource->encoding != encoding) {
        throw ArtifactError("resource encoding does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

const ObjectDescriptor& Binder::descriptor(ObjectHandle handle) const {
    if (handle.index >= reader_.objects().size()) {
        throw ArtifactError("artifact object handle is out of range");
    }
    return reader_.objects()[handle.index];
}

PayloadSpan Binder::payload(ObjectHandle handle) const {
    return reader_.payload(descriptor(handle));
}

void Binder::materialize_on_device(ObjectHandle handle) {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("resource cannot be materialized as a device tensor");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(tensor->name));
    }
    const std::uint64_t alignment = tensor_alignment(tensor->layout);
    const std::uint64_t offset    = align_up(materialization_.device_capacity_bytes, alignment);
    if (tensor->bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    materialization_.device_objects.push_back(
        DeviceMaterialization{.object         = handle,
                              .offset         = offset,
                              .bytes          = tensor->bytes,
                              .alignment      = alignment,
                              .primary_offset = offset,
                              .primary_bytes  = tensor->bytes});
    materialization_.device_capacity_bytes        = offset + tensor->bytes;
    materialization_.source_device_capacity_bytes = materialization_.device_capacity_bytes;
    planned_[handle.index]                        = true;
}

void Binder::shard_row_split_across_devices(ObjectHandle handle, RowSplitShardAxis axis,
                                            std::uint64_t split) {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr || tensor->layout != StorageLayout::RowSplitK128V1 ||
        tensor->shape.size() != 2) {
        throw ArtifactError("only rank-two row-split tensors can be sharded across devices");
    }
    const auto placement = std::find_if(
        materialization_.device_objects.begin(), materialization_.device_objects.end(),
        [&](const DeviceMaterialization& item) { return item.object.index == handle.index; });
    if (placement == materialization_.device_objects.end()) {
        throw ArtifactError("cross-device shard tensor is not materialized on the device");
    }
    const auto duplicate = std::find_if(
        materialization_.row_split_shards.begin(), materialization_.row_split_shards.end(),
        [&](const RowSplitShardMaterialization& item) {
            return item.object.index == handle.index;
        });
    if (duplicate != materialization_.row_split_shards.end()) {
        throw ArtifactError("tensor was assigned more than one cross-device shard");
    }

    const std::uint64_t rows = tensor->shape[0];
    const std::uint64_t cols = tensor->shape[1];
    if (axis == RowSplitShardAxis::PairedRows) {
        if ((rows % 2) != 0 || split == 0 || split >= rows / 2) {
            throw ArtifactError("paired-row shard split is outside the logical row range");
        }
    } else if (axis == RowSplitShardAxis::QkvHeadHalf) {
        // split = Q-band row count; the remaining rows are the K/V band. Each band halves by head
        // across the two cards, so both must be even. Rows are output channels; the row-split-k128
        // grouping runs along the columns and is unaffected by a row split, so no column-group check.
        if (split == 0 || split >= rows || (split % 2) != 0 || ((rows - split) % 2) != 0) {
            throw ArtifactError("qkv-head-half shard split is outside the logical row range");
        }
    } else {
        if (split == 0 || split >= cols) {
            throw ArtifactError("column shard split is outside the logical column range");
        }
        const std::array<std::uint64_t, 2> split_shape = {1, split};
        if (row_split_geometry(tensor->format, split_shape).padded_columns != split) {
            throw ArtifactError(
                "column shard split must preserve the row-split-k128 group boundary");
        }
    }

    materialization_.row_split_shards.push_back(RowSplitShardMaterialization{
        .object = handle,
        .axis   = axis,
        .split  = split,
    });
}

void Binder::retain_on_host(ObjectHandle handle) {
    const auto* resource = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("tensor cannot be retained as a host resource");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(resource->name));
    }
    materialization_.host_objects.push_back(HostMaterialization{handle});
    planned_[handle.index] = true;
}

void Binder::validate_only(ObjectHandle handle) {
    const ObjectDescriptor& object = descriptor(handle);
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(object_name(object)));
    }
    planned_[handle.index] = true;
}

MaterializationPlan Binder::finish() {
    const auto it = std::find(consumed_.begin(), consumed_.end(), false);
    if (it != consumed_.end()) {
        const auto index = static_cast<std::size_t>(it - consumed_.begin());
        throw ArtifactError("artifact object was not consumed by the selected target: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    const auto unplanned = std::find(planned_.begin(), planned_.end(), false);
    if (unplanned != planned_.end()) {
        const auto index = static_cast<std::size_t>(unplanned - planned_.begin());
        throw ArtifactError("artifact object has no materialization placement: " +
                            std::string(object_name(reader_.objects()[index])));
    }

    if (!materialization_.row_split_shards.empty()) {
        materialization_.source_device_capacity_bytes =
            materialization_.device_capacity_bytes;
        std::uint64_t primary_capacity   = 0;
        std::uint64_t secondary_capacity = 0;
        for (DeviceMaterialization& placement : materialization_.device_objects) {
            placement.primary_offset = align_up(primary_capacity, placement.alignment);
            const auto shard = std::find_if(
                materialization_.row_split_shards.begin(),
                materialization_.row_split_shards.end(),
                [&](const RowSplitShardMaterialization& item) {
                    return item.object.index == placement.object.index;
                });
            if (shard == materialization_.row_split_shards.end()) {
                placement.primary_bytes = placement.bytes;
            } else {
                const auto& tensor = std::get<TensorDescriptor>(descriptor(placement.object));
                const std::uint64_t rows = tensor.shape[0];
                const std::uint64_t cols = tensor.shape[1];
                std::array<std::uint64_t, 2> primary_shape{};
                std::array<std::uint64_t, 2> secondary_shape{};
                if (shard->axis == RowSplitShardAxis::PairedRows) {
                    const std::uint64_t intermediate = rows / 2;
                    primary_shape   = {2 * shard->split, cols};
                    secondary_shape = {2 * (intermediate - shard->split), cols};
                } else if (shard->axis == RowSplitShardAxis::QkvHeadHalf) {
                    // split = Q-band rows; each band (Q, then K/V) halves by head.
                    const std::uint64_t q_total  = shard->split;
                    const std::uint64_t kv_total = rows - shard->split;
                    primary_shape   = {q_total / 2 + kv_total / 2, cols};
                    secondary_shape = {(q_total - q_total / 2) + (kv_total - kv_total / 2), cols};
                } else {
                    primary_shape   = {rows, shard->split};
                    secondary_shape = {rows, cols - shard->split};
                }
                placement.primary_bytes =
                    row_split_geometry(tensor.format, primary_shape).encoded_bytes;
                shard->secondary_offset = align_up(secondary_capacity, placement.alignment);
                shard->secondary_bytes =
                    row_split_geometry(tensor.format, secondary_shape).encoded_bytes;
                if (shard->secondary_bytes >
                    std::numeric_limits<std::uint64_t>::max() - shard->secondary_offset) {
                    throw ArtifactError("secondary materialization plan size overflows u64");
                }
                secondary_capacity = shard->secondary_offset + shard->secondary_bytes;
            }
            if (placement.primary_bytes >
                std::numeric_limits<std::uint64_t>::max() - placement.primary_offset) {
                throw ArtifactError("primary materialization plan size overflows u64");
            }
            primary_capacity = placement.primary_offset + placement.primary_bytes;
        }
        materialization_.device_capacity_bytes           = primary_capacity;
        materialization_.secondary_device_capacity_bytes = secondary_capacity;
    }
    return std::move(materialization_);
}

} // namespace ninfer::artifact
