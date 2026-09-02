#include "runtime/engine/kv_capacity.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

void validate_curve(const SequenceCapacityCurve& curve) {
    if (curve.main_page_tokens == 0 || curve.minimum_main_page_groups == 0 ||
        curve.minimum_main_page_groups > curve.maximum_main_page_groups ||
        curve.minimum_device_reservation_bytes == 0) {
        throw std::invalid_argument("sequence capacity curve is invalid");
    }
    if (curve.minimum_main_page_groups < curve.maximum_main_page_groups &&
        curve.bytes_per_additional_main_page_group == 0) {
        throw std::invalid_argument("expandable sequence capacity curve has zero byte stride");
    }
    if (curve.minimum_secondary_device_reservation_bytes == 0 &&
        curve.secondary_bytes_per_additional_main_page_group != 0) {
        throw std::invalid_argument("secondary sequence capacity curve has no minimum reservation");
    }
    if (curve.minimum_secondary_device_reservation_bytes != 0 &&
        curve.minimum_main_page_groups < curve.maximum_main_page_groups &&
        curve.secondary_bytes_per_additional_main_page_group == 0) {
        throw std::invalid_argument("expandable secondary sequence capacity curve has zero byte stride");
    }
}

std::uint32_t fitting_pages(std::size_t budget, std::size_t minimum_bytes,
                            std::size_t stride_bytes, const SequenceCapacityCurve& curve,
                            const char* label) {
    if (budget < minimum_bytes) {
        throw std::invalid_argument(std::string(label) + " minimum Engine runtime reservation requires " +
                                    std::to_string(minimum_bytes) + " bytes, but only " +
                                    std::to_string(budget) + " bytes are available");
    }
    if (curve.minimum_main_page_groups == curve.maximum_main_page_groups) {
        return curve.minimum_main_page_groups;
    }
    const std::size_t additional = (budget - minimum_bytes) / stride_bytes;
    const std::uint64_t candidate =
        static_cast<std::uint64_t>(curve.minimum_main_page_groups) + additional;
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(candidate, curve.maximum_main_page_groups));
}

std::uint32_t explicit_page_groups(const KvCapacityPolicy& policy,
                                   const SequenceCapacityCurve& curve) {
    if (policy.explicit_tokens == 0) {
        throw std::invalid_argument("explicit KV capacity must be positive");
    }
    const std::uint64_t pages =
        1ULL + (static_cast<std::uint64_t>(policy.explicit_tokens) - 1ULL) / curve.main_page_tokens;
    if (pages > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("explicit KV page count exceeds uint32");
    }
    return static_cast<std::uint32_t>(pages);
}

} // namespace

std::size_t SequenceCapacityCurve::reservation_bytes(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const std::size_t additional =
        static_cast<std::size_t>(main_page_groups - minimum_main_page_groups);
    return checked_add(minimum_device_reservation_bytes,
                       checked_mul(additional, bytes_per_additional_main_page_group,
                                   "sequence capacity increment overflows size_t"),
                       "sequence capacity reservation overflows size_t");
}

std::size_t
SequenceCapacityCurve::secondary_reservation_bytes(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    if (minimum_secondary_device_reservation_bytes == 0) { return 0; }
    const std::size_t additional =
        static_cast<std::size_t>(main_page_groups - minimum_main_page_groups);
    return checked_add(
        minimum_secondary_device_reservation_bytes,
        checked_mul(additional, secondary_bytes_per_additional_main_page_group,
                    "secondary sequence capacity increment overflows size_t"),
        "secondary sequence capacity reservation overflows size_t");
}

std::uint32_t SequenceCapacityCurve::resolved_tokens(std::uint32_t main_page_groups) const {
    validate_curve(*this);
    if (main_page_groups < minimum_main_page_groups ||
        main_page_groups > maximum_main_page_groups) {
        throw std::invalid_argument("Main KV page count is outside the target capacity curve");
    }
    const std::uint64_t tokens = static_cast<std::uint64_t>(main_page_groups) * main_page_tokens;
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("resolved KV token capacity exceeds uint32");
    }
    return static_cast<std::uint32_t>(tokens);
}

KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                         const SequenceCapacityCurve& curve,
                                         std::size_t available_runtime_bytes,
                                         std::size_t secondary_available_runtime_bytes) {
    validate_curve(curve);

    const bool has_secondary     = curve.minimum_secondary_device_reservation_bytes != 0;
    std::uint32_t pages          = curve.minimum_main_page_groups;
    std::size_t capacity_budget  = available_runtime_bytes;
    std::size_t secondary_budget = secondary_available_runtime_bytes;
    switch (policy.mode) {
    case KvCapacityMode::Explicit:
        if (policy.automatic_headroom_bytes != 0) {
            throw std::invalid_argument("explicit KV capacity must not carry automatic headroom");
        }
        pages = explicit_page_groups(policy, curve);
        break;
    case KvCapacityMode::Automatic:
        if (available_runtime_bytes < policy.automatic_headroom_bytes) {
            throw std::invalid_argument(
                "automatic KV headroom requires " +
                std::to_string(policy.automatic_headroom_bytes) + " bytes, but only " +
                std::to_string(available_runtime_bytes) + " bytes are available after weights");
        }
        capacity_budget -= policy.automatic_headroom_bytes;
        if (has_secondary) {
            if (secondary_available_runtime_bytes < policy.automatic_headroom_bytes) {
                throw std::invalid_argument(
                    "secondary automatic KV headroom exceeds memory available after weights");
            }
            secondary_budget -= policy.automatic_headroom_bytes;
        }
        pages = fitting_pages(capacity_budget, curve.minimum_device_reservation_bytes,
                              curve.bytes_per_additional_main_page_group, curve, "primary");
        if (has_secondary) {
            pages = std::min(
                pages,
                fitting_pages(secondary_budget,
                              curve.minimum_secondary_device_reservation_bytes,
                              curve.secondary_bytes_per_additional_main_page_group, curve,
                              "secondary"));
        }
        break;
    default:
        throw std::invalid_argument("unknown KV capacity policy");
    }

    const std::size_t reservation = curve.reservation_bytes(pages);
    if (reservation > capacity_budget) {
        throw std::invalid_argument("requested Engine runtime reservation requires " +
                                    std::to_string(reservation) + " bytes, but only " +
                                    std::to_string(capacity_budget) +
                                    " bytes are available for runtime capacity");
    }
    const std::size_t secondary_reservation = curve.secondary_reservation_bytes(pages);
    if (secondary_reservation > secondary_budget) {
        throw std::invalid_argument("requested secondary Engine runtime reservation requires " +
                                    std::to_string(secondary_reservation) + " bytes, but only " +
                                    std::to_string(secondary_budget) +
                                    " bytes are available for runtime capacity");
    }

    return KvCapacityResolution{
        .mode                                 = policy.mode,
        .main_page_groups                     = pages,
        .maximum_main_page_groups             = curve.maximum_main_page_groups,
        .resolved_tokens                      = curve.resolved_tokens(pages),
        .minimum_runtime_reservation_bytes    = curve.minimum_device_reservation_bytes,
        .bytes_per_additional_main_page_group = curve.bytes_per_additional_main_page_group,
        .runtime_reservation_bytes            = reservation,
        .secondary_minimum_runtime_reservation_bytes =
            curve.minimum_secondary_device_reservation_bytes,
        .secondary_bytes_per_additional_main_page_group =
            curve.secondary_bytes_per_additional_main_page_group,
        .secondary_runtime_reservation_bytes = secondary_reservation,
        .available_after_weights_bytes        = available_runtime_bytes,
        .secondary_available_after_weights_bytes = secondary_available_runtime_bytes,
        .automatic_headroom_bytes             = policy.automatic_headroom_bytes,
        .planned_slack_bytes                  = available_runtime_bytes - reservation,
        .secondary_planned_slack_bytes =
            has_secondary ? secondary_available_runtime_bytes - secondary_reservation : 0,
    };
}

} // namespace ninfer::runtime
