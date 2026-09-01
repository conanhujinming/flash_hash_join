#pragma once

#include "flash_join/join_types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace flash_join {

// Direct-address specialization for a unique, dense uint64 build domain. It
// deliberately lives outside UnchainedHashTable: the latter remains the exact
// CedarDB-style general hash table and is the fallback for duplicates, sparse
// domains, and large key ranges.
class PerfectHashTable {
public:
    static constexpr std::uint64_t kMaxRangeSpan = std::uint64_t{1} << 20U;
    static constexpr std::size_t kMaxRangeExpansion = 4;

    static std::unique_ptr<PerfectHashTable> try_build(
        const std::uint64_t* keys,
        const std::uint64_t* values,
        std::size_t size) {
        if (size == 0) {
            return nullptr;
        }

        auto min_key = keys[0];
        auto max_key = keys[0];
        for (std::size_t index = 1; index < size; ++index) {
            min_key = std::min(min_key, keys[index]);
            max_key = std::max(max_key, keys[index]);
            // The range can only grow, so large general joins can reject the
            // specialization without scanning the full build input twice.
            if (max_key - min_key > kMaxRangeSpan) {
                return nullptr;
            }
        }

        const auto range_size = static_cast<std::size_t>(max_key - min_key) + 1U;
        const auto minimum_dense_size =
            (range_size + kMaxRangeExpansion - 1U) / kMaxRangeExpansion;
        if (size < minimum_dense_size) {
            return nullptr;
        }

        auto result = std::unique_ptr<PerfectHashTable>(
            new PerfectHashTable(min_key, max_key, range_size));
        for (std::size_t index = 0; index < size; ++index) {
            const auto slot = static_cast<std::size_t>(keys[index] - min_key);
            if (result->present_[slot] != 0) {
                return nullptr;
            }
            result->present_[slot] = 1;
            result->values_[slot] = values[index];
        }
        return result;
    }

    PerfectHashTable(const PerfectHashTable&) = delete;
    PerfectHashTable& operator=(const PerfectHashTable&) = delete;

    [[nodiscard]] std::size_t count(std::uint64_t key) const noexcept {
        if (key < min_key_ || key > max_key_) {
            return 0;
        }
        return present_[static_cast<std::size_t>(key - min_key_)];
    }

    template <typename Consumer>
    std::size_t find(std::uint64_t key, Consumer&& consumer) const {
        if (key < min_key_ || key > max_key_) {
            return 0;
        }
        const auto slot = static_cast<std::size_t>(key - min_key_);
        if (present_[slot] == 0) {
            return 0;
        }
        consumer(BuildTuple{key, values_[slot]});
        return 1;
    }

private:
    PerfectHashTable(
        std::uint64_t min_key,
        std::uint64_t max_key,
        std::size_t range_size)
        : min_key_(min_key),
          max_key_(max_key),
          present_(new std::uint8_t[range_size]()),
          values_(new std::uint64_t[range_size]) {
    }

    std::uint64_t min_key_;
    std::uint64_t max_key_;
    std::unique_ptr<std::uint8_t[]> present_;
    std::unique_ptr<std::uint64_t[]> values_;
};

}  // namespace flash_join
