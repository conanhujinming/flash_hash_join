#pragma once

#include "flash_join/hash.hpp"
#include "flash_join/join_types.hpp"
#include "flash_join/parallel.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace flash_join {

// CedarDB's unchained join table. When Bloom filtering is enabled, each 64-bit
// directory entry packs the end offset of a dense collision range into its
// high 48 bits and a 16-bit register-blocked Bloom filter into its low bits.
// The preceding entry stores the range start, so a hit is scanned sequentially
// without pointer chasing.
template <bool UseBloomFilter = true>
class UnchainedHashTable {
public:
    UnchainedHashTable(
        const std::uint64_t* keys,
        const std::uint64_t* values,
        std::size_t size,
        std::size_t requested_threads)
        : size_(size), capacity_(directory_capacity(size)), shift_(slot_shift(capacity_)) {
        if constexpr (UseBloomFilter) {
            if (size_ >= (std::uint64_t{1} << 44U)) {
                throw std::length_error("build side is too large for a 48-bit packed byte offset");
            }
        }
        build(keys, values, requested_threads);
    }

    UnchainedHashTable(const UnchainedHashTable&) = delete;
    UnchainedHashTable& operator=(const UnchainedHashTable&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    [[nodiscard]] std::size_t count(std::uint64_t key) const noexcept {
        const auto hash = hash64(key);
        const auto slot = slot_for(hash);
        const auto entry = directory_[slot];
        if constexpr (UseBloomFilter) {
            if (!could_contain(entry, hash)) {
                return 0;
            }
        }

        const auto* current = tuple_at(directory_[slot - 1] >> kMetadataBits);
        const auto* end = tuple_at(entry >> kMetadataBits);
        std::size_t matches = 0;
        for (; current != end; ++current) {
            matches += static_cast<std::size_t>(current->key == key);
        }
        return matches;
    }

    template <typename Consumer>
    std::size_t find(std::uint64_t key, Consumer&& consumer) const {
        const auto hash = hash64(key);
        const auto slot = slot_for(hash);
        const auto entry = directory_[slot];
        if constexpr (UseBloomFilter) {
            if (!could_contain(entry, hash)) {
                return 0;
            }
        }

        const auto* current = tuple_at(directory_[slot - 1] >> kMetadataBits);
        const auto* end = tuple_at(entry >> kMetadataBits);
        std::size_t matches = 0;
        for (; current != end; ++current) {
            if (current->key == key) {
                consumer(*current);
                ++matches;
            }
        }
        return matches;
    }

private:
    struct HashedTuple {
        std::uint64_t hash;
        std::uint64_t key;
        std::uint64_t value;
    };

    static constexpr unsigned kMetadataBits = UseBloomFilter ? 16U : 0U;
    static constexpr std::uint64_t kTagMask = UseBloomFilter
        ? (std::uint64_t{1} << kMetadataBits) - 1U
        : 0U;
    static constexpr std::size_t kTargetPartitions = 256;
    static constexpr std::size_t kBuildGrain = 32U * 1024U;
    static constexpr std::size_t kScatterBufferTuples = 4;

    struct ScatterBuffer {
        std::array<HashedTuple, kScatterBufferTuples> tuples{};
        std::size_t size = 0;
    };

    static std::size_t next_power_of_two(std::size_t value) {
        if (value <= 1) {
            return 1;
        }
        --value;
        for (std::size_t shift = 1; shift < sizeof(value) * 8; shift <<= 1U) {
            value |= value >> shift;
        }
        if (value == std::numeric_limits<std::size_t>::max()) {
            throw std::length_error("hash directory capacity overflow");
        }
        return value + 1;
    }

    static std::size_t directory_capacity(std::size_t tuple_count) {
        if (tuple_count == 0) {
            return 1;
        }
        const auto extra = tuple_count / 8U + static_cast<std::size_t>(tuple_count % 8U != 0);
        if (tuple_count > std::numeric_limits<std::size_t>::max() - extra) {
            throw std::length_error("hash directory capacity overflow");
        }
        // m = 2^ceil(log2(1.125 n)), as evaluated in the CedarDB paper.
        return next_power_of_two(tuple_count + extra);
    }

    static unsigned slot_shift(std::size_t capacity) noexcept {
        if (capacity == 1) {
            return 64;
        }
        unsigned index_bits = 0;
        for (auto value = capacity; value > 1; value >>= 1U) {
            ++index_bits;
        }
        return 64U - index_bits;
    }

    [[nodiscard]] std::size_t slot_for(std::uint64_t hash) const noexcept {
        return capacity_ == 1 ? 0 : static_cast<std::size_t>(hash >> shift_);
    }

    static bool could_contain(std::uint64_t entry, std::uint64_t hash) noexcept {
        const auto tag = bloom_tag(hash);
        const auto filter = static_cast<std::uint16_t>(entry & kTagMask);
        return static_cast<std::uint16_t>(tag & static_cast<std::uint16_t>(~filter)) == 0;
    }

    [[nodiscard]] const BuildTuple* tuple_at(std::uint64_t byte_offset) const noexcept {
        const auto* base = reinterpret_cast<const std::uint8_t*>(tuples_.get());
        return reinterpret_cast<const BuildTuple*>(base + byte_offset);
    }

    [[nodiscard]] BuildTuple* tuple_at(std::uint64_t byte_offset) noexcept {
        auto* base = reinterpret_cast<std::uint8_t*>(tuples_.get());
        return reinterpret_cast<BuildTuple*>(base + byte_offset);
    }

    void build(
        const std::uint64_t* keys,
        const std::uint64_t* values,
        std::size_t requested_threads) {
        directory_storage_ = std::unique_ptr<std::uint64_t[]>(new std::uint64_t[capacity_ + 1]());
        directory_ = directory_storage_.get() + 1;
        directory_[-1] = 0;  // Branch-free sentinel for slot zero.
        if (size_ == 0) {
            return;
        }

        tuples_ = std::unique_ptr<BuildTuple[]>(new BuildTuple[size_]);
        const auto thread_count = parallel::effective_thread_count(
            requested_threads, size_, kBuildGrain);
        // Partition ownership exists to remove synchronization from a parallel
        // build. With one worker it adds no value, so small builds go straight
        // from collected tuples to the final count/prefix/copy stage.
        const auto partition_count = thread_count == 1
            ? std::size_t{1}
            : std::min(capacity_, kTargetPartitions);
        const auto partition_shift = slot_shift(partition_count);
        const auto partition_for = [partition_count, partition_shift](std::uint64_t hash) {
            return partition_count == 1
                ? std::size_t{0}
                : static_cast<std::size_t>(hash >> partition_shift);
        };

        std::vector<std::size_t> partition_offsets(partition_count + 1, 0);
        auto partitioned = std::unique_ptr<HashedTuple[]>(new HashedTuple[size_]);
        if (thread_count == 1) {
            partition_offsets[1] = size_;
            for (std::size_t index = 0; index < size_; ++index) {
                partitioned[index] = HashedTuple{hash64(keys[index]), keys[index], values[index]};
            }
        } else {
            std::vector<std::vector<std::size_t>> histograms(
                thread_count, std::vector<std::size_t>(partition_count, 0));
            auto hashes = std::unique_ptr<std::uint64_t[]>(new std::uint64_t[size_]);

            // Tuple collection: hash once and create thread-local partition counts.
            parallel::for_chunks(size_, thread_count, [&](std::size_t thread, std::size_t begin, std::size_t end) {
                auto& histogram = histograms[thread];
                for (auto index = begin; index < end; ++index) {
                    const auto hash = hash64(keys[index]);
                    hashes[index] = hash;
                    ++histogram[partition_for(hash)];
                }
            });

            std::vector<std::vector<std::size_t>> write_positions(
                thread_count, std::vector<std::size_t>(partition_count, 0));
            for (std::size_t partition = 0; partition < partition_count; ++partition) {
                auto cursor = partition_offsets[partition];
                for (std::size_t thread = 0; thread < thread_count; ++thread) {
                    write_positions[thread][partition] = cursor;
                    cursor += histograms[thread][partition];
                }
                partition_offsets[partition + 1] = cursor;
            }

            parallel::for_chunks(size_, thread_count, [&](std::size_t thread, std::size_t begin, std::size_t end) {
                auto positions = write_positions[thread];
                std::vector<ScatterBuffer> buffers(partition_count);
                const auto flush = [&](std::size_t partition) {
                    auto& buffer = buffers[partition];
                    std::copy_n(buffer.tuples.data(), buffer.size, partitioned.get() + positions[partition]);
                    positions[partition] += buffer.size;
                    buffer.size = 0;
                };

                for (auto index = begin; index < end; ++index) {
                    const auto hash = hashes[index];
                    const auto partition = partition_for(hash);
                    auto& buffer = buffers[partition];
                    buffer.tuples[buffer.size++] = HashedTuple{hash, keys[index], values[index]};
                    if (buffer.size == kScatterBufferTuples) {
                        flush(partition);
                    }
                }
                for (std::size_t partition = 0; partition < partition_count; ++partition) {
                    if (buffers[partition].size != 0) {
                        flush(partition);
                    }
                }
            });
        }

        // Each coarse partition owns a disjoint directory range. It can count,
        // prefix-sum, OR Bloom tags, and copy without locks or atomics.
        parallel::for_tasks(partition_count, thread_count, [&](std::size_t partition) {
            const auto tuple_begin = partition_offsets[partition];
            const auto tuple_end = partition_offsets[partition + 1];
            for (auto index = tuple_begin; index < tuple_end; ++index) {
                const auto slot = slot_for(partitioned[index].hash);
                directory_[slot] += sizeof(BuildTuple) << kMetadataBits;
                if constexpr (UseBloomFilter) {
                    directory_[slot] |= bloom_tag(partitioned[index].hash);
                }
            }

            const auto slot_begin = capacity_ * partition / partition_count;
            const auto slot_end = capacity_ * (partition + 1) / partition_count;
            std::uint64_t cursor = static_cast<std::uint64_t>(tuple_begin * sizeof(BuildTuple));
            for (auto slot = slot_begin; slot < slot_end; ++slot) {
                const auto count_in_bytes = directory_[slot] >> kMetadataBits;
                const auto filter = directory_[slot] & kTagMask;
                directory_[slot] = (cursor << kMetadataBits) | filter;
                cursor += count_in_bytes;
            }

            for (auto index = tuple_begin; index < tuple_end; ++index) {
                const auto& tuple = partitioned[index];
                const auto slot = slot_for(tuple.hash);
                const auto target = directory_[slot] >> kMetadataBits;
                *tuple_at(target) = BuildTuple{tuple.key, tuple.value};
                directory_[slot] += sizeof(BuildTuple) << kMetadataBits;
            }
            assert(cursor == static_cast<std::uint64_t>(tuple_end * sizeof(BuildTuple)));
        });
    }

    std::size_t size_;
    std::size_t capacity_;
    unsigned shift_;
    std::unique_ptr<std::uint64_t[]> directory_storage_;
    std::uint64_t* directory_ = nullptr;
    std::unique_ptr<BuildTuple[]> tuples_;
};

}  // namespace flash_join
