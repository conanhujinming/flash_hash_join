#include <iostream>
#include <vector>
#include <cstdint>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <functional>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <array>
#include <mutex>

// --- Intrinsics Headers ---
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <nmmintrin.h> // For CRC32 and prefetch

// --- Pybind11 Headers ---
#define PYBIND11_NO_ATEXIT
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// --- Mimalloc Headers ---
#define MI_OVERRIDE 0
#define MI_MALLOC_OVERRIDE 0
#define MI_NO_DEINIT 1
#include <mimalloc.h>

namespace py = pybind11;

// ==========================================================================================
// SECTION 1: CORE UTILITIES & HASH TABLE
// ==========================================================================================
constexpr size_t RADIX_BITS = 8;
constexpr size_t NUM_PARTITIONS = 1 << RADIX_BITS;

uint64_t hash64(uint64_t key, uint32_t seed) {
    uint64_t k = 0x8648DBDB;
    uint64_t crc = _mm_crc32_u64(seed, key);
    return crc * ((k << 32) + 1);
}

class SimpleTimer {
public:
    SimpleTimer() : start_time_(std::chrono::high_resolution_clock::now()) {}
    double elapsed_seconds() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time_;
        return elapsed.count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

template<typename KeyType>
struct Hasher {
    uint64_t operator()(const KeyType& key) const { return hash64(key, 0xAAAAAAAA); }
};

namespace internal {
    constexpr size_t TAGS_TABLE_SIZE = 1 << 11;
    constexpr auto create_tags_table() {
        std::array<uint16_t, TAGS_TABLE_SIZE> table{};
        for (uint32_t i = 0; i < TAGS_TABLE_SIZE; ++i) {
            uint32_t h = i * 0x9E3779B9;
            uint16_t b1 = 1 << ((h >> 0) & 15);
            uint16_t b2 = 1 << ((h >> 8) & 15);
            uint16_t b3 = 1 << ((h >> 16) & 15);
            uint16_t b4 = 1 << ((h >> 24) & 15);
            table[i] = b1 | b2 | b3 | b4;
        }
        return table;
    }
}

template<bool UseBloomFilter = false>
class FlashHashTable {
public:
    static constexpr uint8_t EMPTY_TAG = 0xFF;
    static constexpr size_t SIMD_WIDTH = 32;
private:
    struct alignas(16) Slot {
        std::atomic<uint8_t> tag;
        uint64_t key;
        uint64_t value;
    };
    std::unique_ptr<Slot[]> slots_;
    std::unique_ptr<std::atomic<uint16_t>[]> bloom_directory_;
    size_t capacity_;
    size_t capacity_mask_;
    Hasher<uint64_t> hasher_;
    static inline const auto tags_table_ = internal::create_tags_table();
public:
    FlashHashTable(const FlashHashTable&) = delete;
    FlashHashTable& operator=(const FlashHashTable&) = delete;
    static size_t calculate_power_of_2(size_t n) { return n == 0 ? 1 : 1UL << (64 - __builtin_clzll(n - 1)); }

    FlashHashTable(size_t build_size) {
        size_t capacity = calculate_power_of_2(build_size * 1.5 + SIMD_WIDTH);
        this->capacity_ = capacity;
        this->capacity_mask_ = capacity - 1;
        const size_t alloc_size = capacity_ + SIMD_WIDTH - 1;
        slots_ = std::make_unique<Slot[]>(alloc_size);
        for (size_t i = 0; i < capacity_; ++i) { slots_[i].tag.store(EMPTY_TAG, std::memory_order_relaxed); }
        if constexpr (UseBloomFilter) {
            size_t bloom_capacity = capacity_;
            bloom_directory_ = std::make_unique<std::atomic<uint16_t>[]>(bloom_capacity);
            for (size_t i = 0; i < bloom_capacity; ++i) { bloom_directory_[i].store(0, std::memory_order_relaxed); }
        }
    }

    void insert_local(const uint64_t& key, const uint64_t& value) {
        uint64_t hash = this->hasher_(key);
        uint8_t tag = (hash >> 56); if (tag == EMPTY_TAG) tag = 0;
        size_t pos = hash & this->capacity_mask_;
        const size_t initial_pos = pos;
        do {
            if (slots_[pos].tag.load(std::memory_order_relaxed) == EMPTY_TAG) {
                slots_[pos].key = key;
                slots_[pos].value = value;
                slots_[pos].tag.store(tag, std::memory_order_release);
                if constexpr (UseBloomFilter) { bloom_directory_[initial_pos].fetch_or(get_bloom_tag(hash), std::memory_order_relaxed); }
                return;
            }
            if (slots_[pos].key == key) return;
            pos = (pos + 1) & this->capacity_mask_;
        } while (pos != initial_pos);
    }

    void insert_concurrent(const uint64_t& key, const uint64_t& value) {
        uint64_t hash = this->hasher_(key);
        uint8_t tag = (hash >> 56); if (tag == EMPTY_TAG) tag = 0;
        size_t pos = hash & this->capacity_mask_;
        const size_t initial_pos = pos;
        while (true) {
            uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
            if (current_tag == EMPTY_TAG) {
                uint8_t expected_empty = EMPTY_TAG;
                if (slots_[pos].tag.compare_exchange_strong(expected_empty, tag, std::memory_order_acq_rel)) {
                    slots_[pos].key = key;
                    slots_[pos].value = value;
                    if constexpr (UseBloomFilter) { bloom_directory_[initial_pos].fetch_or(get_bloom_tag(hash), std::memory_order_relaxed); }
                    return;
                }
                continue;
            }
            if (current_tag == tag && slots_[pos].key == key) return;
            pos = (pos + 1) & this->capacity_mask_;
            if (pos == initial_pos) return;
        }
    }

    size_t probe_vectorized(const uint64_t* keys, size_t batch_size, uint32_t* out_probe_indices, uint64_t* out_build_values) const {
        size_t match_count = 0;
        constexpr size_t PREFETCH_DISTANCE = 8;
        for (size_t i = 0; i < batch_size; ++i) {
            if (i + PREFETCH_DISTANCE < batch_size) {
                const uint64_t prefetch_key = keys[i + PREFETCH_DISTANCE];
                const uint64_t prefetch_hash = this->hasher_(prefetch_key);
                const size_t prefetch_pos = prefetch_hash & this->capacity_mask_;
                _mm_prefetch(reinterpret_cast<const char*>(&slots_[prefetch_pos]), _MM_HINT_T0);
            }
            const uint64_t key = keys[i];
            uint64_t hash = this->hasher_(key);
            if constexpr (UseBloomFilter) { if (!check_bloom_filter(hash)) continue; }
            uint8_t tag = (hash >> 56); if (tag == EMPTY_TAG) tag = 0;
            size_t pos = hash & this->capacity_mask_;
            const size_t initial_pos = pos;
            do {
                uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
                if (current_tag == EMPTY_TAG) break;
                if (current_tag == tag && slots_[pos].key == key) {
                    out_probe_indices[match_count] = i;
                    out_build_values[match_count] = slots_[pos].value;
                    match_count++;
                    break;
                }
                pos = (pos + 1) & this->capacity_mask_;
            } while (pos != initial_pos);
        }
        return match_count;
    }

    inline uint16_t get_bloom_tag(uint64_t hash) const { return tags_table_[(static_cast<uint32_t>(hash)) >> (32 - 11)]; }

    inline bool check_bloom_filter(uint64_t hash) const {
        size_t slot = (hash & this->capacity_mask_);
        const uint16_t entry = bloom_directory_[slot].load(std::memory_order_relaxed);
        return (get_bloom_tag(hash) & entry) == get_bloom_tag(hash);
    }

    void build_local(const uint64_t* keys, const uint64_t* values, size_t size) { for (size_t i = 0; i < size; ++i) { this->insert_local(keys[i], values[i]); } }
    
    void build_concurrent(const uint64_t* keys, const uint64_t* values, size_t size) {
        size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
        size_t work_per_thread = (size + num_threads - 1) / num_threads;
        std::vector<std::thread> threads;
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, size);
            if (start >= end) continue;
            threads.emplace_back([this, keys, values, start, end]() { for (size_t j = start; j < end; ++j) { this->insert_concurrent(keys[j], values[j]); } });
        }
        for (auto& t : threads) { t.join(); }
    }
};

// ==========================================================================================
// SECTION 2: OPTIMIZED RADIX PARTITIONING
// ==========================================================================================
inline size_t get_partition_idx(uint64_t hash) { return hash >> (64 - RADIX_BITS); }
std::tuple<std::vector<uint64_t>, std::vector<uint64_t>, std::vector<size_t>>
parallel_radix_partition_kv(const uint64_t* keys, const uint64_t* values, size_t size, size_t num_threads) {
    std::vector<uint64_t> out_keys(size), out_values(size);
    std::vector<size_t> partition_offsets(NUM_PARTITIONS + 1, 0);
    Hasher<uint64_t> hasher;
    std::vector<std::vector<size_t>> histograms(num_threads, std::vector<size_t>(NUM_PARTITIONS, 0));
    size_t work_per_thread = (size + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            for (size_t j = start; j < end; ++j) { histograms[i][get_partition_idx(hasher(keys[j]))]++; }
        });
    }
    for (auto& t : threads) { t.join(); }
    std::vector<std::vector<size_t>> write_positions(num_threads, std::vector<size_t>(NUM_PARTITIONS));
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        size_t current_offset = partition_offsets[p];
        for (size_t t = 0; t < num_threads; ++t) {
            write_positions[t][p] = current_offset;
            current_offset += histograms[t][p];
        }
        partition_offsets[p + 1] = current_offset;
    }
    threads.clear();
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            for (size_t j = start; j < end; ++j) {
                const uint64_t key = keys[j];
                const uint64_t val = values[j];
                const size_t p_idx = get_partition_idx(hasher(key));
                size_t pos = write_positions[i][p_idx]++;
                out_keys[pos] = key;
                out_values[pos] = val;
            }
        });
    }
    for (auto& t : threads) { t.join(); }
    return {std::move(out_keys), std::move(out_values), std::move(partition_offsets)};
}

std::tuple<std::vector<uint64_t>, std::vector<size_t>>
parallel_radix_partition_k(const uint64_t* keys, size_t size, size_t num_threads) {
    std::vector<uint64_t> out_keys(size);
    std::vector<size_t> partition_offsets(NUM_PARTITIONS + 1, 0);
    Hasher<uint64_t> hasher;
    std::vector<std::vector<size_t>> histograms(num_threads, std::vector<size_t>(NUM_PARTITIONS, 0));
    size_t work_per_thread = (size + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() { for (size_t j = start; j < end; ++j) { histograms[i][get_partition_idx(hasher(keys[j]))]++; } });
    }
    for (auto& t : threads) { t.join(); }
    std::vector<std::vector<size_t>> write_positions(num_threads, std::vector<size_t>(NUM_PARTITIONS));
    for (size_t p = 0; p < NUM_PARTITIONS; ++p) {
        size_t current_offset = partition_offsets[p];
        for (size_t t = 0; t < num_threads; ++t) {
            write_positions[t][p] = current_offset;
            current_offset += histograms[t][p];
        }
        partition_offsets[p + 1] = current_offset;
    }
    threads.clear();
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            for (size_t j = start; j < end; ++j) {
                const uint64_t key = keys[j];
                const size_t p_idx = get_partition_idx(hasher(key));
                size_t pos = write_positions[i][p_idx]++;
                out_keys[pos] = key;
            }
        });
    }
    for (auto& t : threads) { t.join(); }
    return {std::move(out_keys), std::move(partition_offsets)};
}

// ==========================================================================================
// SECTION 3 & 4: FINALIZED HYBRID JOIN FUNCTIONS & PYBIND MODULE
// ==========================================================================================
#if defined(__cpp_lib_hardware_interference_size)
    constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
#else
    constexpr size_t CACHE_LINE_SIZE = 64;
#endif
constexpr size_t PROBE_BATCH_SIZE = 2048;
struct alignas(CACHE_LINE_SIZE) PaddedCounter { std::atomic<size_t> value{0}; };

template <typename HashTableType>
py::tuple _hash_join_radix_materialize(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) {
    auto b_k_buf = b_k.request(), b_v_buf = b_v.request(), p_k_buf = p_k.request();
    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    SimpleTimer timer;
    auto [p_build_keys, p_build_values, build_offsets] = parallel_radix_partition_kv(static_cast<uint64_t*>(b_k_buf.ptr), static_cast<uint64_t*>(b_v_buf.ptr), b_k_buf.size, num_threads);
    auto [p_probe_keys, probe_offsets] = parallel_radix_partition_k(static_cast<uint64_t*>(p_k_buf.ptr), p_k_buf.size, num_threads);
    std::vector<std::vector<uint64_t>> local_result_keys(num_threads);
    std::vector<std::vector<uint64_t>> local_result_values(num_threads);
    std::vector<std::thread> threads;
    size_t work_per_thread_part = (NUM_PARTITIONS + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_p = i * work_per_thread_part; size_t end_p = std::min(start_p + work_per_thread_part, NUM_PARTITIONS);
        if (start_p >= end_p) continue;
        threads.emplace_back([&, thread_id = i, start_p, end_p]() {
            size_t probe_keys_in_task = 0;
            for(size_t p_idx = start_p; p_idx < end_p; ++p_idx) { probe_keys_in_task += probe_offsets[p_idx + 1] - probe_offsets[p_idx]; }
            if (probe_keys_in_task == 0) return;
            local_result_keys[thread_id].resize(probe_keys_in_task);
            local_result_values[thread_id].resize(probe_keys_in_task);
            uint64_t* r_keys_ptr = local_result_keys[thread_id].data();
            uint64_t* r_vals_ptr = local_result_values[thread_id].data();
            size_t local_match_count = 0;
            std::vector<uint32_t> probe_indices(PROBE_BATCH_SIZE);
            std::vector<uint64_t> build_values(PROBE_BATCH_SIZE);
            for (size_t p_idx = start_p; p_idx < end_p; ++p_idx) {
                size_t build_size = build_offsets[p_idx + 1] - build_offsets[p_idx];
                size_t probe_size = probe_offsets[p_idx + 1] - probe_offsets[p_idx];
                if (build_size == 0 || probe_size == 0) continue;
                HashTableType local_ht(build_size);
                local_ht.build_local(&p_build_keys[build_offsets[p_idx]], &p_build_values[build_offsets[p_idx]], build_size);
                const uint64_t* partition_probe_keys = &p_probe_keys[probe_offsets[p_idx]];
                for (size_t j = 0; j < probe_size; j += PROBE_BATCH_SIZE) {
                    size_t current_batch_size = std::min(PROBE_BATCH_SIZE, probe_size - j);
                    size_t match_count = local_ht.probe_vectorized(&partition_probe_keys[j], current_batch_size, probe_indices.data(), build_values.data());
                    for (size_t k = 0; k < match_count; ++k) {
                        r_keys_ptr[local_match_count] = partition_probe_keys[j + probe_indices[k]];
                        r_vals_ptr[local_match_count] = build_values[k];
                        local_match_count++;
                    }
                }
            }
            local_result_keys[thread_id].resize(local_match_count);
            local_result_values[thread_id].resize(local_match_count);
        });
    }
    for(auto& t : threads) { t.join(); }
    std::vector<size_t> result_offsets(num_threads + 1, 0);
    for(size_t i = 0; i < num_threads; ++i) { result_offsets[i+1] = result_offsets[i] + local_result_keys[i].size(); }
    size_t total_results = result_offsets[num_threads];
    py::array_t<uint64_t> result_keys(total_results);
    py::array_t<uint64_t> result_values(total_results);
    auto* result_keys_ptr = static_cast<uint64_t*>(result_keys.request().ptr);
    auto* result_values_ptr = static_cast<uint64_t*>(result_values.request().ptr);
    threads.clear();
    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i] {
            if (!local_result_keys[i].empty()) {
                memcpy(result_keys_ptr + result_offsets[i], local_result_keys[i].data(), local_result_keys[i].size() * sizeof(uint64_t));
                memcpy(result_values_ptr + result_offsets[i], local_result_values[i].data(), local_result_values[i].size() * sizeof(uint64_t));
            }
        });
    }
    for(auto& t : threads) { t.join(); }
    double core_duration_sec = timer.elapsed_seconds();
    return py::make_tuple(py::int_(total_results), core_duration_sec);
}

template<typename HashTableType>
py::tuple _hash_join_scalar_materialize(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) {
    auto b_k_buf = b_k.request(), b_v_buf = b_v.request(), p_k_buf = p_k.request();
    const auto* probe_keys_ptr = static_cast<const uint64_t*>(p_k_buf.ptr);
    size_t probe_size = p_k_buf.size;
    size_t build_size = b_k_buf.size;
    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    
    SimpleTimer timer;
    
    HashTableType ht(build_size);
    ht.build_concurrent(static_cast<uint64_t*>(b_k_buf.ptr), static_cast<uint64_t*>(b_v_buf.ptr), build_size);

    // --- HYBRID STRATEGY ---
    // Heuristic: If build table is small, it fits in L3 cache. The two-pass
    // method is faster as it avoids an extra memory copy of the results.
    constexpr size_t SMALL_TABLE_THRESHOLD = 500'000;

    if (build_size <= SMALL_TABLE_THRESHOLD) {
        // --- PATH 1: Small Table (Two-Pass, Cache-Hot Re-Probe) ---
        std::vector<PaddedCounter> counts_per_thread(num_threads);
        std::vector<std::thread> threads;
        size_t work_per_thread = (probe_size + num_threads - 1) / num_threads;
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, probe_size);
            if (start >= end) continue;
            threads.emplace_back([&, thread_id = i, start, end]() {
                std::vector<uint32_t> temp_indices(PROBE_BATCH_SIZE);
                std::vector<uint64_t> temp_values(PROBE_BATCH_SIZE);
                size_t local_count = 0;
                for (size_t j = start; j < end; j += PROBE_BATCH_SIZE) {
                    size_t current_batch_size = std::min(PROBE_BATCH_SIZE, end - j);
                    local_count += ht.probe_vectorized(&probe_keys_ptr[j], current_batch_size, temp_indices.data(), temp_values.data());
                }
                counts_per_thread[thread_id].value.fetch_add(local_count, std::memory_order_relaxed);
            });
        }
        for (auto& t : threads) { t.join(); }
        std::vector<size_t> result_offsets(num_threads + 1, 0);
        for(size_t i = 0; i < num_threads; ++i) { result_offsets[i+1] = result_offsets[i] + counts_per_thread[i].value.load(); }
        size_t total_results = result_offsets[num_threads];
        py::array_t<uint64_t> result_keys(total_results);
        py::array_t<uint64_t> result_values(total_results);
        auto* result_keys_ptr = static_cast<uint64_t*>(result_keys.request().ptr);
        auto* result_values_ptr = static_cast<uint64_t*>(result_values.request().ptr);
        std::vector<PaddedCounter> write_counters(num_threads);
        for(size_t i = 0; i < num_threads; ++i) { write_counters[i].value.store(result_offsets[i]); }
        threads.clear();
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, probe_size);
            if (start >= end) continue;
            threads.emplace_back([&, thread_id = i, start, end]() {
                std::vector<uint32_t> probe_indices(PROBE_BATCH_SIZE);
                std::vector<uint64_t> build_values(PROBE_BATCH_SIZE);
                for (size_t j = start; j < end; j += PROBE_BATCH_SIZE) {
                    size_t current_batch_size = std::min(PROBE_BATCH_SIZE, end - j);
                    size_t match_count = ht.probe_vectorized(&probe_keys_ptr[j], current_batch_size, probe_indices.data(), build_values.data());
                    if (match_count > 0) {
                        size_t write_pos = write_counters[thread_id].value.fetch_add(match_count, std::memory_order_relaxed);
                        for (size_t k = 0; k < match_count; ++k) {
                            result_keys_ptr[write_pos + k] = probe_keys_ptr[j + probe_indices[k]];
                            result_values_ptr[write_pos + k] = build_values[k];
                        }
                    }
                }
            });
        }
        for (auto& t : threads) { t.join(); }
        double core_duration_sec = timer.elapsed_seconds();
        return py::make_tuple(py::int_(total_results), core_duration_sec);

    } else {
        // --- PATH 2: Large Table (Single-Pass with Local Buffers) ---
        // This avoids a costly memory-bound re-probe.
        std::vector<std::vector<uint64_t>> local_result_keys(num_threads);
        std::vector<std::vector<uint64_t>> local_result_values(num_threads);
        std::vector<std::thread> threads;
        size_t work_per_thread = (probe_size + num_threads - 1) / num_threads;
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, probe_size);
            if (start >= end) continue;
            threads.emplace_back([&, thread_id = i, start, end]() {
                size_t work_size = end - start;
                local_result_keys[thread_id].resize(work_size);
                local_result_values[thread_id].resize(work_size);
                uint64_t* r_keys_ptr = local_result_keys[thread_id].data();
                uint64_t* r_vals_ptr = local_result_values[thread_id].data();
                size_t local_match_count = 0;
                std::vector<uint32_t> probe_indices(PROBE_BATCH_SIZE);
                std::vector<uint64_t> build_values(PROBE_BATCH_SIZE);
                for (size_t j = start; j < end; j += PROBE_BATCH_SIZE) {
                    size_t current_batch_size = std::min(PROBE_BATCH_SIZE, end - j);
                    size_t match_count = ht.probe_vectorized(&probe_keys_ptr[j], current_batch_size, probe_indices.data(), build_values.data());
                    for (size_t k = 0; k < match_count; ++k) {
                        r_keys_ptr[local_match_count] = probe_keys_ptr[j + probe_indices[k]];
                        r_vals_ptr[local_match_count] = build_values[k];
                        local_match_count++;
                    }
                }
                local_result_keys[thread_id].resize(local_match_count);
                local_result_values[thread_id].resize(local_match_count);
            });
        }
        for (auto& t : threads) { t.join(); }
        std::vector<size_t> result_offsets(num_threads + 1, 0);
        for(size_t i = 0; i < num_threads; ++i) { result_offsets[i+1] = result_offsets[i] + local_result_keys[i].size(); }
        size_t total_results = result_offsets[num_threads];
        py::array_t<uint64_t> result_keys(total_results);
        py::array_t<uint64_t> result_values(total_results);
        auto* result_keys_ptr = static_cast<uint64_t*>(result_keys.request().ptr);
        auto* result_values_ptr = static_cast<uint64_t*>(result_values.request().ptr);
        threads.clear();
        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back([&, i] {
                if (!local_result_keys[i].empty()) {
                    memcpy(result_keys_ptr + result_offsets[i], local_result_keys[i].data(), local_result_keys[i].size() * sizeof(uint64_t));
                    memcpy(result_values_ptr + result_offsets[i], local_result_values[i].data(), local_result_values[i].size() * sizeof(uint64_t));
                }
            });
        }
        for (auto& t : threads) { t.join(); }
        double core_duration_sec = timer.elapsed_seconds();
        return py::make_tuple(py::int_(total_results), core_duration_sec);
    }
}

template <typename HashTableType>
py::tuple _hash_join_radix_count(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) {
    auto b_k_buf = b_k.request(), b_v_buf = b_v.request(), p_k_buf = p_k.request();
    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    SimpleTimer timer;
    auto [p_build_keys, p_build_values, build_offsets] = parallel_radix_partition_kv(static_cast<uint64_t*>(b_k_buf.ptr), static_cast<uint64_t*>(b_v_buf.ptr), b_k_buf.size, num_threads);
    auto [p_probe_keys, probe_offsets] = parallel_radix_partition_k(static_cast<uint64_t*>(p_k_buf.ptr), p_k_buf.size, num_threads);
    std::vector<PaddedCounter> counters(num_threads);
    std::vector<std::thread> threads;
    size_t work_per_thread_part = (NUM_PARTITIONS + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_p = i * work_per_thread_part; size_t end_p = std::min(start_p + work_per_thread_part, NUM_PARTITIONS);
        if (start_p >= end_p) continue;
        threads.emplace_back([&, thread_id = i, start_p, end_p]() {
            size_t local_count = 0;
            std::vector<uint32_t> temp_indices(PROBE_BATCH_SIZE);
            std::vector<uint64_t> temp_values(PROBE_BATCH_SIZE);
            for (size_t p_idx = start_p; p_idx < end_p; ++p_idx) {
                size_t build_size = build_offsets[p_idx + 1] - build_offsets[p_idx];
                size_t probe_size = probe_offsets[p_idx + 1] - probe_offsets[p_idx];
                if (build_size == 0 || probe_size == 0) continue;
                HashTableType local_ht(build_size);
                local_ht.build_local(&p_build_keys[build_offsets[p_idx]], &p_build_values[build_offsets[p_idx]], build_size);
                for (size_t j = 0; j < probe_size; j += PROBE_BATCH_SIZE) {
                    size_t current_batch_size = std::min(PROBE_BATCH_SIZE, probe_size - j);
                    local_count += local_ht.probe_vectorized(&p_probe_keys[probe_offsets[p_idx] + j], current_batch_size, temp_indices.data(), temp_values.data());
                }
            }
            counters[thread_id].value.fetch_add(local_count, std::memory_order_relaxed);
        });
    }
    for(auto& t : threads) { t.join(); }
    size_t total_results = 0;
    for(const auto& counter : counters) { total_results += counter.value.load(); }
    double core_duration_sec = timer.elapsed_seconds();
    return py::make_tuple(py::int_(total_results), core_duration_sec);
}

template<typename HashTableType>
py::tuple _hash_join_scalar_count(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) {
    auto b_k_buf = b_k.request(), b_v_buf = b_v.request(), p_k_buf = p_k.request();
    const auto* probe_keys_ptr = static_cast<const uint64_t*>(p_k_buf.ptr);
    size_t probe_size = p_k_buf.size;
    SimpleTimer timer;
    HashTableType ht(b_k_buf.size);
    ht.build_concurrent(static_cast<uint64_t*>(b_k_buf.ptr), static_cast<uint64_t*>(b_v_buf.ptr), b_k_buf.size);
    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<PaddedCounter> counters(num_threads);
    std::vector<std::thread> threads;
    size_t work_per_thread = (probe_size + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, probe_size);
        if (start >= end) continue;
        threads.emplace_back([&, thread_id = i, start, end]() {
            size_t local_count = 0;
            std::vector<uint32_t> temp_indices(PROBE_BATCH_SIZE);
            std::vector<uint64_t> temp_values(PROBE_BATCH_SIZE);
            for (size_t j = start; j < end; j += PROBE_BATCH_SIZE) {
                size_t current_batch_size = std::min(PROBE_BATCH_SIZE, end - j);
                local_count += ht.probe_vectorized(&probe_keys_ptr[j], current_batch_size, temp_indices.data(), temp_values.data());
            }
            counters[thread_id].value.fetch_add(local_count, std::memory_order_relaxed);
        });
    }
    for (auto& t : threads) { t.join(); }
    size_t total_results = 0;
    for(const auto& counter : counters) { total_results += counter.value.load(); }
    double core_duration_sec = timer.elapsed_seconds();
    return py::make_tuple(py::int_(total_results), core_duration_sec);
}

void initialize_memory_system() { mi_version(); }

PYBIND11_MODULE(flash_join, m) {
    initialize_memory_system();
    m.doc() = "A high-performance hash join with micro-architectural optimizations"; 
    m.def("hash_join_radix", &_hash_join_radix_materialize<FlashHashTable<false>>, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join", &_hash_join_scalar_materialize<FlashHashTable<false>>, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_radix_bloom", &_hash_join_radix_materialize<FlashHashTable<true>>, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_bloom", &_hash_join_scalar_materialize<FlashHashTable<true>>, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_count_radix", &_hash_join_radix_count<FlashHashTable<false>>, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_count", &_hash_join_scalar_count<FlashHashTable<false>>, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_count_radix_bloom", &_hash_join_radix_count<FlashHashTable<true>>, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_count_bloom", &_hash_join_scalar_count<FlashHashTable<true>>, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("initialize", &initialize_memory_system, "Initializes the custom memory allocator.");
}