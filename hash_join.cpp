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

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <nmmintrin.h>

#define PYBIND11_NO_ATEXIT
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#define MI_OVERRIDE 0
#define MI_MALLOC_OVERRIDE 0
#define MI_NO_DEINIT 1
#include <mimalloc.h>

namespace py = pybind11;

constexpr size_t RADIX_BITS = 8;
constexpr size_t NUM_PARTITIONS = 1 << RADIX_BITS;

uint64_t hash64(uint64_t key, uint32_t seed) {
    uint64_t k = 0x8648DBDB;
    uint64_t crc = _mm_crc32_u64(seed, key);
    return crc * ((k << 32) + 1);
}

template<typename KeyType>
struct Hasher {
    uint64_t operator()(const KeyType& key) const { return hash64(key, 0xAAAAAAAA); }
};

// This namespace contains helper functions and constants for the hash table.
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
    struct alignas(16) Slot { // alignas can sometimes help with performance
        std::atomic<uint8_t> tag;
        uint64_t key;
        uint64_t value;
    };
    
    std::unique_ptr<Slot[]> slots_;
    // Bloom filter remains separate as it's a different data structure
    std::unique_ptr<std::atomic<uint16_t>[]> bloom_directory_;

    size_t bloom_shift_;
    static inline const auto tags_table_ = internal::create_tags_table();
    size_t capacity_;
    size_t capacity_mask_;
    Hasher<uint64_t> hasher_;

public:
    FlashHashTable(const FlashHashTable&) = delete;
    FlashHashTable& operator=(const FlashHashTable&) = delete;
    
    static size_t calculate_power_of_2(size_t n) {
        return n == 0 ? 1 : 1UL << (64 - __builtin_clzll(n - 1));
    }

    FlashHashTable(size_t build_size) {
        size_t capacity = calculate_power_of_2(build_size * 1.5 + SIMD_WIDTH);
        this->capacity_ = capacity;
        this->capacity_mask_ = capacity - 1;
        const size_t alloc_size = capacity_ + SIMD_WIDTH - 1;

        slots_ = std::make_unique<Slot[]>(alloc_size);
        for (size_t i = 0; i < capacity_; ++i) {
            slots_[i].tag.store(EMPTY_TAG, std::memory_order_relaxed);
            // It's good practice to zero-initialize the data too
            slots_[i].key = 0;
            slots_[i].value = 0;
        }

        if constexpr (UseBloomFilter) {
            size_t bloom_capacity = capacity_;
            bloom_directory_ = std::make_unique<std::atomic<uint16_t>[]>(bloom_capacity);
            for (size_t i = 0; i < bloom_capacity; ++i) {
                bloom_directory_[i].store(0, std::memory_order_relaxed);
            }
            if (bloom_capacity > 0) {
                bloom_shift_ = 64 - __builtin_ctzll(bloom_capacity);
            }
        }
    }

    // Single-threaded insert
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
                if constexpr (UseBloomFilter) {
                    bloom_directory_[pos & capacity_mask_].fetch_or(get_bloom_tag(hash), std::memory_order_relaxed);
                }
                return;
            }
            if (slots_[pos].key == key) return;
            pos = (pos + 1) & this->capacity_mask_;
        } while (pos != initial_pos);
    }

    void insert_concurrent(const uint64_t& key, const uint64_t& value) {
        uint64_t hash = this->hasher_(key);
        uint8_t tag = (hash >> 56);
        if (tag == EMPTY_TAG) tag = 0;

        size_t pos = hash & this->capacity_mask_;
        const size_t initial_pos = pos;

        while (true) {
            // Step 1: Atomically load the tag with acquire semantics.
            // This ensures that if we see a non-empty tag, the associated key/value writes
            // from the owner thread are visible to us.
            uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);

            // Step 2: Handle empty slot
            if (current_tag == EMPTY_TAG) {
                uint8_t expected_empty = EMPTY_TAG;
                // Try to claim this empty slot using CAS.
                // acq_rel ensures this synchronizes with other potential writers and readers.
                if (slots_[pos].tag.compare_exchange_strong(expected_empty, tag, std::memory_order_acq_rel)) {
                    // Success! We claimed the slot. Now we can safely write our data.
                    slots_[pos].key = key;
                    slots_[pos].value = value;
                    if constexpr (UseBloomFilter) {
                        bloom_directory_[pos & capacity_mask_].fetch_or(get_bloom_tag(hash), std::memory_order_relaxed);
                    }
                    return; // Insert complete
                }
                // If CAS failed, it means another thread just took this slot.
                // The `while(true)` loop will simply re-evaluate this slot in the next iteration,
                // but this time it will fall into the "occupied slot" logic below.
                continue;
            }
            
            // Step 3: Handle occupied slot
            // Since we loaded the tag with acquire, reading key is now safe.
            if (current_tag == tag && slots_[pos].key == key) {
                // It's a duplicate of the key we are trying to insert.
                return; // Duplicate found, nothing to do.
            }

            // Step 4: Collision, move to the next slot (linear probing)
            pos = (pos + 1) & this->capacity_mask_;
            if (pos == initial_pos) {
                // We've looped all the way around. The table is likely full for this probe sequence.
                // In a real-world scenario, you might want to throw an exception or resize.
                return; 
            }
        }
    }

    // The probe function is now adapted to the new layout
    template<typename Func>
    void probe_scalar(const uint64_t& key, Func&& callback) const {
        uint64_t hash = this->hasher_(key);

        if constexpr (UseBloomFilter) {
            if (!check_bloom_filter(hash)) {
                return;
            }
        }
        
        uint8_t tag = (hash >> 56); if (tag == EMPTY_TAG) tag = 0;
        size_t pos = hash & this->capacity_mask_;
        const size_t initial_pos = pos;

        do {
            uint8_t current_tag = slots_[pos].tag.load(std::memory_order_acquire);
            if (current_tag == EMPTY_TAG) {
                return; // Found empty slot, key does not exist.
            }
            if (current_tag == tag && slots_[pos].key == key) {
                callback(key, slots_[pos].value);
                return; // Assume unique keys
            }
            pos = (pos + 1) & this->capacity_mask_;
        } while (pos != initial_pos);
    }
    
    // Helper function for bloom filter
    inline uint16_t get_bloom_tag(uint64_t hash) const {
        return tags_table_[(static_cast<uint32_t>(hash)) >> (32 - 11)];
    }
    
    inline bool check_bloom_filter(uint64_t hash) const {
        size_t slot = (hash & this->capacity_mask_);
        const uint16_t entry = bloom_directory_[slot].load(std::memory_order_relaxed);
        return (get_bloom_tag(hash) & entry) == get_bloom_tag(hash);
    }

    // build/probe methods call the correct insert/probe methods
    void build_local(const uint64_t* keys, const uint64_t* values, size_t size) {
        for (size_t i = 0; i < size; ++i) { this->insert_local(keys[i], values[i]); }
    }

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

    template<typename Func>
    void probe_batch_simd(const uint64_t* keys, size_t batch_size, Func&& callback) const {
        for (size_t i = 0; i < batch_size; ++i) { probe_scalar(keys[i], callback); }
    }
};

// ==========================================================================================
// SECTION 2: ROBUST RADIX PARTITIONING LOGIC (Using Atomics)
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
    for (size_t i = 0; i < num_threads; ++i) {
        for (size_t j = 0; j < NUM_PARTITIONS; ++j) { partition_offsets[j + 1] += histograms[i][j]; }
    }
    for (size_t i = 1; i <= NUM_PARTITIONS; ++i) { partition_offsets[i] += partition_offsets[i - 1]; }
    std::vector<std::atomic<size_t>> atomic_write_offsets(NUM_PARTITIONS);
    for(size_t p = 0; p < NUM_PARTITIONS; ++p) { atomic_write_offsets[p].store(partition_offsets[p], std::memory_order_relaxed); }
    threads.clear();
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, size);
        if (start >= end) continue;
        threads.emplace_back([&, start, end]() {
            for (size_t j = start; j < end; ++j) {
                size_t p_idx = get_partition_idx(hasher(keys[j]));
                size_t write_pos = atomic_write_offsets[p_idx].fetch_add(1, std::memory_order_relaxed);
                out_keys[write_pos] = keys[j]; out_values[write_pos] = values[j];
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
        threads.emplace_back([&, i, start, end]() {
            for (size_t j = start; j < end; ++j) { histograms[i][get_partition_idx(hasher(keys[j]))]++; }
        });
    }
    for (auto& t : threads) { t.join(); }
    for (size_t i = 0; i < num_threads; ++i) {
        for (size_t j = 0; j < NUM_PARTITIONS; ++j) { partition_offsets[j + 1] += histograms[i][j]; }
    }
    for (size_t i = 1; i <= NUM_PARTITIONS; ++i) { partition_offsets[i] += partition_offsets[i - 1]; }
    std::vector<std::atomic<size_t>> atomic_write_offsets(NUM_PARTITIONS);
    for(size_t p = 0; p < NUM_PARTITIONS; ++p) { atomic_write_offsets[p].store(partition_offsets[p], std::memory_order_relaxed); }
    threads.clear();
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, size);
        if (start >= end) continue;
        threads.emplace_back([&, start, end]() {
            for (size_t j = start; j < end; ++j) {
                size_t p_idx = get_partition_idx(hasher(keys[j]));
                size_t write_pos = atomic_write_offsets[p_idx].fetch_add(1, std::memory_order_relaxed);
                out_keys[write_pos] = keys[j];
            }
        });
    }
    for (auto& t : threads) { t.join(); }
    return {std::move(out_keys), std::move(partition_offsets)};
}

// ==========================================================================================
// SECTION 3: REFACTORED JOIN FUNCTIONS & PROCESSORS
// ==========================================================================================
#if defined(__cpp_lib_hardware_interference_size)
    constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
#else
    constexpr size_t CACHE_LINE_SIZE = 64;
#endif

struct alignas(CACHE_LINE_SIZE) PaddedCounter { size_t value{0}; };

struct MaterializeProcessor {
    std::vector<std::vector<std::pair<uint64_t, uint64_t>>> thread_local_results;
    void init(size_t num_threads) { thread_local_results.resize(num_threads); }
    auto get_callback(size_t thread_id) {
        return [this, thread_id](const uint64_t& key, uint64_t val){ 
            thread_local_results[thread_id].emplace_back(key, val); 
        };
    }
    py::tuple finalize() {
        size_t total_results = 0;
        for(const auto& vec : thread_local_results) total_results += vec.size();
        py::array_t<uint64_t> result_keys(total_results);
        py::array_t<uint64_t> result_values(total_results);
        uint64_t* result_keys_ptr = static_cast<uint64_t*>(result_keys.request().ptr);
        uint64_t* result_values_ptr = static_cast<uint64_t*>(result_values.request().ptr);
        size_t current_offset = 0;
        for(const auto& vec : thread_local_results) {
            for(const auto& p : vec) {
                result_keys_ptr[current_offset] = p.first;
                result_values_ptr[current_offset] = p.second;
                current_offset++;
            }
        }
        return py::make_tuple(result_keys, result_values);
    }
};

struct CountProcessor {
    std::vector<PaddedCounter> thread_local_counts;
    void init(size_t num_threads) { thread_local_counts.resize(num_threads); }
    auto get_callback(size_t thread_id) {
        return [this, thread_id](const uint64_t&, uint64_t) { this->thread_local_counts[thread_id].value++; };
    }
    py::int_ finalize() {
        size_t total_results = 0;
        for (const auto& counter : thread_local_counts) { total_results += counter.value; }
        return py::int_(total_results);
    }
};

template <typename HashTableType, typename ResultProcessor>
auto _hash_join_radix_generic(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k, ResultProcessor rp) {
    auto b_k_buf = b_k.request(), b_v_buf = b_v.request(), p_k_buf = p_k.request();
    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    auto [p_build_keys, p_build_values, build_offsets] = parallel_radix_partition_kv(
        static_cast<uint64_t*>(b_k_buf.ptr), static_cast<uint64_t*>(b_v_buf.ptr), b_k_buf.size, num_threads);
    auto [p_probe_keys, probe_offsets] = parallel_radix_partition_k(
        static_cast<uint64_t*>(p_k_buf.ptr), p_k_buf.size, num_threads);
    rp.init(num_threads);
    std::vector<std::thread> threads;
    size_t work_per_thread_part = (NUM_PARTITIONS + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_p = i * work_per_thread_part; size_t end_p = std::min(start_p + work_per_thread_part, NUM_PARTITIONS);
        if (start_p >= end_p) continue;
        threads.emplace_back([&, thread_id = i, start_p, end_p]() {
            auto callback = rp.get_callback(thread_id);
            for (size_t p_idx = start_p; p_idx < end_p; ++p_idx) {
                size_t build_size = build_offsets[p_idx + 1] - build_offsets[p_idx];
                size_t probe_size = probe_offsets[p_idx + 1] - probe_offsets[p_idx];
                if (build_size == 0 || probe_size == 0) continue;
                HashTableType local_ht(build_size);
                local_ht.build_local(&p_build_keys[build_offsets[p_idx]], &p_build_values[build_offsets[p_idx]], build_size);
                local_ht.probe_batch_simd(&p_probe_keys[probe_offsets[p_idx]], probe_size, callback);
            }
        });
    }
    for(auto& t : threads) { t.join(); }
    return rp.finalize();
}

template<typename HashTableType, typename ResultProcessor>
auto _hash_join_scalar_generic(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k, ResultProcessor rp) {
    auto b_k_buf = b_k.request(), b_v_buf = b_v.request(), p_k_buf = p_k.request();
    const auto* probe_keys_ptr = static_cast<const uint64_t*>(p_k_buf.ptr);
    size_t probe_size = p_k_buf.size;
    HashTableType ht(b_k_buf.size);
    ht.build_concurrent(static_cast<uint64_t*>(b_k_buf.ptr), static_cast<uint64_t*>(b_v_buf.ptr), b_k_buf.size);
    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    rp.init(num_threads);
    std::vector<std::thread> threads;
    size_t work_per_thread = (probe_size + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, probe_size);
        if (start >= end) continue;
        threads.emplace_back([&, thread_id = i, start, end]() {
            auto callback = rp.get_callback(thread_id);
            for (size_t j = start; j < end; ++j) { ht.probe_scalar(probe_keys_ptr[j], callback); }
        });
    }
    for (auto& t : threads) { t.join(); }
    return rp.finalize();
}

py::tuple hash_join_radix(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) {
    return _hash_join_radix_generic<FlashHashTable<false>>(b_k, b_v, p_k, MaterializeProcessor{});
}

py::tuple hash_join_radix_bloom(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) {
    return _hash_join_radix_generic<FlashHashTable<true>>(b_k, b_v, p_k, MaterializeProcessor{});
}

py::int_ hash_join_count_radix(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) {
    return _hash_join_radix_generic<FlashHashTable<false>>(b_k, b_v, p_k, CountProcessor{}); 
}

py::int_ hash_join_count_radix_bloom(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) { 
    return _hash_join_radix_generic<FlashHashTable<true>>(b_k, b_v, p_k, CountProcessor{}); 
}

py::tuple hash_join_scalar(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) { 
    return _hash_join_scalar_generic<FlashHashTable<false>>(b_k, b_v, p_k, MaterializeProcessor{}); 
}

py::tuple hash_join_bloom(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) { 
    return _hash_join_scalar_generic<FlashHashTable<true>>(b_k, b_v, p_k, MaterializeProcessor{}); 
}

py::int_ hash_join_count_scalar(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) { 
    return _hash_join_scalar_generic<FlashHashTable<false>>(b_k, b_v, p_k, CountProcessor{}); 
}

py::int_ hash_join_count_bloom(py::array_t<uint64_t> b_k, py::array_t<uint64_t> b_v, py::array_t<uint64_t> p_k) { 
    return _hash_join_scalar_generic<FlashHashTable<true>>(b_k, b_v, p_k, CountProcessor{}); 
}

// ==========================================================================================
// SECTION 4: PYBIND11 MODULE DEFINITION
// ==========================================================================================
void initialize_memory_system() {
    // mi_version();
}

PYBIND11_MODULE(flash_join, m) {
    initialize_memory_system();
    m.doc() = "A high-performance hash join with various optimization strategies"; 
    m.def("hash_join_radix", &hash_join_radix, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join", &hash_join_scalar, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_radix_bloom", &hash_join_radix_bloom, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_bloom", &hash_join_bloom, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_count_radix", &hash_join_count_radix, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_count", &hash_join_count_scalar, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_count_radix_bloom", &hash_join_count_radix_bloom, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("hash_join_count_bloom", &hash_join_count_bloom, "", py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
    m.def("initialize", &initialize_memory_system, "Initializes the custom memory allocator.");
}