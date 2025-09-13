#include <iostream>
#include <vector>
#include <cstdint>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <memory>
#include <functional>
#include <numeric>

// Pybind11 for Python interoperability
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// High-performance memory allocator
#include <mimalloc.h>

// High-performance hash function
#define XXH_INLINE_ALL
#include "xxhash.h"

// For parallel loops (used in result materialization)
#include <omp.h>

namespace py = pybind11;

// --- Hasher & Constants ---
template<typename KeyType>
struct Hasher {
    uint64_t operator()(const KeyType& key) const { return XXH3_64bits(&key, sizeof(KeyType)); }
};

constexpr size_t CHUNK_SIZE = 256;
constexpr size_t NUM_LOCKS = 256;

// A cache-line aligned struct to hold thread-local counters, preventing false sharing.
struct alignas(64) PaddedCounter {
    std::atomic<size_t> value{0};
};


template<typename KeyType, typename ValueType>
struct UnchainedHashTable {
private:
    struct Entry {
        uint8_t tag;
        KeyType key;
        ValueType value;
    };

    // Align chunks to cache line boundaries to potentially improve performance.
    struct alignas(64) Chunk {
        std::atomic<uint32_t> count{0};
        std::atomic<Chunk*> next{nullptr};
        Entry entries[CHUNK_SIZE];
    };

    std::vector<std::atomic<Chunk*>> buckets_;
    std::vector<std::mutex> locks_;
    Hasher<KeyType> hasher_;

    struct MiMallocDeleter { void operator()(Chunk* p) const { mi_free(p); } };
    using ChunkPtr = std::unique_ptr<Chunk, MiMallocDeleter>;
    std::vector<ChunkPtr> chunk_pool_;
    std::mutex pool_mutex_;

    inline uint64_t get_bucket_idx(uint64_t hash) const { return hash & (buckets_.size() - 1); }
    inline uint8_t get_hash_tag(uint64_t hash) const { return (hash >> 32) & 0xFF; }

public:
    static size_t calculate_size(size_t initial_size) {
        size_t size = 1;
        while (size < initial_size) size <<= 1;
        return size;
    }

    UnchainedHashTable(size_t initial_size = 1024*1024) 
        : buckets_(calculate_size(initial_size)), 
          locks_(NUM_LOCKS) 
    {}
    
    // Build logic remains unchanged, it is already highly parallel.
    void insert(const KeyType& key, const ValueType& value) {
        uint64_t hash = hasher_(key);
        uint64_t bucket_idx = get_bucket_idx(hash);
        uint8_t tag = get_hash_tag(hash);

        std::lock_guard<std::mutex> lock(locks_[bucket_idx % NUM_LOCKS]);
        Chunk* current_chunk = buckets_[bucket_idx].load(std::memory_order_acquire);

        if (current_chunk == nullptr) {
            Chunk* new_chunk = (Chunk*)mi_malloc(sizeof(Chunk)); new (new_chunk) Chunk();
            new_chunk->entries[0] = {tag, key, value};
            new_chunk->count.store(1, std::memory_order_release);
            buckets_[bucket_idx].store(new_chunk, std::memory_order_release);
            std::lock_guard<std::mutex> pool_lock(pool_mutex_); chunk_pool_.emplace_back(new_chunk); return;
        }

        while (true) {
            uint32_t slot = current_chunk->count.fetch_add(1, std::memory_order_acq_rel);
            if (slot < CHUNK_SIZE) { current_chunk->entries[slot] = {tag, key, value}; return; }
            current_chunk->count.fetch_sub(1, std::memory_order_release);

            Chunk* next_chunk = current_chunk->next.load(std::memory_order_acquire);
            if (next_chunk == nullptr) {
                Chunk* new_chunk = (Chunk*)mi_malloc(sizeof(Chunk)); new (new_chunk) Chunk();
                new_chunk->entries[0] = {tag, key, value};
                new_chunk->count.store(1, std::memory_order_release);
                Chunk* expected = nullptr;
                if (current_chunk->next.compare_exchange_strong(expected, new_chunk, std::memory_order_acq_rel)) {
                    std::lock_guard<std::mutex> pool_lock(pool_mutex_); chunk_pool_.emplace_back(new_chunk); return;
                } else { mi_free(new_chunk); current_chunk = expected; }
            } else { current_chunk = next_chunk; }
        }
    }

    void build(const KeyType* keys, const ValueType* values, size_t size) {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t chunk_size = (size + num_threads - 1) / num_threads;
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size; size_t end = std::min(start + chunk_size, size);
            if (start >= end) continue;
            threads.emplace_back([this, keys, values, start, end]() {
                for (size_t j = start; j < end; ++j) { this->insert(keys[j], values[j]); }
            });
        }
        for (auto& t : threads) { t.join(); }
    }
    
    // Original probe function for materializing results.
    void probe(const KeyType& key, std::vector<ValueType>& results) const {
        uint64_t hash = hasher_(key);
        uint64_t bucket_idx = get_bucket_idx(hash);
        uint8_t probe_tag = get_hash_tag(hash);
        Chunk* current_chunk = buckets_[bucket_idx].load(std::memory_order_acquire);
        while (current_chunk != nullptr) {
            uint32_t count = current_chunk->count.load(std::memory_order_acquire);
            for (uint32_t i = 0; i < count; ++i) {
                if (current_chunk->entries[i].tag == probe_tag && current_chunk->entries[i].key == key) {
                    results.push_back(current_chunk->entries[i].value);
                }
            }
            current_chunk = current_chunk->next.load(std::memory_order_acquire);
        }
    }

    // A highly optimized probe function that only counts matches.
    size_t probe_and_count(const KeyType& key) const {
        uint64_t hash = hasher_(key);
        uint64_t bucket_idx = get_bucket_idx(hash);
        uint8_t probe_tag = get_hash_tag(hash);
        size_t match_count = 0;
        
        Chunk* current_chunk = buckets_[bucket_idx].load(std::memory_order_acquire);
        while (current_chunk != nullptr) {
            uint32_t count = current_chunk->count.load(std::memory_order_acquire);
            for (uint32_t i = 0; i < count; ++i) {
                if (current_chunk->entries[i].tag == probe_tag && current_chunk->entries[i].key == key) {
                    match_count++; // Just increment a local counter, no heap allocation or vector ops.
                }
            }
            current_chunk = current_chunk->next.load(std::memory_order_acquire);
        }
        return match_count;
    }
};

// --- Pybind11 Wrapper for Materializing Full Results ---
py::tuple hash_join_two_pass(py::array_t<uint64_t> build_keys,
                             py::array_t<uint64_t> build_values,
                             py::array_t<uint64_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();

    const uint64_t* build_keys_ptr = static_cast<uint64_t*>(build_keys_buf.ptr);
    const uint64_t* build_values_ptr = static_cast<uint64_t*>(build_values_buf.ptr);
    const uint64_t* probe_keys_ptr = static_cast<uint64_t*>(probe_keys_buf.ptr);
    size_t build_size = build_keys_buf.size;
    size_t probe_size = probe_keys_buf.size;

    UnchainedHashTable<uint64_t, uint64_t> ht(build_size);
    ht.build(build_keys_ptr, build_values_ptr, build_size);

    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t chunk_size = (probe_size + num_threads - 1) / num_threads;

    std::vector<size_t> counts(num_threads, 0);
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size; size_t end = std::min(start + chunk_size, probe_size);
        if (start >= end) continue;
        threads.emplace_back([&ht, &counts, i, probe_keys_ptr, start, end]() {
            std::vector<uint64_t> local_results; size_t local_count = 0;
            for (size_t j = start; j < end; ++j) {
                local_results.clear(); ht.probe(probe_keys_ptr[j], local_results); local_count += local_results.size();
            }
            counts[i] = local_count;
        });
    }
    for (auto& t : threads) { t.join(); }
    threads.clear();

    size_t total_results = 0;
    std::vector<size_t> offsets(num_threads + 1, 0);
    for (size_t i = 0; i < num_threads; ++i) {
        total_results += counts[i]; offsets[i+1] = offsets[i] + counts[i];
    }
    
    py::array_t<uint64_t> result_keys(total_results);
    py::array_t<uint64_t> result_values(total_results);
    uint64_t* result_keys_ptr = static_cast<uint64_t*>(result_keys.request().ptr);
    uint64_t* result_values_ptr = static_cast<uint64_t*>(result_values.request().ptr);

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size; size_t end = std::min(start + chunk_size, probe_size);
        if (start >= end) continue;
        threads.emplace_back([&ht, probe_keys_ptr, result_keys_ptr, result_values_ptr, &offsets, i, start, end]() {
            std::vector<uint64_t> local_results; size_t current_offset = offsets[i];
            for (size_t j = start; j < end; ++j) {
                local_results.clear(); ht.probe(probe_keys_ptr[j], local_results);
                for (const auto& val : local_results) {
                    result_keys_ptr[current_offset] = probe_keys_ptr[j]; result_values_ptr[current_offset] = val; current_offset++;
                }
            }
        });
    }
    for (auto& t : threads) { t.join(); }

    return py::make_tuple(result_keys, result_values);
}

// --- Pybind11 Wrapper for Optimized Counting ---
py::int_ hash_join_count_optimized(py::array_t<uint64_t> build_keys,
                                   py::array_t<uint64_t> build_values,
                                   py::array_t<uint64_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();

    const uint64_t* build_keys_ptr = static_cast<uint64_t*>(build_keys_buf.ptr);
    const uint64_t* build_values_ptr = static_cast<uint64_t*>(build_values_buf.ptr);
    const uint64_t* probe_keys_ptr = static_cast<uint64_t*>(probe_keys_buf.ptr);
    size_t build_size = build_keys_buf.size;
    size_t probe_size = probe_keys_buf.size;

    UnchainedHashTable<uint64_t, uint64_t> ht(build_size);
    ht.build(build_keys_ptr, build_values_ptr, build_size);

    size_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    size_t chunk_size = (probe_size + num_threads - 1) / num_threads;
    
    // Use the cache-line-padded counters to prevent false sharing.
    std::vector<PaddedCounter> counts(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * chunk_size;
        size_t end = std::min(start + chunk_size, probe_size);
        if (start >= end) continue;
        threads.emplace_back([&ht, &counts, i, probe_keys_ptr, start, end]() {
            size_t local_count = 0;
            // The hot loop now calls the highly optimized counting probe.
            for (size_t j = start; j < end; ++j) {
                local_count += ht.probe_and_count(probe_keys_ptr[j]);
            }
            counts[i].value.store(local_count, std::memory_order_relaxed);
        });
    }
    for (auto& t : threads) { t.join(); }

    size_t total_results = 0;
    for (size_t i = 0; i < num_threads; ++i) {
        total_results += counts[i].value.load();
    }

    return py::int_(total_results);
}


// --- Pybind11 Module Definition ---
PYBIND11_MODULE(fast_join, m) {
    m.doc() = "A high-performance hash join with multiple execution modes"; 
    
    m.def("hash_join", &hash_join_two_pass, 
          "Performs a hash join and returns the full result",
          py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));

    m.def("hash_join_count", &hash_join_count_optimized, 
          "Performs a hash join and returns only the count of results (Optimized)",
          py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));
}