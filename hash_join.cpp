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
#include <algorithm> // For std::sort
#include <random>

// AVX2 and other intrinsics
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <nmmintrin.h> // Header for SSE4.2 intrinsics

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Re-enable mimalloc with proper configuration
#define MI_OVERRIDE 0  // Don't override system malloc
#define MI_MALLOC_OVERRIDE 0  // Don't override malloc
#include <mimalloc.h>

namespace py = pybind11;

uint64_t hash32(uint32_t key, uint32_t seed) {
    uint64_t k = 0x8648DBDB;          // Mixing constant
    uint32_t crc = _mm_crc32_u32(seed, key);   // crc32
    return crc * ((k << 32) + 1); // imul
}

template<typename KeyType>
struct Hasher {
    uint64_t operator()(const KeyType& key) const { return hash32(key, 0xAAAAAAAA); }
};

struct alignas(64) PaddedCounter {
    std::atomic<size_t> value{0};
};

struct MimallocDeleter {
    void operator()(void* ptr) const {
        if (ptr) mi_free(ptr);
    }
};

template<typename KeyType, typename ValueType, size_t ChunkSize = 256, size_t NumLocks = 8192>
struct UnchainedHashTable {
public:
    struct alignas(64) Chunk {
        std::atomic<uint32_t> count{0};
        std::atomic<Chunk*> next{nullptr};
        uint8_t   tags[ChunkSize];
        KeyType   keys[ChunkSize];
        ValueType values[ChunkSize];
    };
    static constexpr size_t StaticChunkSize = ChunkSize;

private:

    std::unique_ptr<char, MimallocDeleter> memory_pool_;
    std::atomic<size_t> pool_offset_{0};
    size_t pool_size_bytes_{0};

    // Buckets directly store pointers to Chunks.
    std::vector<std::atomic<Chunk*>> buckets_;
    std::vector<std::mutex> locks_;
    Hasher<KeyType> hasher_;

public:
    inline uint64_t get_bucket_idx(uint64_t hash) const { return hash & (buckets_.size() - 1); }
    inline uint8_t get_hash_tag(uint64_t hash) const { return (hash >> 32) & 0xFF; }
    

    static size_t calculate_size(size_t initial_size) {
        if (initial_size == 0) return 1;
        return 1UL << (64 - __builtin_clzll(initial_size - 1));
    }

    UnchainedHashTable(size_t initial_size = 1024*1024, size_t build_size_hint = 0) 
        : buckets_(calculate_size(initial_size)), 
          locks_(NumLocks) {
        
        if (build_size_hint > 0) {
            size_t num_chunks_needed = (size_t)(build_size_hint * 1.5) / ChunkSize + 1;
            pool_size_bytes_ = num_chunks_needed * sizeof(Chunk);
            size_t alignment = alignof(Chunk);
            memory_pool_ = std::unique_ptr<char, MimallocDeleter>((char*)mi_malloc_aligned(pool_size_bytes_, alignment));
        }
    }

    ~UnchainedHashTable() {}
    
    UnchainedHashTable(const UnchainedHashTable&) = delete;
    UnchainedHashTable& operator=(const UnchainedHashTable&) = delete;
    UnchainedHashTable(UnchainedHashTable&&) = delete;
    UnchainedHashTable& operator=(UnchainedHashTable&&) = delete;

    Chunk* alloc_chunk_from_pool() {
        size_t offset = pool_offset_.fetch_add(sizeof(Chunk), std::memory_order_relaxed);
        if (offset + sizeof(Chunk) <= pool_size_bytes_) {
            char* ptr = memory_pool_.get() + offset;
            return new (ptr) Chunk();
        }
        return new (mi_malloc_aligned(sizeof(Chunk), alignof(Chunk))) Chunk();
    }
    
    void free_chunk(Chunk* chunk) {
        char* chunk_addr = reinterpret_cast<char*>(chunk);
        char* pool_start = memory_pool_.get();
        char* pool_end = pool_start + pool_size_bytes_;
        
        if (!(chunk_addr >= pool_start && chunk_addr < pool_end)) {
            chunk->~Chunk();
            mi_free(chunk); 
        }
    }
    
    void insert(const KeyType& key, const ValueType& value) {
        uint64_t hash = hasher_(key);
        uint64_t bucket_idx = get_bucket_idx(hash);
        uint8_t tag = get_hash_tag(hash);

        std::lock_guard<std::mutex> lock(locks_[bucket_idx % NumLocks]);

        Chunk* current_chunk = buckets_[bucket_idx].load(std::memory_order_acquire);

        if (current_chunk == nullptr) {
            Chunk* new_chunk = alloc_chunk_from_pool();
            new_chunk->tags[0] = tag;
            new_chunk->keys[0] = key;
            new_chunk->values[0] = value;
            new_chunk->count.store(1, std::memory_order_release);
            buckets_[bucket_idx].store(new_chunk, std::memory_order_release);
            return;
        }

        while (true) {
            uint32_t slot = current_chunk->count.fetch_add(1, std::memory_order_relaxed);
            if (slot < ChunkSize) {
                current_chunk->tags[slot] = tag;
                current_chunk->keys[slot] = key;
                current_chunk->values[slot] = value;
                return;
            }
            current_chunk->count.fetch_sub(1, std::memory_order_relaxed);

            Chunk* next_chunk = current_chunk->next.load(std::memory_order_acquire);
            if (next_chunk == nullptr) {
                Chunk* new_chunk = alloc_chunk_from_pool();
                new_chunk->tags[0] = tag;
                new_chunk->keys[0] = key;
                new_chunk->values[0] = value;
                new_chunk->count.store(1, std::memory_order_release);
                Chunk* expected = nullptr;
                if (current_chunk->next.compare_exchange_strong(expected, new_chunk, std::memory_order_release, std::memory_order_relaxed)) {
                    return; 
                } else {
                    free_chunk(new_chunk);
                    current_chunk = expected; 
                }
            } else { current_chunk = next_chunk; }
        }
    }

    void build(const KeyType* keys, const ValueType* values, size_t size) {
        size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        size_t work_per_thread = (size + num_threads - 1) / num_threads;
        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * work_per_thread; size_t end = std::min(start + work_per_thread, size);
            if (start >= end) continue;
            threads.emplace_back([this, keys, values, start, end]() {
                for (size_t j = start; j < end; ++j) { this->insert(keys[j], values[j]); }
            });
        }
        for (auto& t : threads) { t.join(); }
    }
    
    void probe(const KeyType& key, std::vector<ValueType>& results) const {
        uint64_t hash = hasher_(key);
        uint64_t bucket_idx = get_bucket_idx(hash);
        uint8_t probe_tag = get_hash_tag(hash);

        Chunk* current_chunk = buckets_[bucket_idx].load(std::memory_order_acquire);
        
        while (current_chunk != nullptr) {
            if (current_chunk->next.load(std::memory_order_relaxed)) {
                __builtin_prefetch(current_chunk->next.load(std::memory_order_relaxed), 0, 0);
            }
            uint32_t count = std::min((uint32_t)ChunkSize, current_chunk->count.load(std::memory_order_acquire));
            for (uint32_t i = 0; i < count; ++i) {
                if (current_chunk->tags[i] == probe_tag && current_chunk->keys[i] == key) {
                    results.push_back(current_chunk->values[i]);
                }
            }
            current_chunk = current_chunk->next.load(std::memory_order_acquire);
        }
    }
    
    size_t probe_and_count(const KeyType& key) const {
        uint64_t hash = hasher_(key);
        uint64_t bucket_idx = get_bucket_idx(hash);
        uint8_t probe_tag = get_hash_tag(hash);
        size_t match_count = 0;

        Chunk* current_chunk = buckets_[bucket_idx].load(std::memory_order_acquire);
        
        while (current_chunk != nullptr) {
            if (current_chunk->next.load(std::memory_order_relaxed)) {
                __builtin_prefetch(current_chunk->next.load(std::memory_order_relaxed), 0, 0);
            }
            uint32_t count = std::min((uint32_t)ChunkSize, current_chunk->count.load(std::memory_order_acquire));
            for (uint32_t i = 0; i < count; ++i) {
                if (current_chunk->tags[i] == probe_tag && current_chunk->keys[i] == key) {
                    match_count++;
                }
            }
            current_chunk = current_chunk->next.load(std::memory_order_acquire);
        }
        return match_count;
    }

    size_t probe_and_count_batch(const KeyType* probe_keys, size_t batch_size) const {
        size_t total_match_count = 0;
        for (size_t i = 0; i < batch_size; ++i) {
            const auto& key = probe_keys[i];
            uint64_t hash = hasher_(key);
            uint64_t bucket_idx = get_bucket_idx(hash);
            uint8_t probe_tag = get_hash_tag(hash);
            
            Chunk* current_chunk = buckets_[bucket_idx].load(std::memory_order_acquire);
            
            while (current_chunk != nullptr) {
                if (current_chunk->next.load(std::memory_order_relaxed)) {
                    __builtin_prefetch(current_chunk->next.load(std::memory_order_relaxed), 0, 0);
                }
                uint32_t count = std::min((uint32_t)ChunkSize, current_chunk->count.load(std::memory_order_acquire));

#if defined(__AVX2__)
                const __m256i v_probe_tag = _mm256_set1_epi8(probe_tag);
                size_t j = 0;
                for (; j + 31 < count; j += 32) {
                    __m256i v_chunk_tags = _mm256_loadu_si256((const __m256i*)&current_chunk->tags[j]);
                    __m256i v_cmp_mask = _mm256_cmpeq_epi8(v_probe_tag, v_chunk_tags);
                    uint32_t bitmask = _mm256_movemask_epi8(v_cmp_mask);
                    
                    while (bitmask != 0) {
                        int pos = __builtin_ctz(bitmask);
                        if (current_chunk->keys[j + pos] == key) {
                            total_match_count++;
                        }
                        bitmask &= bitmask - 1;
                    }
                }
                for (; j < count; ++j) {
                    if (current_chunk->tags[j] == probe_tag && current_chunk->keys[j] == key) {
                        total_match_count++;
                    }
                }
#else
                for (uint32_t j = 0; j < count; ++j) {
                    if (current_chunk->tags[j] == probe_tag && current_chunk->keys[j] == key) {
                        total_match_count++;
                    }
                }
#endif
                current_chunk = current_chunk->next.load(std::memory_order_acquire);
            }
        }
        return total_match_count;
    }
    
    size_t probe_batch_write(const KeyType* probe_keys, size_t batch_size,
                             KeyType* out_keys_ptr, ValueType* out_values_ptr) const {
        size_t matches_written = 0;
        for (size_t i = 0; i < batch_size; ++i) {
            const auto& key = probe_keys[i];
            uint64_t hash = hasher_(key);
            uint64_t bucket_idx = get_bucket_idx(hash);
            uint8_t probe_tag = get_hash_tag(hash);
            
            Chunk* current_chunk = buckets_[bucket_idx].load(std::memory_order_acquire);
            
            while (current_chunk != nullptr) {
                if (current_chunk->next.load(std::memory_order_relaxed)) {
                    __builtin_prefetch(current_chunk->next.load(std::memory_order_relaxed), 0, 0);
                }
                uint32_t count = std::min((uint32_t)ChunkSize, current_chunk->count.load(std::memory_order_acquire));

#if defined(__AVX2__)
                const __m256i v_probe_tag = _mm256_set1_epi8(probe_tag);
                size_t j = 0;
                for (; j + 31 < count; j += 32) {
                    __m256i v_chunk_tags = _mm256_loadu_si256((const __m256i*)&current_chunk->tags[j]);
                    __m256i v_cmp_mask = _mm256_cmpeq_epi8(v_probe_tag, v_chunk_tags);
                    uint32_t bitmask = _mm256_movemask_epi8(v_cmp_mask);
                    
                    while (bitmask != 0) {
                        int pos = __builtin_ctz(bitmask);
                        if (current_chunk->keys[j + pos] == key) {
                            out_keys_ptr[matches_written] = key;
                            out_values_ptr[matches_written] = current_chunk->values[j + pos];
                            matches_written++;
                        }
                        bitmask &= bitmask - 1;
                    }
                }
                for (; j < count; ++j) {
                     if (current_chunk->tags[j] == probe_tag && current_chunk->keys[j] == key) {
                        out_keys_ptr[matches_written] = key;
                        out_values_ptr[matches_written] = current_chunk->values[j];
                        matches_written++;
                    }
                }
#else
                for (uint32_t j = 0; j < count; ++j) {
                    if (current_chunk->tags[j] == probe_tag && current_chunk->keys[j] == key) {
                        out_keys_ptr[matches_written] = key;
                        out_values_ptr[matches_written] = current_chunk->values[j];
                        matches_written++;
                    }
                }
#endif
                current_chunk = current_chunk->next.load(std::memory_order_acquire);
            }
        }
        return matches_written;
    }
};

using HT = UnchainedHashTable<uint32_t, uint32_t>;

// --- Public Functions and Passes ---
// (The following code remains unchanged as it depends on the HT class interface, not its implementation)

std::pair<size_t, std::vector<size_t>>
count_scalar_pass(const HT& ht, const uint32_t* probe_keys_ptr, size_t probe_size, size_t num_threads) {
    std::vector<std::thread> threads;
    size_t work_per_thread = (probe_size + num_threads - 1) / num_threads;
    std::vector<PaddedCounter> counts(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread;
        size_t end = std::min(start + work_per_thread, probe_size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            size_t local_count = 0;
            for (size_t j = start; j < end; ++j) {
                local_count += ht.probe_and_count(probe_keys_ptr[j]);
            }
            counts[i].value.store(local_count, std::memory_order_relaxed);
        });
    }
    for (auto& t : threads) { t.join(); }

    size_t total_results = 0;
    std::vector<size_t> offsets(num_threads + 1, 0);
    for (size_t i = 0; i < num_threads; ++i) {
        size_t count = counts[i].value.load(std::memory_order_relaxed);
        offsets[i + 1] = offsets[i] + count;
        total_results += count;
    }
    return {total_results, offsets};
}


std::pair<size_t, std::vector<size_t>>
count_batch_pass(const HT& ht, const uint32_t* probe_keys_ptr, size_t probe_size, size_t num_threads) {
    std::vector<std::thread> threads;
    size_t work_per_thread = (probe_size + num_threads - 1) / num_threads;
    std::vector<PaddedCounter> counts(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread;
        size_t end = std::min(start + work_per_thread, probe_size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            size_t local_count = ht.probe_and_count_batch(&probe_keys_ptr[start], end - start);
            counts[i].value.store(local_count, std::memory_order_relaxed);
        });
    }
    for (auto& t : threads) { t.join(); }

    size_t total_results = 0;
    std::vector<size_t> offsets(num_threads + 1, 0);
    for (size_t i = 0; i < num_threads; ++i) {
        size_t count = counts[i].value.load(std::memory_order_relaxed);
        offsets[i + 1] = offsets[i] + count;
        total_results += count;
    }
    return {total_results, offsets};
}

py::int_ hash_join_count_scalar(py::array_t<uint32_t> build_keys,
                                py::array_t<uint32_t> build_values,
                                py::array_t<uint32_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();

    HT ht(build_keys_buf.size, build_keys_buf.size);
    ht.build(static_cast<uint32_t*>(build_keys_buf.ptr),
             static_cast<uint32_t*>(build_values_buf.ptr),
             build_keys_buf.size);

    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "num_threads = " << num_threads << std::endl;
    auto [total_results, offsets] = count_scalar_pass(ht, static_cast<uint32_t*>(probe_keys_buf.ptr), probe_keys_buf.size, num_threads);
    
    return py::int_(total_results);
}

py::tuple hash_join_scalar(py::array_t<uint32_t> build_keys,
                           py::array_t<uint32_t> build_values,
                           py::array_t<uint32_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();
    const uint32_t* probe_keys_ptr = static_cast<uint32_t*>(probe_keys_buf.ptr);
    size_t probe_size = probe_keys_buf.size;

    HT ht(build_keys_buf.size, build_keys_buf.size);
    ht.build(static_cast<uint32_t*>(build_keys_buf.ptr),
             static_cast<uint32_t*>(build_values_buf.ptr),
             build_keys_buf.size);

    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "num_threads = " << num_threads << std::endl;
    auto [total_results, offsets] = count_scalar_pass(ht, probe_keys_ptr, probe_size, num_threads);

    py::array_t<uint32_t> result_keys(total_results);
    py::array_t<uint32_t> result_values(total_results);
    uint32_t* result_keys_ptr = static_cast<uint32_t*>(result_keys.request().ptr);
    uint32_t* result_values_ptr = static_cast<uint32_t*>(result_values.request().ptr);

    std::vector<std::thread> threads;
    size_t work_per_thread = (probe_size + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread;
        size_t end = std::min(start + work_per_thread, probe_size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            std::vector<uint32_t> local_results;
            size_t current_offset = offsets[i];
            for (size_t j = start; j < end; ++j) {
                local_results.clear();
                ht.probe(probe_keys_ptr[j], local_results);
                for (const auto& val : local_results) {
                    result_keys_ptr[current_offset] = probe_keys_ptr[j];
                    result_values_ptr[current_offset] = val;
                    current_offset++;
                }
            }
        });
    }
    for (auto& t : threads) { t.join(); }

    return py::make_tuple(result_keys, result_values);
}


py::int_ hash_join_count_batch(py::array_t<uint32_t> build_keys,
                               py::array_t<uint32_t> build_values,
                               py::array_t<uint32_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();

    HT ht(build_keys_buf.size, build_keys_buf.size);
    ht.build(static_cast<uint32_t*>(build_keys_buf.ptr),
             static_cast<uint32_t*>(build_values_buf.ptr),
             build_keys_buf.size);

    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "num_threads = " << num_threads << std::endl;
    auto [total_results, offsets] = count_batch_pass(ht, static_cast<uint32_t*>(probe_keys_buf.ptr), probe_keys_buf.size, num_threads);

    return py::int_(total_results);
}


py::tuple hash_join_batch(py::array_t<uint32_t> build_keys,
                          py::array_t<uint32_t> build_values,
                          py::array_t<uint32_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();
    const uint32_t* probe_keys_ptr = static_cast<uint32_t*>(probe_keys_buf.ptr);
    size_t probe_size = probe_keys_buf.size;

    HT ht(build_keys_buf.size, build_keys_buf.size);
    ht.build(static_cast<uint32_t*>(build_keys_buf.ptr),
             static_cast<uint32_t*>(build_values_buf.ptr),
             build_keys_buf.size);

    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "num_threads = " << num_threads << std::endl;
    auto [total_results, offsets] = count_batch_pass(ht, probe_keys_ptr, probe_size, num_threads);

    py::array_t<uint32_t> result_keys(total_results);
    py::array_t<uint32_t> result_values(total_results);
    uint32_t* result_keys_ptr = static_cast<uint32_t*>(result_keys.request().ptr);
    uint32_t* result_values_ptr = static_cast<uint32_t*>(result_values.request().ptr);

    std::vector<std::thread> threads;
    size_t work_per_thread = (probe_size + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread;
        size_t end = std::min(start + work_per_thread, probe_size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            ht.probe_batch_write(&probe_keys_ptr[start], end - start,
                                 result_keys_ptr + offsets[i],
                                 result_values_ptr + offsets[i]);
        });
    }
    for (auto& t : threads) { t.join(); }

    return py::make_tuple(result_keys, result_values);
}

void initialize_memory_system() {
    // Initialize mimalloc without overriding system malloc
    // This allows coexistence with numpy/pandas memory allocators
    mi_version();  // Simple call to ensure mimalloc is initialized
    return;
}

PYBIND11_MODULE(flash_join, m) {
    m.doc() = "A high-performance hash join with various optimization strategies"; 
    m.def("hash_join", &hash_join_batch, 
          "Performs a hash join and returns the full result",
          py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));

    m.def("hash_join_count", &hash_join_count_batch, 
          "Performs a hash join and returns only the count of results",
          py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));

    m.def("hash_join_scalar", &hash_join_scalar, "Original two-pass hash join (scalar probe)");
    m.def("hash_join_count_scalar", &hash_join_count_scalar, "Original optimized count hash join (scalar probe)");
    m.def("initialize", &initialize_memory_system, "Initializes the custom memory allocator.");
}