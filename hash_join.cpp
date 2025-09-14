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
#include <tuple>

// AVX2 and other intrinsics
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <nmmintrin.h> // Header for SSE4.2 intrinsics

#define PYBIND11_NO_ATEXIT
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Re-enable mimalloc with proper configuration
#define MI_OVERRIDE 0  // Don't override system malloc
#define MI_MALLOC_OVERRIDE 0  // Don't override malloc
#define MI_USENOPROCS 1
#define MI_NO_DEINIT 1  
#include <mimalloc.h>

namespace py = pybind11;

// --- Constants for Radix Partitioning ---
// The number of bits to use for partitioning. 8 bits = 256 partitions.
constexpr size_t RADIX_BITS = 8;
// The total number of partitions.
constexpr size_t NUM_PARTITIONS = 1 << RADIX_BITS;


uint64_t hash64(uint64_t key, uint32_t seed) {
    uint64_t k = 0x8648DBDB;          // Mixing constant
    uint64_t crc = _mm_crc32_u64(seed, key);   // crc32
    return crc * ((k << 32) + 1); // imul
}

template<typename KeyType>
struct Hasher {
    uint64_t operator()(const KeyType& key) const { return hash64(key, 0xAAAAAAAA); }
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

    ~UnchainedHashTable() {
        for (size_t i = 0; i < buckets_.size(); ++i) {
            Chunk* current_chunk = buckets_[i].load(std::memory_order_relaxed);
            while (current_chunk != nullptr) {
                Chunk* next = current_chunk->next.load(std::memory_order_relaxed);
                // We must decide how to free it.
                // The free_chunk logic correctly handles pool vs. non-pool chunks.
                free_chunk(current_chunk);
                current_chunk = next;
            }
        }
        // memory_pool_ is freed automatically by unique_ptr after this loop.
    }
    
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
        // Safety check
        if (!chunk) return;

        char* chunk_addr = reinterpret_cast<char*>(chunk);
        char* pool_start = memory_pool_.get();
        
        // Check if the chunk is NOT from the pool
        // Also check if pool_start is valid before dereferencing
        if (!pool_start || !(chunk_addr >= pool_start && chunk_addr < pool_start + pool_size_bytes_)) {
            chunk->~Chunk(); // Explicitly call destructor for non-pod members
            mi_free(chunk); 
        } else {
            // It's from the pool. We could optionally call its destructor if needed.
            // For current Chunk struct, it's not strictly necessary.
             chunk->~Chunk(); 
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

    // Single-threaded build, no locks needed. Used in Radix Join.
    void build_local(const KeyType* keys, const ValueType* values, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            insert_local(keys[i], values[i]);
        }
    }
    
    // Single-threaded insert, no locks needed.
    void insert_local(const KeyType& key, const ValueType& value) {
        uint64_t hash = hasher_(key);
        uint64_t bucket_idx = get_bucket_idx(hash);
        uint8_t tag = get_hash_tag(hash);
        
        Chunk* current_chunk = buckets_[bucket_idx].load(std::memory_order_relaxed);

        if (current_chunk == nullptr) {
            Chunk* new_chunk = alloc_chunk_from_pool();
            new_chunk->tags[0] = tag;
            new_chunk->keys[0] = key;
            new_chunk->values[0] = value;
            new_chunk->count.store(1, std::memory_order_relaxed);
            buckets_[bucket_idx].store(new_chunk, std::memory_order_relaxed);
            return;
        }
        
        while (true) {
            // Relaxed ordering is sufficient for single-threaded context.
            uint32_t slot = current_chunk->count.load(std::memory_order_relaxed);
            if (slot < ChunkSize) {
                current_chunk->tags[slot] = tag;
                current_chunk->keys[slot] = key;
                current_chunk->values[slot] = value;
                current_chunk->count.store(slot + 1, std::memory_order_relaxed);
                return;
            }
            
            Chunk* next_chunk = current_chunk->next.load(std::memory_order_relaxed);
            if (next_chunk == nullptr) {
                Chunk* new_chunk = alloc_chunk_from_pool();
                new_chunk->tags[0] = tag;
                new_chunk->keys[0] = key;
                new_chunk->values[0] = value;
                new_chunk->count.store(1, std::memory_order_relaxed);
                current_chunk->next.store(new_chunk, std::memory_order_relaxed);
                return;
            }
            current_chunk = next_chunk;
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

    size_t probe_write(const KeyType& key, KeyType* out_keys_ptr, ValueType* out_values_ptr) const {
        uint64_t hash = hasher_(key);
        uint64_t bucket_idx = get_bucket_idx(hash);
        uint8_t probe_tag = get_hash_tag(hash);
        size_t matches_written = 0;

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
            // Scalar remainder
            for (; j < count; ++j) {
                if (current_chunk->tags[j] == probe_tag && current_chunk->keys[j] == key) {
                    out_keys_ptr[matches_written] = key;
                    out_values_ptr[matches_written] = current_chunk->values[j];
                    matches_written++;
                }
            }
    #else
            // Original scalar implementation
            for (uint32_t i = 0; i < count; ++i) {
                if (current_chunk->tags[i] == probe_tag && current_chunk->keys[i] == key) {
                    out_keys_ptr[matches_written] = key;
                    out_values_ptr[matches_written] = current_chunk->values[i];
                    matches_written++;
                }
            }
    #endif
            current_chunk = current_chunk->next.load(std::memory_order_acquire);
        }
        return matches_written;
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

using HT = UnchainedHashTable<uint64_t, uint64_t>;

// --- Original Implementation (Single Large Hash Table) ---

std::pair<size_t, std::vector<size_t>>
count_scalar_pass(const HT& ht, const uint64_t* probe_keys_ptr, size_t probe_size, size_t num_threads) {
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
count_batch_pass(const HT& ht, const uint64_t* probe_keys_ptr, size_t probe_size, size_t num_threads) {
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

py::int_ hash_join_count_scalar(py::array_t<uint64_t> build_keys,
                                py::array_t<uint64_t> build_values,
                                py::array_t<uint64_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();

    HT ht(build_keys_buf.size, build_keys_buf.size);
    ht.build(static_cast<uint64_t*>(build_keys_buf.ptr),
             static_cast<uint64_t*>(build_values_buf.ptr),
             build_keys_buf.size);

    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    auto [total_results, offsets] = count_scalar_pass(ht, static_cast<uint64_t*>(probe_keys_buf.ptr), probe_keys_buf.size, num_threads);
    
    return py::int_(total_results);
}

py::tuple hash_join_scalar(py::array_t<uint64_t> build_keys,
                           py::array_t<uint64_t> build_values,
                           py::array_t<uint64_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info probe_keys_buf = probe_keys.request();
    const uint64_t* probe_keys_ptr = static_cast<uint64_t*>(probe_keys_buf.ptr);
    size_t probe_size = probe_keys_buf.size;

    HT ht(build_keys_buf.size, build_keys_buf.size);
    ht.build(static_cast<uint64_t*>(build_keys_buf.ptr),
             static_cast<uint64_t*>(build_values.request().ptr),
             build_keys_buf.size);

    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    auto [total_results, offsets] = count_scalar_pass(ht, probe_keys_ptr, probe_size, num_threads);

    py::array_t<uint64_t> result_keys(total_results);
    py::array_t<uint64_t> result_values(total_results);
    uint64_t* result_keys_ptr = static_cast<uint64_t*>(result_keys.request().ptr);
    uint64_t* result_values_ptr = static_cast<uint64_t*>(result_values.request().ptr);

    std::vector<std::thread> threads;
    size_t work_per_thread = (probe_size + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread;
        size_t end = std::min(start + work_per_thread, probe_size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            size_t current_offset = offsets[i];
            for (size_t j = start; j < end; ++j) {
                size_t matches_found = ht.probe_write(
                    probe_keys_ptr[j],
                    result_keys_ptr + current_offset,
                    result_values_ptr + current_offset
                );
                current_offset += matches_found;
            }
        });
    }
    for (auto& t : threads) { t.join(); }

    return py::make_tuple(result_keys, result_values);
}


py::int_ hash_join_count_batch(py::array_t<uint64_t> build_keys,
                               py::array_t<uint64_t> build_values,
                               py::array_t<uint64_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();

    HT ht(build_keys_buf.size, build_keys_buf.size);
    ht.build(static_cast<uint64_t*>(build_keys_buf.ptr),
             static_cast<uint64_t*>(build_values_buf.ptr),
             build_keys_buf.size);

    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    auto [total_results, offsets] = count_batch_pass(ht, static_cast<uint64_t*>(probe_keys_buf.ptr), probe_keys_buf.size, num_threads);

    return py::int_(total_results);
}


py::tuple hash_join_batch(py::array_t<uint64_t> build_keys,
                          py::array_t<uint64_t> build_values,
                          py::array_t<uint64_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();
    const uint64_t* probe_keys_ptr = static_cast<uint64_t*>(probe_keys_buf.ptr);
    size_t probe_size = probe_keys_buf.size;

    HT ht(build_keys_buf.size, build_keys_buf.size);
    ht.build(static_cast<uint64_t*>(build_keys_buf.ptr),
             static_cast<uint64_t*>(build_values_buf.ptr),
             build_keys_buf.size);

    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    auto [total_results, offsets] = count_batch_pass(ht, probe_keys_ptr, probe_size, num_threads);

    py::array_t<uint64_t> result_keys(total_results);
    py::array_t<uint64_t> result_values(total_results);
    uint64_t* result_keys_ptr = static_cast<uint64_t*>(result_keys.request().ptr);
    uint64_t* result_values_ptr = static_cast<uint64_t*>(result_values.request().ptr);

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

// --- New Radix Partitioning Implementation ---

// Helper function to extract the partition index from a hash
inline size_t get_partition_idx(uint64_t hash) {
    // Use the most significant bits for partitioning
    return hash >> (64 - RADIX_BITS);
}

// Partitions key-value pairs
std::tuple<std::vector<uint64_t>, std::vector<uint64_t>, std::vector<size_t>>
parallel_radix_partition_kv(const uint64_t* keys, const uint64_t* values, size_t size, size_t num_threads) {
    std::vector<uint64_t> out_keys(size);
    std::vector<uint64_t> out_values(size);
    std::vector<size_t> partition_offsets(NUM_PARTITIONS + 1, 0);
    Hasher<uint64_t> hasher;

    // Phase 1: Build histograms in parallel
    std::vector<std::vector<size_t>> histograms(num_threads, std::vector<size_t>(NUM_PARTITIONS, 0));
    size_t work_per_thread = (size + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread;
        size_t end = std::min(start + work_per_thread, size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            for (size_t j = start; j < end; ++j) {
                uint64_t hash = hasher(keys[j]);
                histograms[i][get_partition_idx(hash)]++;
            }
        });
    }
    for (auto& t : threads) { t.join(); }

    // Phase 2: Calculate prefix sums to determine write positions
    for (size_t i = 0; i < num_threads; ++i) {
        for (size_t j = 0; j < NUM_PARTITIONS; ++j) {
            partition_offsets[j + 1] += histograms[i][j];
        }
    }
    for (size_t i = 1; i <= NUM_PARTITIONS; ++i) {
        partition_offsets[i] += partition_offsets[i - 1];
    }
    
    // Phase 3: Scatter data to partitioned buffers in parallel
    std::vector<size_t> current_offsets = partition_offsets;
    threads.clear();
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread;
        size_t end = std::min(start + work_per_thread, size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            std::vector<size_t> local_write_offsets(NUM_PARTITIONS);
            // Determine this thread's starting write position for each partition
            for(size_t p=0; p < NUM_PARTITIONS; ++p) {
                size_t offset = partition_offsets[p];
                for(size_t t=0; t < i; ++t) {
                    offset += histograms[t][p];
                }
                local_write_offsets[p] = offset;
            }

            for (size_t j = start; j < end; ++j) {
                uint64_t hash = hasher(keys[j]);
                size_t p_idx = get_partition_idx(hash);
                size_t write_pos = local_write_offsets[p_idx]++;
                out_keys[write_pos] = keys[j];
                out_values[write_pos] = values[j];
            }
        });
    }
    for (auto& t : threads) { t.join(); }
    
    return {std::move(out_keys), std::move(out_values), std::move(partition_offsets)};
}

// Overload for partitioning only keys (for the probe side)
std::tuple<std::vector<uint64_t>, std::vector<size_t>>
parallel_radix_partition_k(const uint64_t* keys, size_t size, size_t num_threads) {
    std::vector<uint64_t> out_keys(size);
    std::vector<size_t> partition_offsets(NUM_PARTITIONS + 1, 0);
    Hasher<uint64_t> hasher;

    // This logic is identical to the KV version, just without handling values.
    std::vector<std::vector<size_t>> histograms(num_threads, std::vector<size_t>(NUM_PARTITIONS, 0));
    size_t work_per_thread = (size + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;

    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread;
        size_t end = std::min(start + work_per_thread, size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            for (size_t j = start; j < end; ++j) {
                uint64_t hash = hasher(keys[j]);
                histograms[i][get_partition_idx(hash)]++;
            }
        });
    }
    for (auto& t : threads) { t.join(); }

    for (size_t i = 0; i < num_threads; ++i) {
        for (size_t j = 0; j < NUM_PARTITIONS; ++j) {
            partition_offsets[j + 1] += histograms[i][j];
        }
    }
    for (size_t i = 1; i <= NUM_PARTITIONS; ++i) {
        partition_offsets[i] += partition_offsets[i - 1];
    }
    
    threads.clear();
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start = i * work_per_thread;
        size_t end = std::min(start + work_per_thread, size);
        if (start >= end) continue;
        threads.emplace_back([&, i, start, end]() {
            std::vector<size_t> local_write_offsets(NUM_PARTITIONS);
            for(size_t p=0; p < NUM_PARTITIONS; ++p) {
                size_t offset = partition_offsets[p];
                for(size_t t=0; t < i; ++t) {
                    offset += histograms[t][p];
                }
                local_write_offsets[p] = offset;
            }

            for (size_t j = start; j < end; ++j) {
                uint64_t hash = hasher(keys[j]);
                size_t p_idx = get_partition_idx(hash);
                size_t write_pos = local_write_offsets[p_idx]++;
                out_keys[write_pos] = keys[j];
            }
        });
    }
    for (auto& t : threads) { t.join(); }
    
    return {std::move(out_keys), std::move(partition_offsets)};
}


py::int_ hash_join_count_radix(py::array_t<uint64_t> build_keys,
                                py::array_t<uint64_t> build_values,
                                py::array_t<uint64_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();
    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());

    // 1. Partition both build and probe sides
    auto [p_build_keys, p_build_values, build_offsets] = parallel_radix_partition_kv(
        static_cast<uint64_t*>(build_keys_buf.ptr), static_cast<uint64_t*>(build_values_buf.ptr), build_keys_buf.size, num_threads);
    
    auto [p_probe_keys, probe_offsets] = parallel_radix_partition_k(
        static_cast<uint64_t*>(probe_keys_buf.ptr), probe_keys_buf.size, num_threads);

    // 2. Join partitions in parallel
    std::atomic<size_t> total_results{0};
    std::vector<std::thread> threads;
    for (size_t p_idx = 0; p_idx < NUM_PARTITIONS; ++p_idx) {
        threads.emplace_back([&, p_idx]() {
            size_t build_start = build_offsets[p_idx];
            size_t build_end = build_offsets[p_idx + 1];
            size_t build_size = build_end - build_start;

            size_t probe_start = probe_offsets[p_idx];
            size_t probe_end = probe_offsets[p_idx + 1];
            size_t probe_size = probe_end - probe_start;

            if (build_size == 0 || probe_size == 0) return;

            // Build a local, non-thread-safe hash table for this partition
            HT local_ht(build_size, build_size);
            local_ht.build_local(&p_build_keys[build_start], &p_build_values[build_start], build_size);
            
            // Probe and count
            size_t local_count = local_ht.probe_and_count_batch(&p_probe_keys[probe_start], probe_size);
            total_results.fetch_add(local_count, std::memory_order_relaxed);
        });
    }
    for(auto& t : threads) { t.join(); }

    return py::int_(total_results.load());
}

py::tuple hash_join_radix(py::array_t<uint64_t> build_keys,
                          py::array_t<uint64_t> build_values,
                          py::array_t<uint64_t> probe_keys) {
    py::buffer_info build_keys_buf = build_keys.request();
    py::buffer_info build_values_buf = build_values.request();
    py::buffer_info probe_keys_buf = probe_keys.request();
    size_t num_threads = std::max(1u, std::thread::hardware_concurrency());

    // 1. Partition both build and probe sides
    auto [p_build_keys, p_build_values, build_offsets] = parallel_radix_partition_kv(
        static_cast<uint64_t*>(build_keys_buf.ptr), static_cast<uint64_t*>(build_values_buf.ptr), build_keys_buf.size, num_threads);
    
    auto [p_probe_keys, probe_offsets] = parallel_radix_partition_k(
        static_cast<uint64_t*>(probe_keys_buf.ptr), probe_keys_buf.size, num_threads);

    // 2. First pass: count results per partition to pre-allocate memory
    std::vector<PaddedCounter> partition_counts(NUM_PARTITIONS);
    std::vector<std::thread> threads;
    
    size_t work_per_thread_part = (NUM_PARTITIONS + num_threads - 1) / num_threads;
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_p = i * work_per_thread_part;
        size_t end_p = std::min(start_p + work_per_thread_part, NUM_PARTITIONS);
        if (start_p >= end_p) continue;

        threads.emplace_back([&, start_p, end_p]() {
            for (size_t p_idx = start_p; p_idx < end_p; ++p_idx) {
                size_t build_size = build_offsets[p_idx + 1] - build_offsets[p_idx];
                size_t probe_size = probe_offsets[p_idx + 1] - probe_offsets[p_idx];
                if (build_size == 0 || probe_size == 0) {
                    partition_counts[p_idx].value.store(0, std::memory_order_relaxed);
                    continue;
                }
                
                HT local_ht(build_size, build_size);
                local_ht.build_local(&p_build_keys[build_offsets[p_idx]], &p_build_values[build_offsets[p_idx]], build_size);
                
                size_t local_count = local_ht.probe_and_count_batch(&p_probe_keys[probe_offsets[p_idx]], probe_size);
                partition_counts[p_idx].value.store(local_count, std::memory_order_relaxed);
            }
        });
    }
    for(auto& t : threads) { t.join(); }

    // Calculate total size and offsets for writing results
    std::vector<size_t> result_offsets(NUM_PARTITIONS + 1, 0);
    for (size_t i = 0; i < NUM_PARTITIONS; ++i) {
        result_offsets[i+1] = result_offsets[i] + partition_counts[i].value.load(std::memory_order_relaxed);
    }
    size_t total_results = result_offsets[NUM_PARTITIONS];

    // Allocate final result arrays
    py::array_t<uint64_t> result_keys(total_results);
    py::array_t<uint64_t> result_values(total_results);
    uint64_t* result_keys_ptr = static_cast<uint64_t*>(result_keys.request().ptr);
    uint64_t* result_values_ptr = static_cast<uint64_t*>(result_values.request().ptr);

    // 3. Second pass: build and probe again to materialize results
    threads.clear();
    for (size_t i = 0; i < num_threads; ++i) {
        size_t start_p = i * work_per_thread_part;
        size_t end_p = std::min(start_p + work_per_thread_part, NUM_PARTITIONS);
        if (start_p >= end_p) continue;
        
        threads.emplace_back([&, start_p, end_p]() {
            for (size_t p_idx = start_p; p_idx < end_p; ++p_idx) {
                size_t build_size = build_offsets[p_idx + 1] - build_offsets[p_idx];
                size_t probe_size = probe_offsets[p_idx + 1] - probe_offsets[p_idx];
                if (build_size == 0 || probe_size == 0) continue;
                
                HT local_ht(build_size, build_size);
                local_ht.build_local(&p_build_keys[build_offsets[p_idx]], &p_build_values[build_offsets[p_idx]], build_size);
                
                local_ht.probe_batch_write(
                    &p_probe_keys[probe_offsets[p_idx]], probe_size,
                    result_keys_ptr + result_offsets[p_idx],
                    result_values_ptr + result_offsets[p_idx]
                );
            }
        });
    }
    for(auto& t : threads) { t.join(); }
    
    return py::make_tuple(result_keys, result_values);
}


void initialize_memory_system() {
    mi_version();
    return;
}

PYBIND11_MODULE(flash_join, m) {
    initialize_memory_system();
    m.doc() = "A high-performance hash join with various optimization strategies"; 

    // Original implementation (single global hash table)
    m.def("hash_join", &hash_join_batch, 
          "Performs a hash join using a single concurrent hash table",
          py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));

    m.def("hash_join_count", &hash_join_count_batch, 
          "Performs a hash join and returns only the count (single concurrent hash table)",
          py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));

    // Radix partitioning implementation
    m.def("hash_join_radix", &hash_join_radix, 
          "Performs a hash join using radix partitioning",
          py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));

    m.def("hash_join_count_radix", &hash_join_count_radix, 
          "Performs a hash join and returns only the count (radix partitioning)",
          py::arg("build_keys"), py::arg("build_values"), py::arg("probe_keys"));

    // Exposing older scalar versions for benchmarking
    m.def("hash_join_scalar", &hash_join_scalar, "Original two-pass hash join (scalar probe)");
    m.def("hash_join_count_scalar", &hash_join_count_scalar, "Original optimized count hash join (scalar probe)");

    m.def("initialize", &initialize_memory_system, "Initializes the custom memory allocator.");
}