#pragma once

#include <cstdint>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <list>
#include <vector>
#include <array>
#include <atomic>
#include <utility>

struct ggml_tensor;

// Identifies one expert's one weight matrix
struct llama_expert_key {
    int32_t layer;        // 0..n_layer-1
    int32_t expert_idx;   // 0..n_expert-1
    int32_t weight_type;  // 0=up, 1=gate, 2=down

    bool operator==(const llama_expert_key & other) const {
        return layer == other.layer
            && expert_idx == other.expert_idx
            && weight_type == other.weight_type;
    }
};

struct llama_expert_key_hash {
    size_t operator()(const llama_expert_key & k) const {
        // Combine the three fields into a single hash
        size_t h = 0;
        h ^= std::hash<int32_t>{}(k.layer)       + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(k.expert_idx)   + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int32_t>{}(k.weight_type)  + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Where to find an expert's weights on disk
struct llama_expert_disk_info {
    int      fd;              // open file descriptor for GGUF
    size_t   file_offset;     // absolute byte offset in GGUF file
    size_t   size_bytes;      // bytes for this one expert slice
    int      ggml_type;       // quantization type (enum ggml_type)
};

// One cached expert weight slab
struct llama_expert_entry {
    llama_expert_key key;
    void           * data;          // page-aligned buffer
    size_t           size_bytes;
    std::list<llama_expert_key>::iterator lru_it;
};

struct llama_expert_cache_stats {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    size_t   bytes_used;
    size_t   bytes_capacity;

    double hit_rate() const {
        uint64_t total = hits + misses;
        return total > 0 ? (double)hits / total : 0.0;
    }
};

class llama_expert_cache {
public:
    explicit llama_expert_cache(size_t max_bytes);
    ~llama_expert_cache();

    // Non-copyable, non-movable
    llama_expert_cache(const llama_expert_cache &) = delete;
    llama_expert_cache & operator=(const llama_expert_cache &) = delete;

    // Returns pointer to cached expert data.
    // Loads from disk on miss. Thread-safe.
    void * ensure(const llama_expert_key & key,
                  const llama_expert_disk_info & disk_info);

    // Get cached data or allocate empty slot for caller to fill.
    // Returns {pointer, true} on cache hit, {pointer, false} on miss (caller must fill).
    // Returns {nullptr, false} on allocation failure.
    std::pair<void *, bool> get_or_alloc(const llama_expert_key & key, size_t size_bytes);

    // Update LRU ordering (call when expert is accessed but already loaded)
    void touch(const llama_expert_key & key);

    // Check if an expert is cached without loading
    bool contains(const llama_expert_key & key) const;

    // Get statistics
    llama_expert_cache_stats get_stats() const;

    // Reset stats counters
    void reset_stats();

private:
    void evict_until_free(size_t needed);
    void * alloc_aligned(size_t size);
    void free_aligned(void * ptr, size_t size);
    void * load_from_disk(const llama_expert_disk_info & info);

    size_t max_bytes_;
    size_t used_bytes_;

    std::list<llama_expert_key> lru_order_;  // front = most recent
    std::unordered_map<llama_expert_key, llama_expert_entry,
                       llama_expert_key_hash> cache_;

    mutable std::mutex mutex_;
    llama_expert_cache_stats stats_;
};
