#include "llama-expert-cache.h"

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <algorithm>

#ifdef __APPLE__
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#endif

// Page size for aligned allocation (matches Apple Silicon and most x86)
static constexpr size_t ALLOC_ALIGNMENT = 4096;

static size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

llama_expert_cache::llama_expert_cache(size_t max_bytes)
    : max_bytes_(max_bytes)
    , used_bytes_(0)
    , stats_{0, 0, 0, 0, max_bytes} {
}

llama_expert_cache::~llama_expert_cache() {
    for (auto & [key, entry] : cache_) {
        free_aligned(entry.data, entry.size_bytes);
    }
    cache_.clear();
    lru_order_.clear();
    used_bytes_ = 0;
}

void * llama_expert_cache::alloc_aligned(size_t size) {
    size_t alloc_size = align_up(size, ALLOC_ALIGNMENT);
#ifdef _WIN32
    void * ptr = _aligned_malloc(alloc_size, ALLOC_ALIGNMENT);
#else
    void * ptr = nullptr;
    int ret = posix_memalign(&ptr, ALLOC_ALIGNMENT, alloc_size);
    if (ret != 0) {
        ptr = nullptr;
    }
#endif
    return ptr;
}

void llama_expert_cache::free_aligned(void * ptr, size_t /*size*/) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void * llama_expert_cache::load_from_disk(const llama_expert_disk_info & info) {
    void * buf = alloc_aligned(info.size_bytes);
    if (!buf) return nullptr;

#ifdef _WIN32
    // Windows: use _lseeki64 + _read or ReadFile
    _lseeki64(info.fd, info.file_offset, SEEK_SET);
    size_t remaining = info.size_bytes;
    char * dst = (char *)buf;
    while (remaining > 0) {
        int chunk = (int)std::min(remaining, (size_t)INT_MAX);
        int n = _read(info.fd, dst, chunk);
        if (n <= 0) {
            free_aligned(buf, info.size_bytes);
            return nullptr;
        }
        dst += n;
        remaining -= n;
    }
#else
    // POSIX: use pread for thread-safe positional read (no seek mutex needed)
    size_t remaining = info.size_bytes;
    char * dst = (char *)buf;
    off_t offset = (off_t)info.file_offset;
    while (remaining > 0) {
        ssize_t n = pread(info.fd, dst, remaining, offset);
        if (n <= 0) {
            free_aligned(buf, info.size_bytes);
            return nullptr;
        }
        dst += n;
        offset += n;
        remaining -= n;
    }
#endif

    return buf;
}

void llama_expert_cache::evict_until_free(size_t needed) {
    while (used_bytes_ + needed > max_bytes_ && !lru_order_.empty()) {
        // Evict least recently used (back of list)
        auto evict_key = lru_order_.back();
        lru_order_.pop_back();

        auto it = cache_.find(evict_key);
        if (it != cache_.end()) {
            used_bytes_ -= it->second.size_bytes;
            free_aligned(it->second.data, it->second.size_bytes);
            cache_.erase(it);
            stats_.evictions++;
        }
    }
}

void * llama_expert_cache::ensure(const llama_expert_key & key,
                                   const llama_expert_disk_info & disk_info) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check cache
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Hit: move to front of LRU
        stats_.hits++;
        lru_order_.erase(it->second.lru_it);
        lru_order_.push_front(key);
        it->second.lru_it = lru_order_.begin();
        return it->second.data;
    }

    // Miss: load from disk
    stats_.misses++;

    size_t alloc_size = align_up(disk_info.size_bytes, ALLOC_ALIGNMENT);

    // Evict until we have space
    evict_until_free(alloc_size);

    // Load from disk (this does I/O while holding the lock —
    // acceptable for now, can be optimized with async prefetch later)
    void * data = load_from_disk(disk_info);
    if (!data) {
        return nullptr;
    }

    // Insert into cache
    lru_order_.push_front(key);
    llama_expert_entry entry;
    entry.key = key;
    entry.data = data;
    entry.size_bytes = alloc_size;
    entry.lru_it = lru_order_.begin();
    cache_[key] = entry;
    used_bytes_ += alloc_size;
    stats_.bytes_used = used_bytes_;

    return data;
}

std::pair<void *, bool> llama_expert_cache::get_or_alloc(
        const llama_expert_key & key, size_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check cache
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        stats_.hits++;
        lru_order_.erase(it->second.lru_it);
        lru_order_.push_front(key);
        it->second.lru_it = lru_order_.begin();
        return {it->second.data, true};  // hit
    }

    // Miss
    stats_.misses++;

    size_t alloc_size = align_up(size_bytes, ALLOC_ALIGNMENT);
    evict_until_free(alloc_size);

    void * data = alloc_aligned(alloc_size);
    if (!data) {
        return {nullptr, false};
    }

    lru_order_.push_front(key);
    llama_expert_entry entry;
    entry.key = key;
    entry.data = data;
    entry.size_bytes = alloc_size;
    entry.lru_it = lru_order_.begin();
    cache_[key] = entry;
    used_bytes_ += alloc_size;
    stats_.bytes_used = used_bytes_;

    return {data, false};  // miss — caller must fill
}

void llama_expert_cache::touch(const llama_expert_key & key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        lru_order_.erase(it->second.lru_it);
        lru_order_.push_front(key);
        it->second.lru_it = lru_order_.begin();
    }
}

bool llama_expert_cache::contains(const llama_expert_key & key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.find(key) != cache_.end();
}

llama_expert_cache_stats llama_expert_cache::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto s = stats_;
    s.bytes_used = used_bytes_;
    s.bytes_capacity = max_bytes_;
    return s;
}

void llama_expert_cache::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.hits = 0;
    stats_.misses = 0;
    stats_.evictions = 0;
}
