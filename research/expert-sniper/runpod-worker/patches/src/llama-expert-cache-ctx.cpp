#include "llama-expert-cache-ctx.h"
#include "llama-model.h"
#include "llama-hparams.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <algorithm>

#if !defined(_WIN32)
#include <sys/mman.h>
#endif

#ifndef MADV_WILLNEED
#define MADV_WILLNEED 3
#endif

// Initialize expert cache from model metadata
void llama_expert_cache_ctx::init(const llama_model & model, size_t cache_bytes) {
    const auto & hparams = model.hparams;

    n_expert      = (int)hparams.n_expert;
    n_expert_used = (int)hparams.n_expert_used;
    n_layers      = (int)hparams.n_layer;

    if (n_expert == 0 || n_expert_used == 0) {
        // Not a MoE model, nothing to do
        return;
    }

    // Create the cache
    cache = std::make_unique<llama_expert_cache>(cache_bytes);

    // Map layer expert tensors
    expert_tensors.resize(n_layers);
    expert_strides.resize(n_layers);

    for (int il = 0; il < n_layers; il++) {
        const auto & layer = model.layers[il];

        expert_tensors[il] = {
            layer.ffn_up_exps,    // 0 = up
            layer.ffn_gate_exps,  // 1 = gate
            layer.ffn_down_exps,  // 2 = down
        };

        for (int wt = 0; wt < 3; wt++) {
            ggml_tensor * t = expert_tensors[il][wt];
            if (t && t->ne[2] > 1) {
                // Stride per expert = nb[2] (stride along the expert dimension)
                expert_strides[il][wt] = t->nb[2];
            } else {
                expert_strides[il][wt] = 0;
            }
        }
    }

    // Allocate the active expert buffer
    // Worst case: n_expert_used experts × 3 weight types × max expert stride
    size_t max_stride = 0;
    for (int il = 0; il < n_layers; il++) {
        for (int wt = 0; wt < 3; wt++) {
            max_stride = std::max(max_stride, expert_strides[il][wt]);
        }
    }
    active_buffer_size = (size_t)n_expert_used * max_stride;
    active_buffer = malloc(active_buffer_size);

    GGML_ASSERT(active_buffer != nullptr);

    fprintf(stderr, "llama_expert_cache_ctx: initialized for %d layers, %d experts (%d used), "
            "cache = %.1f MB, stride = %.2f MB\n",
            n_layers, n_expert, n_expert_used,
            (double)cache_bytes / (1024*1024),
            (double)max_stride / (1024*1024));
}

std::pair<int, int> llama_expert_cache_ctx::identify_tensor(const ggml_tensor * t) const {
    for (int il = 0; il < n_layers; il++) {
        for (int wt = 0; wt < 3; wt++) {
            if (expert_tensors[il][wt] == t) {
                return {il, wt};
            }
        }
    }
    return {-1, -1};
}

void * llama_expert_cache_ctx::build_active_buffer(
        int layer, int weight_type,
        const int32_t * expert_ids, int n_ids) {

    const size_t stride = expert_strides[layer][weight_type];
    const ggml_tensor * stacked = expert_tensors[layer][weight_type];

    if (!stacked || stride == 0) return nullptr;

    // For each selected expert, either get from cache or copy from mmap'd tensor
    char * dst = (char *)active_buffer;
    for (int i = 0; i < n_ids; i++) {
        int eid = expert_ids[i];
        if (eid < 0 || eid >= n_expert) continue;

        llama_expert_key key = {(int32_t)layer, (int32_t)eid, (int32_t)weight_type};

        const char * expert_src = nullptr;

        if (cache) {
            auto [buf, hit] = cache->get_or_alloc(key, stride);
            if (buf) {
                if (!hit) {
                    // Cache miss: copy from mmap'd tensor data into cache.
                    // This may cause a page fault on first access, but subsequent
                    // accesses will hit the cache and avoid the page fault.
                    const char * src = (const char *)stacked->data + (size_t)eid * stride;
                    memcpy(buf, src, stride);
                }
                expert_src = (const char *)buf;
            }
        }

        if (!expert_src) {
            // No cache or alloc failed — read directly from stacked tensor
            expert_src = (const char *)stacked->data + (size_t)eid * stride;
        }

        memcpy(dst, expert_src, stride);
        dst += stride;
    }

    return active_buffer;
}

// Static eval callback — pre-caches expert weight pages before ggml_mul_mat_id.
// Phase 1: read-through cache that pre-faults mmap pages for active experts,
// keeping hot expert data in our LRU cache to prevent OS eviction.
// Phase 2 (future): tensor patching to avoid mmap entirely.
bool llama_expert_cache_ctx::eval_callback(
        struct ggml_tensor * t,
        bool ask,
        void * user_data) {

    if (!ask) {
        return true;  // "done" notification, nothing to restore yet
    }

    // Only intercept MUL_MAT_ID operations
    if (t->op != GGML_OP_MUL_MAT_ID) {
        return true;
    }

    auto * ctx = (llama_expert_cache_ctx *)user_data;

    // src[0] = stacked expert weights [ne0, ne1, n_expert]
    // src[2] = selected expert indices (from router top-k)
    ggml_tensor * expert_weights = t->src[0];
    ggml_tensor * expert_indices = t->src[2];

    if (!expert_weights || !expert_indices || !ctx->cache) {
        return true;
    }

    // Identify which layer and weight type
    auto [layer, weight_type] = ctx->identify_tensor(expert_weights);
    if (layer < 0) {
        return true;  // not an expert tensor we manage
    }

    // expert_indices data may not be accessible from CPU if on GPU.
    // For now, just pre-cache ALL expert slices for this layer/weight_type
    // that we haven't seen before. This warms the cache progressively.
    // The OS will keep our cache pages resident while evicting cold mmap pages.

    const size_t stride = ctx->expert_strides[layer][weight_type];
    if (stride == 0) {
        return true;
    }

    // Guard: only access tensors if they're in host-accessible memory.
    // When layers are on GPU (ngl > 0), tensor data pointers are CUDA device
    // pointers — dereferencing them from CPU would segfault or hang.
    bool indices_on_host = !expert_indices->buffer ||
                           ggml_backend_buffer_is_host(expert_indices->buffer);
    bool weights_on_host = !expert_weights->buffer ||
                           ggml_backend_buffer_is_host(expert_weights->buffer);

    if (!indices_on_host || !weights_on_host) {
        // Expert tensors are on GPU — skip CPU-side caching for this layer.
        // The GPU already has the data in VRAM, no mmap paging to optimize.
        static int skip_count = 0;
        if (++skip_count <= 10) {
            fprintf(stderr, "expert_cache: skip layer %d wt %d (GPU-resident)\n", layer, weight_type);
        } else if (skip_count == 11) {
            fprintf(stderr, "expert_cache: (suppressing further skip messages)\n");
        }
        return true;
    }

    // Mode selection via environment variable:
    // EXPERT_CACHE_NOOP=1 → callback fires but does nothing (isolates callback overhead)
    // Default → madvise prefetch
    {
        static int mode = -1;
        if (mode < 0) {
            const char * noop = getenv("EXPERT_CACHE_NOOP");
            mode = (noop && noop[0] == '1') ? 1 : 0;
            fprintf(stderr, "expert_cache: mode=%s\n", mode ? "NOOP" : "MADVISE");
        }
        if (mode == 1) {
            return true;  // no-op: callback fires, identifies tensor, but does nothing
        }
    }

    // madvise path: tell the kernel which expert pages we need
    {
        static int advise_count = 0;
        if (++advise_count <= 10) {
            fprintf(stderr, "expert_cache: ADVISE layer %d wt %d (CPU, madvise)\n", layer, weight_type);
        } else if (advise_count == 11) {
            fprintf(stderr, "expert_cache: (suppressing further advise messages)\n");
        }
    }
#if !defined(_WIN32)
    if (expert_indices->data) {
        const int32_t * ids = (const int32_t *)expert_indices->data;
        int n_ids = (int)(ggml_nelements(expert_indices));

        for (int i = 0; i < n_ids; i++) {
            int eid = ids[i];
            if (eid < 0 || eid >= ctx->n_expert) continue;

            const char * src = (const char *)expert_weights->data + (size_t)eid * stride;
            uintptr_t page_start = (uintptr_t)src & ~(uintptr_t)(4096 - 1);
            size_t advise_len = stride + ((uintptr_t)src - page_start);
            madvise((void *)page_start, advise_len, MADV_WILLNEED);
        }
    }
#endif

    // Let the normal ggml_mul_mat_id proceed — it will access the mmap'd data.
    // But because we've copied the hot experts into our cache, the OS is less
    // likely to evict those mmap pages (our cache pins the data in user-space).
    // Over time, the cache reaches steady state and prevents thrashing.
    return true;
}
