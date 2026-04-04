#pragma once

#include "llama-expert-cache.h"

#include <memory>
#include <vector>
#include <array>
#include <cstring>

struct ggml_tensor;
struct llama_model;

// Context that lives alongside llama_context, managing the expert cache
// and intercepting ggml_mul_mat_id operations via the eval callback.
struct llama_expert_cache_ctx {
    std::unique_ptr<llama_expert_cache> cache;

    // Per-layer expert tensor pointers: [layer][0=up, 1=gate, 2=down]
    std::vector<std::array<ggml_tensor *, 3>> expert_tensors;

    // Per-layer expert stride (bytes per expert slice in stacked tensor)
    std::vector<std::array<size_t, 3>> expert_strides;

    int n_expert      = 0;
    int n_expert_used = 0;
    int n_layers      = 0;

    // Active expert buffer: temporary contiguous buffer for selected experts
    // Rebuilt before each ggml_mul_mat_id operation
    void * active_buffer      = nullptr;
    size_t active_buffer_size = 0;

    // Saved state for restoring after tensor patching
    struct patch_state {
        ggml_tensor * tensor;
        void * original_data;
        int32_t original_ne3;  // original n_expert dimension
    };
    std::vector<patch_state> pending_restores;

    ~llama_expert_cache_ctx() {
        if (active_buffer) {
            free(active_buffer);
        }
    }

    // Initialize from model — call after model tensors are loaded
    void init(const llama_model & model, size_t cache_bytes);

    // The eval callback — intercepts ggml_mul_mat_id to use cached experts
    static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

private:
    // Find which layer and weight type a tensor belongs to
    // Returns {layer, weight_type} or {-1, -1} if not found
    std::pair<int, int> identify_tensor(const ggml_tensor * t) const;

    // Build active expert buffer from cache for given experts
    // Returns pointer to contiguous buffer with selected experts packed sequentially
    void * build_active_buffer(int layer, int weight_type,
                               const int32_t * expert_ids, int n_ids);
};
