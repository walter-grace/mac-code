#!/bin/bash
# Apply Expert Sniper patches to stock llama.cpp
# Run from the llama.cpp root directory
set -e

echo "Applying Expert Sniper patches..."

# 1. Add new source files to CMakeLists.txt
if ! grep -q "llama-expert-cache" src/CMakeLists.txt; then
    sed -i '/llama-kv-cache-recurrent.cpp/a\            llama-expert-cache.cpp\n            llama-expert-cache-ctx.cpp' src/CMakeLists.txt
    echo "  [OK] CMakeLists.txt: added expert cache source files"
else
    echo "  [SKIP] CMakeLists.txt: already patched"
fi

# 2. Add expert_cache_size to common_params struct
if ! grep -q "expert_cache_size" common/common.h; then
    sed -i '/int32_t n_gpu_layers/a\    size_t expert_cache_size = 0;   // expert LRU cache size in bytes for MoE models (0 = disabled)' common/common.h
    echo "  [OK] common.h: added expert_cache_size field"
else
    echo "  [SKIP] common.h: already patched"
fi

# 3. Add --expert-cache-size CLI argument
if ! grep -q "expert-cache-size" common/arg.cpp; then
    # Insert before --override-tensor
    sed -i '/\"--override-tensor\"/i\        add_opt(common_arg(\n            {"--expert-cache-size"}, "N",\n            "size of expert LRU cache in MB for MoE models (default: 0 = disabled)",\n            [](common_params \& params, const std::string \& value) {\n                params.expert_cache_size = std::stoull(value) * 1024ULL * 1024ULL;\n            }\n        ).set_env("LLAMA_ARG_EXPERT_CACHE_SIZE"));' common/arg.cpp
    echo "  [OK] arg.cpp: added --expert-cache-size argument"
else
    echo "  [SKIP] arg.cpp: already patched"
fi

# 4. Add expert cache include and initialization to common.cpp
if ! grep -q "llama-expert-cache-ctx.h" common/common.cpp; then
    # Add include at top
    sed -i '1i#include "../src/llama-expert-cache-ctx.h"' common/common.cpp
    echo "  [OK] common.cpp: added include"
else
    echo "  [SKIP] common.cpp: include already present"
fi

if ! grep -q "expert_cache_size" common/common.cpp; then
    # Add initialization after model loading
    # This is the trickiest patch — find where cparams is set up and add expert cache init
    python3 -c "
import re
with open('common/common.cpp', 'r') as f:
    content = f.read()

# Find the common_init_result function and add expert cache init
# Look for where cb_eval could be set
init_code = '''
    // Expert Sniper: initialize expert cache for MoE models
    if (params.expert_cache_size > 0) {
        if (llama_model_n_expert(model) > 0) {
            static auto expert_cache = std::make_unique<llama_expert_cache_ctx>();
            expert_cache->init(*model, params.expert_cache_size);
            cparams.cb_eval = llama_expert_cache_ctx::eval_callback;
            cparams.cb_eval_user_data = expert_cache.get();
            LOG_INF(\"%s: expert cache enabled: %.1f MB\\n\", __func__,
                    (double)params.expert_cache_size / (1024*1024));
        } else {
            LOG_WRN(\"%s: --expert-cache-size specified but model has no experts\\n\", __func__);
        }
    }
'''

# Insert after 'auto cparams' line in common_init_result
if 'auto cparams' in content:
    content = content.replace(
        'auto cparams',
        'auto cparams',
        1  # only first occurrence
    )
    # Find the line after auto cparams and insert
    lines = content.split('\\n')
    for i, line in enumerate(lines):
        if 'auto cparams' in line and 'common_init_result' not in line:
            # Insert after the semicolon line
            j = i
            while j < len(lines) and ';' not in lines[j]:
                j += 1
            lines.insert(j + 1, init_code)
            break
    content = '\\n'.join(lines)

with open('common/common.cpp', 'w') as f:
    f.write(content)
"
    echo "  [OK] common.cpp: added expert cache initialization"
else
    echo "  [SKIP] common.cpp: already patched"
fi

echo "Expert Sniper patches applied successfully."
echo "Build with: cmake -B build -DGGML_CUDA=ON && cmake --build build -j\$(nproc) --target llama-server"
