# Gemma 4-26B-A4B — MoE on Apple Silicon

Run Google's Gemma 4-26B MoE model on your Mac. 128 experts, top-8 routing, 4B active of 26B total.

## Quick Start (llama.cpp — works today)

### 1. Install llama.cpp

```bash
# macOS (Homebrew — may not have Gemma 4 yet, build from source if needed)
brew install llama.cpp

# Or build from source (guaranteed to have Gemma 4 support):
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu) --target llama-server
```

### 2. Download the model

Pick the quantization that fits your Mac:

```bash
pip3 install huggingface-hub

# IQ2_M — 9.3 GB (fits in 8 GB RAM, lower quality)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/gemma-4-26B-A4B-it-GGUF',
    'gemma-4-26B-A4B-it-UD-IQ2_M.gguf', local_dir='$HOME/models/gguf')
"

# Q4_K_M — 16.9 GB (best quality/speed on 16 GB Macs)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/gemma-4-26B-A4B-it-GGUF',
    'gemma-4-26B-A4B-it-UD-Q4_K_M.gguf', local_dir='$HOME/models/gguf')
"
```

### 3. Run

```bash
# If installed via Homebrew:
llama-server \
  -m ~/models/gguf/gemma-4-26B-A4B-it-UD-IQ2_M.gguf \
  -ngl 99 -c 2048 --reasoning off --port 8201

# If built from source:
./llama.cpp/build/bin/llama-server \
  -m ~/models/gguf/gemma-4-26B-A4B-it-UD-IQ2_M.gguf \
  -ngl 99 -c 2048 --reasoning off --port 8201
```

### 4. Test

```bash
curl http://localhost:8201/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"What is the capital of Australia?"}],"max_tokens":50,"temperature":0}'
```

## Benchmarks

All measurements on Apple Silicon, stock llama.cpp with mmap, no expert cache.

| Mac | RAM | Quant | Model size | Speed | Notes |
|-----|-----|-------|-----------|-------|-------|
| M2 MacBook Air | 8 GB | IQ2_M | 9.3 GB | **1.37 tok/s** | Model exceeds RAM — MoE sparsity prevents thrash |
| M4 Mac Mini | 16 GB | IQ2_M | 9.3 GB | **36.5 tok/s** | Fits in RAM, full GPU speed |
| M4 Mac Mini | 16 GB | Q4_K_M | 16.9 GB | **5.18 tok/s** | Exceeds RAM — MoE sparsity handles it |
| M4 Mac Mini | 16 GB | Q8_0 | 26.9 GB | **0 tok/s** | Thrash — llama.cpp CPU_REPACK doubles memory |

## Which quantization should I use?

| Your Mac | RAM | Recommended | Speed |
|----------|-----|-------------|-------|
| Any Mac | 8 GB | IQ2_M (9.3 GB) | ~1.4 tok/s |
| M1/M2/M3/M4 | 16 GB | Q4_K_M (16.9 GB) | ~5 tok/s |
| M1/M2/M3/M4 | 16 GB | IQ2_M (9.3 GB) | ~36 tok/s (fits in RAM) |
| Pro/Max | 32 GB+ | Q8_0 (26.9 GB) | ~15-20 tok/s (estimated) |

**IQ2_M on 16 GB** is the speed champion (36.5 tok/s) because it fits entirely in GPU memory. **Q4_K_M** gives better output quality but runs 7x slower because the OS has to page experts in/out.

## Key Finding: MoE Sparsity Handles Memory Pressure

Gemma 4 activates only 4B of 26B parameters per token (15.4% activation ratio). This means the OS page cache naturally handles memory pressure for modest oversubscription:

- 9.3 GB on 8 GB (1.16x) → works at 1.37 tok/s
- 16.9 GB on 16 GB (1.06x) → works at 5.18 tok/s
- 26.9 GB on 16 GB (1.68x) → thrashes (llama.cpp CPU_REPACK doubles memory)

This contrasts with Qwen 35B where stock llama.cpp thrashes at 10.6 GB on 8 GB — Qwen's shared experts increase the effective working set.

## MLX Expert Sniper (coming soon)

We have a custom Gemma 4 MLX model class (551 lines) and a Q8→streaming split script. The MLX path streams Q8 experts from SSD with no CPU_REPACK overhead, enabling Q8 quality on 16 GB Macs.

Status: architecture implemented, testing in progress.

## Model source

- GGUF: [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- Source: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
- Kaggle: [google/gemma-4](https://www.kaggle.com/models/google/gemma-4)

## Architecture

```
Gemma 4-26B-A4B:
  26B total parameters, 4B active per token
  128 experts per layer, top-8 routing
  30 decoder layers
  Sliding window attention (1024) + full attention (every 6th layer)
  Dense MLP runs alongside MoE (not instead of)
  gelu_pytorch_tanh activation
  K=V sharing in attention
```
