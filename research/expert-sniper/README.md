# Expert Sniper — Run MoE Models Larger Than Your RAM

Two approaches to running 30-35B MoE models on machines that can't hold them in memory.

## Overview

MoE (Mixture-of-Experts) models activate only 8 of 256 experts per token — 96.9% of weights are unused per computation. Expert Sniper exploits this sparsity: pin the small always-needed weights in RAM, stream only the active experts from SSD.

```
Model: 17-21 GB
RAM:   0.87-1.4 GB used
The rest streams from disk on demand
```

## Two Paths

```
expert-sniper/
├── mlx-sniper/          ← Apple Silicon (MLX) — 1.3 tok/s on 16 GB Mac Mini
│   ├── Install: pip install -e .
│   ├── CLI: mlx-sniper chat ~/models/qwen3-30b
│   └── Server: mlx-sniper server ~/models/qwen3-30b --port 8899
│
├── llama-cpp/           ← Cross-platform (CUDA/Metal/CPU) — 44.7 tok/s on RTX 3090
│   ├── Agent: python3 sniper.py --model iq2
│   └── Uses stock llama.cpp with partial GPU offload
│
├── sniper-router/       ← Remote client — run agent locally, inference remotely
│   └── python3 router.py --server http://gpu-server:8201
│
├── docker/              ← Pre-built GPU server — no compilation needed
│   └── docker build -f docker/Dockerfile -t moe-sniper .
│
├── scripts/             ← Reproducible benchmarks
│   └── gpu_benchmark.sh
│
└── RESEARCH.md          ← Full technical writeup
```

## MLX Sniper (Apple Silicon)

Best for: Mac Mini, MacBook Pro, Mac Studio with 16+ GB RAM.

```bash
# Install
cd mlx-sniper && pip install -e .

# Preprocess model (one-time, downloads ~17 GB)
mlx-sniper preprocess mlx-community/Qwen3-30B-A3B-4bit -o ~/models/qwen3-30b

# Or use the streaming preprocessor (lower peak memory):
python3 stream_preprocess.py

# Interactive chat
mlx-sniper chat ~/models/qwen3-30b

# OpenAI-compatible server (other machines can connect)
mlx-sniper server ~/models/qwen3-30b --port 8899 --host 0.0.0.0
```

Full CLI and Python API: [huggingface.co/waltgrace/mlx-expert-sniper](https://huggingface.co/waltgrace/mlx-expert-sniper)

### MLX Sniper Results (M4 Mac Mini, 16 GB)

| Metric | Value |
|--------|-------|
| Model | Qwen3-30B-A3B (17.2 GB, 4-bit) |
| Standard mlx_lm | OOM |
| **Sniper speed (varied prompts)** | **1.3 tok/s** |
| **Sniper speed (same prompt, warm cache)** | **2.5 tok/s** |
| Cache hit rate | 73–99% (varies by prompt novelty) |
| RAM used | 0.87 GB pinned + up to 13 GB expert cache |

## llama.cpp Path (Cross-Platform GPU)

Best for: NVIDIA GPUs, cloud servers (RunPod, Lambda, etc.), or any machine with llama.cpp.

```bash
# Build llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-server

# Download model
pip install huggingface-hub
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-Q4_K_M.gguf --local-dir ./models

# Start server
./build/bin/llama-server -m ./models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -ngl 35 -c 2048 --port 8201 --host 0.0.0.0
```

Then connect the interactive agent:
```bash
pip install ddgs
python3 sniper-router/router.py --server http://localhost:8201
```

Or from another machine:
```bash
python3 sniper-router/router.py --server http://gpu-server:8201
```

### llama.cpp Results — Expert-Aware Prefetching (madvise)

**Headline:** madvise prefetch enables 0.57 tok/s on 8 GB where stock produces zero output.

| Hardware | Model | Speed | Notes |
|----------|-------|-------|-------|
| **M2 MacBook Air (8 GB)** | IQ2_M (10.6 GB), madvise | **0.57 tok/s** | 1 MB overhead, stock thrashes |
| M2 MacBook Air (8 GB) | IQ2_M (10.6 GB), LRU cache | 0.24 tok/s | 5 GB overhead (madvise is 2.4x better) |
| M2 MacBook Air (8 GB) | Q4_K_M (21 GB), madvise | Generates | Model is 2.6x larger than RAM |

**GPU baselines (stock llama.cpp, no expert cache):**

| Hardware | Model | Speed | Notes |
|----------|-------|-------|-------|
| RTX 3090 (24 GB) | Q4_K_M (21 GB), ngl 35 | 44.7 tok/s | Stock llama.cpp, warm cache |
| A6000 (48 GB) | 122B IQ2_M (39 GB), ngl 999 | 66.0 tok/s | Stock llama.cpp, full offload |
| A100 (80 GB) | 235B Q2_K (86 GB), ngl 58 | 1.32 tok/s | Stock llama.cpp, model exceeds VRAM |

Note: Prior GPU benchmarks had a device pointer bug where the eval callback dereferenced CUDA device pointers as host pointers, causing silent failures. All GPU speed numbers above are stock llama.cpp baselines. The expert cache was not contributing to these results.

## Docker (GPU — no build needed)

```bash
# Build from repo root
docker build -f research/expert-sniper/docker/Dockerfile -t moe-sniper .

# Run (auto-downloads 21 GB model)
docker run --gpus all -p 8201:8201 moe-sniper
```

## Remote Agent (router)

Run the agent UI on your laptop, inference on a remote GPU:

```bash
# On GPU server: start llama-server or mlx-sniper server
# On your laptop:
python3 sniper-router/router.py --server http://gpu-server:8201
```

Supports: chat, `/search` (web search), `/image` (vision), `/screenshot`, `/shell`

## Model

**Qwen3.5-35B-A3B** (llama.cpp path) / **Qwen3-30B-A3B** (MLX path)

Both are Mixture-of-Experts with 256 experts per layer, 8 active per token. The key property: only ~3% of FFN weights are needed per computation, making expert streaming viable.

| Quantization | Size | Source |
|-------------|------|--------|
| Q4_K_M (GGUF) | 21 GB | [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) |
| IQ2_M (GGUF) | 10.6 GB | [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) |
| 4-bit (MLX) | 17.2 GB | [mlx-community/Qwen3-30B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3-30B-A3B-4bit) |
| mmproj (vision) | 858 MB | [unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) |

## Research

See [RESEARCH.md](RESEARCH.md) for:
- Expert streaming architecture details
- Measured results with verified claims
- GPU offload analysis
- Multi-machine clustering design
- Apple Silicon scaling projections
