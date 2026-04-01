# MLX Expert Sniper — Apple Silicon MoE Inference

Pip-installable CLI for running MoE models larger than RAM on Apple Silicon via expert streaming.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Preprocess model (one-time)
mlx-sniper preprocess mlx-community/Qwen3-30B-A3B-4bit -o ~/models/qwen3-30b

# Interactive chat
mlx-sniper chat ~/models/qwen3-30b

# OpenAI-compatible server
mlx-sniper server ~/models/qwen3-30b --port 8899 --host 0.0.0.0

# Profile performance
mlx-sniper profile ~/models/qwen3-30b --tokens 100
```

## Results (M4 Mac Mini, 16 GB)

| Metric | Value |
|--------|-------|
| Model | Qwen3-30B-A3B (17.2 GB, 4-bit) |
| Standard mlx_lm | OOM |
| **Sniper speed (varied prompts)** | **3.3 tok/s** |
| **Sniper speed (cold start)** | **2.9 tok/s** |
| Cache hit rate | 86% (right-sized LRU + co-activation prefetch) |
| RAM used | 0.87 GB pinned + ~8 GB expert cache |
| TTFT | 6–7s (stable across prompts) |

## Full Package

The complete pip-installable package with CLI, server, and Python API is at:

**[huggingface.co/waltgrace/mlx-expert-sniper](https://huggingface.co/waltgrace/mlx-expert-sniper)**

## How It Works

Same technique as the llama.cpp path:
1. Pin attention + router + shared experts in RAM (~0.87 GB)
2. Stream only active experts (8 of 256) from SSD via `F_NOCACHE` + `pread`
3. LRU cache keeps hot experts resident (85-88% hit rate)
4. `gather_qmm` fuses quantized matmul across active experts

The files in this directory are the core research implementations. The production CLI wraps these into the `mlx-sniper` command.
