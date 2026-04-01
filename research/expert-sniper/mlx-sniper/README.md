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
| **Qwen3.5-35B-A3B (19.5 GB)** | **5.18 tok/s** (256 experts, cache=4000, REAP+bias=1.0) |
| **Qwen3-30B-A3B (17.2 GB)** | **3.34 tok/s** (128 experts, cache=3000, co-activation) |
| Cache hit rate | 94% (35B) / 79% (30B) |
| TTFT | 2.6s (35B) / 6s (30B) |
| RAM | 8.8 GB (35B) / 9.1 GB (30B) |
| Hardware | M4 Mac Mini 16 GB, 5 varied prompts |

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
