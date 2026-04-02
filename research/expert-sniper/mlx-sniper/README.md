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
| **Qwen3.5-35B-A3B (19.5 GB)** | **5.37 tok/s** (256 experts, cache=4000, bias=1.0, co-activation) |
| **Qwen3-30B-A3B (17.2 GB)** | **3.34 tok/s** (128 experts, cache=3000, co-activation) |
| Cache hit rate | 92% (35B) / 79% (30B) |
| TTFT | 2.9s (35B) / 6s (30B) |
| RAM | 8.7 GB (35B) / 9.1 GB (30B) |
| Hardware | M4 Mac Mini 16 GB, 5 varied prompts, cold start |

## Supported Models

| Model | Size | Experts | tok/s (M4 16GB) | Status |
|-------|------|---------|-----------------|--------|
| Qwen3.5-35B-A3B | 19.5 GB | 256/layer | 5.4 tok/s | Verified |
| Qwen3-30B-A3B | 17.2 GB | 128/layer | 3.3 tok/s | Verified |

More models coming. To request a model, open an issue.

### Hardware Requirements

| Mac | RAM | What you can run |
|-----|-----|-----------------|
| Any Apple Silicon | 8 GB | llama.cpp path only (0.57 tok/s) |
| M1/M2/M3/M4 | 16 GB | Qwen3.5-35B-A3B at 5.4 tok/s |
| M1/M2/M3/M4 Pro/Max | 32 GB+ | Larger models, faster speeds |

### Disk Requirements

| Model | Processed size | Free disk needed |
|-------|---------------|-----------------|
| 30B / Coder-30B | 17 GB | ~20 GB |
| 35B | 19 GB | ~25 GB |
| 80B | 43 GB | ~50 GB |
| 122B | ~65 GB | ~70 GB |
| 235B | ~130 GB | ~140 GB |

For larger models, use an external NVMe drive:
```bash
mlx-sniper download qwen3-next-80b -o /Volumes/MySSD/qwen3-next-80b
mlx-sniper run /Volumes/MySSD/qwen3-next-80b -p "hello" -v
```

Expert streaming works from any mounted volume. Speed scales with drive read bandwidth (~1-3 GB/s on USB-C NVMe, ~5 GB/s on internal).

### Limitations
- Dense models (only MoE architectures)
- Non-Qwen models (architecture-specific for now)
- Windows/Linux (Apple Silicon only for MLX path)
- llama.cpp path has no calibration (madvise only)

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
