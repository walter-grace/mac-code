# mac code

**Run models that don't fit in RAM on your Mac. $0/month.**

## Can I run this on my Mac?

| Your Mac | RAM | What you can run | Speed |
|----------|-----|-----------------|-------|
| Any Mac | 8 GB | Qwen3.5-9B (Q4_K_M, 5.3 GB), 4K context | 16-20 tok/s |
| Any Mac | 16 GB | Qwen3.5-9B (Q4_K_M, 5.3 GB), 64K context | 16-20 tok/s |
| **Mac mini M4** | **16 GB** | **Qwen3.5-35B-A3B (IQ2_M, 10.6 GB)** | **30 tok/s** |
| **Mac mini M4** | **16 GB** | **Qwen3-30B-A3B Q4 (17.2 GB) via Expert Sniper** | **3.3 tok/s** |
| **Mac mini M4** | **16 GB** | **Qwen3.5-35B-A3B Q4 (19.5 GB) via Expert Sniper** | **5.2 tok/s** |
| Mac mini M4 | 16 GB | Qwen3.5-35B-A3B Q4_K_M (22 GB) via Flash Streaming | 1.54 tok/s |
| Mac mini M4 | 16 GB | Qwen3.5-27B (16.1 GB) via Flash Streaming | 0.18 tok/s |
| Mac mini M4 Pro | 48 GB | 35B at full Q4 in RAM | 30+ tok/s |

> **"I wanted to run the Qwen 27B on my M2 16GB but failed. That's not possible, right?"**
>
> It is possible. We stream FFN weights from SSD — only 5.5 GB stays in RAM. The output is coherent, full 4-bit quality. It's slow (0.18 tok/s on a Mac mini M4) but the method works on any 16 GB Apple Silicon Mac. No 2-bit compression, no mmap thrashing, no swap death. [See how it works.](#how-flash-streaming-works)

---

## Quick Start

### 35B Agent (recommended — 30 tok/s on 16 GB)

The fastest option. Uses llama.cpp with a 2-bit quantization (IQ2_M) that fits entirely in RAM.

```bash
brew install llama.cpp
pip3 install rich ddgs --break-system-packages

# Download model (10.6 GB)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"

# Start server + agent
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 12288 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -np 1 -t 4

python3 agent.py
```

### 9B with 64K Context

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"

llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 65536 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -t 4

python3 agent.py
```

---

## Flash Streaming

Run models that **genuinely don't fit in RAM** at full quality. No 2-bit compression. No mmap thrashing.

### Measured Results

Every number below was measured on a 16 GB Mac mini M4. Nothing estimated.

| Model | Total Size | RAM Used | Speed | Quality |
|-------|-----------|---------|-------|---------|
| Qwen3-32B (dense) | 18.4 GB | 4.5 GB | 0.15 tok/s | Full 4-bit |
| **Qwen3.5-27B (dense hybrid)** | **16.1 GB** | **5.5 GB** | **0.18 tok/s** | **Full 4-bit** |
| Qwen3-30B-A3B (MoE) | 17.2 GB | 9.1 GB | 3.3 tok/s | Full 4-bit |
| Qwen3.5-35B-A3B (MoE) | 19.5 GB | 8.8 GB | 5.2 tok/s | Full 4-bit |

All numbers measured on M4 Mac Mini 16 GB across 5 varied prompts (not cherry-picked same-prompt results). MoE models are ~28x faster than dense — only 8 of 128-256 experts activate per token. 35B: REAP dead-expert masking + cache-aware routing bias + co-activation prefetch (6.6x speedup from software alone). 30B: right-sized LRU + co-activation prefetch (REAP/bias not yet applied).

### How Flash Streaming Works

Split the model by access pattern:

**Pinned in RAM (4-6 GB):** Attention weights, embeddings, norms, KV cache. Loaded once, stays forever.

**Streamed from SSD per token:** FFN weights (the bulk of the model). Loaded layer-by-layer, used for one matmul, discarded. Memory never grows.

```
For each token:
  For each layer:
    1. Run attention (from RAM — instant)
    2. Load FFN weights from SSD (~165-221 MB)
    3. Run FFN matmul on GPU
    4. Discard FFN weights — memory stays flat
```

For MoE models, step 2 loads only the 8 active experts (~14 MB), not all 256. That's why MoE is 10x faster.

### Run the 35B MoE Agent (1.54 tok/s, 1.42 GB RAM)

Interactive agent with web search, shell commands, and chain-of-thought. The 22 GB model on a 16 GB Mac.

**Requires pre-built stream files** — see [`research/flash-streaming/`](research/flash-streaming/) for the split/rebuild tools.

```bash
cd research/flash-streaming
python3 moe_agent.py
```

### Run the 27B Dense on 16 GB (0.18 tok/s, 5.5 GB RAM)

```bash
cd research/flash-streaming
pip3 install mlx-lm transformers --break-system-packages

# One-time: download model (~16 GB) and split for streaming
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen3.5-27B-4bit', local_dir='$HOME/models/qwen35-27b-mlx-4bit')
"
python3 split_dense_27b.py

# Run
python3 flash_stream_27b.py
```

### Batched Union-of-Experts (5.1 tok/s)

A research prototype that verifies 8 tokens in one forward pass. Instead of loading experts for each token separately, it computes the set union of active experts across all 8 tokens (~27 unique experts per layer instead of 64) and loads them once.

```bash
cd research/flash-streaming
python3 batched_moe.py
```

This is verification speed (checking draft tokens), not generation speed. Useful for speculative decoding.

---

## Agent Commands

| Command | Action |
|---------|--------|
| `/agent` | Agent mode (default) — search, shell, chat |
| `/raw` | Direct streaming, no tools |
| `/search <q>` | Quick web search |
| `/stats` | Session statistics |
| `/clear` | Reset conversation |
| `/quit` | Exit |

---

## Files

### Top-level (daily use)

| File | What it does |
|------|-------------|
| `agent.py` | Production agent — routes to search/shell/chat via llama.cpp |
| `chat.py` | Simple streaming chat with llama.cpp |
| `dashboard.py` | Real-time monitoring dashboard for llama.cpp |
| `setup.sh` | One-command install (llama.cpp + model download + config) |
| `config.example.json` | Example configuration |
| `web/` | Web UI (server.py + index.html) |

### MLX backend (`mlx/`)

| File | What it does |
|------|-------------|
| `mlx_engine.py` | MLX inference server with 64K context and KV cache persistence |
| `kv_cache.py` | KV cache save/load for session persistence |
| `paged_inference.py` | Paged attention experiment |
| `turboquant.py` | TurboQuant quantization experiments |
| `benchmark.py` | Benchmarking tools |

### Flash Streaming research (`research/flash-streaming/`)

The research journey: we built, measured, and iterated. Each file represents a step.

| File | What it does | Key discovery |
|------|-------------|---------------|
| `flash_stream.py` | v1: mmap-based streaming (0.12 tok/s) | Split-model architecture works |
| `flash_stream_v2.py` | v2: F_NOCACHE direct I/O (0.15 tok/s) | 27% faster than mmap |
| `flash_stream_27b.py` | 27B dense streaming (0.18 tok/s) | Method works on dense + hybrid SSM models |
| `flash_moe.py` | MoE expert-level streaming engine | Only load active experts from SSD |
| `moe_agent.py` | **Working 35B agent** (1.54 tok/s, 1.42 GB) | Coherent 22 GB model on 16 GB Mac |
| `batched_moe.py` | Batched Union-of-Experts (5.1 tok/s) | ~27 unique experts/layer, not 64 |
| `expert_io.py` | F_NOCACHE + pread expert reader (8 threads) | Saturate NVMe queue depth |
| `direct_io.py` | F_NOCACHE + pread for dense FFN layers | Bypass macOS Unified Buffer Cache |
| `split_mlx_model.py` | Split 35B MoE into pinned + per-layer experts | 16KB alignment for DART IOMMU |
| `split_dense_27b.py` | Split 27B dense into pinned + per-layer FFN | Same technique, different architecture |
| `convert_split.py` | GGUF → split safetensors conversion | GGUF is column-major |
| `convert_aligned.py` | Safetensors → 16KB-aligned binary | Required for F_NOCACHE direct I/O |
| `dequant_gguf.py` | Custom Q4_K/Q6_K dequantization (numpy) | MLX can't read GGUF Q4_K blocks |
| `rebuild_pinned.py` | Rebuild pinned weights from MLX golden model | Fix SSM weight dtype issues |
| `flash_agent.py` | 32B dense streaming agent (early version) | Proof of concept |
| `flash_stream_batched.py` | Batched eval experiment | Proved eval sync isn't the bottleneck |
| `README.md` | Detailed research writeup with all measurements | Full methodology and results |

---

## Key Discoveries

These are things we learned the hard way. Each links to the file where it was discovered/fixed.

1. **GGUF is column-major** — `flat.reshape(ne[1], ne[0])`, not `.reshape(ne[0], ne[1]).T`. The wrong reshape gives correct shapes but garbage output. (`dequant_gguf.py`, `convert_split.py`)

2. **MLX 4-bit is 15% larger than expected** — scales + biases at group_size=64 add 0.031 bytes/param. A 32B model is 18.4 GB, not 16 GB. This is why the model doesn't fit in 16 GB RAM even at 4-bit. (`research/flash-streaming/README.md`)

3. **`nn.quantize()` silently skips MoE experts** — `SwitchLinear` is not a subclass of `nn.Linear`. You must pass a `class_predicate` that explicitly includes it. Without this, experts run in float16 and produce garbage. (`moe_agent.py`)

4. **`gather_qmm` eliminates accumulator divergence** — 8 separate `quantized_matmul` calls compound rounding errors across 40 layers. One batched `gather_qmm` call matches the reference model exactly. (`batched_moe.py`, `flash_moe.py`)

5. **F_NOCACHE is 27% faster than mmap** — macOS Unified Buffer Cache adds overhead for sequential streaming workloads. `fcntl(F_NOCACHE)` + `os.pread()` with 16KB alignment bypasses it entirely. (`direct_io.py`, `expert_io.py`)

6. **`setattr` on `nn.Module` leaks memory** — Injecting FFN weights into the model tree via `setattr` prevents garbage collection. Memory grew 3.6 GB per 16 layers. Fix: use `mx.quantized_matmul` directly on loaded arrays, never touch the model tree. (`flash_stream.py`)

7. **Batching layers doesn't help** — We tested 8-layer batches (16 evals vs 128). Zero speedup. The bottleneck is SSD I/O, not GPU sync overhead. (`flash_stream_batched.py`)

---

## Architecture

```
┌──────────────────────────────────────────────┐
│  agent.py — LLM-as-Router                    │
│  search / shell / chat                       │
├──────────┬───────────────────────────────────┤
│ llama.cpp│  MLX backend                      │
│ (fast)   │  + KV cache save/load             │
│          │  + Flash Streaming (out-of-core)   │
│          │  + MoE Expert Sniper (SSD)        │
├──────────┴───────────────────────────────────┤
│  Apple Silicon — Unified Memory + NVMe SSD   │
└──────────────────────────────────────────────┘
```

---

## Credits

- **[Qwen3.5](https://huggingface.co/Qwen)** — the models
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — inference engine
- **[MLX](https://github.com/ml-explore/mlx)** — Apple's ML framework
- **[Unsloth](https://huggingface.co/unsloth)** — GGUF quantizations
- **[mlx-community](https://huggingface.co/mlx-community)** — pre-converted MLX models

## License

MIT
