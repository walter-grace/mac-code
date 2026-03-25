# 🍎 mac code

**Open-source AI agent that runs powerful models on your Mac — leveraging Apple Silicon like it was meant to be used.**

There are 100 million Macs with Apple Silicon in the world. Every one of them has a GPU, unified memory, and a fast SSD. That's the largest untapped AI compute network on Earth — and it's sitting on people's desks doing nothing.

mac code is an open-source project to change that. We run frontier-class open-weight models locally on Macs, with no cloud dependency, no API keys, and no cost per token. The goal is to make every Mac an AI workstation.

---

## The Breakthrough

**Two models on a $600 Mac mini. Both with full tool calling. Zero cloud.**

| Model | What | Speed | Context | On 16 GB RAM |
|---|---|---|---|---|
| **Qwen3.5-35B-A3B MoE** | 35 billion params, 3B active per token | **30 tok/s** | 4K | Via SSD flash-paging |
| **Qwen3.5-9B** | 9 billion params, all active | 16 tok/s | **64K** | Fits entirely in RAM |

The 35B model doesn't fit in memory — it pages from the SSD at 30 tok/s. On NVIDIA hardware, the same paging gives you 1.6 tok/s. Apple Silicon's unified memory makes the difference.

The 9B runs with 64K context using quantized KV cache — enough to read entire codebases. Two flags (`--cache-type-k q4_0 --cache-type-v q4_0`) doubled the context window for free.

Both models do web search, shell commands, and file operations through our LLM-as-Router architecture — the model classifies its own intent and generates its own commands, no external agent framework needed.

---

## The Mission

**Put open-source AI models on every Mac.**

Apple Silicon is uniquely suited for local LLM inference. Unlike NVIDIA GPUs, Apple's unified memory architecture lets the CPU, GPU, and SSD share the same address space. This means:

- Models that don't fit in RAM can **page from the SSD** and the GPU still processes everything
- No PCIe bottleneck, no CPU fallback, no data copies between devices
- A $600 Mac mini can run a 35-billion parameter model at 30 tokens/second

We proved this works. Now we're open-sourcing the entire stack so anyone with a Mac can do it too.

## How it works

### LLM-as-Router

Most AI agents use hardcoded rules or a separate framework to decide what to do. mac code uses **the LLM itself as the router**. Every message gets classified in ~1 second:

```
You: "find me videos on my desktop"
 → LLM classifies: "shell"
 → LLM generates: find ~/Desktop -type f \( -name "*.mp4" -o -name "*.mov" \)
 → Executes command, feeds results back to LLM
 → LLM presents results

You: "who do the lakers play next?"
 → LLM classifies: "search"
 → LLM rewrites query: "Lakers schedule March 24, 2026"
 → DuckDuckGo search + page fetch
 → LLM answers with real data

You: "explain quantum computing"
 → LLM classifies: "chat"
 → Streams directly from LLM
```

Both models support all three paths:

| Model | Speed | Context | Best for |
|---|---|---|---|
| **Qwen3.5-35B-A3B MoE** | **30 tok/s** | 4K | Fast agent tasks, reasoning |
| Qwen3.5-9B | 16 tok/s | **64K** | Long conversations, large files |

Three paths, all powered by the same model:

| Intent | What happens | Speed |
|---|---|---|
| **search** | LLM rewrites query → DuckDuckGo → page fetch → LLM answers | **~8-20s** |
| **shell** | LLM generates command → execute → LLM summarizes results | **~5-15s** |
| **chat** | Streams directly from LLM | **~2-3s** |

### Why this matters

**Because the LLM is the router, upgrading the model upgrades every capability simultaneously.** A smarter model generates better shell commands, writes better search queries, classifies intent more accurately, and gives better answers — without changing a single line of agent code.

```
Same agent.py on every Mac:

  16GB Mac mini  →  9B model   →  good agent
  48GB Mac Pro   →  35B model  →  great agent
  192GB Studio   →  397B model →  frontier agent

  Zero code changes. Just swap the model file.
```

### Test results (10/10 passed)

| Category | Query | Time | Accurate? |
|---|---|---|---|
| Sports live | NBA scores last night | 30.9s | Yes — real scores |
| Sports schedule | Next UFC fight card | 24.6s | Yes — date + fighters |
| Finance | Bitcoin price | 24.6s | Yes — current price |
| Tech news | Latest OpenAI news | 30.5s | Yes — recent events |
| Weather | NYC weather today | 22.5s | Yes — temperature |
| Science | Latest SpaceX launch | 18.5s | Yes — mission details |
| Awards | Best Picture 2026 | 20.4s | Yes — winner named |
| Coding | Latest Python version | 33.7s | Yes — 3.14.3 |
| Markets | Stock market today | 39.8s | Yes — S&P data |
| Local | SF restaurants | 39.8s | Yes — real restaurants |

## What makes this different

**The 35B model doesn't fit in RAM.** That's the whole point.

Qwen3.5-35B-A3B is a 10.6 GB model. A Mac mini M4 has 16 GB of RAM. After macOS takes its share, there's not enough room. The overflow pages from the SSD.

On any other hardware, this kills performance:

| Setup | Speed | Cost/hr | What happens |
|---|---|---|---|
| **Mac mini M4 + SSD paging** | **29.8 tok/s** | **$0.00** | **GPU processes everything via unified memory** |
| NVIDIA GPU + NVMe paging | 1.6 tok/s | $0.44 | CPU bottleneck — GPU can't access paged data |
| NVIDIA GPU + FUSE paging | 0.075 tok/s | $0.44 | Network storage — barely functional |
| NVIDIA GPU in-VRAM (no paging) | 42.5 tok/s | $0.34 | Fast, but costs money and needs big GPU |
| Claude Code (API) | ~80 tok/s | ~$0.50+ | Fastest, but every token costs money |

**Apple Silicon is 18.6x faster than NVIDIA when the model doesn't fit in memory.**

---

## Quick Start

### What you need

- Mac with Apple Silicon (M1 or later, 16GB+ RAM)
- [Homebrew](https://brew.sh)

### One-command setup

```bash
git clone https://github.com/walter-grace/mac-code.git
cd mac-code
chmod +x setup.sh && ./setup.sh
```

### Or step by step

**1 — Install dependencies**

```bash
brew install llama.cpp
pip3 install huggingface-hub rich ddgs --break-system-packages
```

**2 — Download the model**

```bash
mkdir -p ~/models

# 9B model (recommended — tools + web search work reliably)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"
```

Optional — 35B MoE for faster reasoning (manual swap with `/model 35b`):
```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"
```

**3 — Start the server**

```bash
llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 65536 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -t 4
```

**4 — Run the agent**

```bash
python3 agent.py
```

**5 — (Optional) Run the web UI**

```bash
# Build PicoClaw for file operations
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw && make deps && make build && cd ..
mkdir -p ~/.picoclaw/workspace
cp config.example.json ~/.picoclaw/config.json

# Start the web backend
python3 web/server.py

# Open the retro Mac UI
open http://localhost:8080
```

---

## What it looks like

```
  🍎 mac code
  claude code, but it runs on your Mac for free

  model  Qwen3.5-9B  8.95B dense · Q4_K_M · 32K ctx
  tools  search · fetch · exec · files
  cost   $0.00/hr  Apple M4 Metal · localhost:8000

  type / to see all commands

  auto 9b > who do the lakers play next?

  ⠋ rewriting query  1s
  ⠙ searching the web  2s
  ⠹ reading results  4s
  ⠸ generating answer  6s

  The Lakers play the Detroit Pistons on Monday, March 24, 2026
  at Little Caesars Arena in Detroit.

  ▸ search  8.8s  ·  16.3 tok/s

  auto 9b > what is 2*4^6?

  2 × 4^6 = 2 × 4096 = 8,192

  16.8 tok/s  ·  45 tokens  ·  2.7s
```

## Commands

Type `/` to see all commands:

| Command | Action |
|---|---|
| `/agent` | Agent mode — tools + web search (default) |
| `/raw` | Raw mode — direct streaming, no tools |
| `/model 9b` | Switch to 9B (32K ctx, tool calling) |
| `/model 35b` | Switch to 35B MoE (8K ctx, faster reasoning) |
| `/auto` | Toggle smart auto-routing |
| `/btw <q>` | Side question without context |
| `/search <q>` | Quick web search |
| `/bench` | Speed benchmark |
| `/loop 5m <p>` | Run prompt on recurring interval |
| `/stop` | Stop a running loop |
| `/branch` | Save conversation checkpoint |
| `/restore` | Restore to checkpoint |
| `/add-dir <path>` | Set working directory |
| `/save <file>` | Export conversation to JSON |
| `/clear` | Reset conversation |
| `/stats` | Session statistics |
| `/tools` | List available tools |
| `/system <msg>` | Set system prompt |
| `/compact` | Toggle markdown rendering |
| `/cost` | Show cost savings vs cloud |
| `/quit` | Exit |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  mac code agent            Python + Rich             │
│                                                      │
│  ┌─────────────────────────────────────────────┐     │
│  │  LLM-as-Router (1 fast classification call) │     │
│  │  "search" / "shell" / "chat"                │     │
│  └──────┬──────────┬──────────────┬────────────┘     │
│         │          │              │                   │
│     search      shell          chat                  │
│    DuckDuckGo   LLM→command   stream                 │
│    + page fetch  execute      direct                 │
│    + LLM answer  LLM summary                        │
│         │          │              │                   │
│  ┌──────┴──────────┴──────────────┴────────────┐     │
│  │  llama.cpp (Metal GPU, localhost:8000)       │     │
│  │  Any GGUF model — 9B, 35B, 397B, 1T         │     │
│  └─────────────────────────────────────────────┘     │
│                                                      │
│  ┌─────────────────────────────────────────────┐     │
│  │  Apple Silicon — Unified Memory + SSD       │     │
│  │  Metal GPU + flash paging for large models  │     │
│  └─────────────────────────────────────────────┘     │
├──────────────────────────────────────────────────────┤
│  PicoClaw (optional plugin layer)                    │
│  Telegram · Discord · Slack · WhatsApp · 17+ apps   │
└──────────────────────────────────────────────────────┘
```

### Example flows

**"find me videos on my desktop":**
```
LLM classifies → shell
LLM generates  → find ~/Desktop -type f \( -name "*.mp4" -o -name "*.mov" \)
Agent executes → [list of video files]
LLM presents   → "Found 3 videos on your Desktop: ..."
Total: ~8s
```

**"who do the lakers play next?":**
```
LLM classifies → search
LLM rewrites   → "Lakers schedule March 24, 2026"
DuckDuckGo     → [search results + page content]
LLM answers    → "The Lakers play the Pistons on Monday March 24"
Total: ~10s
```

**"explain quantum computing":**
```
LLM classifies → chat
Streams directly from LLM
Total: ~2-3s
```

---

## Files

| File | What |
|---|---|
| `agent.py` | CLI agent — search, chat, slash commands, animated loading |
| `chat.py` | Lightweight streaming chat (no tools) |
| `dashboard.py` | Real-time server monitor with tok/s sparklines |
| `web/index.html` | Retro Macintosh web UI with live LLM streaming |
| `web/server.py` | Web backend — bridges browser to search + LLM |
| `config.example.json` | PicoClaw config for file operations |
| `setup.sh` | One-command install |
| `CLAUDE.md` | Setup instructions for Claude Code users |

---

## Scaling — Same Agent, Bigger Brain

The agent code never changes. You just swap the model file and everything gets smarter:

| Mac | RAM | Model | Agent Quality | Speed |
|---|---|---|---|---|
| Any Mac | 8 GB | Qwen3.5-9B (4K ctx) | Good — chat + basic tools | ~15 tok/s |
| **Mac mini M4** | **16 GB** | **Qwen3.5-9B (32K ctx)** | **Good — search, shell, chat** | **~16 tok/s** |
| Mac mini M4 | 16 GB | Qwen3.5-35B MoE (8K ctx, SSD paging) | Better reasoning, no tools | ~57 tok/s |
| Mac mini M4 Pro | 48 GB | Qwen3.5-35B MoE Q4_K_M (32K ctx) | **Great — tools work at higher quant** | ~40+ tok/s |
| Mac mini M4 Pro | 48 GB | Both models (speculative decoding) | Great + fast | ~60-90 tok/s |
| Mac Studio M4 Max | 128 GB | Qwen3.5-397B-A17B (frontier MoE) | **Frontier — 17B active/token** | ~10-20 tok/s |
| Mac Studio M4 Ultra | 192 GB | Qwen3.5-397B-A17B Q4_K_M | Frontier + quality | ~15-30 tok/s |
| Mac Pro M4 Ultra | 512 GB | Kimi K2.5 (1T MoE, 32B active) | **Beyond frontier** | ~5-15 tok/s |

### What more RAM unlocks

**16 GB (what we proved):**
- 9B handles all agent tasks — LLM classifies intent, generates shell commands, rewrites search queries
- 35B available for manual reasoning (faster but no tool calling at IQ2 quantization)

**48 GB (next step):**
- 35B at Q4_K_M — tool calling works because higher quantization preserves instruction following
- Speculative decoding — 9B drafts tokens, 35B verifies, 2-3x speed boost
- 32K+ context on the 35B — long conversations and complex file analysis

**128-192 GB (frontier):**
- 397B-A17B — the same MoE architecture, 17B active per token
- The LLM-as-router gets dramatically smarter — better command generation, better search queries, better intent classification
- Same `python3 agent.py`, same slash commands, same web search pipeline

**512 GB (the 1T frontier — Kimi K2.5):**
- **Kimi K2.5** — 1 trillion parameters, MoE with 32B active per token
- Currently requires 8x H100 GPUs (~$25/hr in the cloud)
- A Mac Pro with 512 GB unified memory could run a Q2 quantized version (~300 GB on disk)
- The active 32B parameters stay hot in RAM; the other 968B page from SSD
- Same `python3 agent.py` — the LLM-as-router would operate at GPT-4+ level
- Intent classification becomes near-perfect, shell commands are expert-level, search queries are optimally crafted
- **What costs $25/hr on cloud GPUs today runs on a desk for $0/hr**

**The key insight:** because the LLM is the router, planner, and executor, every capability scales with the model. A 1T model on a Mac Pro would generate perfect shell commands, write optimal search queries, and give expert-level answers — all running the same agent.py that works on a $600 Mac mini today. Zero code changes. Just swap the model file.

### Plugin layer (optional)

For messaging app integration, [PicoClaw](https://github.com/sipeed/picoclaw) connects mac code to 17+ platforms:

```
┌─────────────────────────────────────────┐
│  mac code (core agent)                  │
│  Python · LLM router · search · shell   │
├─────────────────────────────────────────┤
│  PicoClaw (optional plugin layer)       │
│  Telegram · Discord · Slack · WhatsApp  │
│  WeChat · Matrix · IRC · Line · QQ      │
└─────────────────────────────────────────┘
```

Your local LLM answering Telegram messages, responding in Discord, monitoring Slack — all at $0/month.

---

## Benchmarks

### Math (212 problems, SymPy verified)

| Category | Score |
|---|---|
| Linear Algebra | **100%** (22/22) |
| Number Theory | **100%** (22/22) |
| Logic | **100%** (20/20) |
| Differential Equations | 95% |
| Geometry | 91% |
| Algebra | 86% |
| **Overall** | **86.3%** (183/212) |

### Speed comparison (3 models on same hardware)

| Model | Speed | Accuracy |
|---|---|---|
| Qwen3.5-9B (Q4_K_M) | 18.3 tok/s | 3/4 |
| Qwen3.5-27B (IQ2_M) | 7.7 tok/s | 3/4 |
| **Qwen3.5-35B-A3B MoE (IQ2_M)** | **30.9 tok/s** | **3/4** |

The 27B dense model is the worst option — slower than both alternatives with the same accuracy. The MoE architecture wins because only 3B of 35B activates per token.

---

## Common Issues

- **GPU OOM after long sessions**: Reboot your Mac to clear Metal GPU memory
- **"Connection refused"**: The llama-server crashed or wasn't started. Restart it.
- **Slow web search (>30s)**: Page fetch timeout on slow sites. The search still works, just slower.
- **35B tool calling via PicoClaw fails**: Expected — JSON function calls break at 2.6 bpw. But our LLM-as-Router works perfectly (8/8 correct). Use `agent.py`, not PicoClaw directly.

---

## Speed Experiments

We tested three techniques to push past 30 tok/s. All need more RAM than 16 GB:

| Experiment | Result | Needs |
|---|---|---|
| Speculative decoding (9B drafts, 35B verifies) | GPU OOM | 48 GB+ |
| Multi-Token Prediction | No gain in llama.cpp | vLLM/SGLang |
| mmap vs no-mmap | ~3% improvement | Already optimal |

**If you have 48 GB+ RAM**, try speculative decoding:

```bash
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --model-draft ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --flash-attn on --ctx-size 8192 \
    --n-gpu-layers 99 -t 4 --draft-max 8 --draft-min 2
```

Expected: **60-90 tok/s**. Report results in an issue.

### MLX backend

We benchmarked Apple's [MLX framework](https://arxiv.org/abs/2511.05502) against llama.cpp on the same hardware:

| Model | llama.cpp | MLX | Winner |
|---|---|---|---|
| **9B** | 16 tok/s | **20 tok/s** | **MLX (+25%)** |
| **35B MoE** | **29 tok/s** | Won't fit (4-bit too large) | **llama.cpp** |

**Why llama.cpp wins for the 35B:** MLX's smallest quantization is 4-bit (4.0 bpw). Our 35B MoE uses IQ2_M (2.6 bpw) to fit in 16GB. Only llama.cpp supports quantizations below 3-bit.

**When to use MLX:**
- **9B model** — 25% faster out of the box
- **48GB+ Macs** — run the 35B at 4-bit quantization for better quality AND speed
- **Any model that fits in RAM** — MLX's zero-copy tensors avoid llama.cpp's Metal overhead

```bash
pip3 install mlx-lm
python3 mlx_server.py              # 9B at ~20 tok/s (25% faster)
python3 mlx_server.py --model 35b  # 35B — needs 48GB+ Mac
```

Same OpenAI-compatible API at localhost:8000. `agent.py` works with either backend.

### Self-improving agent

Every interaction is logged to `~/.mac-code/logs/`. Grade responses with `/good` or `/bad` to build training data. View stats with `/improve`.

The logged JSONL data can be used for:
- **TinyLoRA fine-tuning** (13 parameters, RL-based — [paper](https://arxiv.org/html/2602.04118v1))
- **Prompt optimization** based on failure patterns
- **Autoresearch-style** improvement loops ([Karpathy's autoresearch](https://github.com/karpathy/autoresearch))

---

## License

MIT

## Credits

- **[Qwen3.5](https://huggingface.co/Qwen)** — the models (Alibaba)
- **[llama.cpp](https://github.com/ggergov/llama.cpp)** — inference engine (Georgi Gerganov)
- **[PicoClaw](https://github.com/sipeed/picoclaw)** — agent framework for file ops (Sipeed)
- **[Unsloth](https://huggingface.co/unsloth)** — GGUF quantizations
- **[Rich](https://github.com/Textualize/rich)** — terminal UI (Will McGugan)
- **[DuckDuckGo](https://duckduckgo.com)** — web search (no API key needed)
