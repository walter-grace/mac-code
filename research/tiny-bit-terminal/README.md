# Tiny Bit Terminal

A local AI agent that runs entirely on your Mac. Tool calling, web search, document parsing, and multi-model compare — all from a retro terminal UI.

```
          ______________
         /             /|
        /             / |
       /____________ /  |
      | ___________ |   |
      ||           ||   |
      ||  tiny bit ||   |
      ||           ||   |
      ||___________||   |
      |   _______   |  /
     /|  (_______)  | /
    ( |_____________|/
     \
 .=======================.
 | ::::::::::::::::  ::: |
 | ::::::::::::::[]  ::: |
 |   -----------     ::: |
 `-----------------------'
```

## Install

```bash
# 1. Clone
git clone https://github.com/walter-grace/mac-code.git
cd mac-code/research/tiny-bit-terminal
npm install

# 2. Install python search (one time)
pip3 install duckduckgo-search

# 3. Build llama.cpp and start a model server (pick one):

# Option A: Bonsai-8B (1-bit, 1.16 GB — runs on ANY Mac)
git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp && cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu) --target llama-server
huggingface-cli download prism-ml/Bonsai-8B-gguf Bonsai-8B.gguf --local-dir ./models
./build/bin/llama-server -m ./models/Bonsai-8B.gguf -ngl 999 -c 2048 --port 8203

# Option B: Qwen3.5-9B (2-bit, 3.19 GB — smarter, needs 8+ GB RAM)
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp && cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu) --target llama-server
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-IQ2_XXS.gguf --local-dir ./models
./build/bin/llama-server -m ./models/Qwen3.5-9B-UD-IQ2_XXS.gguf -ngl 999 -c 4096 --port 8204

# Option C: Gemma 4-26B MoE (IQ2, 9.3 GB — Google's 26B MoE, 36 tok/s on 16GB)
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
cd ~/llama.cpp && cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu) --target llama-server
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/gemma-4-26B-A4B-it-GGUF',
    'gemma-4-26B-A4B-it-UD-IQ2_M.gguf', local_dir='./models')
"
./build/bin/llama-server -m ./models/gemma-4-26B-A4B-it-UD-IQ2_M.gguf -ngl 99 -c 2048 --reasoning off --port 8205

# 4. Run tiny bit (in another terminal)
cd mac-code/research/tiny-bit-terminal
npx tsx src/index.tsx --server http://localhost:8203
```

## What It Does

Just type naturally. The model calls tools automatically:

```
▶ you: what's on my desktop?
◆ Running: $ ls ~/Desktop
◆ Output: photo.png  notes.txt  project/
tiny bit: Your desktop has a photo, a text file, and a project folder.

▶ you: search for today's AI news
◆ Searching: AI LLM news March 2026
◆ Results: [2026-03-31] OpenAI drops Sora...
tiny bit: Here are today's highlights...
```

## Commands

| Command | What it does |
|---------|-------------|
| (just type) | Chat with tool calling |
| `/search <query>` | Web search + AI summary |
| `/shell <task>` | Run shell commands (or describe what you want) |
| `/document <path>` | Parse PDF, DOCX, images |
| `/compare` | Race two models side-by-side |
| `/models` | List available models + download commands |
| `/stats` | Server performance stats |
| `/clear` | Clear chat history |
| `/help` | Show all commands |
| `/quit` | Exit |

## Architecture (PicoClaw-inspired)

The agent uses patterns from [PicoClaw](https://github.com/sipeed/picoclaw):

- **Context Budget** — auto-trims conversation history before each LLM call (2.5 chars/token heuristic). Never overflows the context window.
- **Tool Discovery** — core tools always visible, hidden tools (write_file, list_dir, screenshot, system_info) discoverable via `search_tools` meta-tool with auto-demotion TTL.
- **Deterministic Tool Order** — tools sorted alphabetically for better KV cache hits in llama.cpp.
- **Steering Queue** — user messages during tool execution are queued and injected at the next safe point.
- **Parallel Tool Execution** — multiple tool calls in one response run concurrently with per-tool error recovery.

## Models

| Model | Size | Speed (M2 Air) | Notes |
|-------|------|----------------|-------|
| Bonsai-8B (1-bit) | 1.16 GB | 9-20 tok/s | Needs [PrismML fork](https://github.com/PrismML-Eng/llama.cpp) |
| Qwen3-0.6B | 0.4 GB | 50+ tok/s | Ultra-fast, basic |
| Qwen3-1.7B | 1.1 GB | 30+ tok/s | Lightweight |
| Ministral-3B | 2.15 GB | 15-25 tok/s | Balanced |
| Qwen3-4B | 2.5 GB | 15-25 tok/s | Good quality |
| Qwen3.5-9B (IQ2) | 3.19 GB | 1-5 tok/s | Smart, 64K context |
| **Gemma 4-26B MoE (IQ2)** | **9.3 GB** | **36 tok/s (16GB) / 1.4 tok/s (8GB)** | **Google's MoE, 128 experts** |
| Gemma 4-26B MoE (Q4) | 16.9 GB | 5 tok/s (16GB) | Higher quality, needs 16GB |

Run `/models` in the terminal to check which are online and get download commands.

## Requirements

- macOS with Apple Silicon (M1+)
- Node.js 18+
- Python 3 with `duckduckgo-search` (for web search)
- llama.cpp (built from source)
- 4 GB RAM minimum (Bonsai), 8 GB for larger models
