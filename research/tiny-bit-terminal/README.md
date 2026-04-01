# Tiny Bit Terminal — Compare Tiny AI Models

A retro CRT-style terminal for comparing tiny AI models locally on your Mac. No cloud, no API keys. Start with a 1-bit 8B model (1.16 GB) and a 2-bit 9B model (3.19 GB) — both fit on any Mac.

```
          ______________
         /             /|
        /             / |
       /____________ /  |
      | ___________ |   |
      ||           ||   |
      ||  mac-code ||   |
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

## Quick Start

### Option A: Bonsai-8B (1-bit, 1.16 GB — runs on ANY Mac)

The fastest option. A 1-bit 8B model that fits in ~1 GB of RAM.

```bash
# 1. Install Prism's llama.cpp fork (supports 1-bit models)
git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-server

# 2. Download the model (1.16 GB)
pip install huggingface-hub
huggingface-cli download prism-ml/Bonsai-8B-gguf Bonsai-8B.gguf --local-dir ./models

# 3. Start the server
./build/bin/llama-server -m ./models/Bonsai-8B.gguf -ngl 999 -c 2048 --port 8203 --host 127.0.0.1

# 4. Run mac-code (in another terminal)
cd mac-code-ui
npm install
npx tsx src/index.tsx --server http://localhost:8203
```

**Speed:** ~9-20 tok/s on M2, ~50-130 tok/s on M4 Pro

### Option B: Qwen3.5-9B (IQ2_XXS, 3.19 GB — 64K context)

Smarter model with long context. Needs 8+ GB RAM.

```bash
# 1. Build stock llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --target llama-server

# 2. Download the model (3.19 GB)
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-IQ2_XXS.gguf --local-dir ./models

# 3. Start the server
./build/bin/llama-server -m ./models/Qwen3.5-9B-UD-IQ2_XXS.gguf -ngl 999 -c 4096 --port 8204 --host 127.0.0.1 --reasoning off

# 4. Run mac-code
cd mac-code-ui
npm install
npx tsx src/index.tsx --server http://localhost:8204
```

**Speed:** ~1-5 tok/s on 8 GB, ~10-20 tok/s on 16+ GB

### Option C: Both models + compare mode

Run both servers, then:
```bash
npx tsx src/index.tsx --server http://localhost:8203
```

Use `/compare <prompt>` to race both models side-by-side.

## Features

| Command | What it does |
|---------|-------------|
| (just type) | Chat — model auto-uses tools when needed |
| `/search <query>` | Web search via DuckDuckGo + AI synthesis |
| `/shell <task>` | Run shell commands (or ask naturally) |
| `/image <path>` | Describe an image |
| `/screenshot` | Capture + analyze your screen |
| `/compare <prompt>` | Race Bonsai vs Qwen3.5-9B side-by-side |
| `/stats` | Server stats |
| `/clear` | Clear chat |
| `/help` | Show commands |
| `/quit` | Exit |

## Tool Calling

The model automatically uses tools when needed. Just ask naturally:

```
> what's on my desktop?
◆ Running: $ ls ~/Desktop
◆ Output:
  photo.png
  notes.txt
  project/

Your desktop has a photo, a text file, and a project folder.

> read notes.txt
◆ Reading: ~/Desktop/notes.txt
◆ Contents: ...
```

No slash commands needed — the model decides when to search the web, run commands, or read files.

## Models

| Model | Size | Speed (est.) | Best For |
|-------|------|-------------|----------|
| **Qwen3-0.6B** | 0.4 GB | 50+ tok/s | Ultra-fast, testing |
| **Bonsai-8B (1-bit)** | 1.16 GB | 9-20 tok/s | Fast chat, tool calling |
| **Qwen3-1.7B** | 1.1 GB | 30+ tok/s | Lightweight assistant |
| **Ministral-3B** | 2.15 GB | 15-25 tok/s | Balanced quality/speed |
| **Qwen3-4B** | 2.5 GB | 15-25 tok/s | Good quality, small |
| **Qwen3.5-9B (IQ2)** | 3.19 GB | 1-5 tok/s | Smart, 64K context |
| **Qwen3-8B** | 5.0 GB | 5-15 tok/s | Best quality under 8B |

Use `/models` in the terminal to see download commands and check which are running.
Use `/compare` to race any two models side-by-side.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Node.js 18+ or Bun
- llama.cpp (built from source)
- 8 GB RAM minimum (Bonsai runs on 4 GB)
