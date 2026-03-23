#!/bin/bash
# mac code setup — one-command install
set -e

echo ""
echo "  🍎 mac code setup"
echo "  ─────────────────────"
echo ""

# 1. Dependencies
echo "  [1/5] Installing dependencies..."
brew install llama.cpp 2>/dev/null || echo "  llama.cpp already installed"
pip3 install huggingface-hub rich --break-system-packages -q 2>/dev/null

# 2. Download model
echo "  [2/5] Downloading Qwen3.5-35B-A3B (10.6 GB)..."
echo "         This may take 5-15 minutes."
mkdir -p ~/models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
print('  Model downloaded.')
"

# 3. Build PicoClaw
echo "  [3/5] Building PicoClaw agent..."
if [ ! -d "picoclaw" ]; then
    git clone --quiet https://github.com/sipeed/picoclaw.git
fi
cd picoclaw && make deps -s && make build -s && cd ..
echo "  PicoClaw built."

# 4. Configure
echo "  [4/5] Setting up config..."
mkdir -p ~/.picoclaw/workspace
if [ ! -f ~/.picoclaw/config.json ]; then
    cp config.example.json ~/.picoclaw/config.json
    echo "  Config created at ~/.picoclaw/config.json"
else
    echo "  Config already exists, skipping."
fi

# 5. Done
echo "  [5/5] Done!"
echo ""
echo "  ─────────────────────"
echo "  Start the LLM server:"
echo ""
echo "    llama-server \\"
echo "      --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \\"
echo "      --port 8000 --host 127.0.0.1 \\"
echo "      --flash-attn on --ctx-size 8192 \\"
echo "      --n-gpu-layers 99 --reasoning off -np 1 -t 4"
echo ""
echo "  Then run the agent:"
echo ""
echo "    python3 agent.py"
echo ""
