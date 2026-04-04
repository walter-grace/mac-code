#!/bin/bash
set -e -o pipefail

# Expert Sniper — RunPod Serverless Worker
# Runs llama.cpp with madvise expert cache for MoE models

cleanup() {
    echo "start.sh: Cleaning up..."
    pkill -P $$
    exit 0
}

trap cleanup SIGINT SIGTERM

# Cached model support
CACHED_LLAMA_ARGS=""
if [ -n "$LLAMA_CACHED_MODEL" ]; then
    CACHED_LLAMA_ARGS="-m $(python ./find_cached.py $LLAMA_CACHED_MODEL $LLAMA_CACHED_GGUF_PATH)"
    echo "start.sh: Using cached model: $CACHED_LLAMA_ARGS"
fi

# Default model if not set
if [ -z "$LLAMA_SERVER_CMD_ARGS" ]; then
    LLAMA_SERVER_CMD_ARGS="-hf unsloth/Qwen3.5-35B-A3B-GGUF:IQ2_M --ctx-size 4096"
    echo "start.sh: Defaulting to Qwen3.5-35B-A3B IQ2_M (Expert Sniper mode)"
fi

# Block --port (we always use 3098)
if [[ "$LLAMA_SERVER_CMD_ARGS" == *"--port"* ]]; then
    echo "start.sh: Error: Do not define --port. Port 3098 is required."
    exit 1
fi

# Expert Sniper: auto-enable madvise expert cache
EXPERT_CACHE_ARGS=""
if [ -n "$EXPERT_CACHE_SIZE" ]; then
    EXPERT_CACHE_ARGS="--expert-cache-size $EXPERT_CACHE_SIZE"
    echo "start.sh: Expert Sniper cache size: $EXPERT_CACHE_SIZE"
fi

# Auto-detect MoE models and enable expert cache
if [ -z "$EXPERT_CACHE_ARGS" ]; then
    if [[ "$LLAMA_SERVER_CMD_ARGS" == *"35B"* ]] || \
       [[ "$LLAMA_SERVER_CMD_ARGS" == *"30B"* ]] || \
       [[ "$LLAMA_SERVER_CMD_ARGS" == *"26B"* ]] || \
       [[ "$LLAMA_SERVER_CMD_ARGS" == *"MoE"* ]] || \
       [[ "$LLAMA_SERVER_CMD_ARGS" == *"moe"* ]] || \
       [[ "$LLAMA_SERVER_CMD_ARGS" == *"A3B"* ]] || \
       [[ "$LLAMA_SERVER_CMD_ARGS" == *"A4B"* ]]; then
        EXPERT_CACHE_ARGS="--expert-cache-size 1"
        echo "start.sh: Expert Sniper auto-enabled (detected MoE model)"
    fi
fi

# GPU layers: if not specified and expert cache is on, default to ngl 0 (CPU + expert streaming)
if [[ "$LLAMA_SERVER_CMD_ARGS" != *"-ngl"* ]] && [ -n "$EXPERT_CACHE_ARGS" ]; then
    LLAMA_SERVER_CMD_ARGS="$LLAMA_SERVER_CMD_ARGS -ngl 0"
    echo "start.sh: Expert Sniper mode: CPU inference with expert streaming"
fi

# Kill existing instances
pkill llama-server 2>/dev/null || true

echo "start.sh: Starting llama-server..."
echo "  Args: $CACHED_LLAMA_ARGS $LLAMA_SERVER_CMD_ARGS $EXPERT_CACHE_ARGS --port 3098"

touch llama.server.log
LD_LIBRARY_PATH=/app/lib /app/llama-server $CACHED_LLAMA_ARGS $LLAMA_SERVER_CMD_ARGS $EXPERT_CACHE_ARGS --port 3098 2>&1 | tee llama.server.log &
LLAMA_SERVER_PID=$!

# Wait for server to start
echo "start.sh: Waiting for llama-server..."
for i in $(seq 1 240); do
    if grep -q "listening" llama.server.log 2>/dev/null; then
        echo "start.sh: llama-server is ready!"
        break
    fi
    if ! kill -0 $LLAMA_SERVER_PID 2>/dev/null; then
        echo "start.sh: Error: llama-server exited unexpectedly"
        cat llama.server.log
        exit 1
    fi
    sleep 0.5
done

if ! grep -q "listening" llama.server.log 2>/dev/null; then
    echo "start.sh: Error: llama-server did not start within 120s"
    cat llama.server.log
    exit 1
fi

echo "start.sh: Delegating to RunPod handler"
python -u handler.py $1
