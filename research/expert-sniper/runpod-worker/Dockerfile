# Expert Sniper — RunPod Serverless Worker
# llama.cpp with madvise expert cache for MoE models exceeding VRAM
#
# Stock llama.cpp: 0 tok/s on 16GB GPU with 19.5GB model
# Expert Sniper:   runs via expert streaming from disk
#
# Patch approach: adds ~65 lines of C++ to stock llama.cpp
# - 4 new files (expert cache context + LRU cache)
# - apply.sh inserts CLI flag + init into existing files (version-tolerant)

# ── Stage 1: Build ──────────────────────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq cmake git python3 && \
    rm -rf /var/lib/apt/lists/*

# Clone latest llama.cpp
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /build/llama.cpp

# Copy Expert Sniper files
COPY patches/src/llama-expert-cache-ctx.cpp /build/llama.cpp/src/
COPY patches/src/llama-expert-cache-ctx.h /build/llama.cpp/src/
COPY patches/src/llama-expert-cache.cpp /build/llama.cpp/src/
COPY patches/src/llama-expert-cache.h /build/llama.cpp/src/
COPY patches/apply.sh /build/apply.sh
RUN chmod +x /build/apply.sh

# Apply patches (idempotent — safe to run multiple times)
RUN cd /build/llama.cpp && /build/apply.sh

# Build with CUDA (A100=80, RTX 3090=86, RTX 4090=89, H100=90)
RUN cd /build/llama.cpp && \
    cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc) --target llama-server

# ── Stage 2: Runtime ────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --yes --quiet && \
    apt-get install --yes --quiet --no-install-recommends \
    software-properties-common gpg-agent && \
    add-apt-repository --yes ppa:deadsnakes/ppa && \
    apt-get update --yes --quiet && \
    apt-get install --yes --quiet --no-install-recommends \
    python3.11 python3.11-dev python3.11-distutils \
    bash curl ca-certificates && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy binary + shared libs from builder
COPY --from=builder /build/llama.cpp/build/bin/llama-server /app/llama-server
RUN mkdir -p /app/lib
COPY --from=builder /build/llama.cpp/build/ggml/src/ /tmp/ggml-libs/
COPY --from=builder /build/llama.cpp/build/src/ /tmp/llama-libs/
RUN find /tmp/ggml-libs /tmp/llama-libs -name "*.so*" -exec cp {} /app/lib/ \; 2>/dev/null; \
    rm -rf /tmp/ggml-libs /tmp/llama-libs

ENV LD_LIBRARY_PATH=/app/lib

# RunPod serverless handler
WORKDIR /work
COPY src/ /work/
RUN pip install -r ./requirements.txt && chmod +x /work/start.sh

ENTRYPOINT ["/bin/sh", "-c", "/work/start.sh"]
