#!/usr/bin/env python3
"""
Benchmark: SSD pread vs 1-bit fallback for MoE cache-miss serving.

Tests the exact code paths in expert_io.py and flash_moe.py:
  Config A: cache miss → SSD pread (blocks) → gather_qmm
  Config B: cache miss → 1-bit mmap dequant → mx.matmul
            → async SSD backfill → next call is cache hit

Measures: latency per expert, tok/s equivalent, output quality.
"""

import os
import sys
import time
import gc
import json
import numpy as np

# Add parent dirs to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn

from expert_io import MoEExpertReader, FallbackBuffer

MODEL_DIR = os.path.expanduser("~/models/qwen35-35b-moe-stream")
EXPERT_DIR = f"{MODEL_DIR}/experts"
FALLBACK_PATH = "/Volumes/USB DISK/expert_fallback_1bit.bin"
NUM_LAYERS = 21  # layers 0-20 are complete (256 experts each), layer 21 is truncated
BITS = 4
GROUP_SIZE = 64
HIDDEN_DIM = 2048
MOE_INTERMEDIATE = 512
TOP_K = 8
NUM_EXPERTS = 256


def run_expert_ffn_4bit(x, expert_data, expert_ids):
    """SwiGLU via gather_qmm — 4-bit quantized path."""
    active_ids = sorted(expert_data.keys())
    K = len(active_ids)
    id_to_local = {eid: i for i, eid in enumerate(active_ids)}

    def stack_proj(proj_name):
        w = mx.stack([expert_data[eid][f"mlp.switch_mlp.{proj_name}.weight"] for eid in active_ids])
        s = mx.stack([expert_data[eid][f"mlp.switch_mlp.{proj_name}.scales"] for eid in active_ids])
        b = mx.stack([expert_data[eid][f"mlp.switch_mlp.{proj_name}.biases"] for eid in active_ids])
        return w, s, b

    gate_w, gate_s, gate_b = stack_proj("gate_proj")
    up_w, up_s, up_b = stack_proj("up_proj")
    down_w, down_s, down_b = stack_proj("down_proj")

    # Create indices mapping to local expert IDs
    local_ids = mx.array([[[[id_to_local[eid] for eid in expert_ids]]]])
    x_exp = mx.expand_dims(mx.expand_dims(x, 0), 0)  # [1, 1, 1, 1, D]
    x_exp = mx.expand_dims(x_exp, -2)

    gate_out = mx.gather_qmm(x_exp, gate_w, scales=gate_s, biases=gate_b,
                              rhs_indices=local_ids, transpose=True,
                              group_size=GROUP_SIZE, bits=BITS)
    up_out = mx.gather_qmm(x_exp, up_w, scales=up_s, biases=up_b,
                            rhs_indices=local_ids, transpose=True,
                            group_size=GROUP_SIZE, bits=BITS)
    hidden = nn.silu(gate_out) * up_out
    down_out = mx.gather_qmm(hidden, down_w, scales=down_s, biases=down_b,
                              rhs_indices=local_ids, transpose=True,
                              group_size=GROUP_SIZE, bits=BITS)
    return down_out.squeeze()


def run_expert_ffn_1bit(x, expert_data, expert_ids):
    """SwiGLU via mx.matmul — float16 fallback path."""
    results = []
    for eid in expert_ids:
        d = expert_data[eid]
        gate_w = d["mlp.switch_mlp.gate_proj.weight"]
        up_w = d["mlp.switch_mlp.up_proj.weight"]
        down_w = d["mlp.switch_mlp.down_proj.weight"]

        gate_o = mx.matmul(x, gate_w.T)
        up_o = mx.matmul(x, up_w.T)
        h = nn.silu(gate_o) * up_o
        down_o = mx.matmul(h, down_w.T)
        results.append(down_o)

    return mx.stack(results)


def benchmark_config(name, reader, num_runs=3, num_tokens=20):
    """
    Simulate `num_tokens` decode steps across `NUM_LAYERS` layers.
    Each step: router picks TOP_K random experts, get_experts(), compute FFN.
    """
    print(f"\n{'='*60}")
    print(f"  CONFIG {name}")
    print(f"  {num_tokens} tokens × {NUM_LAYERS} layers × {TOP_K} experts")
    print(f"{'='*60}")

    results = []
    np.random.seed(42)

    for run in range(num_runs):
        # Reset cache between runs
        if reader.lru:
            reader.lru.cache.clear()
            reader.lru.hits = 0
            reader.lru.misses = 0
        reader.cache_hits = 0
        reader.reads = 0
        reader.bytes_read = 0
        reader.read_time = 0.0
        if reader.fallback:
            reader.fallback.fallback_hits = 0

        gc.collect()
        mx.clear_cache()

        # Purge disk cache for cold start
        os.system("sudo purge 2>/dev/null")
        time.sleep(0.5)

        x = mx.random.normal((1, 1, HIDDEN_DIM)).astype(mx.float16)
        mx.eval(x)

        token_times = []
        t_total_start = time.time()

        for token in range(num_tokens):
            t_token = time.time()

            for layer_idx in range(NUM_LAYERS):
                # Simulate router: pick TOP_K random experts
                active_ids = list(np.random.choice(NUM_EXPERTS, TOP_K, replace=False))

                # Prefetch next layer
                if layer_idx + 1 < NUM_LAYERS:
                    reader.prefetch_experts(layer_idx + 1,
                                           list(np.random.choice(NUM_EXPERTS, TOP_K, replace=False)))

                experts_4bit, experts_1bit = reader.get_experts(layer_idx, active_ids)

                # Compute FFN through both paths
                if experts_4bit:
                    out_4bit = run_expert_ffn_4bit(x, experts_4bit,
                                                   [e for e in active_ids if e in experts_4bit])
                    mx.eval(out_4bit)

                if experts_1bit:
                    out_1bit = run_expert_ffn_1bit(x, experts_1bit,
                                                   [e for e in active_ids if e in experts_1bit])
                    mx.eval(out_1bit)

            token_time = time.time() - t_token
            token_times.append(token_time)

        total_time = time.time() - t_total_start
        tok_per_sec = num_tokens / total_time
        ttft = token_times[0]

        # Collect stats
        cache_stats = ""
        if reader.lru:
            cache_stats = reader.lru.stats()
        fallback_stats = ""
        if reader.fallback:
            fallback_stats = reader.fallback.stats()

        result = {
            "run": run + 1,
            "total_time": total_time,
            "tok_per_sec": tok_per_sec,
            "ttft": ttft,
            "avg_token": np.mean(token_times),
            "min_token": np.min(token_times),
            "max_token": np.max(token_times),
            "ssd_reads": reader.reads - reader.cache_hits,
            "cache_hits": reader.cache_hits,
            "total_reads": reader.reads,
            "bytes_read_gb": reader.bytes_read / 1e9,
            "io_time": reader.read_time,
            "cache_stats": cache_stats,
            "fallback_stats": fallback_stats,
        }
        results.append(result)

        print(f"\n  Run {run+1}/{num_runs}:")
        print(f"    tok/s:          {tok_per_sec:.2f}")
        print(f"    TTFT:           {ttft*1000:.0f} ms")
        print(f"    Avg token:      {result['avg_token']*1000:.0f} ms")
        print(f"    SSD reads:      {result['ssd_reads']}")
        print(f"    Cache hits:     {result['cache_hits']}")
        print(f"    I/O time:       {result['io_time']:.2f}s ({result['io_time']/total_time*100:.0f}% of total)")
        print(f"    Bytes from SSD: {result['bytes_read_gb']:.2f} GB")
        if cache_stats:
            print(f"    {cache_stats}")
        if fallback_stats:
            print(f"    {fallback_stats}")

    return results


def benchmark_quality(reader_a, reader_b, num_samples=10):
    """Compare output quality: 4-bit gather_qmm vs 1-bit matmul on same inputs."""
    print(f"\n{'='*60}")
    print("  QUALITY COMPARISON: 4-bit vs 1-bit on same experts")
    print(f"{'='*60}")

    np.random.seed(123)
    cosines = []

    for sample in range(num_samples):
        layer_idx = np.random.randint(0, NUM_LAYERS)
        active_ids = list(np.random.choice(NUM_EXPERTS, TOP_K, replace=False))
        x = mx.random.normal((HIDDEN_DIM,)).astype(mx.float16)

        # Get 4-bit experts from SSD
        e4bit, _ = reader_a.get_experts(layer_idx, active_ids)
        # Get 1-bit experts from fallback
        fallback = reader_b.fallback
        e1bit = {}
        for eid in active_ids:
            e1bit[eid] = fallback.get_expert_f16(layer_idx, eid)

        # Run both FFN paths
        out_4bit = run_expert_ffn_4bit(x.reshape(1, 1, -1), e4bit, active_ids)
        out_1bit = run_expert_ffn_1bit(x.reshape(1, 1, -1), e1bit, active_ids)
        mx.eval(out_4bit, out_1bit)

        # Cosine similarity
        a = out_4bit.reshape(-1).astype(mx.float32)
        b = out_1bit.reshape(-1).astype(mx.float32)
        cos = (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b) + 1e-8)).item()
        cosines.append(cos)

        print(f"  Sample {sample+1}: layer={layer_idx:2d}, experts={active_ids[:3]}... "
              f"cosine={cos:.4f}")

    print(f"\n  Mean cosine similarity: {np.mean(cosines):.4f}")
    print(f"  Min:  {np.min(cosines):.4f}")
    print(f"  Max:  {np.max(cosines):.4f}")
    return cosines


def main():
    print("=" * 60)
    print("  EXPERT SNIPER — 1-Bit Fallback Benchmark")
    print(f"  {NUM_LAYERS} layers × {NUM_EXPERTS} experts × {TOP_K} active/layer")
    print(f"  Fallback: {FALLBACK_PATH}")
    print("=" * 60)

    # Config A: no fallback, no cache (pure SSD)
    print("\n[Config A] Initializing: SSD-only (no fallback, no cache)...")
    reader_a = MoEExpertReader(EXPERT_DIR, NUM_LAYERS, num_workers=8, cache_size=0)

    # Config B: with fallback + LRU cache
    print("[Config B] Initializing: 1-bit fallback + LRU cache...")
    reader_b = MoEExpertReader(EXPERT_DIR, NUM_LAYERS, num_workers=8,
                                cache_size=200,
                                fallback_path=FALLBACK_PATH)

    # --- Benchmark throughput ---
    print("\n\n" + "=" * 60)
    print("  PHASE 1: THROUGHPUT (20 tokens, cold start)")
    print("=" * 60)

    results_a = benchmark_config("A (SSD-only)", reader_a, num_runs=3, num_tokens=20)
    results_b = benchmark_config("B (1-bit fallback + cache)", reader_b, num_runs=3, num_tokens=20)

    # --- Quality comparison ---
    print("\n")
    cosines = benchmark_quality(reader_a, reader_b, num_samples=10)

    # --- Summary ---
    avg_a = np.mean([r["tok_per_sec"] for r in results_a])
    avg_b = np.mean([r["tok_per_sec"] for r in results_b])
    speedup = avg_b / avg_a if avg_a > 0 else 0

    print(f"\n\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Config A (SSD-only):        {avg_a:.2f} tok/s avg")
    print(f"  Config B (1-bit fallback):   {avg_b:.2f} tok/s avg")
    print(f"  Speedup:                     {speedup:.2f}x")
    print(f"  Quality (cosine sim):        {np.mean(cosines):.4f} avg")
    print(f"{'='*60}")

    ttft_a = np.mean([r["ttft"] for r in results_a])
    ttft_b = np.mean([r["ttft"] for r in results_b])
    print(f"  TTFT A: {ttft_a*1000:.0f} ms")
    print(f"  TTFT B: {ttft_b*1000:.0f} ms")

    io_pct_a = np.mean([r["io_time"]/r["total_time"]*100 for r in results_a])
    io_pct_b = np.mean([r["io_time"]/r["total_time"]*100 for r in results_b])
    print(f"  I/O % of total A: {io_pct_a:.0f}%")
    print(f"  I/O % of total B: {io_pct_b:.0f}%")

    ssd_a = np.mean([r["bytes_read_gb"] for r in results_a])
    ssd_b = np.mean([r["bytes_read_gb"] for r in results_b])
    print(f"  SSD read A: {ssd_a:.2f} GB")
    print(f"  SSD read B: {ssd_b:.2f} GB")

    reader_a.close()
    reader_b.close()


if __name__ == "__main__":
    main()
