#!/usr/bin/env python3
"""
Quick benchmark: SSD pread vs 1-bit mmap fallback.
Tests 5 tokens × 5 layers × 8 experts. Unbuffered output.
"""
import os, sys, time, gc, json
import numpy as np

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # line-buffered

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
import mlx.nn as nn
from expert_io import MoEExpertReader

EXPERT_DIR = os.path.expanduser("~/models/qwen35-35b-moe-stream/experts")
FALLBACK_PATH = "/Volumes/USB DISK/expert_fallback_1bit.bin"
NUM_LAYERS = 10  # use first 10 layers only
TOP_K = 8
NUM_EXPERTS = 256
HIDDEN_DIM = 2048
BITS = 4
GROUP_SIZE = 64
NUM_TOKENS = 5
NUM_RUNS = 2


def swiglu_4bit(x, expert_data, active_ids):
    """gather_qmm path"""
    ids = sorted(expert_data.keys())
    id_map = {eid: i for i, eid in enumerate(ids)}

    def stack(proj):
        w = mx.stack([expert_data[e][f"mlp.switch_mlp.{proj}.weight"] for e in ids])
        s = mx.stack([expert_data[e][f"mlp.switch_mlp.{proj}.scales"] for e in ids])
        b = mx.stack([expert_data[e][f"mlp.switch_mlp.{proj}.biases"] for e in ids])
        return w, s, b

    gw, gs, gb = stack("gate_proj")
    uw, us, ub = stack("up_proj")
    dw, ds, db = stack("down_proj")

    local = mx.array([[[[id_map[e] for e in active_ids]]]])
    xe = mx.expand_dims(mx.expand_dims(x, 0), 0)
    xe = mx.expand_dims(xe, -2)

    g = mx.gather_qmm(xe, gw, scales=gs, biases=gb, rhs_indices=local,
                       transpose=True, group_size=GROUP_SIZE, bits=BITS)
    u = mx.gather_qmm(xe, uw, scales=us, biases=ub, rhs_indices=local,
                       transpose=True, group_size=GROUP_SIZE, bits=BITS)
    h = nn.silu(g) * u
    d = mx.gather_qmm(h, dw, scales=ds, biases=db, rhs_indices=local,
                       transpose=True, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(d)
    return d.squeeze()


def swiglu_1bit(x, expert_data, active_ids):
    """matmul path"""
    outs = []
    for eid in active_ids:
        d = expert_data[eid]
        g = mx.matmul(x, d["mlp.switch_mlp.gate_proj.weight"].T)
        u = mx.matmul(x, d["mlp.switch_mlp.up_proj.weight"].T)
        h = nn.silu(g) * u
        o = mx.matmul(h, d["mlp.switch_mlp.down_proj.weight"].T)
        outs.append(o)
    out = mx.stack(outs)
    mx.eval(out)
    return out


def run_config(name, reader, num_tokens, num_runs):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  {num_tokens} tokens × {NUM_LAYERS} layers × {TOP_K} experts")
    print(f"{'='*50}")

    all_tps = []
    np.random.seed(42)

    for run in range(num_runs):
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
        os.system("sudo purge 2>/dev/null")
        time.sleep(0.3)

        x = mx.random.normal((1, 1, HIDDEN_DIM)).astype(mx.float16)
        mx.eval(x)

        times = []
        t0 = time.time()

        for tok in range(num_tokens):
            tt = time.time()
            for layer in range(NUM_LAYERS):
                active = list(np.random.choice(NUM_EXPERTS, TOP_K, replace=False))

                e4, e1 = reader.get_experts(layer, active)

                if e4:
                    swiglu_4bit(x.squeeze(), e4, [e for e in active if e in e4])
                if e1:
                    swiglu_1bit(x.squeeze(), e1, [e for e in active if e in e1])

            times.append(time.time() - tt)
            print(f"    token {tok}: {times[-1]*1000:.0f}ms", flush=True)

        total = time.time() - t0
        tps = num_tokens / total

        print(f"  Run {run+1}: {tps:.2f} tok/s, TTFT={times[0]*1000:.0f}ms, "
              f"avg={np.mean(times)*1000:.0f}ms")
        print(f"    reads={reader.reads}, cache_hits={reader.cache_hits}, "
              f"SSD={reader.reads - reader.cache_hits}, "
              f"I/O={reader.read_time:.2f}s ({reader.read_time/total*100:.0f}%)")
        if reader.lru:
            print(f"    {reader.lru.stats()}")
        if reader.fallback:
            print(f"    {reader.fallback.stats()}")
        all_tps.append(tps)

    return all_tps


def quality_test(reader_a, reader_b):
    print(f"\n{'='*50}")
    print("  QUALITY: 4-bit vs 1-bit (cosine similarity)")
    print(f"{'='*50}")

    np.random.seed(99)
    cosines = []

    for i in range(5):
        layer = np.random.randint(0, NUM_LAYERS)
        active = list(np.random.choice(NUM_EXPERTS, TOP_K, replace=False))
        x = mx.random.normal((1, 1, HIDDEN_DIM)).astype(mx.float16)

        e4, _ = reader_a.get_experts(layer, active)
        fb = reader_b.fallback
        e1 = {eid: fb.get_expert_f16(layer, eid) for eid in active}

        out4 = swiglu_4bit(x.squeeze(), e4, active).reshape(-1).astype(mx.float32)
        out1 = swiglu_1bit(x.squeeze(), e1, active).reshape(-1).astype(mx.float32)
        mx.eval(out4, out1)

        cos = (mx.sum(out4 * out1) / (mx.linalg.norm(out4) * mx.linalg.norm(out1) + 1e-8)).item()
        cosines.append(cos)
        print(f"  sample {i+1}: layer={layer}, cosine={cos:.4f}")

    print(f"  MEAN cosine: {np.mean(cosines):.4f}")
    return cosines


def main():
    print("EXPERT SNIPER — Quick Fallback Benchmark")
    print(f"MLX {mx.__version__}, {NUM_LAYERS} layers, {TOP_K} experts/layer")

    # Config A: SSD only, no cache
    print("\n[A] SSD-only reader...")
    ra = MoEExpertReader(EXPERT_DIR, NUM_LAYERS, num_workers=8, cache_size=0)

    # Config B: 1-bit fallback + LRU cache
    print("[B] 1-bit fallback + cache reader...")
    rb = MoEExpertReader(EXPERT_DIR, NUM_LAYERS, num_workers=8,
                         cache_size=200, fallback_path=FALLBACK_PATH)

    tps_a = run_config("CONFIG A: SSD-only (no cache)", ra, NUM_TOKENS, NUM_RUNS)
    tps_b = run_config("CONFIG B: 1-bit fallback + LRU cache", rb, NUM_TOKENS, NUM_RUNS)

    cosines = quality_test(ra, rb)

    avg_a = np.mean(tps_a)
    avg_b = np.mean(tps_b)
    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"  Config A (SSD-only):   {avg_a:.2f} tok/s")
    print(f"  Config B (fallback):   {avg_b:.2f} tok/s")
    print(f"  Speedup:               {avg_b/avg_a:.2f}x")
    print(f"  Quality (cosine):      {np.mean(cosines):.4f}")
    print(f"{'='*50}")

    ra.close()
    rb.close()


if __name__ == "__main__":
    main()
