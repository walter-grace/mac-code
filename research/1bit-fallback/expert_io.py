"""
MoE Expert Sniper — Read only active experts from SSD via F_NOCACHE + pread.

For a 256-expert model with 8 active per token:
  - Each expert: ~1.69 MB (4-bit quantized, moe_intermediate_size=512, hidden_size=2048)
  - Per layer: 8 × 1.69 MB = 13.5 MB
  - Per token (40 layers): ~540 MB
  - At 3-5 GB/s NVMe: ~108-180ms = 5.6-9.3 tok/s theoretical

Uses multi-threaded pread (8 workers) to saturate NVMe queue depth.
1-bit fallback buffer serves cache misses from mmap'd RAM while SSD
backfills the 4-bit version for the next token.
"""

import os
import json
import fcntl
import mmap
import time
import numpy as np
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
PAGE_SIZE = 16384


class DownProjFallback:
    """
    1-bit fallback buffer for down_proj only (mixed-precision strategy).

    On cache miss:
      gate_proj → pread from SSD (4-bit, full quality)
      up_proj   → pread from SSD (4-bit, full quality)
      down_proj → instant dequant from mmap'd 1-bit buffer (0.81 cosine)

    Saves 33% of SSD I/O per cache miss. Buffer is ~792 MB (down_proj only)
    vs 2.38 GB for all projections.

    File format: 16KB JSON header + [layer][expert] down_proj data
    Per expert: fp16 scales + packed sign bits for down_proj
    Reconstruction: weight = scale * (2*bit - 1)
    """

    DOWN_SHAPE = (2048, 512)
    VALUES = 1048576  # 2048 * 512

    def __init__(self, path, group_size=128):
        self.group_size = group_size
        self.enabled = False
        self.fallback_hits = 0
        self.dequant_time = 0.0

        if not os.path.exists(path):
            print(f"  [fallback] file not found: {path}")
            return

        with open(path, "rb") as f:
            raw = f.read(PAGE_SIZE)
        depth = 0
        for i, b in enumerate(raw):
            if b == ord("{"):
                depth += 1
            elif b == ord("}"):
                depth -= 1
                if depth == 0:
                    self.header = json.loads(raw[: i + 1])
                    break

        self.num_layers = self.header["num_layers"]
        self.num_experts = self.header["num_experts"]
        self.expert_1bit_size = self.header["expert_1bit_size"]
        self.data_start = self.header["data_start"]

        padded = self.VALUES + (-self.VALUES % group_size)
        n_groups = padded // group_size
        self.n_groups = n_groups
        self.scales_bytes = n_groups * 2
        self.packed_bytes = n_groups * (group_size // 8)

        self._fd = os.open(path, os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        self.enabled = True

        # Precompute bit masks for GPU unpack (128, 64, 32, 16, 8, 4, 2, 1)
        import mlx.core as mx
        self._bit_masks = mx.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=mx.uint8)

        file_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [fallback] mmap'd {file_mb:.0f} MB down_proj 1-bit buffer "
              f"({self.num_layers} layers × {self.num_experts} experts)")

    def get_down_proj_f16(self, layer_idx, expert_id):
        """
        Dequantize down_proj from 1-bit buffer to float16 mx.array [2048, 512].
        Uses MLX GPU ops for bit unpacking — no numpy unpackbits.
        """
        import mlx.core as mx

        t0 = time.time()
        offset = self.data_start + (layer_idx * self.num_experts + expert_id) * self.expert_1bit_size

        s_end = offset + self.scales_bytes
        p_end = s_end + self.packed_bytes

        # Read raw bytes from mmap into numpy (small copy)
        scales_np = np.frombuffer(self._mm[offset:s_end], dtype=np.float16).copy()
        packed_np = np.frombuffer(self._mm[s_end:p_end], dtype=np.uint8).copy()

        # Move to MLX — GPU does the rest
        scales = mx.array(scales_np).reshape(self.n_groups, 1)
        packed = mx.array(packed_np).reshape(self.n_groups, self.group_size // 8)

        # Unpack sign bits on GPU: each uint8 → 8 bits via bitwise AND
        bits = (mx.expand_dims(packed, -1) & self._bit_masks) > 0
        bits = bits.reshape(self.n_groups, self.group_size).astype(mx.float16)

        # Dequant: weight = scale * (2*bit - 1)
        weights = (2.0 * bits - 1.0) * scales.astype(mx.float16)
        weights = weights.reshape(-1)[:self.VALUES].reshape(self.DOWN_SHAPE)
        mx.eval(weights)

        self.fallback_hits += 1
        self.dequant_time += time.time() - t0
        return weights

    def close(self):
        if hasattr(self, "_mm") and self._mm:
            self._mm.close()
        if hasattr(self, "_fd"):
            os.close(self._fd)

    def stats(self):
        avg_ms = (self.dequant_time / self.fallback_hits * 1000) if self.fallback_hits > 0 else 0
        return (f"fallback_hits={self.fallback_hits}, "
                f"avg_dequant={avg_ms:.1f}ms, "
                f"total_dequant={self.dequant_time:.2f}s")


class LRUExpertCache:
    """LRU cache for parsed expert data. Skips SSD reads on cache hits."""

    def __init__(self, max_experts=100):
        self.max_experts = max_experts
        self.cache = OrderedDict()  # (layer_idx, expert_id) → parsed expert dict
        self.hits = 0
        self.misses = 0

    def get(self, layer_idx, expert_id):
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, layer_idx, expert_id, data):
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_experts:
                self.cache.popitem(last=False)
            self.cache[key] = data

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def stats(self):
        total = self.hits + self.misses
        return (f"cache: {len(self.cache)}/{self.max_experts} entries, "
                f"hit_rate={self.hit_rate():.1%} ({self.hits}/{total})")


class MoEExpertReader:
    """
    Reads specific experts from concatenated layer files via F_NOCACHE + pread.
    Expert offset = data_start + expert_id × expert_block_size

    With fallback_path set, cache misses are served instantly from a 1-bit
    mmap buffer while the 4-bit version loads from SSD for the next token.
    """

    def __init__(self, expert_dir, num_layers, num_workers=8, cache_size=0,
                 fallback_path=None):
        self.expert_dir = expert_dir
        self.num_layers = num_layers
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # LRU cache (0 = disabled)
        self.lru = LRUExpertCache(max_experts=cache_size) if cache_size > 0 else None

        # 1-bit down_proj fallback buffer
        self.fallback = None
        if fallback_path:
            self.fallback = DownProjFallback(fallback_path)
            if not self.fallback.enabled:
                self.fallback = None

        # Parse all layer headers
        self.headers = {}
        self.fds = {}
        for i in range(num_layers):
            path = f"{expert_dir}/layer_{i:02d}.bin"
            with open(path, "rb") as f:
                raw = f.read(PAGE_SIZE)
            self.headers[i] = json.loads(raw.rstrip(b"\x00"))

        # Precompute layout info
        h0 = self.headers[0]["layout"]
        self.expert_block_size = h0["expert_block_size"]
        self.data_start = h0["data_start"]
        self.tensor_layout = h0["tensors"]

        # Stats
        self.read_time = 0.0
        self.reads = 0
        self.bytes_read = 0
        self.cache_hits = 0

        # Prefetch state
        self.prefetch_futures = {}

    def _get_fd(self, layer_idx):
        if layer_idx not in self.fds:
            path = f"{self.expert_dir}/layer_{layer_idx:02d}.bin"
            fd = os.open(path, os.O_RDONLY)
            fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[layer_idx] = fd
        return self.fds[layer_idx]

    def _read_expert(self, layer_idx, expert_id):
        """Read one expert's data via pread. Thread-safe."""
        fd = self._get_fd(layer_idx)
        offset = self.data_start + expert_id * self.expert_block_size

        # Read the full expert block
        data = os.pread(fd, self.expert_block_size, offset)
        return data

    def _parse_expert_data(self, raw_data, expert_id):
        """Parse raw bytes into MLX arrays for one expert."""
        import mlx.core as mx

        # Map dtype strings to MLX dtypes
        MLX_DTYPES = {
            "uint32": mx.uint32, "float16": mx.float16, "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }

        result = {}
        for name, info in self.tensor_layout.items():
            inner_offset = info["inner_offset"]
            nbytes = info["nbytes"]
            shape = info["shape_per_expert"]
            dtype_str = info["dtype"].replace("mlx.core.", "")
            mlx_dtype = MLX_DTYPES.get(dtype_str, mx.float16)

            arr_bytes = raw_data[inner_offset:inner_offset + nbytes]
            # Create MLX array directly from bytes (handles bfloat16 correctly)
            flat = mx.array(np.frombuffer(arr_bytes, dtype=np.uint8))
            arr = flat.view(mlx_dtype).reshape(shape)
            result[name] = arr

        return result

    def prefetch_experts(self, layer_idx, expert_ids):
        """Launch parallel pread for experts not in cache. Non-blocking."""
        futures = {}
        for eid in expert_ids:
            # Skip prefetch if already cached
            if self.lru and (layer_idx, eid) in self.lru.cache:
                continue
            future = self.executor.submit(self._read_expert, layer_idx, eid)
            futures[eid] = future
        self.prefetch_futures[layer_idx] = futures

    def _read_expert_partial(self, layer_idx, expert_id):
        """Read gate_proj + up_proj from SSD (skip down_proj). Thread-safe."""
        fd = self._get_fd(layer_idx)
        offset = self.data_start + expert_id * self.expert_block_size
        # Read only gate + up + their scales/biases (first 1,179,648 bytes)
        # down_proj starts at inner_offset 1,179,648
        down_offset = self.tensor_layout["mlp.switch_mlp.down_proj.weight"]["inner_offset"]
        data = os.pread(fd, down_offset, offset)
        return data

    def _parse_expert_partial(self, raw_data, expert_id):
        """Parse gate_proj + up_proj from partial SSD read."""
        import mlx.core as mx
        MLX_DTYPES = {
            "uint32": mx.uint32, "float16": mx.float16, "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }
        result = {}
        for name, info in self.tensor_layout.items():
            # Skip down_proj tensors — they come from fallback
            if "down_proj" in name:
                continue
            inner_offset = info["inner_offset"]
            nbytes = info["nbytes"]
            shape = info["shape_per_expert"]
            dtype_str = info["dtype"].replace("mlx.core.", "")
            mlx_dtype = MLX_DTYPES.get(dtype_str, mx.float16)
            if inner_offset + nbytes > len(raw_data):
                continue
            arr_bytes = raw_data[inner_offset:inner_offset + nbytes]
            flat = mx.array(np.frombuffer(arr_bytes, dtype=np.uint8))
            arr = flat.view(mlx_dtype).reshape(shape)
            result[name] = arr
        return result

    def get_experts(self, layer_idx, expert_ids):
        """
        Get parsed expert data for active experts.

        Mixed-precision fallback strategy:
          Cache HIT  → all 3 projections from 4-bit cache (gather_qmm)
          Cache MISS → gate+up from SSD pread (4-bit, 2/3 I/O)
                     → down from 1-bit mmap buffer (instant, 0.81 cosine)
                     → async backfill: full expert from SSD into cache

        Returns: dict[expert_id] → dict[tensor_name → mx.array]
          For cache hits: all tensors are 4-bit quantized (uint32 + bf16 scales/biases)
          For fallback:   gate+up are 4-bit, down_proj.weight is float16 mx.array

        The caller (run_expert_ffn) checks dtype to decide gather_qmm vs matmul per projection.
        """
        t0 = time.time()

        experts = {}
        futures = self.prefetch_futures.pop(layer_idx, {})
        backfill_futures = {}

        for eid in expert_ids:
            # 1. Check LRU cache
            if self.lru:
                cached = self.lru.get(layer_idx, eid)
                if cached is not None:
                    experts[eid] = cached
                    self.cache_hits += 1
                    continue

            # 2. Check prefetched data (already read from SSD)
            if eid in futures:
                raw = futures[eid].result()
                parsed = self._parse_expert_data(raw, eid)
                experts[eid] = parsed
                self.bytes_read += len(raw)
                if self.lru:
                    self.lru.put(layer_idx, eid, parsed)
                continue

            # 3. Cache miss
            if self.fallback:
                # Mixed precision: gate+up from SSD, down from 1-bit
                raw_partial = self._read_expert_partial(layer_idx, eid)
                parsed = self._parse_expert_partial(raw_partial, eid)
                self.bytes_read += len(raw_partial)

                # down_proj from 1-bit fallback (instant)
                down_f16 = self.fallback.get_down_proj_f16(layer_idx, eid)
                parsed["mlp.switch_mlp.down_proj.weight"] = down_f16

                experts[eid] = parsed

                # Async backfill: read FULL expert from SSD for next token's cache
                backfill_futures[eid] = self.executor.submit(
                    self._read_expert, layer_idx, eid
                )
            else:
                # No fallback — full synchronous SSD read
                raw = self._read_expert(layer_idx, eid)
                parsed = self._parse_expert_data(raw, eid)
                experts[eid] = parsed
                self.bytes_read += len(raw)
                if self.lru:
                    self.lru.put(layer_idx, eid, parsed)

        if backfill_futures:
            self._schedule_backfill(layer_idx, backfill_futures)

        self.read_time += time.time() - t0
        self.reads += len(expert_ids)
        return experts

    def _schedule_backfill(self, layer_idx, futures):
        """Parse and cache full 4-bit expert from SSD in background."""
        def _do_backfill():
            for eid, future in futures.items():
                try:
                    raw = future.result()
                    parsed = self._parse_expert_data(raw, eid)
                    self.bytes_read += len(raw)
                    if self.lru:
                        self.lru.put(layer_idx, eid, parsed)
                except Exception:
                    pass
        self.executor.submit(_do_backfill)

    def stats(self):
        if self.reads == 0:
            return "No reads yet"
        ssd_reads = self.reads - self.cache_hits
        avg_ms = self.read_time / self.reads * 1000
        throughput = self.bytes_read / self.read_time / 1e9 if self.read_time > 0 else 0
        s = (f"reads={self.reads}, ssd_reads={ssd_reads}, cache_hits={self.cache_hits}, "
             f"avg={avg_ms:.1f}ms/expert, "
             f"throughput={throughput:.1f} GB/s, "
             f"total_bytes={self.bytes_read/1e9:.2f} GB, "
             f"total_time={self.read_time:.1f}s")
        if self.lru:
            s += f"\n  {self.lru.stats()}"
        if self.fallback:
            s += f"\n  {self.fallback.stats()}"
        return s

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.executor.shutdown(wait=False)
        if self.fallback:
            self.fallback.close()
