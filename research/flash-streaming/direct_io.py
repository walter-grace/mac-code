"""
Direct I/O reader for 16KB-aligned FFN weight files.

Bypasses macOS Unified Buffer Cache (UBC) entirely:
  - fcntl(F_NOCACHE) prevents cache pollution
  - pread for atomic positioned reads (no seek mutex)
  - posix_memalign-equivalent aligned buffers via numpy
  - Multi-threaded reads to saturate NVMe bus

This eliminates the page fault overhead that limited
mmap-based streaming to 0.12 tok/s.
"""

import os
import json
import fcntl
import struct
import ctypes
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

PAGE_SIZE = 16384  # 16KB Apple Silicon page size
F_NOCACHE = 48     # macOS fcntl constant


class AlignedBuffer:
    """16KB-aligned memory buffer for direct I/O."""

    def __init__(self, size):
        # Allocate with extra space for alignment
        self.size = size
        aligned_size = size + PAGE_SIZE
        self._raw = np.empty(aligned_size, dtype=np.uint8)
        # Find the 16KB-aligned start within the buffer
        addr = self._raw.ctypes.data
        offset = (PAGE_SIZE - (addr % PAGE_SIZE)) % PAGE_SIZE
        self.buf = self._raw[offset:offset + size]
        assert self.buf.ctypes.data % PAGE_SIZE == 0, "Buffer not aligned!"

    def as_numpy(self, dtype, shape):
        return np.frombuffer(self.buf[:np.prod(shape) * np.dtype(dtype).itemsize],
                            dtype=dtype).reshape(shape)


class DirectFFNReader:
    """
    Reads FFN layer weights using direct I/O (F_NOCACHE + pread).
    No mmap. No UBC. No page faults.
    """

    def __init__(self, aligned_dir, num_layers=64):
        self.aligned_dir = aligned_dir
        self.num_layers = num_layers
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.prefetch_future = None
        self.prefetch_idx = -1

        # Pre-parse all layer headers
        self.headers = {}
        for i in range(num_layers):
            path = f"{aligned_dir}/layer_{i:02d}.bin"
            with open(path, "rb") as f:
                header_bytes = f.read(PAGE_SIZE)
            # Strip null padding
            header_json = header_bytes.rstrip(b"\x00").decode("utf-8")
            self.headers[i] = json.loads(header_json)

        # Open file descriptors lazily with F_NOCACHE
        self.fds = {}
        self._fd_paths = {}
        for i in range(num_layers):
            self._fd_paths[i] = f"{aligned_dir}/layer_{i:02d}.bin"

        # Timing stats
        self.read_time = 0.0
        self.reads = 0

    def _read_tensor(self, fd, offset, nbytes):
        """Read tensor data using single pread (atomic, no seek)."""
        data = os.pread(fd, nbytes, offset)
        if len(data) != nbytes:
            raise IOError(f"pread returned {len(data)} bytes, expected {nbytes} at offset {offset}")
        return np.frombuffer(data, dtype=np.uint8)

    def _read_layer(self, layer_idx):
        """Read all tensors for one FFN layer using direct I/O."""
        t0 = time.time()
        import mlx.core as mx

        header = self.headers[layer_idx]
        # Lazy FD open
        if layer_idx not in self.fds:
            fd = os.open(self._fd_paths[layer_idx], os.O_RDONLY)
            fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[layer_idx] = fd
        fd = self.fds[layer_idx]
        result = {}

        for name, info in header["tensors"].items():
            offset = info["offset"]
            nbytes = info["nbytes"]
            shape = info["shape"]
            dtype_str = info["dtype"]

            # Map MLX dtype strings to numpy dtypes (handles "mlx.core.float16" format)
            clean_dtype = dtype_str.replace("mlx.core.", "")
            np_dtype = {
                "float16": np.float16,
                "float32": np.float32,
                "uint32": np.uint32,
                "bfloat16": np.float16,
            }.get(clean_dtype, np.uint8)

            raw = self._read_tensor(fd, offset, nbytes)
            arr = np.frombuffer(raw[:np.prod(shape) * np.dtype(np_dtype).itemsize],
                               dtype=np_dtype).reshape(shape)
            result[name] = mx.array(arr)

        self.read_time += time.time() - t0
        self.reads += 1
        return result

    def prefetch(self, layer_idx):
        if self.prefetch_future is not None and self.prefetch_idx == layer_idx:
            return
        self.prefetch_future = self.executor.submit(self._read_layer, layer_idx)
        self.prefetch_idx = layer_idx

    def get(self, layer_idx):
        if self.prefetch_future is not None and self.prefetch_idx == layer_idx:
            data = self.prefetch_future.result()
            self.prefetch_future = None
            return data
        return self._read_layer(layer_idx)

    def stats(self):
        avg = (self.read_time / self.reads * 1000) if self.reads > 0 else 0
        throughput = (self.reads * 221e6) / self.read_time / 1e9 if self.read_time > 0 else 0
        return f"reads={self.reads}, avg={avg:.0f}ms/layer, throughput={throughput:.1f} GB/s, total_io={self.read_time:.1f}s"

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.executor.shutdown(wait=False)
