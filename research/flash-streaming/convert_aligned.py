"""
Convert FFN layers to 16KB-aligned raw binary for direct I/O.

Format per layer file (layer_XX.bin):
  [16KB header] — JSON metadata padded to 16384 bytes
  [tensor data]  — each tensor starts at 16KB-aligned offset

This bypasses safetensors' misaligned headers and enables
F_NOCACHE + pread at full NVMe bandwidth.
"""

import os
import json
import struct
import time
import numpy as np
import mlx.core as mx

MODEL_DIR = "/Users/bigneek/models/qwen3-32b-flash-stream"
OUTPUT_DIR = "/Users/bigneek/models/qwen3-32b-flash-stream/ffn_aligned"
PAGE_SIZE = 16384  # 16KB — Apple Silicon page size


def align_up(offset, alignment=PAGE_SIZE):
    return (offset + alignment - 1) & ~(alignment - 1)


def convert_layer(layer_idx):
    """Convert one safetensors FFN layer to 16KB-aligned binary."""
    # Load from existing safetensors
    st_path = f"{MODEL_DIR}/ffn/layer_{layer_idx:02d}.safetensors"
    data = mx.load(st_path)

    # Plan the layout
    # Tensor order: gate.weight, gate.scales, gate.biases,
    #               up.weight, up.scales, up.biases,
    #               down.weight, down.scales, down.biases
    tensor_order = [
        "mlp.gate_proj.weight", "mlp.gate_proj.scales", "mlp.gate_proj.biases",
        "mlp.up_proj.weight", "mlp.up_proj.scales", "mlp.up_proj.biases",
        "mlp.down_proj.weight", "mlp.down_proj.scales", "mlp.down_proj.biases",
    ]

    # Compute offsets — data starts after 16KB header
    offset = PAGE_SIZE  # First tensor starts at 16KB
    layout = {}
    for name in tensor_order:
        arr = data[name]
        mx.eval(arr)
        nbytes = arr.nbytes
        dtype = str(arr.dtype)
        shape = list(arr.shape)

        layout[name] = {
            "offset": offset,
            "nbytes": nbytes,
            "dtype": dtype,
            "shape": shape,
        }
        # Next tensor aligned to 16KB
        offset = align_up(offset + nbytes)

    total_size = offset

    # Build header JSON
    header = {
        "format": "flash_aligned_v1",
        "page_size": PAGE_SIZE,
        "layer_idx": layer_idx,
        "total_size": total_size,
        "tensors": layout,
    }
    header_json = json.dumps(header).encode("utf-8")
    assert len(header_json) < PAGE_SIZE, f"Header too large: {len(header_json)} bytes"

    # Write the file
    out_path = f"{OUTPUT_DIR}/layer_{layer_idx:02d}.bin"
    with open(out_path, "wb") as f:
        # Write header padded to PAGE_SIZE
        f.write(header_json)
        f.write(b"\x00" * (PAGE_SIZE - len(header_json)))

        # Write each tensor at its aligned offset
        for name in tensor_order:
            arr = data[name]
            arr_np = np.array(arr)
            current = f.tell()
            target = layout[name]["offset"]
            if current < target:
                f.write(b"\x00" * (target - current))
            f.write(arr_np.tobytes())

        # Pad to final size
        current = f.tell()
        if current < total_size:
            f.write(b"\x00" * (total_size - current))

    return total_size


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Converting FFN layers to 16KB-aligned binary...")
    t0 = time.time()
    total = 0
    for i in range(64):
        size = convert_layer(i)
        total += size
        if i % 16 == 0:
            print(f"  Layer {i}: {size/1e6:.1f} MB")

    print(f"\n  64 layers: {total/1e9:.2f} GB")
    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"  Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
