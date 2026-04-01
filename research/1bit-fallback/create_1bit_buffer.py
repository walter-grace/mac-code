#!/usr/bin/env python3
"""
Expert Sniper 1-Bit Fallback Buffer Creator

Reads 4-bit group-quantized MoE expert files (moe_flash_v1 format),
dequantizes to float32, then re-quantizes to 1-bit (sign + scale per group).

Output: a single binary file with all experts from all layers.

Format of output file:
  [JSON header, padded to 16384 bytes]
  [layer 0: expert 0 data, expert 1 data, ..., expert 255 data]
  [layer 1: ...]
  ...

Per expert data (3 projections × (scales + packed_bits)):
  gate_proj.scales (fp16), gate_proj.packed (uint8)
  up_proj.scales (fp16), up_proj.packed (uint8)
  down_proj.scales (fp16), down_proj.packed (uint8)
"""

import json, os, sys, time
import numpy as np

# Config
GROUP_SIZE = 128
INPUT_DIR = os.path.expanduser("~/models/qwen35-35b-moe-stream/experts/")
OUTPUT_PATH = "/Volumes/USB DISK/expert_fallback_1bit.bin"

# Projection definitions (name, weight_key, scales_key, biases_key)
PROJECTIONS = [
    ("gate_proj", "mlp.switch_mlp.gate_proj"),
    ("up_proj", "mlp.switch_mlp.up_proj"),
    ("down_proj", "mlp.switch_mlp.down_proj"),
]


def parse_header(path):
    with open(path, "rb") as f:
        raw = f.read(16384)
    depth = 0
    for i, b in enumerate(raw):
        if b == ord("{"):
            depth += 1
        elif b == ord("}"):
            depth -= 1
            if depth == 0:
                return json.loads(raw[: i + 1])
    raise ValueError("No valid JSON header found")


def read_bf16_as_f32(f, nbytes):
    u16 = np.frombuffer(f.read(nbytes), dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32


def dequantize_projection(f, expert_offset, tensors, proj_prefix):
    w_info = tensors[f"{proj_prefix}.weight"]
    s_info = tensors[f"{proj_prefix}.scales"]
    b_info = tensors[f"{proj_prefix}.biases"]

    # Read packed uint32 weights
    f.seek(expert_offset + w_info["inner_offset"])
    raw_w = np.frombuffer(f.read(w_info["nbytes"]), dtype=np.uint32)

    # Read scales and biases (bf16 -> f32)
    f.seek(expert_offset + s_info["inner_offset"])
    scales = read_bf16_as_f32(f, s_info["nbytes"])

    f.seek(expert_offset + b_info["inner_offset"])
    biases = read_bf16_as_f32(f, b_info["nbytes"])

    # Unpack uint32 -> 8 x 4-bit nibbles
    unpacked = np.zeros(len(raw_w) * 8, dtype=np.uint8)
    for bit in range(8):
        unpacked[bit::8] = ((raw_w >> (bit * 4)) & 0xF).astype(np.uint8)

    # Dequantize: w_float = uint4 * scale + bias (group_size=64)
    orig_group_size = len(unpacked) // len(scales)
    groups = unpacked.reshape(len(scales), orig_group_size).astype(np.float32)
    dequant = groups * scales[:, None] + biases[:, None]

    return dequant.flatten()


def quantize_to_1bit(float_weights):
    """1-bit quantization: store sign + mean(abs) per group of 128."""
    flat = float_weights.astype(np.float32)
    # Pad to multiple of GROUP_SIZE
    remainder = len(flat) % GROUP_SIZE
    if remainder:
        flat = np.pad(flat, (0, GROUP_SIZE - remainder))

    n_groups = len(flat) // GROUP_SIZE
    groups = flat.reshape(n_groups, GROUP_SIZE)

    scales = np.mean(np.abs(groups), axis=1).astype(np.float16)
    sign_bits = (groups >= 0).astype(np.uint8)
    packed = np.packbits(sign_bits, axis=1)  # (n_groups, GROUP_SIZE//8)

    return scales, packed.flatten()


def calc_1bit_expert_size():
    """Calculate bytes per expert in the 1-bit format."""
    total = 0
    # gate_proj and up_proj: 512*2048 = 1,048,576 values each
    # down_proj: 2048*512 = 1,048,576 values
    for n_values in [1048576, 1048576, 1048576]:
        padded = n_values + (-n_values % GROUP_SIZE)
        n_groups = padded // GROUP_SIZE
        total += n_groups * 2  # fp16 scales
        total += n_groups * (GROUP_SIZE // 8)  # packed bits
    return total


def main():
    layer_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".bin"))
    n_layers = len(layer_files)
    print(f"Found {n_layers} layer files")

    # Parse first header to get layout
    header = parse_header(os.path.join(INPUT_DIR, layer_files[0]))
    layout = header["layout"]
    n_experts = layout["num_experts"]
    expert_block_size = layout["expert_block_size"]
    data_start = layout["data_start"]
    tensors = layout["tensors"]

    expert_1bit_size = calc_1bit_expert_size()
    total_size = n_layers * n_experts * expert_1bit_size
    header_size = 16384  # page-aligned

    print(f"Experts: {n_experts} per layer, {n_layers} layers")
    print(f"1-bit expert size: {expert_1bit_size:,} bytes ({expert_1bit_size/1024:.1f} KB)")
    print(f"Total data: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    print(f"Total file: {(header_size + total_size):,} bytes ({(header_size + total_size)/1024/1024:.1f} MB)")
    print(f"Output: {OUTPUT_PATH}")
    print()

    # Write output header
    out_header = {
        "format": "expert_fallback_1bit_v1",
        "group_size": GROUP_SIZE,
        "num_layers": n_layers,
        "num_experts": n_experts,
        "expert_1bit_size": expert_1bit_size,
        "data_start": header_size,
        "projections": ["gate_proj", "up_proj", "down_proj"],
        "values_per_projection": 1048576,
        "source_format": "moe_flash_v1",
        "source_quantization": "4bit_group64",
        "dequant_formula": "uint4 * scale + bias",
        "1bit_formula": "sign * scale (scale = mean_abs per group)",
        "reconstruction": "weight = scale * (2*bit - 1)",
    }

    out_header_bytes = json.dumps(out_header, indent=2).encode("utf-8")
    if len(out_header_bytes) > header_size:
        raise ValueError(f"Header too large: {len(out_header_bytes)} > {header_size}")

    # Pad header to page boundary
    out_header_padded = out_header_bytes + b"\x00" * (header_size - len(out_header_bytes))

    with open(OUTPUT_PATH, "wb") as out_f:
        out_f.write(out_header_padded)

    t_start = time.time()

    for layer_idx, layer_file in enumerate(layer_files):
        t_layer = time.time()
        layer_path = os.path.join(INPUT_DIR, layer_file)

        # Parse this layer's header (in case layout varies)
        lh = parse_header(layer_path)
        ltensors = lh["layout"]["tensors"]
        ldata_start = lh["layout"]["data_start"]
        lexpert_size = lh["layout"]["expert_block_size"]

        layer_buffer = bytearray()

        with open(layer_path, "rb") as f:
            for expert_idx in range(n_experts):
                expert_offset = ldata_start + expert_idx * lexpert_size

                expert_bytes = bytearray()
                for proj_name, proj_prefix in PROJECTIONS:
                    float_w = dequantize_projection(f, expert_offset, ltensors, proj_prefix)
                    scales, packed = quantize_to_1bit(float_w)
                    expert_bytes.extend(scales.tobytes())
                    expert_bytes.extend(packed.tobytes())

                assert len(expert_bytes) == expert_1bit_size, \
                    f"Expert size mismatch: {len(expert_bytes)} != {expert_1bit_size}"
                layer_buffer.extend(expert_bytes)

        # Write entire layer at once
        with open(OUTPUT_PATH, "ab") as out_f:
            out_f.write(layer_buffer)

        elapsed = time.time() - t_layer
        total_elapsed = time.time() - t_start
        written = header_size + (layer_idx + 1) * n_experts * expert_1bit_size
        print(f"  Layer {layer_idx:2d}/{n_layers} done in {elapsed:.1f}s | "
              f"Written: {written/1024/1024:.0f} MB | "
              f"Elapsed: {total_elapsed:.0f}s")

        # Free memory
        del layer_buffer

    total_time = time.time() - t_start
    final_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nDone! {final_size:,} bytes ({final_size/1024/1024:.1f} MB) in {total_time:.0f}s")
    print(f"Throughput: {n_layers * n_experts / total_time:.1f} experts/sec")


if __name__ == "__main__":
    main()
