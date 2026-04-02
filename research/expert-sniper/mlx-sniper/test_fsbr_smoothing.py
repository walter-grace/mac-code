#!/usr/bin/env python3
"""Test FSBR smoothing: per-channel scaling to suppress activation spikes before ternary quantization."""
import os, sys, json, numpy as np

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

EXPERT_DIR = os.path.expanduser("~/models/qwen35-35b-moe-stream/experts/")
GROUP_SIZE_TERNARY = 128
THRESHOLD = 0.5
HIDDEN = 2048
MOE_DIM = 512
N_CALIB = 32

def parse_header(path):
    with open(path, "rb") as f:
        raw = f.read(16384)
    depth = 0
    for i, b in enumerate(raw):
        if b == ord("{"): depth += 1
        elif b == ord("}"):
            depth -= 1
            if depth == 0: return json.loads(raw[:i+1])

def read_bf16_as_f32(f, nbytes):
    u16 = np.frombuffer(f.read(nbytes), dtype=np.uint16)
    f32 = np.zeros(len(u16), dtype=np.float32)
    f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
    return f32

def dequant_4bit(layer_path, expert_offset, tensors, proj):
    prefix = f"mlp.switch_mlp.{proj}"
    w_info = tensors[f"{prefix}.weight"]
    s_info = tensors[f"{prefix}.scales"]
    b_info = tensors[f"{prefix}.biases"]
    with open(layer_path, "rb") as f:
        f.seek(expert_offset + w_info["inner_offset"])
        raw_w = np.frombuffer(f.read(w_info["nbytes"]), dtype=np.uint32)
        f.seek(expert_offset + s_info["inner_offset"])
        scales = read_bf16_as_f32(f, s_info["nbytes"])
        f.seek(expert_offset + b_info["inner_offset"])
        biases = read_bf16_as_f32(f, b_info["nbytes"])
    unpacked = np.zeros(len(raw_w) * 8, dtype=np.uint8)
    for bit in range(8):
        unpacked[bit::8] = ((raw_w >> (bit * 4)) & 0xF).astype(np.uint8)
    gs = len(unpacked) // len(scales)
    groups = unpacked.reshape(len(scales), gs).astype(np.float32)
    return (groups * scales[:, None] + biases[:, None]).flatten()

def to_ternary(float_weights, shape):
    flat = float_weights.astype(np.float32)
    r = len(flat) % GROUP_SIZE_TERNARY
    if r: flat = np.pad(flat, (0, GROUP_SIZE_TERNARY - r))
    ng = len(flat) // GROUP_SIZE_TERNARY
    groups = flat.reshape(ng, GROUP_SIZE_TERNARY)
    scales = np.mean(np.abs(groups), axis=1)
    safe_s = np.where(scales > 0, scales, 1e-7)
    normalized = groups / safe_s[:, None]
    recon = np.zeros_like(normalized)
    recon[normalized >= THRESHOLD] = 1.0
    recon[normalized <= -THRESHOLD] = -1.0
    return (recon * scales[:, None]).flatten()[:shape[0]*shape[1]].reshape(shape)

def silu(x):
    return x / (1.0 + np.exp(-x))

def cos(a, b):
    d = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return d / (na * nb) if na > 0 and nb > 0 else 0.0

layer_files = sorted(f for f in os.listdir(EXPERT_DIR) if f.endswith('.bin'))
np.random.seed(42)

test_cases = []
for _ in range(10):
    li = np.random.randint(5, min(21, len(layer_files)))
    ei = np.random.randint(0, 200)
    test_cases.append((li, ei))

cos_baseline, cos_smoothed = [], []

for li, ei in test_cases:
    layer_path = os.path.join(EXPERT_DIR, layer_files[li])
    hdr = parse_header(layer_path)
    lt = hdr["layout"]["tensors"]
    ds = hdr["layout"]["data_start"]
    ebs = hdr["layout"]["expert_block_size"]
    n_full = (os.path.getsize(layer_path) - ds) // ebs
    if ei >= n_full:
        continue

    eo = ds + ei * ebs
    gate_f = dequant_4bit(layer_path, eo, lt, "gate_proj").reshape(MOE_DIM, HIDDEN)
    up_f = dequant_4bit(layer_path, eo, lt, "up_proj").reshape(MOE_DIM, HIDDEN)
    down_f = dequant_4bit(layer_path, eo, lt, "down_proj").reshape(HIDDEN, MOE_DIM)

    # Calibration: 32 random activations
    X = np.random.randn(N_CALIB, HIDDEN).astype(np.float32) * 0.01
    gate_out = silu(gate_f @ X.T)  # [MOE_DIM, N_CALIB]
    up_out = up_f @ X.T            # [MOE_DIM, N_CALIB]
    intermediate = gate_out * up_out  # [MOE_DIM, N_CALIB]

    # Per-channel smoothing factor (over MOE_DIM channels)
    S = intermediate.std(axis=1)  # [MOE_DIM]
    S = S / (S.mean() + 1e-10)
    S = S.clip(0.1, 10.0)

    # Smoothed down weights: absorb S into down_proj columns
    # down @ intermediate = (down * S[None,:]) @ (intermediate / S[:,None])
    down_smooth = down_f * S[None, :]  # [HIDDEN, MOE_DIM] * [MOE_DIM] broadcast

    # Ternary quantize both
    down_t_baseline = to_ternary(down_f.flatten(), (HIDDEN, MOE_DIM))
    down_t_smoothed = to_ternary(down_smooth.flatten(), (HIDDEN, MOE_DIM))

    # Test on fresh activations
    x_test = np.random.randn(HIDDEN).astype(np.float32) * 0.01
    ref = down_f @ (silu(gate_f @ x_test) * (up_f @ x_test))

    inter_test = silu(gate_f @ x_test) * (up_f @ x_test)

    # Baseline ternary
    out_base = down_t_baseline @ inter_test

    # Smoothed ternary: must divide intermediate by S before matmul
    out_smooth = down_t_smoothed @ (inter_test / S)

    cb = cos(ref, out_base)
    cs = cos(ref, out_smooth)
    cos_baseline.append(cb)
    cos_smoothed.append(cs)
    print(f"  L{li:02d} E{ei:03d}: baseline={cb:.4f}  smoothed={cs:.4f}  delta={cs-cb:+.4f}")

print(f"\n{'='*50}")
print(f"  Baseline ternary:  {np.mean(cos_baseline):.4f}")
print(f"  FSBR smoothed:     {np.mean(cos_smoothed):.4f}")
print(f"  Delta:             {np.mean(cos_smoothed)-np.mean(cos_baseline):+.4f}")
print(f"{'='*50}")
if np.mean(cos_smoothed) > np.mean(cos_baseline) + 0.005:
    print(f"  FSBR helps! But requires runtime division by S — NOT zero cost.")
else:
    print(f"  FSBR does not help. Skip it.")
