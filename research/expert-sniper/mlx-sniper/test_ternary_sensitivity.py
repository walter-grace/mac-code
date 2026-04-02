#!/usr/bin/env python3
"""Test ternary sensitivity per projection through full SwiGLU."""
import os, sys, json, numpy as np

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

EXPERT_DIR = os.path.expanduser("~/models/qwen35-35b-moe-stream/experts/")
GROUP_SIZE_4BIT = 64
GROUP_SIZE_TERNARY = 128
THRESHOLD = 0.5
HIDDEN = 2048
MOE_DIM = 512

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
    weights = (recon * scales[:, None]).flatten()[:shape[0]*shape[1]]
    return weights.reshape(shape)

def swiglu(x, gate_w, up_w, down_w):
    g = gate_w @ x
    u = up_w @ x
    h = (1.0 / (1.0 + np.exp(-g))) * g * u  # silu(g) * u
    return down_w @ h

def silu(x):
    return x / (1.0 + np.exp(-x))

def swiglu_proper(x, gate_w, up_w, down_w):
    g = gate_w @ x
    u = up_w @ x
    h = silu(g) * u
    return down_w @ h

layer_files = sorted(f for f in os.listdir(EXPERT_DIR) if f.endswith('.bin'))
np.random.seed(42)

# Pick 10 random (layer, expert) pairs from layers 5-20
test_cases = []
for _ in range(10):
    li = np.random.randint(5, min(21, len(layer_files)))
    ei = np.random.randint(0, 200)  # stay in range
    test_cases.append((li, ei))

cosines_a, cosines_b, cosines_c = [], [], []

for li, ei in test_cases:
    layer_path = os.path.join(EXPERT_DIR, layer_files[li])
    hdr = parse_header(layer_path)
    lt = hdr["layout"]["tensors"]
    ds = hdr["layout"]["data_start"]
    ebs = hdr["layout"]["expert_block_size"]

    file_size = os.path.getsize(layer_path)
    n_full = (file_size - ds) // ebs
    if ei >= n_full:
        continue

    eo = ds + ei * ebs

    # Full 4-bit dequant
    gate_f = dequant_4bit(layer_path, eo, lt, "gate_proj").reshape(MOE_DIM, HIDDEN)
    up_f = dequant_4bit(layer_path, eo, lt, "up_proj").reshape(MOE_DIM, HIDDEN)
    down_f = dequant_4bit(layer_path, eo, lt, "down_proj").reshape(HIDDEN, MOE_DIM)

    # Ternary versions
    gate_t = to_ternary(gate_f.flatten(), (MOE_DIM, HIDDEN))
    up_t = to_ternary(up_f.flatten(), (MOE_DIM, HIDDEN))
    down_t = to_ternary(down_f.flatten(), (HIDDEN, MOE_DIM))

    x = np.random.randn(HIDDEN).astype(np.float32) * 0.01
    ref = swiglu_proper(x, gate_f, up_f, down_f)

    # Config A: gate=ternary
    out_a = swiglu_proper(x, gate_t, up_f, down_f)
    # Config B: up=ternary
    out_b = swiglu_proper(x, gate_f, up_t, down_f)
    # Config C: down=ternary (current)
    out_c = swiglu_proper(x, gate_f, up_f, down_t)

    def cos(a, b):
        d = np.dot(a, b)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return d / (na * nb) if na > 0 and nb > 0 else 0.0

    ca, cb, cc = cos(ref, out_a), cos(ref, out_b), cos(ref, out_c)
    cosines_a.append(ca)
    cosines_b.append(cb)
    cosines_c.append(cc)
    print(f"  L{li:02d} E{ei:03d}: gate={ca:.4f}  up={cb:.4f}  down={cc:.4f}")

print(f"\n{'='*50}")
print(f"  Config A (gate=ternary): {np.mean(cosines_a):.4f}")
print(f"  Config B (up=ternary):   {np.mean(cosines_b):.4f}")
print(f"  Config C (down=ternary): {np.mean(cosines_c):.4f}")
print(f"{'='*50}")
best = ['A (gate)', 'B (up)', 'C (down)'][np.argmax([np.mean(cosines_a), np.mean(cosines_b), np.mean(cosines_c)])]
print(f"  Winner: {best}")
