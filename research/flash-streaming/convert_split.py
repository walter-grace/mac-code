"""
Split GGUF into streaming-optimized layout:
  pinned.safetensors  — attention + embed + norms (~4.3 GB, always in RAM)
  ffn/layer_XX.safetensors — per-layer FFN (~238 MB each, streamed from SSD)

All weights at full 4-bit gs=64 quality. No quality compromise.
"""

import time
import gc
import json
import os
import numpy as np
import mlx.core as mx

from sys import path as sys_path
sys_path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dequant_gguf import dequantize as dequant_gguf_tensor
from gguf import GGUFReader

GGUF_PATH = "/Users/bigneek/models/Qwen3-32B-Q4_K_M.gguf"
OUTPUT_DIR = "/Users/bigneek/models/qwen3-32b-flash-stream"
BITS = 4
GROUP_SIZE = 64

GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}
LAYER_MAP = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

FFN_KEYS = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}


def gguf_name_to_mlx(name):
    if name in GLOBAL_MAP:
        return GLOBAL_MAP[name]
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        layer_idx = parts[1]
        tensor_key = ".".join(parts[2:])
        if tensor_key in LAYER_MAP:
            return f"model.layers.{layer_idx}.{LAYER_MAP[tensor_key]}"
    return None


def is_ffn(mlx_name):
    return any(k in mlx_name for k in FFN_KEYS)


def is_linear(mlx_name):
    linear_parts = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}
    return any(p in mlx_name for p in linear_parts)


def convert_tensor(t):
    """Dequant GGUF tensor → MLX 4-bit quantized arrays."""
    shape = tuple(t.shape)
    n_elements = int(np.prod(shape))
    ggml_type = int(t.tensor_type)
    raw = t.data

    if ggml_type == 0:
        flat = np.frombuffer(raw.reshape(-1).tobytes(), dtype=np.float32, count=n_elements)
    elif ggml_type == 1:
        flat = np.frombuffer(raw.reshape(-1).tobytes(), dtype=np.float16, count=n_elements)
    else:
        flat = dequant_gguf_tensor(raw, ggml_type, n_elements)

    return flat, shape


def main():
    os.makedirs(f"{OUTPUT_DIR}/ffn", exist_ok=True)

    print("=" * 60)
    print("  Converting GGUF → Split Streaming Layout")
    print(f"  Quality: {BITS}-bit gs={GROUP_SIZE} (FULL quality)")
    print("=" * 60)

    reader = GGUFReader(GGUF_PATH)
    print(f"\n  {len(reader.tensors)} tensors")

    # Collect tensors by destination
    pinned_weights = {}  # attention + embed + norms
    ffn_layers = {}      # layer_idx -> {name: array}

    t0 = time.time()

    for idx, t in enumerate(reader.tensors):
        mlx_name = gguf_name_to_mlx(t.name)
        if mlx_name is None:
            continue

        flat, shape = convert_tensor(t)

        if is_ffn(mlx_name):
            # Extract layer index
            parts = mlx_name.split(".")
            layer_idx = int(parts[2])  # model.layers.XX.mlp...
            local_name = ".".join(parts[3:])  # mlp.gate_proj.weight

            if layer_idx not in ffn_layers:
                ffn_layers[layer_idx] = {}

            # Quantize: col-major reshape → [out, in] → mx.quantize
            weight_f16 = mx.array(flat.astype(np.float16).reshape(shape[1], shape[0]))
            del flat
            qw, scales, biases = mx.quantize(weight_f16, group_size=GROUP_SIZE, bits=BITS)
            mx.eval(qw, scales, biases)
            del weight_f16

            base = local_name.replace(".weight", "")
            ffn_layers[layer_idx][f"{base}.weight"] = qw
            ffn_layers[layer_idx][f"{base}.scales"] = scales
            ffn_layers[layer_idx][f"{base}.biases"] = biases

        elif is_linear(mlx_name):
            # Attention or lm_head → pinned
            weight_f16 = mx.array(flat.astype(np.float16).reshape(shape[1], shape[0]))
            del flat
            qw, scales, biases = mx.quantize(weight_f16, group_size=GROUP_SIZE, bits=BITS)
            mx.eval(qw, scales, biases)
            del weight_f16

            base = mlx_name.replace(".weight", "")
            pinned_weights[f"{base}.weight"] = qw
            pinned_weights[f"{base}.scales"] = scales
            pinned_weights[f"{base}.biases"] = biases

        elif "embed_tokens" in mlx_name:
            weight_f16 = mx.array(flat.astype(np.float16).reshape(shape[1], shape[0]))
            del flat
            qw, scales, biases = mx.quantize(weight_f16, group_size=GROUP_SIZE, bits=BITS)
            mx.eval(qw, scales, biases)
            del weight_f16

            base = mlx_name.replace(".weight", "")
            pinned_weights[f"{base}.weight"] = qw
            pinned_weights[f"{base}.scales"] = scales
            pinned_weights[f"{base}.biases"] = biases
        else:
            # Norms → pinned as float32
            pinned_weights[mlx_name] = mx.array(flat.astype(np.float32).reshape(shape))
            mx.eval(pinned_weights[mlx_name])

        gc.collect()

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(reader.tensors)}] {time.time()-t0:.0f}s")

    del reader
    gc.collect()

    # Save pinned weights
    print(f"\n  Saving pinned weights ({len(pinned_weights)} arrays)...")
    mx.save_safetensors(f"{OUTPUT_DIR}/pinned.safetensors", pinned_weights)
    pinned_bytes = sum(v.nbytes for v in pinned_weights.values())
    print(f"    {pinned_bytes/1e9:.2f} GB")
    del pinned_weights
    gc.collect()

    # Save per-layer FFN
    print(f"  Saving {len(ffn_layers)} FFN layer files...")
    total_ffn = 0
    for layer_idx in sorted(ffn_layers.keys()):
        fname = f"{OUTPUT_DIR}/ffn/layer_{layer_idx:02d}.safetensors"
        mx.save_safetensors(fname, ffn_layers[layer_idx])
        layer_bytes = sum(v.nbytes for v in ffn_layers[layer_idx].values())
        total_ffn += layer_bytes
        if layer_idx % 16 == 0:
            print(f"    Layer {layer_idx}: {layer_bytes/1e6:.1f} MB")

    print(f"    Total FFN: {total_ffn/1e9:.2f} GB")

    # Save config
    config = {
        "model_type": "qwen3",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "intermediate_size": 25600,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rope_theta": 1000000.0,
        "head_dim": 128,
        "tie_word_embeddings": False,
        "quantization": {"bits": BITS, "group_size": GROUP_SIZE},
        "streaming": {
            "pinned_file": "pinned.safetensors",
            "ffn_dir": "ffn/",
            "ffn_layers": 64,
        }
    }
    with open(f"{OUTPUT_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  Done in {time.time()-t0:.0f}s!")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Pinned (RAM):  {pinned_bytes/1e9:.2f} GB")
    print(f"  FFN (SSD):     {total_ffn/1e9:.2f} GB")
    print(f"  Total:         {(pinned_bytes+total_ffn)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
