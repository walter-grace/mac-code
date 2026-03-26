"""
Flash Stream v2 — Direct I/O, No mmap, No UBC.

Bypasses macOS page fault handler entirely.
Reads FFN weights via F_NOCACHE + pread into aligned buffers.

v1: 0.12 tok/s (mmap page faults)
v2 target: 0.35 tok/s (direct I/O, SSD bandwidth bound)
"""

import time
import json
import gc

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from direct_io import DirectFFNReader

MODEL_DIR = "/Users/bigneek/models/qwen3-32b-flash-stream"
ALIGNED_DIR = f"{MODEL_DIR}/ffn_aligned"
BITS = 4
GROUP_SIZE = 64


def run_ffn(x, ffn_data):
    gate = mx.quantized_matmul(
        x, ffn_data["mlp.gate_proj.weight"],
        scales=ffn_data["mlp.gate_proj.scales"],
        biases=ffn_data["mlp.gate_proj.biases"],
        transpose=True, group_size=GROUP_SIZE, bits=BITS,
    )
    up = mx.quantized_matmul(
        x, ffn_data["mlp.up_proj.weight"],
        scales=ffn_data["mlp.up_proj.scales"],
        biases=ffn_data["mlp.up_proj.biases"],
        transpose=True, group_size=GROUP_SIZE, bits=BITS,
    )
    hidden = nn.silu(gate) * up
    out = mx.quantized_matmul(
        hidden, ffn_data["mlp.down_proj.weight"],
        scales=ffn_data["mlp.down_proj.scales"],
        biases=ffn_data["mlp.down_proj.biases"],
        transpose=True, group_size=GROUP_SIZE, bits=BITS,
    )
    return out


def main():
    print("=" * 60)
    print("  FLASH STREAM v2 — Direct I/O")
    print("  F_NOCACHE + pread. No mmap. No UBC. No page faults.")
    print("=" * 60)

    with open(f"{MODEL_DIR}/config.json") as f:
        config = json.load(f)

    num_layers = config["num_hidden_layers"]

    from mlx_lm.models.qwen3 import Model, ModelArgs
    args = ModelArgs(
        model_type=config["model_type"],
        hidden_size=config["hidden_size"],
        num_hidden_layers=num_layers,
        intermediate_size=config["intermediate_size"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        rms_norm_eps=config["rms_norm_eps"],
        vocab_size=config["vocab_size"],
        max_position_embeddings=config["max_position_embeddings"],
        rope_theta=config["rope_theta"],
        head_dim=config["head_dim"],
        tie_word_embeddings=config["tie_word_embeddings"],
    )

    model = Model(args)
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)

    mx.set_memory_limit(10 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)

    # Load pinned weights (attention + embed)
    print("\nLoading pinned weights...")
    t0 = time.time()
    pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")
    model.load_weights(list(pinned.items()), strict=False)
    params = [p for name, p in tree_flatten(model.parameters()) if "mlp" not in name]
    mx.eval(*params)
    del pinned
    gc.collect()
    mx.clear_cache()
    print(f"  {sum(p.nbytes for p in params)/1e9:.2f} GB in {time.time()-t0:.1f}s")

    # Initialize direct I/O reader
    print("\nInitializing direct I/O reader (F_NOCACHE)...")
    reader = DirectFFNReader(ALIGNED_DIR, num_layers)
    print(f"  {num_layers} layer FDs opened with F_NOCACHE")

    # Warm up: read one layer to verify
    print("  Verifying read...")
    t0 = time.time()
    test = reader.get(0)
    print(f"  Layer 0 read: {time.time()-t0:.3f}s, keys: {list(test.keys())}")
    del test

    # Tokenizer
    from transformers import AutoTokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.base import create_attention_mask

    def forward_direct(input_ids, cache):
        h = model.model.embed_tokens(input_ids)
        mask = create_attention_mask(h, cache[0])

        reader.prefetch(0)

        for i in range(num_layers):
            layer = model.model.layers[i]
            if i + 1 < num_layers:
                reader.prefetch(i + 1)

            # Attention (RAM)
            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed, mask=mask, cache=cache[i])
            h = h + attn_out
            mx.eval(h)

            # FFN (direct I/O from SSD)
            ffn_data = reader.get(i)
            normed = layer.post_attention_layernorm(h)
            ffn_out = run_ffn(normed, ffn_data)
            h = h + ffn_out
            mx.eval(h)

            del ffn_data, ffn_out, normed, attn_out
            mx.clear_cache()

        h = model.model.norm(h)
        return model.lm_head(h)

    # Test
    prompt = "The key insight from the LLM in a Flash paper is that"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    cache = make_prompt_cache(model)

    # Purge OS caches for clean measurement
    import subprocess
    subprocess.run(["sudo", "purge"], capture_output=True)
    print("\n  OS caches purged for clean measurement")

    # Prefill
    print(f"\n--- Prefill ({len(tokens)} tokens) ---")
    t0 = time.time()
    logits = forward_direct(input_ids, cache)
    mx.eval(logits)
    t_pf = time.time() - t0
    print(f"  {t_pf:.2f}s ({len(tokens)/t_pf:.2f} tok/s)")
    print(f"  IO: {reader.stats()}")
    print(f"  Memory: {mx.get_active_memory()/1e9:.2f} GB")

    # Decode
    temperature = 0.7
    rep_penalty = 1.2
    max_tokens = 30

    print(f"\n--- Decode (max {max_tokens}, temp={temperature}) ---")
    generated = []
    t_decode = time.time()

    for step in range(max_tokens):
        next_logits = logits[:, -1, :]

        if generated:
            seen = mx.array(list(set(generated[-50:])))
            pl = next_logits[:, seen]
            pl = mx.where(pl > 0, pl / rep_penalty, pl * rep_penalty)
            next_logits[:, seen] = pl

        probs = mx.softmax(next_logits / temperature, axis=-1)
        token = mx.random.categorical(mx.log(probs + 1e-10))
        mx.eval(token)
        token_id = token.item()

        if token_id in (151645, 151643):
            break

        generated.append(token_id)
        print(tokenizer.decode([token_id]), end="", flush=True)

        t_step = time.time()
        logits = forward_direct(token.reshape(1, 1), cache)
        mx.eval(logits)
        step_time = time.time() - t_step

        if (step + 1) % 5 == 0:
            elapsed = time.time() - t_decode
            tps = (step + 1) / elapsed
            mem = mx.get_active_memory() / 1e9
            print(f" [{tps:.3f} tok/s, {mem:.1f}GB, last={step_time:.1f}s]", flush=True)

    t_total = time.time() - t_decode
    n = len(generated)
    tps = n / t_total if t_total > 0 else 0

    output = tokenizer.decode(generated)
    print(f"\n\nDecode: {n} tokens in {t_total:.1f}s ({tps:.3f} tok/s)")
    print(f"IO: {reader.stats()}")
    print(f"Memory: {mx.get_active_memory()/1e9:.2f} GB")

    import subprocess
    r = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
    print(f"{r.stdout.strip()}")

    print(f"\n{'='*60}")
    print(f"{prompt}{output}")
    print(f"{'='*60}")

    # Compare
    print(f"\n  v1 (mmap):      0.12 tok/s")
    print(f"  v2 (direct IO): {tps:.3f} tok/s")
    if tps > 0.12:
        print(f"  Speedup:        {tps/0.12:.1f}x")

    reader.close()


if __name__ == "__main__":
    main()
