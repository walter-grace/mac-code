"""
Flash Stream Batched — The Speed Breakthrough.

Instead of eval after every layer (128 evals/token = 7.5s overhead),
batch 8 FFN layers at a time (16 evals/token).

Memory: 4.28 GB pinned + 1.77 GB batch buffer (~6 GB total, fits in 16 GB)
Expected: ~2-3 tok/s at full 4-bit quality.
"""

import time
import json
import gc
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

MODEL_DIR = "/Users/bigneek/models/qwen3-32b-flash-stream"
BITS = 4
GROUP_SIZE = 64
BATCH_SIZE = 8  # Process 8 layers between evals


class BatchedFFNStreamer:
    """Pre-loads batches of FFN layers for reduced eval overhead."""

    def __init__(self, ffn_dir, num_layers, batch_size=BATCH_SIZE):
        self.ffn_dir = ffn_dir
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefetch_future = None
        self.prefetch_batch_idx = -1
        self.read_time = 0.0
        self.reads = 0

    def _read_batch(self, start_idx):
        """Read a batch of FFN layers from SSD."""
        t0 = time.time()
        batch = {}
        end_idx = min(start_idx + self.batch_size, self.num_layers)
        for i in range(start_idx, end_idx):
            batch[i] = mx.load(f"{self.ffn_dir}/layer_{i:02d}.safetensors")
            self.reads += 1
        self.read_time += time.time() - t0
        return batch

    def prefetch_batch(self, start_idx):
        if self.prefetch_future is not None and self.prefetch_batch_idx == start_idx:
            return
        self.prefetch_future = self.executor.submit(self._read_batch, start_idx)
        self.prefetch_batch_idx = start_idx

    def get_batch(self, start_idx):
        if self.prefetch_future is not None and self.prefetch_batch_idx == start_idx:
            data = self.prefetch_future.result()
            self.prefetch_future = None
            return data
        return self._read_batch(start_idx)

    def stats(self):
        avg = (self.read_time / self.reads * 1000) if self.reads > 0 else 0
        return f"FFN reads: {self.reads}, avg: {avg:.1f}ms, total IO: {self.read_time:.1f}s"


def run_ffn(x, ffn_data, group_size=GROUP_SIZE, bits=BITS):
    """SwiGLU FFN with quantized matmul."""
    gate = mx.quantized_matmul(
        x, ffn_data["mlp.gate_proj.weight"],
        scales=ffn_data["mlp.gate_proj.scales"],
        biases=ffn_data["mlp.gate_proj.biases"],
        transpose=True, group_size=group_size, bits=bits,
    )
    up = mx.quantized_matmul(
        x, ffn_data["mlp.up_proj.weight"],
        scales=ffn_data["mlp.up_proj.scales"],
        biases=ffn_data["mlp.up_proj.biases"],
        transpose=True, group_size=group_size, bits=bits,
    )
    hidden = nn.silu(gate) * up
    out = mx.quantized_matmul(
        hidden, ffn_data["mlp.down_proj.weight"],
        scales=ffn_data["mlp.down_proj.scales"],
        biases=ffn_data["mlp.down_proj.biases"],
        transpose=True, group_size=group_size, bits=bits,
    )
    return out


def main():
    print("=" * 60)
    print("  FLASH STREAM BATCHED — Speed Breakthrough")
    print(f"  Batch size: {BATCH_SIZE} layers per eval")
    print(f"  Expected evals/token: {(64 // BATCH_SIZE) * 2}")
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

    mx.set_memory_limit(12 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)

    # Load pinned weights
    print("\nLoading pinned weights...")
    t0 = time.time()
    pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")
    model.load_weights(list(pinned.items()), strict=False)
    params = [p for name, p in tree_flatten(model.parameters()) if "mlp" not in name]
    mx.eval(*params)
    del pinned
    gc.collect()
    mx.clear_cache()

    pinned_gb = sum(p.nbytes for p in params) / 1e9
    print(f"  {pinned_gb:.2f} GB pinned in {time.time()-t0:.1f}s")
    print(f"  Active memory: {mx.get_active_memory()/1e9:.2f} GB")

    streamer = BatchedFFNStreamer(
        f"{MODEL_DIR}/{config['streaming']['ffn_dir']}",
        num_layers, BATCH_SIZE,
    )

    # Tokenizer
    from transformers import AutoTokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.base import create_attention_mask

    def forward_batched(input_ids, cache):
        """Forward pass with batched FFN loading."""
        h = model.model.embed_tokens(input_ids)
        mask = create_attention_mask(h, cache[0])

        # Pre-read first batch
        streamer.prefetch_batch(0)

        for batch_start in range(0, num_layers, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, num_layers)

            # Prefetch NEXT batch while processing current
            next_batch = batch_start + BATCH_SIZE
            if next_batch < num_layers:
                streamer.prefetch_batch(next_batch)

            # Load current batch of FFN layers
            ffn_batch = streamer.get_batch(batch_start)

            # Process all layers in this batch
            for i in range(batch_start, batch_end):
                layer = model.model.layers[i]

                # Attention (from RAM)
                normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(normed, mask=mask, cache=cache[i])
                h = h + attn_out

                # FFN (from batch buffer)
                normed = layer.post_attention_layernorm(h)
                ffn_out = run_ffn(normed, ffn_batch[i])
                h = h + ffn_out

            # ONE eval for the entire batch — this is the key optimization
            mx.eval(h)

            # Discard batch
            del ffn_batch
            mx.clear_cache()

        h = model.model.norm(h)
        return model.lm_head(h)

    # Test prompt
    prompt = "The key insight from the LLM in a Flash paper is that"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])
    cache = make_prompt_cache(model)

    # Prefill
    print(f"\n--- Prefill ({len(tokens)} tokens) ---")
    t0 = time.time()
    logits = forward_batched(input_ids, cache)
    mx.eval(logits)
    t_pf = time.time() - t0
    print(f"  {t_pf:.2f}s ({len(tokens)/t_pf:.1f} tok/s)")
    print(f"  {streamer.stats()}")
    print(f"  Active memory: {mx.get_active_memory()/1e9:.2f} GB")

    # Decode
    temperature = 0.7
    rep_penalty = 1.2
    max_tokens = 50

    print(f"\n--- Decode (max {max_tokens}, temp={temperature}) ---")
    generated = []
    t_decode = time.time()

    for step in range(max_tokens):
        next_logits = logits[:, -1, :]

        # Repetition penalty
        if generated:
            seen = mx.array(list(set(generated[-50:])))
            pl = next_logits[:, seen]
            pl = mx.where(pl > 0, pl / rep_penalty, pl * rep_penalty)
            next_logits[:, seen] = pl

        # Temperature sampling
        probs = mx.softmax(next_logits / temperature, axis=-1)
        token = mx.random.categorical(mx.log(probs + 1e-10))
        mx.eval(token)
        token_id = token.item()

        if token_id in (151645, 151643):
            break

        generated.append(token_id)
        print(tokenizer.decode([token_id]), end="", flush=True)

        logits = forward_batched(token.reshape(1, 1), cache)
        mx.eval(logits)

        if (step + 1) % 10 == 0:
            elapsed = time.time() - t_decode
            tps = (step + 1) / elapsed
            mem = mx.get_active_memory() / 1e9
            print(f" [{tps:.2f} tok/s, {mem:.1f}GB]", flush=True)

    t_total = time.time() - t_decode
    n = len(generated)
    tps = n / t_total if t_total > 0 else 0

    output = tokenizer.decode(generated)
    print(f"\n\nDecode: {n} tokens in {t_total:.1f}s ({tps:.2f} tok/s)")
    print(f"{streamer.stats()}")
    print(f"Active memory: {mx.get_active_memory()/1e9:.2f} GB")

    import subprocess
    r = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
    print(f"{r.stdout.strip()}")

    print(f"\n{'='*60}")
    print(f"{prompt}{output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
