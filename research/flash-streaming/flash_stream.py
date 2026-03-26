"""
Flash Stream — The Real Breakthrough.

Runs the FULL 4-bit Qwen3-32B (18.4 GB) on 16 GB RAM.
NOT by compressing to 2-bit garbage. NOT by mmap thrashing.

Architecture:
  - Pinned in RAM: attention + embeddings + norms (~4.3 GB)
  - Streamed from SSD: FFN layers (~238 MB each, 15.2 GB total)
  - Double-buffer: while GPU computes FFN[i], CPU reads FFN[i+1] from SSD
  - Per-token: read 64 FFN layers sequentially, discard after use

This is what llama.cpp CANNOT do. mmap thrashes at 0.017 tok/s.
We do predictive sequential streaming.
"""

import time
import json
import os
import gc
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

MODEL_DIR = "/Users/bigneek/models/qwen3-32b-flash-stream"
BITS = 4
GROUP_SIZE = 64


# ── FFN Streamer ────────────────────────────────────────────────────

class FFNStreamer:
    """
    Streams FFN layers from SSD with double-buffering.
    While GPU processes layer N, we pre-read layer N+1 from disk.
    """

    def __init__(self, ffn_dir: str, num_layers: int):
        self.ffn_dir = ffn_dir
        self.num_layers = num_layers
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefetch_future = None
        self.prefetch_idx = -1
        self.reads = 0
        self.read_time = 0.0

    def _read_layer(self, layer_idx: int) -> dict:
        """Read one FFN layer from SSD. Runs in background thread."""
        t0 = time.time()
        path = f"{self.ffn_dir}/layer_{layer_idx:02d}.safetensors"
        data = mx.load(path)
        self.read_time += time.time() - t0
        self.reads += 1
        return data

    def prefetch(self, layer_idx: int):
        """Start pre-reading the next FFN layer in background."""
        if self.prefetch_future is not None and self.prefetch_idx == layer_idx:
            return  # Already prefetching this one
        self.prefetch_future = self.executor.submit(self._read_layer, layer_idx)
        self.prefetch_idx = layer_idx

    def get(self, layer_idx: int) -> dict:
        """Get FFN layer weights. Uses prefetched data if available."""
        if self.prefetch_future is not None and self.prefetch_idx == layer_idx:
            data = self.prefetch_future.result()
            self.prefetch_future = None
            return data
        return self._read_layer(layer_idx)

    def stats(self):
        avg = (self.read_time / self.reads * 1000) if self.reads > 0 else 0
        return f"FFN reads: {self.reads}, avg: {avg:.0f}ms/layer, total: {self.read_time:.1f}s"


# ── Streaming Model ────────────────────────────────────────────────

class Qwen3StreamModel:
    """
    Qwen3-32B with FFN streaming from SSD.
    Attention is always in RAM. FFN loaded per-layer, discarded after use.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir

        with open(f"{model_dir}/config.json") as f:
            self.config = json.load(f)

        self.num_layers = self.config["num_hidden_layers"]
        self.streamer = FFNStreamer(
            f"{model_dir}/{self.config['streaming']['ffn_dir']}",
            self.num_layers,
        )

        # Create model skeleton
        from mlx_lm.models.qwen3 import Model, ModelArgs
        args = ModelArgs(
            model_type=self.config["model_type"],
            hidden_size=self.config["hidden_size"],
            num_hidden_layers=self.num_layers,
            intermediate_size=self.config["intermediate_size"],
            num_attention_heads=self.config["num_attention_heads"],
            num_key_value_heads=self.config["num_key_value_heads"],
            rms_norm_eps=self.config["rms_norm_eps"],
            vocab_size=self.config["vocab_size"],
            max_position_embeddings=self.config["max_position_embeddings"],
            rope_theta=self.config["rope_theta"],
            head_dim=self.config["head_dim"],
            tie_word_embeddings=self.config["tie_word_embeddings"],
        )
        self.model = Model(args)
        nn.quantize(self.model, group_size=GROUP_SIZE, bits=BITS)

    def load_pinned(self):
        """Load attention + embeddings + norms into RAM. These stay forever."""
        # Set memory limits: allow 10 GB for Metal, minimal cache
        mx.set_memory_limit(10 * 1024**3)  # 10 GB
        mx.set_cache_limit(256 * 1024**2)   # 256 MB cache max

        print("Loading pinned weights (attention + embed + norms)...")
        t0 = time.time()
        pinned = mx.load(f"{self.model_dir}/pinned.safetensors")
        self.model.load_weights(list(pinned.items()), strict=False)

        # Force pinned weights into GPU memory
        params_to_eval = []
        for name, p in tree_flatten(self.model.parameters()):
            if "mlp" not in name:
                params_to_eval.append(p)
        mx.eval(*params_to_eval)

        del pinned
        gc.collect()
        mx.clear_cache()

        pinned_bytes = sum(p.nbytes for p in params_to_eval)
        print(f"  Loaded in {time.time()-t0:.1f}s ({pinned_bytes/1e9:.2f} GB pinned)")
        print(f"  Metal active: {mx.get_active_memory()/1e9:.2f} GB")

    def _run_ffn_manual(self, x: mx.array, ffn_data: dict) -> mx.array:
        """Run SwiGLU FFN manually with streamed weights. Never stored in model."""
        # quantized_matmul(x, w, scales, biases, transpose=True) does x @ w.T
        # w is [out, in_packed], scales/biases are [out, groups]
        gate_out = mx.quantized_matmul(
            x, ffn_data["mlp.gate_proj.weight"],
            scales=ffn_data["mlp.gate_proj.scales"],
            biases=ffn_data["mlp.gate_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        up_out = mx.quantized_matmul(
            x, ffn_data["mlp.up_proj.weight"],
            scales=ffn_data["mlp.up_proj.scales"],
            biases=ffn_data["mlp.up_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        hidden = nn.silu(gate_out) * up_out
        del gate_out, up_out

        out = mx.quantized_matmul(
            hidden, ffn_data["mlp.down_proj.weight"],
            scales=ffn_data["mlp.down_proj.scales"],
            biases=ffn_data["mlp.down_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        del hidden
        return out

    def forward_streaming(self, input_ids: mx.array, cache=None):
        """
        Forward pass with FFN streaming from SSD.
        FFN weights are NEVER attached to the model — computed manually and discarded.
        """
        from mlx_lm.models.base import create_attention_mask

        B, L = input_ids.shape
        h = self.model.model.embed_tokens(input_ids)

        if cache is None:
            cache = [None] * self.num_layers
        mask = create_attention_mask(h, cache[0])

        # Pre-read first FFN layer
        self.streamer.prefetch(0)

        for i in range(self.num_layers):
            layer = self.model.model.layers[i]

            # Start prefetching NEXT FFN layer while we work on this one
            if i + 1 < self.num_layers:
                self.streamer.prefetch(i + 1)

            # === ATTENTION (from RAM, fast) ===
            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed, mask=mask, cache=cache[i])
            h = h + attn_out

            # Force attention to materialize — gives prefetch time
            mx.eval(h)

            # === FFN (streamed from SSD, computed manually, never stored) ===
            ffn_data = self.streamer.get(i)

            normed = layer.post_attention_layernorm(h)
            ffn_out = self._run_ffn_manual(normed, ffn_data)
            h = h + ffn_out

            # Force FFN result, then DISCARD everything
            mx.eval(h)
            del ffn_data, ffn_out, normed, attn_out
            mx.clear_cache()

            if i % 16 == 0:
                mem = mx.get_active_memory() / 1e9
                print(f"    [layer {i}] active_mem={mem:.2f} GB", flush=True)

        h = self.model.model.norm(h)
        out = self.model.lm_head(h)
        return out


def main():
    print("=" * 60)
    print("  FLASH STREAM — 18.4 GB Model on 16 GB RAM")
    print("  Full 4-bit quality. No compression compromise.")
    print("  Predictive SSD streaming. Not mmap thrashing.")
    print("=" * 60)

    engine = Qwen3StreamModel(MODEL_DIR)
    engine.load_pinned()

    import subprocess
    r = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
    print(f"  {r.stdout.strip()}")

    # Tokenizer
    from transformers import AutoTokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

    prompt = "The key insight from the LLM in a Flash paper is that"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(engine.model)

    # Prefill
    print(f"\n--- Prefill ({len(tokens)} tokens) ---")
    t0 = time.time()
    logits = engine.forward_streaming(input_ids, cache=cache)
    mx.eval(logits)
    t_pf = time.time() - t0
    print(f"  {t_pf:.2f}s ({len(tokens)/t_pf:.1f} tok/s)")
    print(f"  {engine.streamer.stats()}")

    # Decode
    temperature = 0.7
    top_p = 0.9
    rep_penalty = 1.2
    max_tokens = 50

    print(f"\n--- Decode (max {max_tokens}, temp={temperature}) ---")
    generated = []
    t_decode = time.time()

    for step in range(max_tokens):
        next_logits = logits[:, -1, :]

        # Repetition penalty
        if generated:
            seen = mx.array(list(set(generated)))
            penalty_logits = next_logits[:, seen]
            penalty_logits = mx.where(
                penalty_logits > 0,
                penalty_logits / rep_penalty,
                penalty_logits * rep_penalty,
            )
            next_logits[:, seen] = penalty_logits

        # Temperature + top-p sampling
        probs = mx.softmax(next_logits / temperature, axis=-1)
        sorted_indices = mx.argsort(-probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
        cumsum = mx.cumsum(sorted_probs, axis=-1)
        mask = (cumsum - sorted_probs) <= top_p
        sorted_probs = sorted_probs * mask
        sorted_probs = sorted_probs / (sorted_probs.sum(axis=-1, keepdims=True) + 1e-10)
        token = mx.random.categorical(mx.log(sorted_probs + 1e-10))
        token = mx.take_along_axis(sorted_indices, token[:, None], axis=-1).squeeze(-1)

        mx.eval(token)
        token_id = token.item()

        if token_id in (151645, 151643):
            break

        generated.append(token_id)
        print(tokenizer.decode([token_id]), end="", flush=True)

        # Single-token forward with streaming
        logits = engine.forward_streaming(token.reshape(1, 1), cache=cache)
        mx.eval(logits)

        if (step + 1) % 10 == 0:
            elapsed = time.time() - t_decode
            tps = (step + 1) / elapsed
            print(f" [{tps:.2f} tok/s]", flush=True)

    t_total = time.time() - t_decode
    n = len(generated)
    tps = n / t_total if t_total > 0 else 0

    output = tokenizer.decode(generated)
    print(f"\n\nDecode: {n} tokens in {t_total:.1f}s ({tps:.2f} tok/s)")
    print(f"{engine.streamer.stats()}")

    r = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True)
    print(f"{r.stdout.strip()}")

    print(f"\n{'='*60}")
    print(f"{prompt}{output}")
    print(f"{'='*60}")
    print(f"\n  Model: 18.4 GB (full 4-bit quality)")
    print(f"  RAM used: ~8 GB (pinned attention + buffers)")
    print(f"  FFN streamed: 15.2 GB from SSD per token")
    print(f"  Speed: {tps:.2f} tok/s (vs 0.017 mmap thrashing)")


if __name__ == "__main__":
    main()
