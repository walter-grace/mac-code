#!/usr/bin/env python3
"""
Distributed Gemma 4 Interactive Agent — 26B-A4B across 3 Mac Minis.

Architecture: Same as Qwen distributed but adapted for Gemma 4:
  - Dense MLP runs in parallel with MoE experts
  - Router has inline RMS norm + per_expert_scale
  - gelu_approx activation
  - layer_scalar per layer
  - Embedding scaling by sqrt(hidden_size)

Usage:
  python gemma4_distributed.py \
    --nodes http://<NODE_A_IP>:8401,http://<NODE_B_IP>:8401
"""

import sys
import os
import re
import time
import argparse
import json
import gc

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distributed_reader_fast import FastDistributedReader

# Import custom Gemma 4 model
sys.path.insert(0, os.path.expanduser("~/cli-agent/src"))
from mlx_expert_sniper.models.gemma4 import Model, ModelArgs


class Gemma4DistributedEngine:
    """Distributed Gemma 4 engine with multi-turn support."""

    def __init__(self, node_urls, model_dir=None):
        self.node_urls = node_urls
        self.model_dir = model_dir or os.path.expanduser("~/models/gemma4-4bit")
        self.model = None
        self.reader = None
        self.tokenizer = None
        self.cache = None
        self.num_layers = 30

    def load(self):
        print("Loading Gemma 4 pinned model...")
        t0 = time.time()

        with open(os.path.join(self.model_dir, "config.json")) as f:
            config = json.load(f)

        text_config = config.get("text_config", config)
        self.num_layers = text_config["num_hidden_layers"]

        args = ModelArgs.from_dict(text_config)
        self.model = Model(args)

        # Mixed quantization: apply per-layer config from model
        quant_config = config.get("quantization", {})
        default_bits = quant_config.get("bits", 4)
        default_gs = quant_config.get("group_size", 64)
        
        # Build per-module quantization map
        # 8-bit modules: mlp.{gate,up,down}_proj, router.proj
        def is_8bit(path, module):
            if not isinstance(module, nn.Linear):
                return False
            # Check if this path has an 8-bit override in config
            full_path = "language_model." + path
            if full_path in quant_config and isinstance(quant_config[full_path], dict):
                return quant_config[full_path].get("bits", default_bits) == 8
            return False
        
        # Quantize 4-bit modules (everything except 8-bit ones)
        def should_quantize_4bit(path, module):
            if isinstance(module, nn.Embedding):
                return True
            if not isinstance(module, nn.Linear):
                return False
            if is_8bit(path, module):
                return False
            if module.weight.shape[-1] < default_gs:
                return False
            return True
        
        nn.quantize(self.model, group_size=default_gs, bits=default_bits,
                     class_predicate=should_quantize_4bit)
        
        # Quantize 8-bit modules
        def should_quantize_8bit(path, module):
            if not isinstance(module, nn.Linear):
                return False
            return is_8bit(path, module)
        
        nn.quantize(self.model, group_size=64, bits=8,
                     class_predicate=should_quantize_8bit)

        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)

        # Load ALL weights (pinned + experts for now — we only eval pinned)
        import glob
        all_weights = {}
        for sf in sorted(glob.glob(os.path.join(self.model_dir, "model*.safetensors"))):
            all_weights.update(mx.load(sf))

        # Strip language_model prefix
        stripped = [(k.replace("language_model.", "", 1), v) for k, v in all_weights.items()]
        self.model.load_weights(stripped, strict=False)

        # Only eval non-expert params
        params = [p for name, p in tree_flatten(self.model.parameters())
                  if "expert" not in name and "switch" not in name]
        mx.eval(*params)

        pinned_gb = sum(p.nbytes for p in params) / 1e9
        elapsed = time.time() - t0
        print(f"  Pinned model loaded: {pinned_gb:.1f} GB in {elapsed:.1f}s")

        # Free expert weights from model (they're on the remote nodes)
        del all_weights
        gc.collect()
        mx.clear_cache()

        # Distributed reader
        self.reader = FastDistributedReader(
            node_urls=self.node_urls,
            num_layers=self.num_layers,
            num_experts=128,
        )
        print(f"  Connected to {len(self.node_urls)} expert nodes")

        # KV cache
        self.cache = self.model.make_cache()

        # Tokenizer
        tok_path = os.path.join(self.model_dir, "tokenizer.json")
        if os.path.exists(tok_path):
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(tok_path)
            self._use_fast_tok = True
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._use_fast_tok = False

        print(f"  Ready!\n")

    def encode(self, text):
        if self._use_fast_tok:
            return self.tokenizer.encode(text).ids
        return self.tokenizer.encode(text)

    def encode_chat(self, prompt):
        """Encode a user message in Gemma 4 chat format.

        Format: <bos><|turn>user\\n{prompt}<turn|>\\n<|turn>model\\n
        Token IDs: bos=2, turn_start=105, turn_end=106, newline=107
        """
        NL = chr(10)
        prompt_toks = self.encode(prompt)
        user_toks = self.encode("user" + NL)
        model_toks = self.encode("model" + NL)
        return [2, 105] + user_toks + prompt_toks + [106, 107, 105] + model_toks

    def decode(self, ids):
        if self._use_fast_tok:
            return self.tokenizer.decode(ids)
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def reset_cache(self):
        self.cache = self.model.make_cache()

    def forward(self, input_ids):
        """Forward pass with distributed expert computation."""
        from mlx_lm.models.base import create_attention_mask

        h = self.model.model.embed_tokens(input_ids)
        h = h * (self.model.args.hidden_size ** 0.5)

        mask = create_attention_mask(h, self.cache[0] if self.cache else None)

        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            cache_i = self.cache[i] if self.cache else None

            # 1. Attention
            residual = h
            h_norm = layer.input_layernorm(h)
            h_attn = layer.self_attn(h_norm, mask=mask, cache=cache_i)
            h_attn = layer.post_attention_layernorm(h_attn)
            h = residual + h_attn
            mx.eval(h)

            # 2. Dense MLP (always runs)
            residual = h
            h_ff = layer.pre_feedforward_layernorm(h)
            h_ff = layer.mlp(h_ff)

            if layer.enable_moe_block:
                h_dense = layer.post_feedforward_layernorm_1(h_ff)

                # 3. Router
                B, L, D = residual.shape
                residual_flat = residual.reshape(-1, D)
                router = layer.router
                x_normed = router._inline_rms_norm(residual_flat)
                x_normed = x_normed * router.scale * (router.hidden_size ** -0.5)
                scores = router.proj(x_normed)
                probs = mx.softmax(scores, axis=-1)

                top_k_indices = mx.argpartition(-probs, kth=router.top_k - 1, axis=-1)[..., :router.top_k]
                top_k_weights = mx.take_along_axis(probs, top_k_indices, axis=-1)
                top_k_weights = top_k_weights / mx.sum(top_k_weights, axis=-1, keepdims=True)
                expert_scales = router.per_expert_scale[top_k_indices]
                top_k_weights = top_k_weights * expert_scales

                # Expert input
                moe_input = layer.pre_feedforward_layernorm_2(residual_flat)
                mx.eval(moe_input, top_k_indices, top_k_weights)

                top_k_indices_r = top_k_indices.reshape(B, L, -1)
                top_k_weights_r = top_k_weights.reshape(B, L, -1)

                active_ids = sorted(set(int(e) for e in np.array(top_k_indices_r).flatten()))

                # 4. Expert FFN (DISTRIBUTED)
                expert_out = self.reader.compute_distributed(
                    layer_idx=i,
                    expert_ids=active_ids,
                    hidden_state=moe_input.reshape(B, L, D),
                    top_k_indices=top_k_indices_r,
                    top_k_weights=top_k_weights_r,
                )

                h_moe = layer.post_feedforward_layernorm_2(expert_out)
                h_ff = h_dense + h_moe

            # Final norm + residual + scalar
            h_ff = layer.post_feedforward_layernorm(h_ff)
            h = residual + h_ff
            h = h * layer.layer_scalar
            mx.eval(h)

            mx.clear_cache()

        h = self.model.model.norm(h)

        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(h)
        else:
            return self.model.lm_head(h)

    def generate(self, prompt, max_tokens=200, temperature=0.7):
        """Generate with streaming output."""
        # Use proper Gemma 4 chat template (turn_start=105, turn_end=106)
        tokens = self.encode_chat(prompt)

        generated = []
        input_ids = mx.array([tokens])
        t_start = time.time()
        printed_len = 0

        for step in range(max_tokens):
            logits = self.forward(input_ids)

            # Gemma 4 soft capping
            if hasattr(self.model, 'args') and hasattr(self.model.args, 'final_logit_softcapping'):
                cap = self.model.args.final_logit_softcapping
                if cap and cap > 0:
                    logits = mx.tanh(logits / cap) * cap

            mx.eval(logits)

            if temperature <= 0:
                next_token = int(mx.argmax(logits[0, -1]).item())
            else:
                probs = mx.softmax(logits[0, -1] / temperature, axis=-1)
                next_token = int(mx.random.categorical(mx.log(probs + 1e-10)).item())

            generated.append(next_token)
            input_ids = mx.array([[next_token]])

            # Stream output
            full = self.decode(generated)
            if len(full) > printed_len:
                new_text = full[printed_len:]
                print(new_text, end='', flush=True)
                printed_len = len(full)

            if (step + 1) % 20 == 0:
                elapsed = time.time() - t_start
                tps = (step + 1) / elapsed
                sys.stdout.write(f' [{tps:.2f} tok/s]')
                sys.stdout.flush()

            # EOS: <eos>=1, <turn|>=106
            if next_token in [1, 106]:
                break

        total = time.time() - t_start
        n = len(generated)
        tps = n / total if total > 0 else 0

        print(f'\n\n--- {n} tokens | {total:.1f}s | {tps:.2f} tok/s ---')
        print(f'    {self.reader.stats()}')
        return self.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="Distributed Gemma 4 Interactive Agent")
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--model-dir", default=os.path.expanduser("~/models/gemma4-4bit"))
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    node_urls = [u.strip() for u in args.nodes.split(",")]

    print("=" * 60)
    print("  Distributed Expert Sniper — Gemma 4 (26B-A4B)")
    print(f"  Nodes: {len(node_urls)} expert partitions")
    for i, url in enumerate(node_urls):
        print(f"    [{i}] {url}")
    print("=" * 60)
    print()

    engine = Gemma4DistributedEngine(node_urls=node_urls, model_dir=args.model_dir)
    engine.load()

    print("Type your message (or 'quit' to exit, 'reset' to clear context)")
    print("-" * 60)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if prompt.lower() == "reset":
            engine.reset_cache()
            engine.reader.reads = 0
            engine.reader.read_time = 0.0
            print("Context cleared.")
            continue

        engine.generate(prompt, max_tokens=args.max_tokens, temperature=args.temperature)


if __name__ == "__main__":
    main()
