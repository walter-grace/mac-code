#!/usr/bin/env python3
"""
MoE Sniper engine for Gemma 4-26B-A4B.

Streams Q8 experts from SSD while pinned weights (attention + dense MLP)
stay in RAM. This enables Q8 quality (26.9 GB) on 16 GB Macs where
stock llama.cpp produces 0 tok/s.
"""
import json, sys, os, time, gc
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .expert_io import MoEExpertReader
from .coactivation import CoActivationTracker

MODEL_DIR = ""
BITS = 8  # Q8 experts
GROUP_SIZE = 32  # Q8 group size from GGUF


def run_expert_ffn_gemma4(x, expert_data, top_k_indices, top_k_weights,
                           per_expert_scale=None):
    """
    Gemma 4 expert FFN with fused gate_up and Q8 quantized experts.

    Expert data keys:
      ffn_gate_up_exps.weight: [1408, 704] uint32 (Q8 packed)
      ffn_gate_up_exps.scales: [1408, 88] float16
      ffn_gate_up_exps.biases: [1408, 88] float16
      ffn_down_exps.weight: [2816, 176] uint32 (Q8 packed)
      ffn_down_exps.scales: [2816, 22] float16
      ffn_down_exps.biases: [2816, 22] float16
    """
    active_ids = sorted(expert_data.keys())
    id_to_local = {eid: i for i, eid in enumerate(active_ids)}

    inds_np = np.array(top_k_indices)
    local_np = np.vectorize(lambda v: id_to_local.get(int(v), 0))(inds_np)
    local_indices = mx.array(local_np)

    # Stack experts for batched gather_qmm
    gate_up_w = mx.stack([expert_data[eid]["ffn_gate_up_exps.weight"] for eid in active_ids])
    gate_up_s = mx.stack([expert_data[eid]["ffn_gate_up_exps.scales"] for eid in active_ids])
    gate_up_b = mx.stack([expert_data[eid]["ffn_gate_up_exps.biases"] for eid in active_ids])

    down_w = mx.stack([expert_data[eid]["ffn_down_exps.weight"] for eid in active_ids])
    down_s = mx.stack([expert_data[eid]["ffn_down_exps.scales"] for eid in active_ids])
    down_b = mx.stack([expert_data[eid]["ffn_down_exps.biases"] for eid in active_ids])

    x_exp = mx.expand_dims(x, (-2, -3))

    # Fused gate+up projection via gather_qmm
    gate_up_out = mx.gather_qmm(x_exp, gate_up_w, scales=gate_up_s, biases=gate_up_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=BITS)

    # Split fused output into gate and up halves
    gate, up = mx.split(gate_up_out, 2, axis=-1)

    # GELU activation (Gemma 4 uses gelu_pytorch_tanh)
    hidden = nn.gelu_approx(gate) * up

    # Down projection
    down_out = mx.gather_qmm(hidden, down_w, scales=down_s, biases=down_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=BITS)

    out = down_out.squeeze(-2)

    # Apply per-expert scale if provided
    if per_expert_scale is not None:
        expert_scales = mx.array([per_expert_scale[int(eid)] for eid in active_ids])
        # Broadcast scale to match gather output
        # top_k_weights already normalized, multiply by per_expert_scale
        scale_per_token = mx.take(expert_scales, local_indices)
        top_k_weights = top_k_weights * scale_per_token

    out = (out * top_k_weights[..., None]).sum(axis=-2)
    return out


class MoESniperEngineGemma4:
    def __init__(self, cache_size=3000, enable_prediction=True):
        self.model = None
        self.reader = None
        self.tokenizer = None
        self.cache = None
        self.num_layers = 30
        self.coact = None
        self._cache_size = cache_size
        self._enable_prediction = enable_prediction
        self.per_expert_scales = {}  # layer -> [128] float array

    def load(self):
        with open(os.path.join(MODEL_DIR, "config.json")) as f:
            config = json.load(f)
        self.num_layers = config["num_hidden_layers"]
        streaming = config.get("streaming", {})

        # Load model using our custom gemma4 model class
        from .models.gemma4 import Model, ModelArgs
        args = ModelArgs(
            model_type=config.get("model_type", "gemma4"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=self.num_layers,
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            num_global_key_value_heads=config.get("num_global_key_value_heads", 2),
            global_head_dim=config.get("global_head_dim", 512),
            head_dim=config.get("head_dim", 256),
            intermediate_size=config.get("intermediate_size", 2112),
            moe_intermediate_size=config.get("moe_intermediate_size", 704),
            num_experts=config.get("num_experts", 128),
            top_k_experts=config.get("top_k_experts", 8),
            vocab_size=config["vocab_size"],
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            sliding_window=config.get("sliding_window", 1024),
            hidden_activation=config.get("hidden_activation", "gelu_pytorch_tanh"),
            final_logit_softcapping=config.get("final_logit_softcapping", 30.0),
            attention_k_eq_v=config.get("attention_k_eq_v", True),
            enable_moe_block=config.get("enable_moe_block", True),
            tie_word_embeddings=config.get("tie_word_embeddings", True),
            max_position_embeddings=config.get("max_position_embeddings", 262144),
            layer_types=config.get("layer_types", []),
        )

        self.model = Model(args)

        # Quantize model structure to match Q8 weights from GGUF
        # This creates QuantizedLinear/QuantizedEmbedding modules
        # that expect weight/scales/biases format
        from mlx_lm.models.switch_layers import SwitchLinear
        def should_quantize(path, module):
            if isinstance(module, nn.Embedding): return True
            if isinstance(module, SwitchLinear): return True
            if not isinstance(module, nn.Linear): return False
            # Router proj is float32 in GGUF, don't quantize
            if "router" in path:
                return False
            return True
        nn.quantize(self.model, group_size=GROUP_SIZE, bits=BITS,
                     class_predicate=should_quantize)

        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)

        pinned = mx.load(os.path.join(MODEL_DIR, streaming.get("pinned_file", "pinned.safetensors")))

        # Sanitize weight names
        sanitized = self.model.sanitize(dict(pinned))
        self.model.load_weights(list(sanitized.items()), strict=False)

        # Eval non-expert params
        params = [p for name, p in tree_flatten(self.model.parameters())
                  if "expert" not in name]
        mx.eval(*params)
        del pinned, sanitized; gc.collect(); mx.clear_cache()

        pinned_gb = sum(p.nbytes for p in params) / 1e9

        # Load expert reader
        expert_dir = os.path.join(MODEL_DIR, streaming.get("expert_dir", "bin"))
        self.reader = MoEExpertReader(expert_dir, self.num_layers,
                                       num_workers=8, cache_size=self._cache_size)

        # Load per-expert scales from bin headers
        for li in range(self.num_layers):
            header = self.reader.headers[li]
            if "per_expert_scale" in header:
                self.per_expert_scales[li] = header["per_expert_scale"]

        self.coact = CoActivationTracker(self.num_layers, warmup_tokens=3)

        # Load tokenizer
        from transformers import AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/gemma-4-26B-A4B-it", trust_remote_code=True)

        return pinned_gb

    def reset_cache(self):
        self.cache = self.model.make_cache()

    def forward(self, input_ids):
        """Forward pass with expert streaming from SSD.

        Matches the fixed DecoderLayer.__call__ residual structure exactly.
        """
        h = self.model.model.embed_tokens(input_ids)
        # Gemma embedding scaling
        h = h * mx.array(self.model.args.hidden_size ** 0.5, dtype=h.dtype)

        # Build masks
        from mlx_lm.models.base import create_attention_mask
        args = self.model.args
        first_global = next((i for i, lt in enumerate(args.layer_types) if lt == "full_attention"), 0)
        first_sliding = next((i for i, lt in enumerate(args.layer_types) if lt == "sliding_attention"), 0)
        global_mask = create_attention_mask(h, self.cache[first_global])
        sliding_mask = create_attention_mask(h, self.cache[first_sliding],
                                              window_size=args.sliding_window)

        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            is_global = args.layer_types[i] == "full_attention"
            mask = global_mask if is_global else sliding_mask

            # 1. Attention with pre/post norms and residual
            residual = h
            h = layer.input_layernorm(h)
            h = layer.self_attn(h, mask, self.cache[i])
            h = layer.post_attention_layernorm(h)
            h = residual + h
            mx.eval(h)

            # 2. Feed-forward
            residual = h
            h = layer.pre_feedforward_layernorm(h)
            dense_out = layer.mlp(h)

            if layer.enable_moe_block:
                # Dense path
                h_dense = layer.post_feedforward_layernorm_1(dense_out)

                # MoE: router takes residual (pre-MLP hidden states)
                B, L, D = residual.shape
                residual_flat = residual.reshape(-1, D)
                router_weights, router_indices = layer.router(residual_flat)
                mx.eval(router_weights, router_indices)

                active_ids = list(set(int(e) for e in np.array(router_indices).flatten()))
                self.coact.record_layer(i, active_ids)

                # Predictive prefetch
                if self._enable_prediction and self.coact.ready and i + 1 < self.num_layers:
                    predicted = self.coact.predict_next_layer(i, active_ids, top_k=6)
                    if predicted:
                        to_fetch = [eid for eid in predicted
                                    if self.reader.lru and self.reader.lru.get(i+1, eid) is None]
                        if to_fetch:
                            self.reader.prefetch_experts(i+1, to_fetch)
                if i + 1 < self.num_layers:
                    self.reader.prefetch_experts(i+1, active_ids)

                # Load experts from SSD
                expert_data = self.reader.get_experts(i, active_ids)

                # Expert input: pre_feedforward_layernorm_2 on residual
                moe_input = layer.pre_feedforward_layernorm_2(residual_flat)

                # Run expert FFN
                per_expert_scale = self.per_expert_scales.get(i)
                expert_out = run_expert_ffn_gemma4(
                    moe_input.reshape(B, L, D), expert_data,
                    router_indices.reshape(B, L, -1),
                    router_weights.reshape(B, L, -1),
                    per_expert_scale=per_expert_scale)

                h_moe = layer.post_feedforward_layernorm_2(expert_out)
                h = h_dense + h_moe
                h = layer.post_feedforward_layernorm(h)  # norm combined dense+moe
            else:
                h = layer.post_feedforward_layernorm(dense_out)

            # Residual + layer scalar
            h = residual + h
            h = h * layer.layer_scalar
            mx.eval(h)
            mx.clear_cache()

        self.coact.end_token()
        h = self.model.model.norm(h)
        return self._apply_lm_head(h)

    def _apply_lm_head(self, h):
        """Apply lm_head with logit softcapping."""
        if self.model.args.tie_word_embeddings:
            logits = self.model.model.embed_tokens.as_linear(h)
        else:
            logits = self.model.lm_head(h)

        cap = self.model.args.final_logit_softcapping
        if cap and cap > 0:
            logits = mx.tanh(logits / cap) * cap

        return logits
