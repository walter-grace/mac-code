"""
MLX model implementation for Google Gemma 4 (26B-A4B) MoE.

Architecture: gemma4_text
- 30 decoder layers with mixed sliding/full attention
- 128 experts, top-8 routing per token
- Dense MLP runs in parallel with MoE on every layer
- K=V weight sharing (attention_k_eq_v)
- Two RoPE configs: sliding (theta=10k) vs full (theta=1M, partial_rotary_factor=0.25)
- Global attention uses larger head_dim (512) vs sliding (256)

Reference: HuggingFace transformers Gemma4TextModel
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def _default_layer_types() -> List[str]:
    """Default Gemma4 layer pattern: 5x sliding then 1x full, repeated."""
    pattern = ["sliding_attention"] * 5 + ["full_attention"]
    return (pattern * 5)[:30]  # 30 layers total


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_text"
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 2
    head_dim: int = 256
    global_head_dim: int = 512
    intermediate_size: int = 2112
    moe_intermediate_size: int = 704
    num_experts: int = 128
    top_k_experts: int = 8
    vocab_size: int = 262144
    rms_norm_eps: float = 1e-6
    hidden_activation: str = "gelu_pytorch_tanh"
    final_logit_softcapping: float = 30.0
    sliding_window: int = 1024
    max_position_embeddings: int = 262144
    attention_k_eq_v: bool = True
    enable_moe_block: bool = True
    tie_word_embeddings: bool = True
    layer_types: List[str] = field(default_factory=_default_layer_types)
    # RoPE
    rope_theta_sliding: float = 10_000.0
    rope_theta_global: float = 1_000_000.0
    partial_rotary_factor: float = 0.25  # only for full attention layers


# --------------------------------------------------------------------------- #
# Norms
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    """Gemma-style RMSNorm: weight is (1 + w) * rms_norm(x)."""
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #

class Attention(nn.Module):
    """
    Gemma4 attention with two modes:
      - sliding_attention: sliding window, standard RoPE, head_dim=256, 8 KV heads
      - full_attention: full context, partial RoPE, global_head_dim=512, 2 KV heads

    attention_k_eq_v: K and V share the same projection weights. The HF weights
    have separate k_proj and v_proj entries but they're identical. We keep both
    for weight loading compatibility but only use k_proj output for both K and V.
    """
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.attention_k_eq_v = args.attention_k_eq_v

        self.n_heads = args.num_attention_heads

        if self.is_sliding:
            self.n_kv_heads = args.num_key_value_heads
            self.head_dim = args.head_dim
            rope_dims = args.head_dim
            rope_theta = args.rope_theta_sliding
        else:
            self.n_kv_heads = args.num_global_key_value_heads
            self.head_dim = args.global_head_dim
            # Partial rotary: only rotate first partial_rotary_factor of dims
            rope_dims = int(args.global_head_dim * args.partial_rotary_factor)
            rope_theta = args.rope_theta_global

        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        # v_proj exists for weight loading but K=V means we use k_proj output for V too
        if not self.attention_k_eq_v:
            self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(rope_dims, traditional=False, base=rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        # K=V: use key projection output as values too
        values = keys if self.attention_k_eq_v else self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


# --------------------------------------------------------------------------- #
# Dense MLP
# --------------------------------------------------------------------------- #

class DenseMLP(nn.Module):
    """Standard gated MLP with gelu_pytorch_tanh activation."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

class Router(nn.Module):
    """
    Gemma4 MoE router with learnable scale and per-expert scales.

    Forward:
      1. Inline RMS norm (no learnable weight)
      2. Scale by self.scale * hidden_size^{-0.5}
      3. Linear projection to num_experts
      4. Softmax -> top-k selection
      5. Renormalize top-k weights, multiply by per_expert_scale
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_experts = args.num_experts
        self.top_k = args.top_k_experts

        self.proj = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        # Learnable scalar scale
        self.scale = mx.ones((1,))
        # Per-expert scales
        self.per_expert_scale = mx.ones((args.num_experts,))

    def _inline_rms_norm(self, x: mx.array) -> mx.array:
        """RMS norm without learnable weight."""
        variance = mx.mean(x * x, axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + 1e-6)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        # x: [B, L, D] or [B*L, D]
        orig_shape = x.shape
        if x.ndim == 3:
            B, L, D = x.shape
            x = x.reshape(-1, D)
        else:
            D = x.shape[-1]

        # Inline RMS norm (no learnable weight)
        x_normed = self._inline_rms_norm(x)
        # Scale
        x_normed = x_normed * self.scale * (self.hidden_size ** -0.5)
        # Project to expert logits
        scores = self.proj(x_normed)
        # Softmax over experts
        probs = mx.softmax(scores, axis=-1)
        # Top-k
        top_k_indices = mx.argpartition(-probs, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        # Gather the weights for selected experts
        top_k_weights = mx.take_along_axis(probs, top_k_indices, axis=-1)
        # Renormalize
        top_k_weights = top_k_weights / mx.sum(top_k_weights, axis=-1, keepdims=True)
        # Per-expert scaling
        expert_scales = self.per_expert_scale[top_k_indices]
        top_k_weights = top_k_weights * expert_scales

        if len(orig_shape) == 3:
            top_k_weights = top_k_weights.reshape(B, L, self.top_k)
            top_k_indices = top_k_indices.reshape(B, L, self.top_k)

        return top_k_weights, top_k_indices


# --------------------------------------------------------------------------- #
# Expert Block
# --------------------------------------------------------------------------- #

class Experts(nn.Module):
    """
    Fused MoE experts using gather_mm for efficient batched expert dispatch.

    Weight shapes (from HuggingFace):
      gate_up_proj: [num_experts, 2 * moe_intermediate_size, hidden_size]
      down_proj:    [num_experts, hidden_size, moe_intermediate_size]

    ### SNIPER ENGINE NOTE ###
    # In production, gate_up_proj and down_proj are NOT resident in memory.
    # They are loaded from SSD on-demand by the expert sniper engine.
    # The weights here serve as placeholders for the model structure.
    # The sniper engine will intercept expert dispatch and handle I/O.
    ### END NOTE ###
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.moe_intermediate_size = args.moe_intermediate_size
        self.hidden_size = args.hidden_size

        # [num_experts, 2*moe_intermediate, hidden_size] -- SSD-backed in production
        self.gate_up_proj = mx.zeros(
            (args.num_experts, 2 * args.moe_intermediate_size, args.hidden_size)
        )
        # [num_experts, hidden_size, moe_intermediate] -- SSD-backed in production
        self.down_proj = mx.zeros(
            (args.num_experts, args.hidden_size, args.moe_intermediate_size)
        )

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
    ) -> mx.array:
        """
        Args:
            x: [B, L, D] input hidden states
            indices: [B, L, top_k] expert indices

        Returns:
            Expert outputs [B, L, top_k, D] before weighted sum
        """
        # Expand for gather_mm: [B*L, 1, 1, D]
        if x.ndim == 3:
            B, L, D = x.shape
            x_flat = x.reshape(-1, D)
        else:
            x_flat = x
            B, L = None, None

        x_expanded = mx.expand_dims(mx.expand_dims(x_flat, -2), -2)

        # gate_up_proj is [E, 2*I, D], we need [E, D, 2*I] for matmul
        # gather_mm expects rhs as [E, D, O] indexed by expert indices
        gate_up_out = mx.gather_mm(
            x_expanded,
            self.gate_up_proj.swapaxes(-1, -2),
            rhs_indices=indices.reshape(-1, indices.shape[-1]) if B is not None else indices,
        )
        # gate_up_out: [B*L, top_k, 1, 2*I]
        gate_up_out = gate_up_out.squeeze(-2)

        # Split into gate and up
        gate, up = mx.split(gate_up_out, 2, axis=-1)
        # Apply gelu activation (gelu_pytorch_tanh)
        hidden = nn.gelu_approx(gate) * up

        # down_proj: [E, D, I] -- already in the right shape for gather_mm
        hidden_expanded = mx.expand_dims(hidden, -2)
        out = mx.gather_mm(
            hidden_expanded,
            self.down_proj.swapaxes(-1, -2),
            rhs_indices=indices.reshape(-1, indices.shape[-1]) if B is not None else indices,
        )
        # out: [B*L, top_k, 1, D]
        out = out.squeeze(-2)

        if B is not None:
            out = out.reshape(B, L, indices.shape[-1], -1)

        return out


# --------------------------------------------------------------------------- #
# Decoder Layer
# --------------------------------------------------------------------------- #

class DecoderLayer(nn.Module):
    """
    Gemma4 decoder layer with dense MLP + MoE parallel residual structure.

    Forward pass:
      1. Attention with residual
      2. Dense MLP with pre/post norms
      3. MoE block (if enabled) added in parallel to dense output
      4. Residual connection + layer_scalar
    """
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.enable_moe_block = args.enable_moe_block

        # Attention
        self.self_attn = Attention(args, layer_idx)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        # Dense MLP
        self.mlp = DenseMLP(args)
        self.pre_feedforward_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        if self.enable_moe_block:
            # MoE norms
            self.post_feedforward_layernorm_1 = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.pre_feedforward_layernorm_2 = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_feedforward_layernorm_2 = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

            # Router + Experts
            self.router = Router(args)
            self.experts = Experts(args)

        # Per-layer learned scalar multiplier
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # 1. Attention
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = residual + h

        # 2. Dense MLP
        residual = h
        dense_in = self.pre_feedforward_layernorm(h)
        dense_out = self.mlp(dense_in)
        h = self.post_feedforward_layernorm(dense_out)

        # 3. MoE (parallel to dense, sharing the same residual)
        if self.enable_moe_block:
            # MoE input: norm applied to (residual + raw dense_out), not the post-normed version
            moe_input = self.pre_feedforward_layernorm_2(residual + dense_out)

            # Route
            top_k_weights, top_k_indices = self.router(moe_input)

            # Expert forward
            expert_out = self.experts(moe_input, top_k_indices)
            # Weighted sum over top-k experts: [B, L, top_k, D] * [B, L, top_k, 1] -> [B, L, D]
            weighted_out = (expert_out * mx.expand_dims(top_k_weights, -1)).sum(axis=-2)
            moe_out = self.post_feedforward_layernorm_2(weighted_out)

            # Combine: dense (post-normed again) + moe
            h = self.post_feedforward_layernorm_1(h) + moe_out

        # 4. Residual + layer scalar
        h = residual + h
        h = h * self.layer_scalar

        return h


# --------------------------------------------------------------------------- #
# Full Model
# --------------------------------------------------------------------------- #

class Gemma4TextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_idx=i)
            for i in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        # Gemma scaling: multiply embeddings by sqrt(hidden_size)
        h = h * mx.array(self.args.hidden_size ** 0.5, dtype=h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        # Build masks: one for global attention, one for sliding window
        # Find first global layer to get its cache for mask creation
        first_global_idx = None
        first_sliding_idx = None
        for i, lt in enumerate(self.args.layer_types):
            if lt == "full_attention" and first_global_idx is None:
                first_global_idx = i
            if lt == "sliding_attention" and first_sliding_idx is None:
                first_sliding_idx = i
            if first_global_idx is not None and first_sliding_idx is not None:
                break

        global_mask = create_attention_mask(
            h, cache[first_global_idx] if first_global_idx is not None else None
        )
        sliding_mask = create_attention_mask(
            h,
            cache[first_sliding_idx] if first_sliding_idx is not None else None,
            window_size=self.args.sliding_window,
        )

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            is_global = self.args.layer_types[i] == "full_attention"
            mask = global_mask if is_global else sliding_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Gemma4TextModel(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)

        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(out)
        else:
            logits = self.lm_head(out)

        # Logit soft-capping
        cap = self.args.final_logit_softcapping
        if cap is not None and cap > 0:
            logits = mx.tanh(logits / cap) * cap

        return logits

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        """Create KV caches: RotatingKVCache for sliding layers, KVCache for global."""
        caches = []
        for i in range(self.args.num_hidden_layers):
            if self.args.layer_types[i] == "sliding_attention":
                caches.append(RotatingKVCache(max_size=self.args.sliding_window))
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Convert HuggingFace weight names to our model's expected names.

        HF prefix: model.language_model.layers.N.xxx
        Our prefix: model.layers.N.xxx

        Also handles:
        - Stripping "model.language_model." prefix -> "model."
        - v_proj weights are dropped when attention_k_eq_v (K=V sharing)
        - embed_tokens maps to model.embed_tokens
        - lm_head is dropped if tie_word_embeddings
        """
        new_weights = {}
        for k, v in weights.items():
            # Strip the "model.language_model." prefix
            new_key = k
            if new_key.startswith("model.language_model."):
                new_key = "model." + new_key[len("model.language_model."):]

            # Drop v_proj when K=V (weights are identical to k_proj)
            if self.args.attention_k_eq_v and "v_proj" in new_key:
                continue

            # Drop lm_head when tied
            if self.args.tie_word_embeddings and new_key == "lm_head.weight":
                continue

            new_weights[new_key] = v

        return new_weights
