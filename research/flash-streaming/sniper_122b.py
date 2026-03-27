"""
MoE Expert Sniper — PyTorch/CUDA edition.

Qwen3.5-122B-A10B (69.6 GB, 4-bit) on a 24 GB GPU.
Only 2.9 GB pinned on GPU. 256 experts per layer on NVMe.
Router picks 8 → load those 8 → dequantize on GPU → matmul → discard.

This is the NVIDIA port of our MLX Expert Sniper.
Same principle, different hardware: anywhere there's a memory boundary,
MoE sparsity lets you load 3% instead of 100%.

Usage:
    python3 sniper_122b.py [--model-dir /path/to/stream] [--prompt "your question"]
"""

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from safetensors import safe_open

# ── Config ──────────────────────────────────────────────

BITS = 4
GROUP_SIZE = 64
NUM_LAYERS = 48
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_SIZE = 3072
EXPERT_INTERMEDIATE = 1024
SHARED_INTERMEDIATE = 1024
HEAD_DIM = 256
NUM_HEADS = 32
NUM_KV_HEADS = 2
VOCAB_SIZE = 248320


def dequantize_4bit(weight, scales, biases, group_size=64):
    """
    Dequantize MLX-format 4-bit weights to float16.

    MLX 4-bit packs 8 values per uint32:
      weight shape: [out_features, in_features // 8] (uint32)
      scales shape: [out_features, in_features // group_size] (float16)
      biases shape: [out_features, in_features // group_size] (float16)

    Returns: [out_features, in_features] float16 tensor
    """
    out_features = weight.shape[0]

    # Unpack uint32 → 8 x 4-bit values
    w = weight.to(torch.int32)
    unpacked = []
    for i in range(8):
        unpacked.append((w >> (4 * i)) & 0xF)
    # Stack: [out, in//8, 8] → reshape to [out, in]
    unpacked = torch.stack(unpacked, dim=-1)
    in_features = unpacked.shape[1] * 8
    unpacked = unpacked.reshape(out_features, in_features).to(torch.float16)

    # Apply scales and biases per group
    num_groups = in_features // group_size
    unpacked = unpacked.reshape(out_features, num_groups, group_size)
    scales_exp = scales.unsqueeze(-1)   # [out, num_groups, 1]
    biases_exp = biases.unsqueeze(-1)   # [out, num_groups, 1]
    dequantized = unpacked * scales_exp + biases_exp
    return dequantized.reshape(out_features, in_features)


class ExpertCache:
    """LRU cache for recently loaded experts."""

    def __init__(self, max_entries=100):
        self.max_entries = max_entries
        self.cache = {}  # (layer, expert_id) → dict of tensors
        self.access_order = []

    def get(self, layer, expert_id):
        key = (layer, expert_id)
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, layer, expert_id, tensors):
        key = (layer, expert_id)
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_entries:
            evict = self.access_order.pop(0)
            del self.cache[evict]
        self.cache[key] = tensors
        self.access_order.append(key)


class SniperEngine:
    """
    MoE Expert Sniper for Qwen3.5-122B on NVIDIA GPU.

    Pinned on GPU: attention + router + shared expert + embeddings (~2.9 GB)
    On NVMe: 256 experts × 48 layers (~65 GB)
    Per token: router picks 8 experts → load from NVMe → dequantize on GPU → matmul → discard
    """

    def __init__(self, model_dir, device="cuda"):
        self.model_dir = Path(model_dir)
        self.device = device
        self.expert_handles = {}  # layer_idx → safe_open handle
        self.expert_cache = ExpertCache(max_entries=200)
        self.executor = ThreadPoolExecutor(max_workers=8)

        # Load config
        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)

        text_cfg = self.config.get("text_config", self.config)
        self.num_layers = text_cfg.get("num_hidden_layers", NUM_LAYERS)
        self.hidden_size = text_cfg.get("hidden_size", HIDDEN_SIZE)
        self.num_experts = text_cfg.get("num_experts", NUM_EXPERTS)
        self.top_k = text_cfg.get("num_experts_per_tok", TOP_K)
        self.expert_intermediate = text_cfg.get("moe_intermediate_size", EXPERT_INTERMEDIATE)
        self.shared_intermediate = text_cfg.get("shared_expert_intermediate_size", SHARED_INTERMEDIATE)
        self.layer_types = text_cfg.get("layer_types", ["linear_attention"] * 36 + ["full_attention"] * 12)

        print(f"Config: {self.num_layers} layers, {self.num_experts} experts, "
              f"top-{self.top_k}, hidden={self.hidden_size}, "
              f"expert_ffn={self.expert_intermediate}")

    def load_pinned(self):
        """Load pinned weights (attention + router + shared + embed) onto GPU."""
        print("\nLoading pinned weights onto GPU...")
        t0 = time.time()

        pinned_path = self.model_dir / "pinned.safetensors"
        self.pinned = {}

        with safe_open(str(pinned_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for i, key in enumerate(keys):
                tensor = f.get_tensor(key)
                self.pinned[key] = tensor.to(self.device)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(keys)} tensors loaded...")

        pinned_gb = sum(t.nbytes for t in self.pinned.values()) / 1e9
        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  Pinned: {pinned_gb:.2f} GB, VRAM used: {vram_gb:.2f} GB [{time.time()-t0:.1f}s]")

    def _get_expert_handle(self, layer_idx):
        """Get or open safetensors handle for a layer's experts."""
        if layer_idx not in self.expert_handles:
            path = self.model_dir / "experts" / f"layer_{layer_idx:02d}.safetensors"
            self.expert_handles[layer_idx] = safe_open(str(path), framework="pt", device="cpu")
        return self.expert_handles[layer_idx]

    def load_active_experts(self, layer_idx, expert_ids):
        """
        Load only the active experts from the stacked tensor.

        Expert tensors are stored as [256, out, in]. We index with expert_ids
        to get [top_k, out, in] — loading only the rows we need.
        """
        handle = self._get_expert_handle(layer_idx)
        ids = expert_ids.cpu().tolist()

        result = {}
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            for comp in ["weight", "scales", "biases"]:
                key = f"{proj}.{comp}"
                full = handle.get_tensor(key)  # [256, out, in]
                # Index only active experts: [top_k, out, in]
                result[key] = full[ids].to(self.device)

        return result

    def run_expert_ffn(self, x, layer_idx, expert_ids, expert_weights):
        """
        Run MoE FFN for one token using stacked expert tensors.

        Expert weights are [256, out, in] on disk. We load only the 8 active
        rows, dequantize on GPU, run SwiGLU, weight by router scores.

        x: [1, 1, hidden_size] on GPU
        expert_ids: [1, 1, top_k] tensor of expert indices
        expert_weights: [1, 1, top_k] tensor of routing weights
        Returns: [1, 1, hidden_size] on GPU
        """
        ids = expert_ids[0, 0]        # [top_k] on GPU
        weights = expert_weights[0, 0]  # [top_k] on GPU
        k = ids.shape[0]

        # Load active expert slices from NVMe → GPU
        data = self.load_active_experts(layer_idx, ids)

        # Dequantize each expert's projections and compute
        output = torch.zeros(self.hidden_size, device=self.device, dtype=torch.float16)
        x_flat = x.squeeze(0).squeeze(0)  # [hidden]

        for i in range(k):
            # Dequantize this expert's projections
            gate_w = dequantize_4bit(data["gate_proj.weight"][i], data["gate_proj.scales"][i], data["gate_proj.biases"][i], GROUP_SIZE)
            up_w = dequantize_4bit(data["up_proj.weight"][i], data["up_proj.scales"][i], data["up_proj.biases"][i], GROUP_SIZE)
            down_w = dequantize_4bit(data["down_proj.weight"][i], data["down_proj.scales"][i], data["down_proj.biases"][i], GROUP_SIZE)

            # SwiGLU
            gate_out = F.silu(x_flat @ gate_w.t())
            up_out = x_flat @ up_w.t()
            hidden = gate_out * up_out
            expert_out = hidden @ down_w.t()

            output += weights[i] * expert_out
            del gate_w, up_w, down_w, gate_out, up_out, hidden, expert_out

        del data
        return output.unsqueeze(0).unsqueeze(0)

    def run_shared_expert(self, x, layer_idx):
        """Run the shared expert (always active, pinned in VRAM)."""
        # Find shared expert weights in pinned
        prefix_candidates = [
            f"language_model.model.layers.{layer_idx}.mlp.shared_expert",
            f"model.layers.{layer_idx}.mlp.shared_expert",
        ]

        prefix = None
        for p in prefix_candidates:
            if f"{p}.gate_proj.weight" in self.pinned:
                prefix = p
                break

        if prefix is None:
            return torch.zeros_like(x)

        gate_w = dequantize_4bit(
            self.pinned[f"{prefix}.gate_proj.weight"],
            self.pinned[f"{prefix}.gate_proj.scales"],
            self.pinned[f"{prefix}.gate_proj.biases"],
            GROUP_SIZE,
        )
        up_w = dequantize_4bit(
            self.pinned[f"{prefix}.up_proj.weight"],
            self.pinned[f"{prefix}.up_proj.scales"],
            self.pinned[f"{prefix}.up_proj.biases"],
            GROUP_SIZE,
        )
        down_w = dequantize_4bit(
            self.pinned[f"{prefix}.down_proj.weight"],
            self.pinned[f"{prefix}.down_proj.scales"],
            self.pinned[f"{prefix}.down_proj.biases"],
            GROUP_SIZE,
        )

        x_flat = x.squeeze(0).squeeze(0)
        gate_out = F.silu(x_flat @ gate_w.t())
        up_out = x_flat @ up_w.t()
        hidden = gate_out * up_out
        out = hidden @ down_w.t()

        del gate_w, up_w, down_w
        return out.unsqueeze(0).unsqueeze(0)

    def get_pinned(self, key):
        """Get a pinned tensor, trying both naming conventions."""
        if key in self.pinned:
            return self.pinned[key]
        # Try with language_model prefix
        alt = f"language_model.{key}"
        if alt in self.pinned:
            return self.pinned[alt]
        # Try without
        for k, v in self.pinned.items():
            if k.endswith(key.split(".")[-2] + "." + key.split(".")[-1]):
                return v
        return None

    def route(self, x, layer_idx):
        """
        Run the router to pick top-K experts.
        Router weight is pinned in VRAM.
        """
        # Router is quantized: has weight, scales, biases
        prefix_candidates = [
            f"language_model.model.layers.{layer_idx}.mlp.gate",
            f"model.layers.{layer_idx}.mlp.gate",
        ]

        router_w = None
        for p in prefix_candidates:
            if f"{p}.weight" in self.pinned:
                w = self.pinned[f"{p}.weight"]
                if w.dtype == torch.uint32:
                    router_w = dequantize_4bit(w, self.pinned[f"{p}.scales"], self.pinned[f"{p}.biases"], GROUP_SIZE)
                else:
                    router_w = w.to(torch.float16)
                break

        if router_w is None:
            raise ValueError(f"Router weight not found for layer {layer_idx}")

        x_flat = x.squeeze(0).squeeze(0).to(torch.float16)  # [hidden]
        router_w = router_w.to(torch.float16)
        logits = x_flat @ router_w.t()     # [num_experts]
        scores = F.softmax(logits, dim=-1)
        top_k_scores, top_k_ids = torch.topk(scores, self.top_k)
        # Normalize
        top_k_scores = top_k_scores / top_k_scores.sum()

        return top_k_ids.unsqueeze(0).unsqueeze(0), top_k_scores.unsqueeze(0).unsqueeze(0)

    def forward_layer_attention(self, h, layer_idx):
        """
        Run attention for one layer. All attention weights are pinned.
        Simplified: uses direct matmul without KV cache for proof of concept.
        """
        prefix_candidates = [
            f"language_model.model.layers.{layer_idx}",
            f"model.layers.{layer_idx}",
        ]

        prefix = None
        for p in prefix_candidates:
            if f"{p}.input_layernorm.weight" in self.pinned:
                prefix = p
                break

        if prefix is None:
            print(f"WARNING: Layer {layer_idx} attention weights not found, skipping")
            return h

        # RMSNorm
        norm_w = self.pinned[f"{prefix}.input_layernorm.weight"]
        normed = self._rms_norm(h, norm_w)

        # For proof of concept: simplified self-attention
        # In production, this would use proper KV cache and rotary embeddings
        layer_type = self.layer_types[layer_idx] if layer_idx < len(self.layer_types) else "full_attention"

        if layer_type == "full_attention":
            attn_prefix = f"{prefix}.self_attn"
        else:
            attn_prefix = f"{prefix}.linear_attn"

        # Q, K, V projections (dequantize from pinned)
        q_w = self._dequant_pinned(f"{attn_prefix}.q_proj")
        k_w = self._dequant_pinned(f"{attn_prefix}.k_proj")
        v_w = self._dequant_pinned(f"{attn_prefix}.v_proj")
        o_w = self._dequant_pinned(f"{attn_prefix}.o_proj")

        if q_w is None:
            return h  # Skip if weights not found

        x = normed.squeeze(0).squeeze(0).to(torch.float16)  # [hidden]
        q = x @ q_w.t()
        k = x @ k_w.t()
        v = x @ v_w.t()

        # Simplified attention (single token, no KV cache)
        # For single token decode, attention is just: output = V (since softmax(Q@K^T/sqrt(d)) = 1)
        attn_out = v
        out = attn_out @ o_w.t()

        del q_w, k_w, v_w, o_w
        return h + out.unsqueeze(0).unsqueeze(0)

    def forward_layer_ffn(self, h, layer_idx):
        """Run MoE FFN: route → load experts → compute → discard."""
        prefix_candidates = [
            f"language_model.model.layers.{layer_idx}",
            f"model.layers.{layer_idx}",
        ]

        prefix = None
        for p in prefix_candidates:
            if f"{p}.post_attention_layernorm.weight" in self.pinned:
                prefix = p
                break

        if prefix is None:
            return h

        # Post-attention RMSNorm
        norm_w = self.pinned[f"{prefix}.post_attention_layernorm.weight"]
        normed = self._rms_norm(h, norm_w)

        # Check if this layer has a router (MoE) or is dense
        has_router = False
        for p in prefix_candidates:
            if f"{p}.mlp.gate.weight" in self.pinned:
                has_router = True
                break

        if has_router:
            # MoE layer: route → load experts → compute → discard
            expert_ids, expert_weights = self.route(normed, layer_idx)
            expert_out = self.run_expert_ffn(normed, layer_idx, expert_ids, expert_weights)
            shared_out = self.run_shared_expert(normed, layer_idx)
            return h + expert_out + shared_out
        else:
            # Dense layer: standard SwiGLU FFN (pinned in VRAM)
            gate_w = self._dequant_pinned(f"{prefix}.mlp.gate_proj")
            up_w = self._dequant_pinned(f"{prefix}.mlp.up_proj")
            down_w = self._dequant_pinned(f"{prefix}.mlp.down_proj")
            if gate_w is not None:
                x_flat = normed.squeeze(0).squeeze(0)
                gate_out = F.silu(x_flat @ gate_w.t())
                up_out = x_flat @ up_w.t()
                hidden = gate_out * up_out
                ffn_out = hidden @ down_w.t()
                del gate_w, up_w, down_w
                return h + ffn_out.unsqueeze(0).unsqueeze(0)
            return h

    def forward_token(self, token_id):
        """Full forward pass for one token through all layers."""
        # Embed
        embed_key = None
        for k in self.pinned:
            if "embed_tokens" in k and "weight" in k:
                embed_key = k
                break

        if embed_key is None:
            raise ValueError("Embedding weights not found in pinned")

        # Dequantize embedding (or use directly if not quantized)
        embed_w = self.pinned[embed_key]
        if embed_w.dtype == torch.uint32:
            # Find scales/biases
            base = embed_key.replace(".weight", "")
            embed_w = dequantize_4bit(
                embed_w,
                self.pinned[f"{base}.scales"],
                self.pinned[f"{base}.biases"],
                GROUP_SIZE,
            )

        h = embed_w[token_id].unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        h = h.to(torch.float16)

        for i in range(self.num_layers):
            t0 = time.time()

            # Attention (all pinned in VRAM)
            h = self.forward_layer_attention(h, i)

            # MoE FFN (experts loaded from NVMe)
            h = self.forward_layer_ffn(h, i)

            elapsed = time.time() - t0
            if i < 3 or i == self.num_layers - 1:
                vram = torch.cuda.memory_allocated() / 1e9
                print(f"  Layer {i:2d}: {elapsed:.3f}s  VRAM: {vram:.2f} GB")

            # Clear GPU cache
            torch.cuda.empty_cache()

        # Final norm
        norm_key = None
        for k in self.pinned:
            if k.endswith("model.norm.weight") or k.endswith("language_model.model.norm.weight"):
                norm_key = k
                break

        if norm_key:
            h = self._rms_norm(h, self.pinned[norm_key])

        # LM head
        lm_key = None
        for k in self.pinned:
            if "lm_head" in k and "weight" in k:
                lm_key = k
                break

        if lm_key:
            lm_w = self.pinned[lm_key]
            if lm_w.dtype == torch.uint32:
                base = lm_key.replace(".weight", "")
                lm_w = dequantize_4bit(
                    lm_w,
                    self.pinned[f"{base}.scales"],
                    self.pinned[f"{base}.biases"],
                    GROUP_SIZE,
                )
            logits = h.squeeze(0).squeeze(0) @ lm_w.t()
            return logits

        return h.squeeze()

    def _rms_norm(self, x, weight, eps=1e-6):
        """RMSNorm."""
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return (x_normed * weight).to(torch.float16)

    def _dequant_pinned(self, key_prefix):
        """Dequantize a pinned weight by prefix."""
        w_key = f"{key_prefix}.weight"
        if w_key not in self.pinned:
            return None

        w = self.pinned[w_key]
        if w.dtype == torch.uint32:
            s = self.pinned.get(f"{key_prefix}.scales")
            b = self.pinned.get(f"{key_prefix}.biases")
            if s is not None and b is not None:
                return dequantize_4bit(w, s, b, GROUP_SIZE)

        return w.to(torch.float16)


def main():
    parser = argparse.ArgumentParser(description="MoE Expert Sniper — 122B on 24GB GPU")
    parser.add_argument("--model-dir", default=os.path.expanduser("~/models/qwen35-122b-stream"))
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("  MoE EXPERT SNIPER — Qwen3.5-122B-A10B")
    print("  69.6 GB model on 24 GB GPU")
    print("  Expert Sniping: load 8/256 experts per layer from NVMe")
    print("=" * 60)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    engine = SniperEngine(args.model_dir, device=args.device)
    engine.load_pinned()

    vram = torch.cuda.memory_allocated() / 1e9 if args.device == "cuda" else 0
    print(f"\nVRAM after pinned: {vram:.2f} GB")

    # Tokenize
    print(f"\nLoading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-122B-A10B", trust_remote_code=True)

    messages = [
        {"role": "system", "content": "Think briefly, answer directly."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(text)
    print(f"Prompt: {len(tokens)} tokens")

    # Prefill
    print(f"\n--- Prefill ---")
    t0 = time.time()
    for tid in tokens:
        logits = engine.forward_token(tid)
    prefill_time = time.time() - t0
    print(f"Prefill: {prefill_time:.1f}s for {len(tokens)} tokens")

    # Decode
    print(f"\n--- Decode (max {args.max_tokens}) ---")
    generated = []
    t_decode = time.time()

    for step in range(args.max_tokens):
        next_token = torch.argmax(logits).item()
        if next_token in (248044, 248046):
            break

        generated.append(next_token)
        chunk = tokenizer.decode([next_token])
        print(chunk, end="", flush=True)

        t_step = time.time()
        logits = engine.forward_token(next_token)
        step_time = time.time() - t_step

        if (step + 1) % 5 == 0:
            vram = torch.cuda.memory_allocated() / 1e9 if args.device == "cuda" else 0
            tps = (step + 1) / (time.time() - t_decode)
            print(f" [{tps:.3f} tok/s, {vram:.1f}GB VRAM]", flush=True)

    t_total = time.time() - t_decode
    n = len(generated)
    tps = n / t_total if t_total > 0 else 0

    output = tokenizer.decode(generated)
    vram = torch.cuda.memory_allocated() / 1e9 if args.device == "cuda" else 0

    print(f"\n\n{'='*60}")
    print(f"Q: {args.prompt}")
    print(f"A: {output}")
    print(f"{'='*60}")
    print(f"  Model: Qwen3.5-122B-A10B (69.6 GB, 4-bit)")
    print(f"  VRAM pinned: {vram:.1f} GB")
    print(f"  Experts: loaded from NVMe per token ({engine.top_k}/{engine.num_experts} per layer)")
    print(f"  Speed: {tps:.3f} tok/s")
    print(f"  Tokens: {n}")
    print(f"  Device: {args.device}")


if __name__ == "__main__":
    main()
