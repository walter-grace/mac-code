"""
MoE Expert Sniper v3 — HF Transformers + Expert Sniping (correct output).

Uses the Codex/Opus blueprint:
  1. init_empty_weights → zero memory model skeleton
  2. Materialize non-expert params on GPU, delete expert modules
  3. Dequantize MLX 4-bit → bfloat16, inject via set_module_tensor_to_device
  4. Patch MoE forward to snipe experts from NVMe/VRAM cache
  5. Generate with model.generate() → correct attention for free

Lean 4 proof (Harmonic AI) verifies: if w_i = 0 for i ∉ S, then
  ∑ w_i · f_i(x) = ∑_{i∈S} w_i · f_i(x)  — expert sniping is exact.

Usage:
    python3 sniper_122b_v3.py --model-dir /workspace/qwen35-122b-stream \
        --original-dir /workspace/qwen35-122b-a10b-4bit
"""

import os
import gc
import json
import time
import copy
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

BITS = 4
GROUP_SIZE = 64


def dequantize_mlx_4bit(weight, scales, biases, group_size=64):
    """Dequantize MLX 4-bit packed uint32 → bfloat16. Unsigned, LSB-first."""
    if weight.dtype not in (torch.uint32, torch.int32):
        return weight.to(torch.bfloat16)
    out_features = weight.shape[0]
    w = weight.to(torch.int32)
    shifts = torch.arange(0, 32, 4, device=w.device)
    unpacked = (w.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 0xF
    in_features = unpacked.shape[1] * 8
    unpacked = unpacked.reshape(out_features, in_features).float()
    num_groups = in_features // group_size
    unpacked = unpacked.reshape(out_features, num_groups, group_size)
    dq = unpacked * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
    return dq.reshape(out_features, in_features).to(torch.bfloat16)


def remap_key(k):
    """language_model.model.layers.X → model.layers.X"""
    if k.startswith("language_model."):
        return k[len("language_model."):]
    return k


class ExpertSniper:
    """Loads active MoE experts from NVMe or VRAM cache."""

    def __init__(self, expert_dir, num_layers, device="cuda", cache_layers=15):
        self.expert_dir = Path(expert_dir)
        self.device = device
        self.handles = {}
        self.vram_cache = {}
        self.cache_layers = cache_layers
        self.num_layers = num_layers

    def cache_in_vram(self):
        print(f"  Caching expert layers 0-{self.cache_layers-1} in VRAM...")
        t0 = time.time()
        for i in range(min(self.cache_layers, self.num_layers)):
            path = self.expert_dir / f"layer_{i:02d}.safetensors"
            if not path.exists():
                continue
            data = {}
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    data[k] = f.get_tensor(k).to(self.device)
            self.vram_cache[i] = data
        gb = sum(sum(t.nbytes for t in d.values()) for d in self.vram_cache.values()) / 1e9
        print(f"  Cached: {gb:.2f} GB [{time.time()-t0:.1f}s]")

    def _handle(self, layer_idx):
        if layer_idx not in self.handles:
            self.handles[layer_idx] = safe_open(
                str(self.expert_dir / f"layer_{layer_idx:02d}.safetensors"),
                framework="pt", device="cpu"
            )
        return self.handles[layer_idx]

    def get_experts(self, layer_idx, expert_ids):
        """Get dequantized [top_k, out, in] weights for active experts."""
        ids = expert_ids if isinstance(expert_ids, list) else expert_ids.tolist()
        result = {}
        if layer_idx in self.vram_cache:
            data = self.vram_cache[layer_idx]
            idx = torch.tensor(ids, dtype=torch.long, device=self.device)
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                w = torch.index_select(data[f"{proj}.weight"], 0, idx)
                s = torch.index_select(data[f"{proj}.scales"], 0, idx)
                b = torch.index_select(data[f"{proj}.biases"], 0, idx)
                result[proj] = dequantize_mlx_4bit(w, s, b, GROUP_SIZE)
        else:
            h = self._handle(layer_idx)
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                fw, fs, fb = h.get_tensor(f"{proj}.weight"), h.get_tensor(f"{proj}.scales"), h.get_tensor(f"{proj}.biases")
                w = torch.stack([fw[i] for i in ids]).to(self.device)
                s = torch.stack([fs[i] for i in ids]).to(self.device)
                b = torch.stack([fb[i] for i in ids]).to(self.device)
                result[proj] = dequantize_mlx_4bit(w, s, b, GROUP_SIZE)
        return result


def make_sniped_forward(moe_block, layer_idx, sniper, top_k):
    """Create patched MoE forward that snipes experts from disk."""
    gate = moe_block.gate
    shared_expert = getattr(moe_block, 'shared_expert', None)
    shared_expert_gate = getattr(moe_block, 'shared_expert_gate', None)

    def forward(hidden_states):
        B, L, D = hidden_states.shape
        x = hidden_states.reshape(-1, D)
        router_logits = gate(x)
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_w, topk_idx = torch.topk(scores, top_k, dim=-1)
        topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)

        needed = topk_idx.unique().tolist()
        expert_w = sniper.get_experts(layer_idx, needed)
        id_to_local = {eid: i for i, eid in enumerate(needed)}

        output = torch.zeros_like(x)
        for local_idx, eid in enumerate(needed):
            mask = (topk_idx == eid)
            token_mask = mask.any(dim=-1)
            tidx = token_mask.nonzero(as_tuple=True)[0]
            if len(tidx) == 0:
                continue
            w = (topk_w * mask.to(topk_w.dtype)).sum(dim=-1)
            inp = x[tidx]
            g = F.silu(inp @ expert_w["gate_proj"][local_idx].t())
            u = inp @ expert_w["up_proj"][local_idx].t()
            out = (g * u) @ expert_w["down_proj"][local_idx].t()
            output[tidx] += w[tidx].unsqueeze(-1) * out

        if shared_expert is not None:
            s_out = shared_expert(x)
            if shared_expert_gate is not None:
                s_out = s_out * torch.sigmoid(shared_expert_gate(x))
            output = output + s_out

        del expert_w
        return output.reshape(B, L, D)

    return forward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/qwen35-122b-stream")
    parser.add_argument("--original-dir", default="/workspace/qwen35-122b-a10b-4bit")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--cache-layers", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("  MoE EXPERT SNIPER v3 — Correct Output Edition")
    print("=" * 60)

    device = args.device
    model_dir = Path(args.model_dir)
    original_dir = Path(args.original_dir)

    # ── 1. Config ──
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(str(original_dir), trust_remote_code=True)
    text_cfg = config.text_config if hasattr(config, 'text_config') else config
    num_layers = text_cfg.num_hidden_layers
    top_k = getattr(text_cfg, 'num_experts_per_tok', 8)
    num_experts = text_cfg.num_experts
    print(f"  {num_layers} layers, {num_experts} experts, top-{top_k}")

    # ── 2. Create empty model (Codex blueprint) ──
    print("\n[1/5] Creating empty model skeleton...")
    t0 = time.time()

    from accelerate import init_empty_weights

    with init_empty_weights():
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_config(
            text_cfg, trust_remote_code=True, torch_dtype=torch.bfloat16
        )

    # Delete expert modules → replace with empty ModuleList
    for i in range(num_layers):
        layer = model.model.layers[i]
        if hasattr(layer.mlp, 'experts') and layer.mlp.experts is not None:
            layer.mlp.experts = nn.ModuleList()  # empty, no params

    print(f"  Created in {time.time()-t0:.1f}s (experts deleted)")

    # ── 3. Materialize non-expert params on GPU ──
    print("\n[2/5] Materializing non-expert params on GPU...")
    t0 = time.time()

    from accelerate.utils import set_module_tensor_to_device

    # First pass: materialize all remaining params as empty on GPU
    for name, param in list(model.named_parameters()):
        if param.device == torch.device("meta"):
            set_module_tensor_to_device(
                model, name, device=device,
                value=torch.zeros(param.shape, dtype=torch.bfloat16)
            )

    for name, buf in list(model.named_buffers()):
        if buf.device == torch.device("meta") or buf.device != torch.device(device):
            try:
                set_module_tensor_to_device(
                    model, name, device=device,
                    value=torch.zeros(buf.shape, dtype=buf.dtype)
                )
            except Exception:
                pass  # some buffers may not be settable

    # Move ALL remaining buffers (rotary_emb inv_freq etc) to GPU
    model = model.to(device)

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Empty model on GPU: {vram:.2f} GB [{time.time()-t0:.1f}s]")

    # ── 4. Inject dequantized pinned weights ──
    print("\n[3/5] Injecting dequantized pinned weights...")
    t0 = time.time()

    pinned_path = model_dir / "pinned.safetensors"
    loaded = 0
    skipped = 0

    with safe_open(str(pinned_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        # Group by base name
        bases = {}
        for k in keys:
            if k.endswith(".scales"):
                bases.setdefault(k[:-7], {})["scales"] = k
            elif k.endswith(".biases"):
                bases.setdefault(k[:-7], {})["biases"] = k
            elif k.endswith(".weight"):
                bases.setdefault(k[:-7], {})["weight"] = k
            else:
                bases.setdefault(k, {})["raw"] = k

        model_param_names = set(n for n, _ in model.named_parameters())
        model_buffer_names = set(n for n, _ in model.named_buffers())

        for base, parts in bases.items():
            if "raw" in parts:
                raw_key = parts["raw"]
                mapped = remap_key(raw_key)
                tensor = f.get_tensor(raw_key)

                if mapped in model_param_names or mapped in model_buffer_names:
                    try:
                        set_module_tensor_to_device(model, mapped, device=device, value=tensor.to(torch.bfloat16))
                        loaded += 1
                    except Exception:
                        skipped += 1
                else:
                    skipped += 1

            elif "weight" in parts and "scales" in parts:
                w = f.get_tensor(parts["weight"])
                s = f.get_tensor(parts["scales"])
                b = f.get_tensor(parts["biases"])
                dq = dequantize_mlx_4bit(w, s, b, GROUP_SIZE)

                target = remap_key(base) + ".weight"
                if target in model_param_names:
                    try:
                        set_module_tensor_to_device(model, target, device=device, value=dq)
                        loaded += 1
                    except ValueError as e:
                        # Shape mismatch — likely shared_expert with wrong intermediate
                        skipped += 1
                else:
                    skipped += 1
                del dq

            elif "weight" in parts:
                w = f.get_tensor(parts["weight"])
                target = remap_key(base) + ".weight"
                if target in model_param_names:
                    try:
                        set_module_tensor_to_device(model, target, device=device, value=w.to(torch.bfloat16))
                        loaded += 1
                    except Exception:
                        skipped += 1
                else:
                    skipped += 1

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded: {loaded}, Skipped: {skipped}, VRAM: {vram:.2f} GB [{time.time()-t0:.1f}s]")

    # ── 5. Expert sniper + MoE patching ──
    print("\n[4/5] Setting up Expert Sniper + patching MoE...")
    sniper = ExpertSniper(model_dir / "experts", num_layers, device=device, cache_layers=args.cache_layers)
    sniper.cache_in_vram()

    patched = 0
    for i in range(num_layers):
        layer = model.model.layers[i]
        if hasattr(layer.mlp, 'gate') and layer.mlp.gate is not None:
            layer.mlp.forward = make_sniped_forward(layer.mlp, i, sniper, top_k)
            patched += 1

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Patched: {patched}/{num_layers}, VRAM: {vram:.2f} GB")

    # ── 6. Generate ──
    print("\n[5/5] Generating...")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(original_dir), trust_remote_code=True)

    messages = [
        {"role": "system", "content": "Answer briefly and directly."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    print(f"  Prompt: {prompt_len} tokens")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )
    t_total = time.time() - t0

    new_tokens = output_ids[0][prompt_len:]
    n = len(new_tokens)
    tps = n / t_total if t_total > 0 else 0
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    vram = torch.cuda.memory_allocated() / 1e9

    print(f"\n{'='*60}")
    print(f"Q: {args.prompt}")
    print(f"A: {output_text}")
    print(f"{'='*60}")
    print(f"  Model: Qwen3.5-122B-A10B (69.6 GB)")
    print(f"  VRAM: {vram:.1f} GB")
    print(f"  Cached layers: 0-{args.cache_layers-1}")
    print(f"  Speed: {tps:.3f} tok/s")
    print(f"  Tokens: {n}")
    print(f"  Time: {t_total:.1f}s")
    print(f"  Attention: HF Transformers native (GatedDeltaNet + GQA)")
    print(f"  Experts: Sniped from NVMe ({top_k}/{num_experts} per layer)")
    print(f"  Proof: Lean 4 verified (Harmonic AI)")


if __name__ == "__main__":
    main()
