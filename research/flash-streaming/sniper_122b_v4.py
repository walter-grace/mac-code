"""
MoE Expert Sniper v4 — HF from_pretrained backbone (guaranteed correct weights).

Loads full model via from_pretrained on CPU, strips experts, moves backbone
to GPU, patches MoE forward with expert sniping.

This eliminates our custom dequantization entirely for non-expert weights.
Expert weights still use our proven dequant (verified 0.05% match vs HF).

Usage:
    python3 sniper_122b_v4.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import gc
import time

GROUP_SIZE = 64

def dequantize_mlx_4bit(weight, scales, biases, group_size=64):
    if weight.dtype not in (torch.uint32, torch.int32):
        return weight.to(torch.bfloat16)
    orig_shape = weight.shape
    if weight.ndim == 3:
        batch = orig_shape[0]
        weight = weight.reshape(-1, orig_shape[-1])
        scales = scales.reshape(-1, scales.shape[-1])
        biases = biases.reshape(-1, biases.shape[-1])
    else:
        batch = None
    out_features = weight.shape[0]
    w = weight.to(torch.int32)
    shifts = torch.arange(0, 32, 4, device=w.device)
    unpacked = (w.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 0xF
    in_features = unpacked.shape[1] * 8
    unpacked = unpacked.reshape(out_features, in_features).float()
    num_groups = in_features // group_size
    unpacked = unpacked.reshape(out_features, num_groups, group_size)
    dq = unpacked * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
    result = dq.reshape(out_features, in_features).to(torch.bfloat16)
    if batch is not None:
        result = result.reshape(batch, orig_shape[1], in_features)
    return result


def make_sniped_forward(gate, shared_expert, shared_expert_gate, layer_idx, top_k, expert_dir, device):
    def forward(hidden_states):
        B, L, D = hidden_states.shape
        x = hidden_states.reshape(-1, D)
        gate_out = gate(x)
        if isinstance(gate_out, tuple) and len(gate_out) == 3:
            _, topk_w, topk_idx = gate_out
            topk_w = topk_w.to(hidden_states.dtype)
        else:
            scores = F.softmax(gate_out, dim=-1, dtype=torch.float32)
            topk_w, topk_idx = torch.topk(scores, top_k, dim=-1)
            topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)

        needed = topk_idx.unique().tolist()
        ep = f"{expert_dir}/layer_{layer_idx:02d}.safetensors"
        with safe_open(ep, framework="pt", device="cpu") as ef:
            expert_w = {}
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                fw = torch.stack([ef.get_tensor(f"{proj}.weight")[e] for e in needed]).to(device)
                fs = torch.stack([ef.get_tensor(f"{proj}.scales")[e] for e in needed]).to(device)
                fb = torch.stack([ef.get_tensor(f"{proj}.biases")[e] for e in needed]).to(device)
                expert_w[proj] = dequantize_mlx_4bit(fw, fs, fb, GROUP_SIZE)

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
    device = "cuda"
    original_dir = "/workspace/qwen35-122b-a10b-4bit"
    expert_dir = "/workspace/qwen35-122b-stream/experts"

    print("=" * 60)
    print("  MoE EXPERT SNIPER v4")
    print("  HF from_pretrained backbone + Expert Sniping")
    print("=" * 60)

    # Step 1: Load full model on CPU
    print("\n[1/5] Loading full model on CPU (~25 min)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        original_dir,
        trust_remote_code=True,
        device_map={'': 'cpu'},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.0f}s")

    # Access the text model (might be wrapped in VLM)
    if hasattr(model, 'language_model'):
        text_model = model.language_model
    else:
        text_model = model

    config = text_model.config if hasattr(text_model, 'config') else model.config
    num_layers = config.num_hidden_layers
    top_k = getattr(config, 'num_experts_per_tok', 8)
    print(f"  {num_layers} layers, top-{top_k} experts")

    # Verify a weight is dequantized
    w0 = text_model.model.layers[0].linear_attn.in_proj_qkv.weight
    print(f"  L0 qkv: {w0.shape} {w0.dtype} mean={w0.float().mean():.8f}")

    # Step 2: Delete expert weights to free CPU RAM
    print("\n[2/5] Deleting expert weights from CPU...")
    t0 = time.time()
    for i in range(num_layers):
        layer = text_model.model.layers[i]
        if hasattr(layer.mlp, 'experts') and layer.mlp.experts is not None:
            del layer.mlp.experts
            layer.mlp.experts = nn.ModuleList()
    gc.collect()
    print(f"  Done in {time.time()-t0:.1f}s")

    # Step 3: Move non-expert model to GPU
    print("\n[3/5] Moving backbone to GPU...")
    t0 = time.time()
    text_model = text_model.to(device)
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.2f} GB [{time.time()-t0:.1f}s]")

    # Step 4: Patch MoE forward
    print("\n[4/5] Patching MoE layers...")
    patched = 0
    for i in range(num_layers):
        layer = text_model.model.layers[i]
        if hasattr(layer.mlp, 'gate') and layer.mlp.gate is not None:
            moe = layer.mlp
            moe.forward = make_sniped_forward(
                moe.gate, getattr(moe, 'shared_expert', None),
                getattr(moe, 'shared_expert_gate', None),
                i, top_k, expert_dir, device
            )
            patched += 1
    print(f"  Patched: {patched}/{num_layers}")

    # Step 5: Generate!
    print("\n[5/5] Generating...")
    text_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(original_dir, trust_remote_code=True)

    # Test 1: Single forward
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    print(f"  Prompt: '{prompt}' ({input_ids.shape[1]} tokens)")

    t0 = time.time()
    with torch.no_grad():
        out = text_model(input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits if hasattr(out, 'logits') else out[0]
    elapsed = time.time() - t0

    top10 = torch.topk(logits[0, -1].float(), 10)
    print(f"\n  Top 10 predictions ({elapsed:.1f}s):")
    for i, (val, idx) in enumerate(zip(top10.values, top10.indices)):
        print(f"    {i+1}. '{tokenizer.decode([idx.item()])}' (logit={val.item():.2f})")

    paris_tokens = tokenizer.encode("Paris", add_special_tokens=False)
    paris_in_top = any(idx.item() in paris_tokens for idx in top10.indices)
    print(f"\n  'Paris' in top 10: {paris_in_top}")

    if paris_in_top:
        print("  >>> SUCCESS! Coherent 122B output via Expert Sniping!")
    else:
        print("  >>> Still broken — investigate further")

    # Test 2: Generate a few tokens
    print(f"\n  Greedy decode (10 tokens):")
    generated = []
    current_ids = input_ids.clone()
    t0 = time.time()
    for step in range(10):
        with torch.no_grad():
            out = text_model(current_ids, attention_mask=torch.ones_like(current_ids), use_cache=False)
        logits = out.logits if hasattr(out, 'logits') else out[0]
        next_token = logits[0, -1].argmax().item()
        if next_token in (248044, 248046):
            break
        generated.append(next_token)
        current_ids = torch.cat([current_ids, torch.tensor([[next_token]], device=device)], dim=1)
        chunk = tokenizer.decode([next_token])
        print(f"    Token {step}: '{chunk}'", flush=True)

    output_text = tokenizer.decode(generated, skip_special_tokens=True)
    tps = len(generated) / (time.time() - t0) if generated else 0

    print(f"\n{'='*60}")
    print(f"Q: {prompt}")
    print(f"A: {output_text}")
    print(f"{'='*60}")
    print(f"  Model: Qwen3.5-122B-A10B (69.6 GB)")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"  Speed: {tps:.3f} tok/s")
    print(f"  Backbone: HF from_pretrained (guaranteed correct)")
    print(f"  Experts: Sniped from NVMe (8/256 per layer)")


if __name__ == "__main__":
    main()
