"""
Split Qwen3.5-122B-A10B-4bit into Flash Stream format for PyTorch.

Input: mlx-community/Qwen3.5-122B-A10B-4bit safetensors (69.6 GB)
Output: pinned.safetensors + per-layer expert safetensors

Key discovery: MLX stores all 256 experts STACKED in dim 0:
  switch_mlp.gate_proj.weight: [256, 1024, 384] (uint32)
  switch_mlp.up_proj.weight:   [256, 1024, 384] (uint32)
  switch_mlp.down_proj.weight: [256, 3072, 128] (uint32)

At runtime, we index expert_tensor[expert_ids] to get only active experts.

Usage:
    python3 split_122b.py [--model-dir /path/to/model] [--output-dir /path/to/output]
"""

import os
import sys
import gc
import json
import time
import argparse
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.path.expanduser("~/models/qwen35-122b-a10b-4bit"))
    parser.add_argument("--output-dir", default=os.path.expanduser("~/models/qwen35-122b-stream"))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experts").mkdir(exist_ok=True)

    # Copy config
    config_src = model_dir / "config.json"
    if config_src.exists():
        import shutil
        shutil.copy(config_src, output_dir / "config.json")
        print(f"Copied config.json")

    # Load weight index
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    key_to_shard = index["weight_map"]
    print(f"Total keys: {len(key_to_shard)}")

    # Classify: pinned vs expert (switch_mlp)
    pinned_keys = []
    expert_keys = {}  # layer_idx -> [keys]

    for key in sorted(key_to_shard.keys()):
        if ".switch_mlp." in key:
            # Extract layer index
            # Pattern: language_model.model.layers.{i}.mlp.switch_mlp.{proj}.{component}
            parts = key.split(".layers.")
            if len(parts) > 1:
                layer_idx = int(parts[1].split(".")[0])
                if layer_idx not in expert_keys:
                    expert_keys[layer_idx] = []
                expert_keys[layer_idx].append(key)
            else:
                pinned_keys.append(key)
        else:
            pinned_keys.append(key)

    print(f"Pinned keys: {len(pinned_keys)}")
    print(f"Expert layers: {len(expert_keys)}")
    expert_key_count = sum(len(v) for v in expert_keys.values())
    print(f"Expert keys: {expert_key_count}")
    if expert_keys:
        first_layer = min(expert_keys.keys())
        print(f"Keys per expert layer: {len(expert_keys[first_layer])}")
        for k in expert_keys[first_layer]:
            print(f"  {k}")

    # Save pinned weights
    print(f"\n--- Saving pinned weights ---")
    pinned = {}
    t0 = time.time()
    shards_opened = {}

    for i, key in enumerate(pinned_keys):
        shard_name = key_to_shard[key]
        shard_path = str(model_dir / shard_name)
        if shard_path not in shards_opened:
            shards_opened[shard_path] = safe_open(shard_path, framework="pt", device="cpu")
        pinned[key] = shards_opened[shard_path].get_tensor(key)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pinned_keys)} loaded...")

    pinned_size = sum(t.nbytes for t in pinned.values()) / 1e9
    print(f"  Pinned: {len(pinned)} tensors, {pinned_size:.2f} GB")
    save_file(pinned, str(output_dir / "pinned.safetensors"))
    print(f"  Saved in {time.time()-t0:.1f}s")
    del pinned
    gc.collect()

    # Save expert weights per layer
    print(f"\n--- Saving expert weights per layer ---")
    for layer_idx in sorted(expert_keys.keys()):
        t0 = time.time()
        layer_tensors = {}

        for key in expert_keys[layer_idx]:
            shard_name = key_to_shard[key]
            shard_path = str(model_dir / shard_name)
            if shard_path not in shards_opened:
                shards_opened[shard_path] = safe_open(shard_path, framework="pt", device="cpu")
            tensor = shards_opened[shard_path].get_tensor(key)
            # Simplify key: strip everything before switch_mlp
            short_key = key.split(".switch_mlp.")[1]
            layer_tensors[short_key] = tensor

        layer_size = sum(t.nbytes for t in layer_tensors.values()) / 1e9
        out_path = output_dir / "experts" / f"layer_{layer_idx:02d}.safetensors"
        save_file(layer_tensors, str(out_path))
        elapsed = time.time() - t0
        print(f"  Layer {layer_idx:2d}: {len(layer_tensors)} tensors, {layer_size:.2f} GB [{elapsed:.1f}s]")

        del layer_tensors
        gc.collect()

    shards_opened.clear()

    print(f"\n=== Split complete ===")
    print(f"  Output: {output_dir}")
    print(f"  Pinned: {output_dir / 'pinned.safetensors'}")
    print(f"  Expert layers: {len(expert_keys)}")


if __name__ == "__main__":
    main()
