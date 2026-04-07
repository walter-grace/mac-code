#!/usr/bin/env python3
"""
Vision-aware Gemma 4 sniper for single-machine inference.

Combines:
  - mlx-vlm vision_tower + embed_vision (~1.2 GB, no SSD streaming)
  - Our own MoESniperEngineGemma4 (~5.8 GB pinned + experts streamed from SSD)

Total RAM: ~7 GB. Verified working on M4 Mac Mini 16 GB.

Requirements:
  pip install "mlx>=0.31" "mlx-vlm>=0.4" "mlx-lm>=0.31"
  (these need Python 3.11+, install in a venv)
"""

import os
import sys
import json
import time
import glob

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# Gemma 4 special token IDs
IMAGE_TOKEN_ID = 262144  # <image_soft_token> — replaced by vision features
BOI_TOKEN_ID = 255999    # <start_of_image>
EOI_TOKEN_ID = 256000    # <end_of_image>


class VisionGemma4Sniper:
    """Single-machine vision-aware Gemma 4 sniper.

    The vision encoder (~1.2 GB, bfloat16) is loaded from a source 4-bit
    Gemma 4 model directory. The LLM (text + experts) is loaded via our
    existing MoESniperEngineGemma4 from a pre-split streaming directory.
    """

    def __init__(self, stream_dir, source_dir, cache_size=4000):
        """
        Args:
            stream_dir: directory with pinned.safetensors + bin/layer_XX.bin
                       (created by split_gemma4.py)
            source_dir: directory with the original mlx-community/gemma-4-26b-a4b-it-4bit
                       safetensors files (used to extract vision weights)
            cache_size: LRU cache size for the expert reader
        """
        self.stream_dir = os.path.expanduser(stream_dir)
        self.source_dir = os.path.expanduser(source_dir)
        self.cache_size = cache_size

        self.sniper = None
        self.vision_tower = None
        self.embed_vision = None
        self.image_processor = None

    def load(self):
        """Load all three components."""
        # Make sure the existing single-machine sniper code is importable
        SNIPER_PATHS = [
            os.path.expanduser("~/expert-sniper-mlx"),
            os.path.expanduser("~/cli-agent/src"),
        ]
        for p in SNIPER_PATHS:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        try:
            from moe_agent_gemma4 import MoESniperEngineGemma4
        except ImportError as e:
            raise ImportError(
                "Could not import moe_agent_gemma4. The vision engine wraps "
                "the existing single-machine sniper. Make sure these directories "
                "exist and contain the sniper code:\n"
                f"  ~/expert-sniper-mlx/moe_agent_gemma4.py\n"
                f"  ~/cli-agent/src/mlx_expert_sniper/models/gemma4.py\n"
                f"Original error: {e}"
            )
        from mlx_vlm.models.gemma4 import VisionModel, VisionConfig, Gemma4ImageProcessor
        from mlx_vlm.models.gemma4.gemma4 import MultimodalEmbedder

        print("Loading Vision Gemma 4 Sniper...")

        # Load configs
        with open(os.path.join(self.stream_dir, "config.json")) as f:
            text_config = json.load(f)
        with open(os.path.join(self.source_dir, "config.json")) as f:
            source_config = json.load(f)
        vision_config_dict = source_config.get("vision_config", {})
        if not vision_config_dict:
            raise RuntimeError(f"No vision_config in {self.source_dir}/config.json")

        # 1. Sniper LLM (proven 4.15 tok/s on M4)
        print("  [1/4] Sniper LLM...")
        self.sniper = MoESniperEngineGemma4(
            model_dir=self.stream_dir, cache_size=self.cache_size
        )
        self.sniper.load()

        # 2. Vision tower (bfloat16, no quantization)
        print("  [2/4] Vision tower...")
        vc = VisionConfig.from_dict(vision_config_dict)
        self.vision_tower = VisionModel(vc)

        vision_weights = {}
        for sf in sorted(glob.glob(os.path.join(self.source_dir, "model-*.safetensors"))):
            w = mx.load(sf)
            for k, v in w.items():
                if k.startswith("vision_tower."):
                    vision_weights[k[len("vision_tower."):]] = v
            del w
        self.vision_tower.load_weights(list(vision_weights.items()), strict=False)
        mx.eval(self.vision_tower.parameters())

        # 3. embed_vision projector (1 quantized layer)
        print("  [3/4] embed_vision...")
        self.embed_vision = MultimodalEmbedder(
            embedding_dim=vision_config_dict["hidden_size"],
            text_hidden_size=text_config["hidden_size"],
            eps=vision_config_dict.get("rms_norm_eps", 1e-6),
        )
        nn.quantize(self.embed_vision, group_size=64, bits=4)

        ev_weights = {}
        for sf in sorted(glob.glob(os.path.join(self.source_dir, "model-*.safetensors"))):
            w = mx.load(sf)
            for k, v in w.items():
                if k.startswith("embed_vision."):
                    ev_weights[k[len("embed_vision."):]] = v
            del w
        self.embed_vision.load_weights(list(ev_weights.items()), strict=False)
        mx.eval(self.embed_vision.parameters())

        # 4. Image processor
        print("  [4/4] Image processor...")
        self.image_processor = Gemma4ImageProcessor.from_pretrained(self.source_dir)

        total_gb = mx.get_active_memory() / 1e9
        print(f"  Total active memory: {total_gb:.2f} GB")
        print("Ready!")

    def encode_image(self, image_path):
        """Encode an image into projected embeddings.

        Returns:
            (image_features, n_tokens) where image_features is
            mx.array [1, n_tokens, text_hidden_size]
        """
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        processed = self.image_processor(images=[img], return_tensors="mlx")

        # Unpack tuple (dict, list)
        if isinstance(processed, tuple):
            pv_obj = processed[0]
            n_tokens_list = processed[1]
        else:
            pv_obj = processed
            n_tokens_list = None

        if isinstance(pv_obj, dict):
            pixel_values = pv_obj["pixel_values"]
        else:
            pixel_values = pv_obj

        if not isinstance(pixel_values, mx.array):
            pixel_values = mx.array(np.array(pixel_values))

        # Run vision tower → embed_vision (the projector)
        image_features = self.vision_tower(pixel_values)
        image_features = self.embed_vision(image_features)
        mx.eval(image_features)

        n_tokens = n_tokens_list[0] if n_tokens_list else image_features.shape[1]
        return image_features, n_tokens

    def encode_chat(self, prompt, image_path=None):
        """Build the input token sequence with optional image placeholders.

        Returns: (token_ids, image_features_or_None, n_image_tokens)
        """
        NL = chr(10)
        prompt_toks = self.sniper.tokenizer.encode(prompt).ids
        user_toks = self.sniper.tokenizer.encode("user" + NL).ids
        model_toks = self.sniper.tokenizer.encode("model" + NL).ids

        image_features = None
        n_image_tokens = 0

        if image_path:
            image_features, n_image_tokens = self.encode_image(image_path)
            tokens = (
                [2, 105]                              # <bos> <|turn>
                + user_toks                            # user\n
                + [BOI_TOKEN_ID]                       # <start_of_image>
                + [IMAGE_TOKEN_ID] * n_image_tokens    # image placeholders
                + [EOI_TOKEN_ID]                       # <end_of_image>
                + prompt_toks                          # text question
                + [106, 107, 105]                      # <turn|> \n <|turn>
                + model_toks                           # model\n
            )
        else:
            tokens = (
                [2, 105] + user_toks + prompt_toks
                + [106, 107, 105] + model_toks
            )

        return tokens, image_features, n_image_tokens

    def generate(self, prompt, image_path=None, max_tokens=200, temperature=0.7,
                  on_chunk=None):
        """Generate a response, optionally conditioned on an image.

        on_chunk: optional callback that receives streaming text chunks.
        """
        tokens, image_features, n_image_tokens = self.encode_chat(prompt, image_path)

        # Run the prefill (with image embedding injection)
        next_token = self._prefill(tokens, image_features)

        # Stream subsequent tokens
        generated = [next_token]
        eos_set = {1, 106}  # <eos>, <turn|>

        from collections import deque
        recent = deque(maxlen=64)
        recent.append(next_token)
        printed = ""

        for step in range(max_tokens - 1):
            # Decode buffer for streaming
            full = self.sniper.tokenizer.decode(generated)
            new_chunk = full[len(printed):]
            if new_chunk and on_chunk:
                on_chunk(new_chunk)
            printed = full

            if next_token in eos_set:
                break

            input_ids = mx.array([[next_token]])
            logits = self.sniper.forward(input_ids)
            mx.eval(logits)

            # Apply repetition penalty (mild)
            if recent:
                last_logits = logits[0, -1]
                last_np = np.array(last_logits.astype(mx.float32))
                for tid in set(recent):
                    if last_np[tid] > 0:
                        last_np[tid] /= 1.1
                    else:
                        last_np[tid] *= 1.1
                last_logits = mx.array(last_np)
            else:
                last_logits = logits[0, -1]

            if temperature <= 0:
                next_token = int(mx.argmax(last_logits).item())
            else:
                probs = mx.softmax(last_logits / temperature, axis=-1)
                next_token = int(mx.random.categorical(mx.log(probs + 1e-10)).item())

            generated.append(next_token)
            recent.append(next_token)

        # Final flush
        full = self.sniper.tokenizer.decode(generated)
        new_chunk = full[len(printed):]
        if new_chunk and on_chunk:
            on_chunk(new_chunk)

        return full

    def _prefill(self, input_token_ids, image_features):
        """Run the first forward pass with image embeddings injected."""
        from mlx_lm.models.base import create_attention_mask
        from mlx_vlm.models.gemma4.gemma4 import masked_scatter
        from moe_agent_gemma4 import run_expert_ffn

        self.sniper.reset_cache()

        input_ids = mx.array([input_token_ids])

        # Get text embeddings + Gemma sqrt(d) scaling
        h = self.sniper.model.model.embed_tokens(input_ids)
        h = h * (self.sniper.model.args.hidden_size ** 0.5)

        # Inject image embeddings at IMAGE_TOKEN_ID positions
        if image_features is not None:
            image_mask = (input_ids == IMAGE_TOKEN_ID)
            image_feats_flat = image_features.reshape(-1, image_features.shape[-1])
            image_feats_flat = image_feats_flat.astype(h.dtype)
            image_mask_expanded = mx.expand_dims(image_mask, -1)
            image_mask_expanded = mx.broadcast_to(image_mask_expanded, h.shape)
            h = masked_scatter(h, image_mask_expanded, image_feats_flat)

        # Run the LLM forward pass — replicates moe_agent_gemma4 forward but
        # starting from h (post-embedding) instead of input_ids
        mask = create_attention_mask(h, self.sniper.cache[0] if self.sniper.cache else None)

        for i in range(self.sniper.num_layers):
            layer = self.sniper.model.model.layers[i]
            cache_i = self.sniper.cache[i] if self.sniper.cache else None

            residual = h
            h_norm = layer.input_layernorm(h)
            h_attn = layer.self_attn(h_norm, mask=mask, cache=cache_i)
            h_attn = layer.post_attention_layernorm(h_attn)
            h = residual + h_attn
            mx.eval(h)

            residual = h
            h_ff = layer.pre_feedforward_layernorm(h)
            h_ff = layer.mlp(h_ff)

            if layer.enable_moe_block:
                h_dense = layer.post_feedforward_layernorm_1(h_ff)
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

                moe_input = layer.pre_feedforward_layernorm_2(residual_flat)
                mx.eval(moe_input, top_k_indices, top_k_weights)
                top_k_indices_r = top_k_indices.reshape(B, L, -1)
                top_k_weights_r = top_k_weights.reshape(B, L, -1)
                active_ids = list(set(int(e) for e in np.array(top_k_indices_r).flatten()))

                expert_data = self.sniper.reader.get_experts(i, active_ids)
                moe_input_r = moe_input.reshape(B, L, D)
                expert_out = run_expert_ffn(moe_input_r, expert_data, top_k_indices_r, top_k_weights_r)
                h_moe = layer.post_feedforward_layernorm_2(expert_out)
                h_ff = h_dense + h_moe

            h_ff = layer.post_feedforward_layernorm(h_ff)
            h = residual + h_ff
            h = h * layer.layer_scalar
            mx.eval(h)
            mx.clear_cache()

        h = self.sniper.model.model.norm(h)

        if self.sniper.model.args.tie_word_embeddings:
            logits = self.sniper.model.model.embed_tokens.as_linear(h)
        else:
            logits = self.sniper.model.lm_head(h)
        mx.eval(logits)

        # Sample first token
        return int(mx.argmax(logits[0, -1]).item())
