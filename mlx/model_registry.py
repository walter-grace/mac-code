#!/usr/bin/env python3
"""
Model registry for mac code.

Discovers and manages GGUF models from ~/models/ (and additional paths).
Replaces the hardcoded MODELS dict with a user-configurable, auto-discovered
registry stored at ~/.mac-code/models.json.

Fallback: if no config exists, ships with the default Qwen3.5 models.
"""

import json
import os
import re
import struct
import sys
from pathlib import Path
from datetime import datetime

CONFIG_DIR = Path.home() / ".mac-code"
CONFIG_FILE = CONFIG_DIR / "models.json"
DEFAULT_MODELS_DIR = Path.home() / "models"

KNOWN_FAMILIES = {
    "qwen": {
        "arch": "qwen2",
        "ctx_default": 32768,
        "moe_sizes": {"35b-a3b", "30b-a3b", "235b-a22b", "72b-a3b"},
    },
    "llama": {"arch": "llama", "ctx_default": 8192},
    "mistral": {"arch": "mistral", "ctx_default": 32768},
    "phi": {"arch": "phi3", "ctx_default": 4096},
    "gemma": {"arch": "gemma", "ctx_default": 8192},
    "deepseek": {"arch": "deepseek2", "ctx_default": 4096, "moe_sizes": {"671b"}},
    "yi": {"arch": "yi", "ctx_default": 4096},
    "starcoder": {"arch": "starcoder", "ctx_default": 8192},
    "codestral": {"arch": "mistral", "ctx_default": 32768},
    "command": {"arch": "command-r", "ctx_default": 8192},
    "granite": {"arch": "granite", "ctx_default": 8192},
    "smollm": {"arch": "llama", "ctx_default": 8192},
    "tinyllama": {"arch": "llama", "ctx_default": 2048},
}

DEFAULT_MODELS = {
    "9b": {
        "path": str(DEFAULT_MODELS_DIR / "Qwen3.5-9B-Q4_K_M.gguf"),
        "ctx": 32768,
        "name": "Qwen3.5-9B",
        "detail": "8.95B dense · Q4_K_M · 32K ctx",
        "flags": "--flash-attn on --n-gpu-layers 99 --reasoning off -t 4",
        "good_for": "tool calling, long conversations, agent tasks",
    },
    "35b": {
        "path": str(DEFAULT_MODELS_DIR / "Qwen3.5-35B-A3B-UD-IQ2_M.gguf"),
        "ctx": 8192,
        "name": "Qwen3.5-35B-A3B",
        "detail": "MoE 34.7B · 3B active · IQ2_M · 8K ctx",
        "flags": "--flash-attn on --n-gpu-layers 99 --reasoning off -np 1 -t 4",
        "good_for": "reasoning, math, knowledge, fast answers",
    },
}


def _parse_gguf_metadata(path):
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            meta = {}
            for _ in range(n_kv):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                vtype = struct.unpack("<I", f.read(4))[0]

                if vtype == 0:
                    val = struct.unpack("<B", f.read(1))[0]
                elif vtype == 1:
                    val = struct.unpack("<b", f.read(1))[0]
                elif vtype == 2:
                    val = struct.unpack("<H", f.read(2))[0]
                elif vtype == 3:
                    val = struct.unpack("<h", f.read(2))[0]
                elif vtype == 4:
                    val = struct.unpack("<I", f.read(4))[0]
                elif vtype == 5:
                    val = struct.unpack("<i", f.read(4))[0]
                elif vtype == 6:
                    val = struct.unpack("<Q", f.read(8))[0]
                elif vtype == 7:
                    val = struct.unpack("<q", f.read(8))[0]
                elif vtype == 8:
                    val = struct.unpack("<f", f.read(4))[0]
                elif vtype == 9:
                    val = struct.unpack("<d", f.read(8))[0]
                elif vtype == 10:
                    slen = struct.unpack("<Q", f.read(8))[0]
                    val = f.read(slen).decode("utf-8", errors="replace")
                elif vtype == 11:
                    arr_type = struct.unpack("<I", f.read(4))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    if arr_type == 10:
                        val = []
                        for _ in range(arr_len):
                            slen = struct.unpack("<Q", f.read(8))[0]
                            val.append(f.read(slen).decode("utf-8", errors="replace"))
                    else:
                        type_sizes = {
                            0: 1,
                            1: 1,
                            2: 2,
                            3: 2,
                            4: 4,
                            5: 4,
                            6: 8,
                            7: 8,
                            8: 4,
                            9: 8,
                        }
                        elem_size = type_sizes.get(arr_type, 4)
                        f.read(arr_len * elem_size)
                        val = None
                else:
                    val = None

                meta[key] = val

            return meta
    except Exception:
        return None


def _detect_model_info(filepath):
    """Detect model name, context, MoE status from GGUF metadata and filename."""
    path = Path(filepath)
    filename = path.stem.lower()
    size_gb = path.stat().st_size / (1024**3) if path.exists() else 0
    gguf_meta = _parse_gguf_metadata(str(path))

    name = path.stem
    detail_parts = []
    is_moe = False
    ctx = 8192

    if gguf_meta:
        arch = gguf_meta.get("general.architecture", "")
        model_name_meta = gguf_meta.get("general.name", "")
        n_embd = gguf_meta.get(f"{arch}.embedding_length", 0)
        n_layers = gguf_meta.get(f"{arch}.block_count", 0)
        n_heads = gguf_meta.get(f"{arch}.attention.head_count", 0)
        ctx_train = gguf_meta.get(f"{arch}.context_length", 0)
        expert_count = gguf_meta.get(f"{arch}.expert_count", 0)
        expert_used = gguf_meta.get(f"{arch}.expert_used_count", 0)

        if model_name_meta:
            name = model_name_meta

        if expert_count and expert_count > 1:
            is_moe = True
            detail_parts.append(f"MoE {expert_count}e{expert_used}")

        if ctx_train:
            ctx = min(ctx_train, 65536)
        elif n_embd:
            ctx = min(n_embd * 4, 32768)

    quant_match = re.search(
        r"(Q[0-4]_K_M|Q[0-5]_K_S|Q4_0|Q5_0|Q5_1|Q8_0|IQ[0-9]_[A-Z]+|F16|F32|BF16)",
        filename,
        re.IGNORECASE,
    )
    quant = quant_match.group(1).upper() if quant_match else "quantized"

    param_match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", filename)
    if param_match:
        detail_parts.insert(0, f"{param_match.group(1)}B")

    detail_parts.append(quant)
    if size_gb > 0:
        detail_parts.append(f"{size_gb:.1f}GB")

    ctx_label = f"{ctx // 1024}K" if ctx >= 1024 else str(ctx)
    detail_parts.append(f"{ctx_label} ctx")

    detail = " · ".join(detail_parts) if detail_parts else "local model"

    flags = "--flash-attn on --n-gpu-layers 99 --reasoning off -t 4"
    if is_moe:
        flags += " -np 1"

    return {
        "name": name,
        "path": str(path),
        "ctx": ctx,
        "flags": flags,
        "detail": detail,
        "good_for": "",
    }


def _alias_from_name(name):
    """Generate a short alias from a model name: Qwen3.5-9B-Q4_K_M -> 9b"""
    name = name.lower()

    moe_match = re.search(r"(\d+)b-a(\d+)b", name)
    if moe_match:
        return f"{moe_match.group(1)}b-a{moe_match.group(2)}b"

    param_match = re.search(r"(\d+(?:\.\d+)?)\s*b", name)
    if param_match:
        size = param_match.group(1).replace(".", "p")
        return f"{size}b"

    family_match = re.search(
        r"(qwen|llama|mistral|phi|gemma|deepseek|yi|codestral|command|granite|smollm|tinyllama)",
        name,
    )
    if family_match:
        return family_match.group(1)

    return name[:12].replace(" ", "-")


class ModelRegistry:
    def __init__(self):
        self._models = {}
        self._load()

    def _load(self):
        self._models = {}

        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    data = json.load(f)
                self._models = data.get("models", {})
                return
            except Exception:
                pass

        for alias, cfg in DEFAULT_MODELS.items():
            self._models[alias] = dict(cfg)

    def _save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(
                {"models": self._models, "updated": datetime.now().isoformat()},
                f,
                indent=2,
            )

    def get(self, alias):
        return self._models.get(alias)

    def list_models(self):
        result = {}
        for alias, cfg in sorted(self._models.items()):
            entry = dict(cfg)
            entry["alias"] = alias
            entry["exists"] = os.path.exists(cfg["path"])
            entry["size_gb"] = (
                round(Path(cfg["path"]).stat().st_size / (1024**3), 2)
                if entry["exists"]
                else 0
            )
            result[alias] = entry
        return result

    def add(self, path, alias=None):
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            return None, f"File not found: {path}"
        if not path.lower().endswith(".gguf"):
            return None, f"Not a GGUF file: {path}"

        info = _detect_model_info(path)
        if alias is None:
            alias = _alias_from_name(info["name"])

        if alias in self._models:
            existing = self._models[alias]["name"]
            return (
                None,
                f"Alias '{alias}' already taken by {existing}. Choose a different alias.",
            )

        self._models[alias] = info
        self._save()
        return alias, f"Added {info['name']} as '{alias}' ({info['detail']})"

    def remove(self, alias):
        if alias not in self._models:
            return False, f"Model '{alias}' not found"
        name = self._models[alias]["name"]
        del self._models[alias]
        self._save()
        return True, f"Removed '{name}' (alias: {alias})"

    def scan(self, directory=None):
        """Scan a directory for GGUF files and add any new ones."""
        scan_dir = Path(directory) if directory else DEFAULT_MODELS_DIR
        if not scan_dir.exists():
            return [], f"Directory not found: {scan_dir}"

        existing_paths = {cfg["path"] for cfg in self._models.values()}
        added = []

        for gguf in sorted(scan_dir.glob("*.gguf")):
            abs_path = str(gguf.resolve())
            if abs_path in existing_paths:
                continue

            alias, msg = self.add(abs_path)
            if alias:
                added.append((alias, self._models[alias]["name"]))

        if added:
            return added, f"Found {len(added)} new model(s)"
        return [], "No new models found"

    def reset(self):
        self._models = {}
        for alias, cfg in DEFAULT_MODELS.items():
            self._models[alias] = dict(cfg)
        self._save()
        return list(self._models.keys())


registry = ModelRegistry()


def get_models():
    return registry.list_models()


def get_model(alias):
    return registry.get(alias)


def add_model(path, alias=None):
    return registry.add(path, alias)


def remove_model(alias):
    return registry.remove(alias)


def scan_models(directory=None):
    return registry.scan(directory)


def reset_models():
    return registry.reset()
