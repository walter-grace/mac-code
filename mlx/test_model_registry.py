#!/usr/bin/env python3
"""
Tests for model_registry.py.

Validates:
1. Default models load correctly when no config exists
2. Auto-discovery scans ~/models/ for GGUF files
3. Add/remove/scan model operations
4. Alias generation from model names
5. GGUF metadata parsing
6. Config persistence to ~/.mac-code/models.json
"""

import json
import os
import sys
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))

import model_registry
from model_registry import (
    ModelRegistry,
    _alias_from_name,
    _detect_model_info,
    DEFAULT_MODELS,
    CONFIG_FILE,
)


def _create_fake_gguf(
    path,
    name="test-model",
    arch="llama",
    ctx_length=8192,
    n_embd=4096,
    n_layers=32,
    n_heads=32,
    expert_count=0,
    expert_used=0,
):
    """Create a minimal valid GGUF file with metadata."""
    kv_pairs = [
        ("general.name", 10, name),
        ("general.architecture", 10, arch),
        (f"{arch}.context_length", 4, ctx_length),
        (f"{arch}.embedding_length", 4, n_embd),
        (f"{arch}.block_count", 4, n_layers),
        (f"{arch}.attention.head_count", 4, n_heads),
    ]
    if expert_count > 0:
        kv_pairs.append((f"{arch}.expert_count", 4, expert_count))
        kv_pairs.append((f"{arch}.expert_used_count", 4, expert_used))

    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 0))
        f.write(struct.pack("<Q", len(kv_pairs)))

        for key, vtype, value in kv_pairs:
            f.write(struct.pack("<Q", len(key)))
            f.write(key.encode("utf-8"))
            f.write(struct.pack("<I", vtype))
            if vtype == 4:
                f.write(struct.pack("<I", value))
            elif vtype == 10:
                f.write(struct.pack("<Q", len(value)))
                f.write(value.encode("utf-8"))


class TestAliasGeneration(unittest.TestCase):
    def test_qwen_with_params(self):
        self.assertEqual(_alias_from_name("Qwen3.5-9B-Q4_K_M"), "9b")

    def test_qwen_moe(self):
        self.assertEqual(_alias_from_name("Qwen3.5-35B-A3B-IQ2_M"), "35b-a3b")

    def test_llama_model(self):
        self.assertEqual(_alias_from_name("Llama-3.1-8B-Instruct-Q4_K_M"), "8b")

    def test_mistral_model(self):
        self.assertEqual(_alias_from_name("Mistral-7B-Instruct-Q4_0"), "7b")

    def test_family_fallback(self):
        self.assertEqual(_alias_from_name("gemma-2-it-Q4_K_M"), "gemma")

    def test_unknown_model(self):
        alias = _alias_from_name("my-custom-model")
        self.assertEqual(alias, "my-custom-mo")


class TestDefaultModels(unittest.TestCase):
    def test_defaults_exist(self):
        self.assertIn("9b", DEFAULT_MODELS)
        self.assertIn("35b", DEFAULT_MODELS)

    def test_defaults_have_required_fields(self):
        for alias, cfg in DEFAULT_MODELS.items():
            self.assertIn("path", cfg)
            self.assertIn("ctx", cfg)
            self.assertIn("name", cfg)
            self.assertIn("flags", cfg)
            self.assertIn("detail", cfg)


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_file = Path(self.tmpdir) / "models.json"
        self.models_dir = Path(self.tmpdir) / "models"
        self.models_dir.mkdir()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_registry(self):
        with (
            patch.object(model_registry, "CONFIG_FILE", self.config_file),
            patch.object(model_registry, "DEFAULT_MODELS_DIR", self.models_dir),
        ):
            return ModelRegistry()

    def test_loads_defaults_when_no_config(self):
        reg = self._make_registry()
        models = reg.list_models()
        self.assertIn("9b", models)
        self.assertIn("35b", models)

    def test_persists_after_add(self):
        reg = self._make_registry()
        gguf = self.models_dir / "test.gguf"
        _create_fake_gguf(str(gguf))

        with (
            patch.object(model_registry, "CONFIG_FILE", self.config_file),
            patch.object(model_registry, "DEFAULT_MODELS_DIR", self.models_dir),
        ):
            alias, msg = reg.add(str(gguf), "test")

        self.assertIsNotNone(alias)
        self.assertTrue(self.config_file.exists())

        data = json.loads(self.config_file.read_text())
        self.assertIn("test", data["models"])

    def test_add_nonexistent_file(self):
        reg = self._make_registry()
        alias, msg = reg.add("/nonexistent/model.gguf")
        self.assertIsNone(alias)
        self.assertIn("not found", msg.lower())

    def test_add_non_gguf(self):
        reg = self._make_registry()
        fake = self.models_dir / "model.bin"
        fake.write_bytes(b"not a gguf")
        alias, msg = reg.add(str(fake))
        self.assertIsNone(alias)
        self.assertIn("gguf", msg.lower())

    def test_add_duplicate_alias(self):
        reg = self._make_registry()
        gguf = self.models_dir / "test.gguf"
        _create_fake_gguf(str(gguf))
        with (
            patch.object(model_registry, "CONFIG_FILE", self.config_file),
            patch.object(model_registry, "DEFAULT_MODELS_DIR", self.models_dir),
        ):
            reg.add(str(gguf), "test")
            alias, msg = reg.add(str(gguf), "test")
        self.assertIsNone(alias)
        self.assertIn("already taken", msg)

    def test_remove_model(self):
        reg = self._make_registry()
        gguf = self.models_dir / "test.gguf"
        _create_fake_gguf(str(gguf))
        with patch.object(model_registry, "CONFIG_FILE", self.config_file):
            reg.add(str(gguf), "mytest")
            ok, msg = reg.remove("mytest")
        self.assertTrue(ok)
        self.assertIsNone(reg.get("mytest"))

    def test_remove_nonexistent(self):
        reg = self._make_registry()
        ok, msg = reg.remove("nonexistent")
        self.assertFalse(ok)

    def test_scan_finds_new_models(self):
        reg = self._make_registry()
        gguf1 = self.models_dir / "Llama-3.1-8B-Q4_K_M.gguf"
        gguf2 = self.models_dir / "Phi-3.5-mini-instruct-Q4_K_M.gguf"
        _create_fake_gguf(str(gguf1), name="Llama-3.1-8B")
        _create_fake_gguf(str(gguf2), name="Phi-3.5-mini")

        with (
            patch.object(model_registry, "CONFIG_FILE", self.config_file),
            patch.object(model_registry, "DEFAULT_MODELS_DIR", self.models_dir),
        ):
            added, msg = reg.scan()

        self.assertEqual(len(added), 2)
        self.assertIn("Found 2", msg)

    def test_scan_skips_existing(self):
        reg = self._make_registry()
        gguf = self.models_dir / "Qwen3.5-9B-Q4_K_M.gguf"
        _create_fake_gguf(str(gguf), name="Qwen3.5-9B")

        defaults = {
            "9b": {
                "path": str(gguf),
                "ctx": 32768,
                "name": "Qwen3.5-9B",
                "detail": "test",
                "flags": "",
                "good_for": "",
            }
        }
        with (
            patch.object(model_registry, "DEFAULT_MODELS", defaults),
            patch.object(model_registry, "CONFIG_FILE", self.config_file),
            patch.object(model_registry, "DEFAULT_MODELS_DIR", self.models_dir),
        ):
            reg2 = ModelRegistry()
            added, msg = reg2.scan()

        self.assertEqual(len(added), 0)

    def test_list_models_includes_exists_flag(self):
        reg = self._make_registry()
        models = reg.list_models()
        for alias, info in models.items():
            self.assertIn("exists", info)
            self.assertIn("alias", info)

    def test_reset_restores_defaults(self):
        reg = self._make_registry()
        gguf = self.models_dir / "extra.gguf"
        _create_fake_gguf(str(gguf))
        with (
            patch.object(model_registry, "CONFIG_FILE", self.config_file),
            patch.object(model_registry, "DEFAULT_MODELS_DIR", self.models_dir),
        ):
            reg.add(str(gguf), "extra")
            aliases = reg.reset()
        self.assertIn("9b", aliases)
        self.assertNotIn("extra", reg._models)


class TestDetectModelInfo(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_detects_from_gguf_metadata(self):
        gguf = Path(self.tmpdir) / "Llama-3-8B-Q4_K_M.gguf"
        _create_fake_gguf(str(gguf), name="Llama 3 8B", ctx_length=8192)
        info = _detect_model_info(str(gguf))
        self.assertEqual(info["name"], "Llama 3 8B")
        self.assertEqual(info["ctx"], 8192)

    def test_detects_moe(self):
        gguf = Path(self.tmpdir) / "Qwen3.5-35B-A3B-Q4.gguf"
        _create_fake_gguf(
            str(gguf),
            name="Qwen3.5-35B",
            arch="qwen2",
            expert_count=128,
            expert_used=8,
            ctx_length=4096,
        )
        info = _detect_model_info(str(gguf))
        self.assertIn("MoE", info["detail"])
        self.assertIn("-np 1", info["flags"])

    def test_nonexistent_file(self):
        info = _detect_model_info("/nonexistent/file.gguf")
        self.assertIsNotNone(info)
        self.assertIn("path", info)


class TestGGUFMetadataParsing(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reads_string_metadata(self):
        gguf = Path(self.tmpdir) / "test.gguf"
        _create_fake_gguf(str(gguf), name="TestModel")
        meta = model_registry._parse_gguf_metadata(str(gguf))
        self.assertEqual(meta["general.name"], "TestModel")

    def test_reads_int_metadata(self):
        gguf = Path(self.tmpdir) / "test.gguf"
        _create_fake_gguf(str(gguf), ctx_length=32768)
        meta = model_registry._parse_gguf_metadata(str(gguf))
        self.assertEqual(meta["llama.context_length"], 32768)

    def test_rejects_non_gguf(self):
        fake = Path(self.tmpdir) / "fake.bin"
        fake.write_bytes(b"NOT_GGUF_DATA")
        result = model_registry._parse_gguf_metadata(str(fake))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
