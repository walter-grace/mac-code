#!/usr/bin/env python3
"""
MLX inference engine with KV cache access.
Drop-in replacement for llama.cpp with persistent context support.

Usage:
    python3 mlx_engine.py                    # Start server on :8000
    python3 mlx_engine.py --model 35b        # Use 35B MoE
    python3 mlx_engine.py --save-context foo  # Save KV after processing
    python3 mlx_engine.py --load-context foo  # Load KV before serving
"""

import argparse
import json
import sys
import os
import re
import time
import uuid
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading

# Model registry
MODELS = {
    "9b": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "35b": "mlx-community/Qwen3.5-35B-A3B-4bit",
}

# Global state
model = None
tokenizer = None
model_name = None


def load_model(model_key="9b"):
    """Load an MLX model."""
    global model, tokenizer, model_name

    try:
        from mlx_lm import load
    except ImportError:
        print("MLX not installed. Run: pip3 install mlx-lm")
        sys.exit(1)

    model_id = MODELS.get(model_key, model_key)
    print(f"  Loading {model_id}...")

    model, tokenizer = load(model_id)
    model_name = model_key

    print(f"  Model loaded: {model_id}")
    return model, tokenizer


STOP_TOKENS = ["</think", "<|im_end|>", "<|im_start|>"]


def _clean_response(text):
    for stop in STOP_TOKENS:
        if stop in text:
            text = text[: text.index(stop)]
    text = re.sub(r"_latency.*?</low>", "", text, flags=re.DOTALL)
    return text.strip()


def generate(messages, max_tokens=2000, temperature=0.7, stream=False):
    """Generate a response from the model. If stream=True, yields tokens."""
    prompt = format_chat(messages)

    if stream:
        return _generate_stream(prompt, max_tokens, temperature)

    from mlx_lm import generate as mlx_generate

    t0 = time.time()
    response = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    elapsed = time.time() - t0

    response = _clean_response(response)
    tokens = len(tokenizer.encode(response)) if response else 0
    speed = tokens / elapsed if elapsed > 0 else 0

    return {
        "content": response,
        "tokens": tokens,
        "elapsed": elapsed,
        "speed": speed,
    }


def _generate_stream(prompt, max_tokens, temperature):
    from mlx_lm import stream_generate as mlx_stream_generate

    t0 = time.time()
    full_text = ""
    token_count = 0
    in_thinking = False
    thinking_ended = False

    for chunk in mlx_stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, temp=temperature
    ):
        text = chunk.text if hasattr(chunk, "text") else str(chunk)
        token_count += 1

        if not thinking_ended:
            if not in_thinking:
                if "_latency" in text or full_text.endswith(""):
                    in_thinking = True
                    full_text += text
                    continue
                if "" in text or "" in text:
                    thinking_ended = True
                    full_text = ""
                    continue
                in_thinking = True
                full_text += text
                continue
            else:
                full_text += text
                if "" in full_text or "" in full_text:
                    thinking_ended = True
                    remaining = re.sub(
                        r".*?(?:|)", "", full_text, count=1, flags=re.DOTALL
                    )
                    full_text = ""
                    if remaining:
                        cleaned = _clean_response(remaining)
                        if cleaned:
                            yield cleaned
                continue

        for stop in STOP_TOKENS:
            if stop in text:
                text = text[: text.index(stop)]
                if text:
                    yield text
                elapsed = time.time() - t0
                return

        if text:
            yield text

    elapsed = time.time() - t0


def format_chat(messages):
    """Format chat messages into a prompt string."""
    # Use Qwen chat template
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    # Add empty thinking block to skip reasoning mode
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "\n".join(parts)


def save_context(name, prompt_tokens=None, metadata=None):
    """Save current KV cache to disk (and optionally R2)."""
    from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache
    from pathlib import Path
    import json as _json

    cache_dir = Path.home() / ".mac-code" / "kv-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = str(cache_dir / f"{name}.safetensors")

    # Create and fill cache
    cache = make_prompt_cache(model)

    if prompt_tokens is not None:
        import mlx.core as mx

        tokens = (
            mx.array(prompt_tokens)
            if not isinstance(prompt_tokens, mx.array)
            else prompt_tokens
        )
        logits = model(tokens[None], cache=cache)
        mx.eval(logits)

    # Save
    meta = {
        "name": name,
        "model": model_name,
        "saved": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if metadata:
        meta.update(metadata)

    save_prompt_cache(cache_path, cache, metadata={k: str(v) for k, v in meta.items()})

    # Save metadata separately for easy reading
    meta_path = cache_dir / f"{name}.meta.json"
    meta["size_mb"] = os.path.getsize(cache_path) / (1024 * 1024)
    with open(meta_path, "w") as f:
        _json.dump(meta, f, indent=2)

    return meta


def load_context(name):
    """Load KV cache from disk into the model."""
    from mlx_lm.models.cache import load_prompt_cache
    from pathlib import Path

    cache_dir = Path.home() / ".mac-code" / "kv-cache"
    cache_path = str(cache_dir / f"{name}.safetensors")

    if not os.path.exists(cache_path):
        return None

    t0 = time.time()
    cache, meta = load_prompt_cache(cache_path, return_metadata=True)
    load_time = time.time() - t0

    return {
        "cache": cache,
        "metadata": meta,
        "load_time": load_time,
    }


class APIHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible API handler."""

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/v1/chat/completions":
            self._handle_chat()
        elif path == "/v1/context/save":
            self._handle_save_context()
        elif path == "/v1/context/load":
            self._handle_load_context()
        elif path == "/v1/context/upload":
            self._handle_upload_context()
        elif path == "/v1/context/download":
            self._handle_download_context()
        else:
            self.send_error(404)

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            self._send_json({"status": "ok", "model": model_name})
        elif path == "/props":
            self._send_json(
                {
                    "model_alias": f"Qwen3.5-{model_name}-MLX",
                    "model_path": MODELS.get(model_name, ""),
                }
            )
        elif path == "/v1/models":
            self._handle_list_models()
        elif path == "/v1/context/list":
            self._handle_list_contexts()
        else:
            self.send_error(404)

    def _handle_save_context(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        name = body.get("name", f"ctx-{int(time.time())}")
        prompt = body.get("prompt", "")

        tokens = tokenizer.encode(prompt) if prompt else None
        meta = save_context(name, prompt_tokens=tokens, metadata=body.get("metadata"))
        self._send_json({"ok": True, **meta})

    def _handle_load_context(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        name = body.get("name", "")

        result = load_context(name)
        if result:
            self._send_json(
                {
                    "ok": True,
                    "load_time": result["load_time"],
                    "metadata": result["metadata"],
                }
            )
        else:
            self._send_json(
                {"ok": False, "error": f"Context not found: {name}"}, status=404
            )

    def _handle_upload_context(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        name = body.get("name", "")

        from r2_store import upload_context

        result = upload_context(name)
        self._send_json(result)

    def _handle_download_context(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        name = body.get("name", "")

        from r2_store import download_context

        result = download_context(name)
        self._send_json(result)

    def _handle_list_contexts(self):
        from r2_store import list_local_contexts, list_remote_contexts, is_configured

        local = list_local_contexts()
        remote = list_remote_contexts() if is_configured() else []
        self._send_json(
            {"local": local, "remote": remote, "r2_configured": is_configured()}
        )

    def _handle_list_models(self):
        current = model_name or "9b"
        model_id = MODELS.get(current, current)
        self._send_json(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    }
                ],
            }
        )

    def _handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 2000)
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", False)

        if stream:
            self._handle_chat_stream(messages, max_tokens, temperature)
        else:
            self._handle_chat_sync(messages, max_tokens, temperature)

    def _handle_chat_sync(self, messages, max_tokens, temperature):
        try:
            result = generate(messages, max_tokens, temperature)

            response = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": result["content"],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": result["tokens"],
                },
                "timings": {
                    "predicted_per_second": result["speed"],
                    "predicted_ms": result["elapsed"] * 1000,
                },
            }

            self._send_json(response)

        except Exception as e:
            self._send_json({"error": {"message": str(e)}}, status=500)

    def _handle_chat_stream(self, messages, max_tokens, temperature):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            t0 = time.time()
            token_count = 0
            full_content = ""

            for text_chunk in generate(messages, max_tokens, temperature, stream=True):
                token_count += 1
                full_content += text_chunk

                sse_data = json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name or "local",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text_chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                self.wfile.write(f"data: {sse_data}\n\n".encode())
                self.wfile.flush()

            elapsed = time.time() - t0
            speed = token_count / elapsed if elapsed > 0 else 0

            final_chunk = json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name or "local",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "timings": {
                        "predicted_per_second": speed,
                        "predicted_ms": elapsed * 1000,
                    },
                    "usage": {
                        "completion_tokens": token_count,
                    },
                }
            )
            self.wfile.write(f"data: {final_chunk}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        except Exception as e:
            error_data = json.dumps({"error": {"message": str(e)}})
            try:
                self.wfile.write(f"data: {error_data}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError:
                pass

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        msg = format % args
        if "favicon" not in msg:
            print(f"  {msg}")


def main():
    parser = argparse.ArgumentParser(description="MLX engine for mac code")
    parser.add_argument(
        "--model",
        default="9b",
        choices=list(MODELS.keys()),
        help="Model to load (default: 9b)",
    )
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--save-context", help="Save KV cache after loading")
    parser.add_argument("--load-context", help="Load KV cache before serving")
    args = parser.parse_args()

    print(f"\n  🍎 mac code MLX engine")
    print(f"  Model: {MODELS[args.model]}")
    print(f"  Port:  {args.port}")
    print()

    # Load model
    load_model(args.model)

    # Load context from disk if requested
    if args.load_context:
        from kv_cache import load_kv_cache

        tensors, meta = load_kv_cache(args.load_context)
        if tensors:
            set_kv_cache(tensors)
            print(
                f"  Loaded context: {args.load_context} ({meta.get('num_layers', '?')} layers)"
            )
        else:
            print(f"  Context not found: {args.load_context}")

    # Start server
    print(f"  Server: http://localhost:{args.port}")
    print(f"  KV cache: persistent context enabled")
    print()

    server = HTTPServer(("127.0.0.1", args.port), APIHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
