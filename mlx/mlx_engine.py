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
import re
import sys
import os
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading

# Model registry
MODELS = {
    "9b": "mlx-community/Qwen3.5-9B-MLX-4bit",
    "35b": "mlx-community/Qwen3.5-35B-A3B-4bit",
}

STOP_STRINGS = ["\x3c|endoftext|\x3e", "\x3c|im_end|\x3e", "\x3c|im_start|\x3e"]

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


def clean_response(text):
    """Strip special tokens and thinking tags from response."""
    for stop in STOP_STRINGS:
        if stop in text:
            text = text[:text.index(stop)]

    if "\x3cthink\x3e" in text:
        text = re.sub(r'\x3cthink\x3e.*?\x3c/think\x3e', '', text, flags=re.DOTALL)

    return text.strip()


def generate(messages, max_tokens=2000, temperature=0.7):
    """Generate a complete response (non-streaming)."""
    from mlx_lm import generate as mlx_generate

    prompt = format_chat(messages)

    t0 = time.time()
    response = mlx_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens,
    )
    elapsed = time.time() - t0

    response = clean_response(response)
    tokens = len(tokenizer.encode(response)) if response else 0
    speed = tokens / elapsed if elapsed > 0 else 0

    return {
        "content": response,
        "tokens": tokens,
        "elapsed": elapsed,
        "speed": speed,
    }


def generate_stream(messages, max_tokens=2000, temperature=0.7):
    """Stream tokens one at a time using mlx_lm.stream_generate."""
    from mlx_lm import stream_generate

    prompt = format_chat(messages)

    # Track state for filtering thinking tags
    in_think = False
    think_done = False
    buffer = ""

    for resp in stream_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens,
    ):
        text = resp.text
        token_id = resp.token
        finish = resp.finish_reason

        # Skip empty text
        if not text:
            if finish:
                yield "", finish, resp
            continue

        # Filter out thinking content
        if not think_done:
            buffer += text
            # Check if we've seen the end of thinking
            if "\x3c/think\x3e" in buffer:
                # Extract content after </think>
                after = buffer.split("\x3c/think\x3e", 1)[-1]
                think_done = True
                buffer = ""
                if after.strip():
                    yield after, finish, resp
                continue
            # Still in thinking region, don't yield
            if "\x3cthink\x3e" in buffer or in_think:
                in_think = True
                continue
            # No thinking tags seen, yield normally
            think_done = True
            yield buffer, finish, resp
            buffer = ""
            continue

        # Check for stop tokens
        skip = False
        for stop in STOP_STRINGS:
            if stop in text:
                text = text[:text.index(stop)]
                skip = True

        if text:
            yield text, finish, resp

        if skip or finish:
            yield "", finish or "stop", resp
            return


def format_chat(messages):
    """Format chat messages into a prompt string."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"\x3c|im_start|\x3esystem\n{content}\x3c|im_end|\x3e")
        elif role == "user":
            parts.append(f"\x3c|im_start|\x3euser\n{content}\x3c|im_end|\x3e")
        elif role == "assistant":
            parts.append(f"\x3c|im_start|\x3eassistant\n{content}\x3c|im_end|\x3e")
    # Add empty thinking block to skip reasoning mode
    parts.append("\x3c|im_start|\x3eassistant\n\x3cthink\x3e\n\n\x3c/think\x3e\n\n")
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
        tokens = mx.array(prompt_tokens) if not isinstance(prompt_tokens, mx.array) else prompt_tokens
        logits = model(tokens[None], cache=cache)
        mx.eval(logits)

    # Save
    meta = {"name": name, "model": model_name, "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
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
            self._send_json({
                "model_alias": f"Qwen3.5-{model_name.upper()}-MLX",
                "model_path": MODELS.get(model_name, ""),
            })
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
            self._send_json({"ok": True, "load_time": result["load_time"], "metadata": result["metadata"]})
        else:
            self._send_json({"ok": False, "error": f"Context not found: {name}"}, status=404)

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
        self._send_json({"local": local, "remote": remote, "r2_configured": is_configured()})

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
            self._handle_chat_normal(messages, max_tokens, temperature)

    def _handle_chat_normal(self, messages, max_tokens, temperature):
        """Non-streaming: return complete response as single JSON."""
        try:
            result = generate(messages, max_tokens, temperature)

            response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": result["content"],
                    },
                    "finish_reason": "stop",
                }],
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
        """Streaming: return SSE (Server-Sent Events) format for agent.py."""
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            for text, finish, resp in generate_stream(messages, max_tokens, temperature):
                chunk = {
                    "choices": [{
                        "delta": {},
                        "finish_reason": finish,
                    }]
                }

                if text:
                    chunk["choices"][0]["delta"]["content"] = text

                if finish:
                    chunk["choices"][0]["delta"] = {}
                    chunk["choices"][0]["finish_reason"] = finish

                line = f"data: {json.dumps(chunk)}\n\n"
                self.wfile.write(line.encode())
                self.wfile.flush()

            # Send [DONE] marker
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        except Exception as e:
            try:
                error_chunk = f"data: {json.dumps({'error': str(e)})}\n\n"
                self.wfile.write(error_chunk.encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
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
    parser.add_argument("--model", default="9b", choices=list(MODELS.keys()),
                       help="Model to load (default: 9b)")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--save-context", help="Save KV cache after loading")
    parser.add_argument("--load-context", help="Load KV cache before serving")
    args = parser.parse_args()

    print(f"\n  \U0001f34e mac code MLX engine")
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
            print(f"  Loaded context: {args.load_context} ({meta.get('num_layers', '?')} layers)")
        else:
            print(f"  Context not found: {args.load_context}")

    # Start server
    print(f"  Server: http://localhost:{args.port}")
    print(f"  KV cache: persistent context enabled")
    print(f"  Streaming: enabled")
    print()

    server = HTTPServer(("127.0.0.1", args.port), APIHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
