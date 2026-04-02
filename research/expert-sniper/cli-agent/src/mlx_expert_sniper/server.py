"""
Ollama-compatible HTTP server for Expert Sniper.

Implements /api/tags, /api/chat, /api/generate, /api/version.
Compatible with Open WebUI, Continue.dev, and any Ollama client.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, sys, os, time
import numpy as np

STOP_TOKENS = {"<|im_end|>", "<|endoftext|>", "<|im_start|>"}

_engine = None
_bias = 0.0
_model_dir = None


def _get_engine():
    global _engine, _bias
    if _engine is not None:
        return _engine

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Loading model...", flush=True)

    from .calibrate import load_calibration, auto_size_cache
    cal = load_calibration(_model_dir)
    if cal:
        cache_size = cal["cache_size"]
        _bias = cal["routing_bias"]
        print(f"  Calibration: cache={cache_size}, bias={_bias}")
    else:
        cache_size, _, _ = auto_size_cache(_model_dir)
        print(f"  No calibration, defaults: cache={cache_size}")

    from . import engine as engine_mod
    engine_mod.MODEL_DIR = _model_dir
    from .engine import MoESniperEngine35B
    _engine = MoESniperEngine35B(cache_size=cache_size, enable_prediction=True)
    _engine.load()
    print("  Model loaded.", flush=True)
    return _engine


def _generate_stream(engine, prompt, max_tokens=200):
    """Generator yielding token strings."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from .engine import run_expert_ffn

    engine.reset_cache()
    tok = engine.tokenizer
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    except Exception:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tok.encode(text)
    input_ids = mx.array([tokens])

    def forward(inp):
        h = engine.model.model.embed_tokens(inp)
        fa_mask = create_attention_mask(h, engine.cache[engine.model.model.fa_idx])
        ssm_mask = create_ssm_mask(h, engine.cache[engine.model.model.ssm_idx])
        for i in range(engine.num_layers):
            layer = engine.model.model.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask
            normed = layer.input_layernorm(h)
            if layer.is_linear:
                attn_out = layer.linear_attn(normed, mask=mask, cache=engine.cache[i])
            else:
                attn_out = layer.self_attn(normed, mask=mask, cache=engine.cache[i])
            h = h + attn_out
            mx.eval(h)
            normed = layer.post_attention_layernorm(h)
            raw_logits = layer.mlp.gate(normed)
            if _bias > 0 and engine.reader.lru is not None:
                cached_mask = np.zeros(256, dtype=np.float32)
                for eid in range(256):
                    if engine.reader.lru.get(i, eid) is not None:
                        cached_mask[eid] = _bias
                raw_logits = raw_logits + mx.array(cached_mask).reshape(1, -1)
            gates = mx.softmax(raw_logits, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)
            active_ids = list(set(int(e) for e in np.array(inds).flatten()))
            engine.coact.record_layer(i, active_ids)
            if engine.coact.ready and i + 1 < engine.num_layers:
                predicted = engine.coact.predict_next_layer(i, active_ids, top_k=6)
                if predicted:
                    to_fetch = [eid for eid in predicted
                                if engine.reader.lru and engine.reader.lru.get(i+1, eid) is None]
                    if to_fetch:
                        engine.reader.prefetch_experts(i+1, to_fetch)
            if i + 1 < engine.num_layers:
                engine.reader.prefetch_experts(i+1, active_ids)
            expert_data = engine.reader.get_experts(i, active_ids)
            expert_out = run_expert_ffn(normed, expert_data, inds, scores)
            shared_out = layer.mlp.shared_expert(normed)
            shared_gate = mx.sigmoid(layer.mlp.shared_expert_gate(normed))
            if shared_gate.ndim < shared_out.ndim:
                shared_gate = shared_gate[..., None]
            expert_out = expert_out + shared_gate * shared_out
            h = h + expert_out
            mx.eval(h)
            del expert_data, expert_out, normed, attn_out
            mx.clear_cache()
        engine.coact.end_token()
        h = engine.model.model.norm(h)
        return engine.model.lm_head(h)

    logits = forward(input_ids)
    mx.eval(logits)

    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tid = token.item()
        if tid in (248044, 248045):
            break
        chunk = tok.decode([tid])
        # Filter stop tokens
        if any(st in chunk for st in STOP_TOKENS):
            break
        yield chunk
        logits = forward(token.reshape(1, 1))
        mx.eval(logits)


class OllamaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/tags":
            self._json_response({
                "models": [{
                    "name": "qwen3.5-35b",
                    "model": "qwen3.5-35b",
                    "size": 19500000000,
                    "details": {"family": "qwen3.5", "parameter_size": "35B",
                                "quantization_level": "Q4_0"},
                }]
            })
        elif self.path in ("/api/version", "/"):
            self._json_response({"version": "0.2.0-sniper"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/generate"):
            self.send_response(404)
            self.end_headers()
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_len))

        if "messages" in body:
            prompt = body["messages"][-1]["content"]
        else:
            prompt = body.get("prompt", "hello")

        stream = body.get("stream", True)
        max_tokens = body.get("options", {}).get("num_predict", 200)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        engine = _get_engine()
        t0 = time.time()
        total_tokens = 0
        full_response = ""

        for token_text in _generate_stream(engine, prompt, max_tokens=max_tokens):
            total_tokens += 1
            full_response += token_text
            if stream:
                self._ndjson({"model": "qwen3.5-35b",
                              "message": {"role": "assistant", "content": token_text},
                              "done": False})

        elapsed = time.time() - t0
        done = {
            "model": "qwen3.5-35b",
            "message": {"role": "assistant", "content": "" if stream else full_response},
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "eval_count": total_tokens,
            "eval_duration": int(elapsed * 1e9),
        }
        self._ndjson(done)
        tps = total_tokens / elapsed if elapsed > 0 else 0
        print(f"  [{total_tokens} tok, {tps:.1f} tok/s, {elapsed:.1f}s] {prompt[:40]}")

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _ndjson(self, data):
        self.wfile.write((json.dumps(data) + "\n").encode())
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


def run_server(model_dir, host="127.0.0.1", port=11434):
    global _model_dir
    _model_dir = model_dir

    print(f"mlx-sniper serve")
    print(f"  Model:  {model_dir}")
    print(f"  Listen: http://{host}:{port}")
    print(f"  API:    Ollama-compatible (/api/tags, /api/chat, /api/generate)")
    print()

    _get_engine()  # Pre-load

    print(f"\nReady. Listening on http://{host}:{port}")
    print(f"  Test: curl http://localhost:{port}/api/tags")
    print(f"  Chat: curl http://localhost:{port}/api/chat -d '{{\"model\":\"qwen3.5-35b\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'")
    print()

    import socket
    HTTPServer.allow_reuse_address = True
    server = HTTPServer((host, port), OllamaHandler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
