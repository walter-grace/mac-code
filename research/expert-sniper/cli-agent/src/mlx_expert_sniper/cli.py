"""
mlx-sniper CLI.

Usage:
    mlx-sniper download qwen3.5-35b [-o ~/models/qwen35-35b]
    mlx-sniper calibrate <model-dir> [--quick] [--force] [--ram N]
    mlx-sniper run <model-dir> -p "prompt" [-v] [--max-tokens N]
"""
import argparse
import sys
import os
import time


def cmd_download(args):
    from .download import download_model, list_models

    if args.model_name == "list":
        list_models()
        return

    output = args.output
    if output:
        output = os.path.expanduser(output)

    download_model(
        args.model_name,
        output_dir=output,
        calibrate_quick=not args.full_calibrate,
        keep_download=args.keep_download,
    )


def cmd_serve(args):
    from .server import run_server
    run_server(
        model_dir=args.model_dir,
        host=args.host,
        port=args.port,
    )


def cmd_calibrate(args):
    from .calibrate import calibrate, load_calibration

    if not args.force:
        existing = load_calibration(args.model_dir)
        if existing:
            print(f"Calibration exists: cache={existing['cache_size']}, "
                  f"bias={existing['routing_bias']}, "
                  f"dead={existing['reap_dead_pct']:.1%}")
            print(f"Use --force to overwrite.")
            return

    calibrate(args.model_dir, ram_gb=args.ram, quick=args.quick)


def cmd_run(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import mlx.core as mx
    import numpy as np
    from .calibrate import load_calibration, auto_size_cache
    from .engine import MoESniperEngine35B, run_expert_ffn
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    # Load calibration if available
    cal = load_calibration(args.model_dir)
    if cal:
        cache_size = cal["cache_size"]
        bias = cal["routing_bias"]
        print(f"Loaded calibration: cache={cache_size}, bias={bias}, "
              f"dead={cal['reap_dead_pct']:.1%}")
    else:
        cache_size, _, _ = auto_size_cache(args.model_dir)
        bias = 0.0
        print(f"No calibration found. Using defaults: cache={cache_size}, bias=0.0")
        print(f"Run 'mlx-sniper calibrate {args.model_dir}' for optimal performance.")

    # Patch MODEL_DIR in engine module
    from . import engine as engine_mod
    engine_mod.MODEL_DIR = args.model_dir

    eng = MoESniperEngine35B(cache_size=cache_size, enable_prediction=True)
    eng.load()
    eng.reset_cache()
    print(f"Model loaded. Metal: {mx.get_active_memory()/1e9:.2f} GB")

    tok = eng.tokenizer
    messages = [{"role": "user", "content": args.prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    except:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tok.encode(text)
    input_ids = mx.array([tokens])

    # Forward with bias
    def biased_forward(inp):
        h = eng.model.model.embed_tokens(inp)
        fa_mask = create_attention_mask(h, eng.cache[eng.model.model.fa_idx])
        ssm_mask = create_ssm_mask(h, eng.cache[eng.model.model.ssm_idx])
        for i in range(eng.num_layers):
            layer = eng.model.model.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask
            normed = layer.input_layernorm(h)
            if layer.is_linear:
                attn_out = layer.linear_attn(normed, mask=mask, cache=eng.cache[i])
            else:
                attn_out = layer.self_attn(normed, mask=mask, cache=eng.cache[i])
            h = h + attn_out
            mx.eval(h)
            normed = layer.post_attention_layernorm(h)
            raw_logits = layer.mlp.gate(normed)
            if bias > 0 and eng.reader.lru is not None:
                cached_mask = np.zeros(256, dtype=np.float32)
                for eid in range(256):
                    if eng.reader.lru.get(i, eid) is not None:
                        cached_mask[eid] = bias
                raw_logits = raw_logits + mx.array(cached_mask).reshape(1, -1)
            gates = mx.softmax(raw_logits, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)
            active_ids = list(set(int(e) for e in np.array(inds).flatten()))
            eng.coact.record_layer(i, active_ids)
            if eng.coact.ready and i + 1 < eng.num_layers:
                predicted = eng.coact.predict_next_layer(i, active_ids, top_k=6)
                if predicted:
                    to_fetch = [eid for eid in predicted
                                if eng.reader.lru and eng.reader.lru.get(i+1, eid) is None]
                    if to_fetch:
                        eng.reader.prefetch_experts(i+1, to_fetch)
            if i + 1 < eng.num_layers:
                eng.reader.prefetch_experts(i+1, active_ids)
            expert_data = eng.reader.get_experts(i, active_ids)
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
        eng.coact.end_token()
        h = eng.model.model.norm(h)
        return eng.model.lm_head(h)

    t0 = time.time()
    logits = biased_forward(input_ids)
    mx.eval(logits)
    ttft = time.time() - t0

    generated = []
    for _ in range(args.max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tid = token.item()
        if tid in (248044, 248045):
            break
        generated.append(tid)
        chunk = tok.decode([tid])
        sys.stdout.write(chunk)
        sys.stdout.flush()
        logits = biased_forward(token.reshape(1, 1))
        mx.eval(logits)

    elapsed = time.time() - t0
    n = len(generated)
    tps = n / (elapsed - ttft) if elapsed > ttft else 0

    if args.verbose:
        print(f"\n\n  {n} tokens | {tps:.2f} tok/s | TTFT: {ttft:.2f}s | "
              f"Total: {elapsed:.2f}s")
        print(f"  Cache: {eng.reader.stats()}")
        print(f"  Metal: {mx.get_active_memory()/1e9:.2f} GB")
    else:
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="mlx-sniper",
        description="Run MoE models larger than RAM on Apple Silicon",
    )
    sub = parser.add_subparsers(dest="command")

    # download
    p = sub.add_parser("download", help="Download, preprocess, and calibrate a model")
    p.add_argument("model_name", help="Model name (e.g. qwen3.5-35b) or 'list'")
    p.add_argument("-o", "--output", default=None, help="Output directory (default: ~/models/<name>)")
    p.add_argument("--full-calibrate", action="store_true", help="Run full calibration with bias sweep")
    p.add_argument("--keep-download", action="store_true", help="Keep raw HF download after preprocessing")

    # serve
    p = sub.add_parser("serve", help="Ollama-compatible HTTP server")
    p.add_argument("model_dir", help="Path to sniper model directory")
    p.add_argument("--port", type=int, default=11434, help="Port (default: 11434)")
    p.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1, use 0.0.0.0 for network)")

    # calibrate
    p = sub.add_parser("calibrate", help="One-time model calibration (~2-8 min)")
    p.add_argument("model_dir", help="Path to sniper model directory")
    p.add_argument("--ram", type=float, default=None, help="Override RAM (GB)")
    p.add_argument("--quick", action="store_true", help="Skip bias sweep (2 min)")
    p.add_argument("--force", action="store_true", help="Overwrite existing calibration")

    # run
    p = sub.add_parser("run", help="Generate text from a prompt")
    p.add_argument("model_dir", help="Path to sniper model directory")
    p.add_argument("--prompt", "-p", required=True, help="Text prompt")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    {"download": cmd_download, "serve": cmd_serve, "calibrate": cmd_calibrate, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
