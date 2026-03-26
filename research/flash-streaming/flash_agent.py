#!/usr/bin/env python3
"""
flash-agent — Qwen3-32B (18.4 GB) on 16 GB Mac via SSD streaming.

Full 4-bit quality. No compression compromise.
Only 4.5 GB in RAM. FFN streamed from SSD per token.
"""

import json, sys, os, time, gc, random, re
from concurrent.futures import ThreadPoolExecutor
from direct_io import DirectFFNReader

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.live import Live
from rich.padding import Padding

console = Console()

# ── Config ────────────────────────────────────────
MODEL_DIR = "/Users/bigneek/models/qwen3-32b-flash-stream"
BITS = 4
GROUP_SIZE = 64
TEMPERATURE = 0.7
TOP_P = 0.9
REP_PENALTY = 1.2
MAX_TOKENS = 512

# ── Animated creatures ────────────────────────────
CREATURES = [
    ["   ⚡( ᐛ )⚡  ", "  ⚡( ᐛ )⚡   ", " ⚡( ᐛ )⚡    ", "  ⚡( ᐛ )⚡   "],
    ["  ⠋  ", "  ⠙  ", "  ⠹  ", "  ⠸  ", "  ⠼  ", "  ⠴  ", "  ⠦  ", "  ⠧  ", "  ⠇  ", "  ⠏  "],
]
CREATURE = CREATURES[random.randint(0, len(CREATURES) - 1)]


# ── FFN Streamer ──────────────────────────────────

ALIGNED_DIR = f"{MODEL_DIR}/ffn_aligned"
USE_DIRECT_IO = os.path.isdir(ALIGNED_DIR)


# ── Flash Stream Engine ───────────────────────────

class FlashStreamEngine:
    def __init__(self):
        self.model = None
        self.streamer = None
        self.tokenizer = None
        self.num_layers = 64
        self.cache = None

    def load(self):
        with open(f"{MODEL_DIR}/config.json") as f:
            config = json.load(f)

        self.num_layers = config["num_hidden_layers"]

        from mlx_lm.models.qwen3 import Model, ModelArgs
        args = ModelArgs(
            model_type=config["model_type"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=self.num_layers,
            intermediate_size=config["intermediate_size"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            vocab_size=config["vocab_size"],
            max_position_embeddings=config["max_position_embeddings"],
            rope_theta=config["rope_theta"],
            head_dim=config["head_dim"],
            tie_word_embeddings=config["tie_word_embeddings"],
        )

        self.model = Model(args)
        nn.quantize(self.model, group_size=GROUP_SIZE, bits=BITS)

        # Set memory limits
        mx.set_memory_limit(10 * 1024**3)
        mx.set_cache_limit(256 * 1024**2)

        # Load pinned weights
        pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")
        self.model.load_weights(list(pinned.items()), strict=False)
        params = [p for name, p in tree_flatten(self.model.parameters()) if "mlp" not in name]
        mx.eval(*params)
        del pinned
        gc.collect()
        mx.clear_cache()

        pinned_gb = sum(p.nbytes for p in params) / 1e9

        # Use direct I/O if aligned files exist, otherwise mmap fallback
        if USE_DIRECT_IO:
            self.streamer = DirectFFNReader(ALIGNED_DIR, self.num_layers)
            self.io_mode = "direct I/O (F_NOCACHE)"
        else:
            from concurrent.futures import ThreadPoolExecutor as TPE
            class MmapStreamer:
                def __init__(self, ffn_dir, n):
                    self.ffn_dir = ffn_dir
                    self.ex = TPE(max_workers=1)
                    self.fut = None; self.idx = -1
                def _read(self, i): return mx.load(f"{self.ffn_dir}/layer_{i:02d}.safetensors")
                def prefetch(self, i):
                    if self.fut and self.idx == i: return
                    self.fut = self.ex.submit(self._read, i); self.idx = i
                def get(self, i):
                    if self.fut and self.idx == i:
                        d = self.fut.result(); self.fut = None; return d
                    return self._read(i)
            self.streamer = MmapStreamer(
                f"{MODEL_DIR}/{config['streaming']['ffn_dir']}", self.num_layers
            )
            self.io_mode = "mmap (safetensors)"

        # Tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-32B", trust_remote_code=True
        )

        return pinned_gb

    def reset_cache(self):
        from mlx_lm.models.cache import make_prompt_cache
        self.cache = make_prompt_cache(self.model)

    def _run_ffn(self, x, ffn_data):
        gate = mx.quantized_matmul(
            x, ffn_data["mlp.gate_proj.weight"],
            scales=ffn_data["mlp.gate_proj.scales"],
            biases=ffn_data["mlp.gate_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        up = mx.quantized_matmul(
            x, ffn_data["mlp.up_proj.weight"],
            scales=ffn_data["mlp.up_proj.scales"],
            biases=ffn_data["mlp.up_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        hidden = nn.silu(gate) * up
        del gate, up
        out = mx.quantized_matmul(
            hidden, ffn_data["mlp.down_proj.weight"],
            scales=ffn_data["mlp.down_proj.scales"],
            biases=ffn_data["mlp.down_proj.biases"],
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
        )
        del hidden
        return out

    def forward(self, input_ids):
        from mlx_lm.models.base import create_attention_mask

        h = self.model.model.embed_tokens(input_ids)
        mask = create_attention_mask(h, self.cache[0])

        self.streamer.prefetch(0)

        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            if i + 1 < self.num_layers:
                self.streamer.prefetch(i + 1)

            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed, mask=mask, cache=self.cache[i])
            h = h + attn_out
            mx.eval(h)

            ffn_data = self.streamer.get(i)
            normed = layer.post_attention_layernorm(h)
            ffn_out = self._run_ffn(normed, ffn_data)
            h = h + ffn_out
            mx.eval(h)
            del ffn_data, ffn_out, normed, attn_out
            mx.clear_cache()

        h = self.model.model.norm(h)
        return self.model.lm_head(h)

    def sample(self, logits, generated):
        next_logits = logits[:, -1, :]

        if generated:
            seen = mx.array(list(set(generated[-100:])))
            pl = next_logits[:, seen]
            pl = mx.where(pl > 0, pl / REP_PENALTY, pl * REP_PENALTY)
            next_logits[:, seen] = pl

        probs = mx.softmax(next_logits / TEMPERATURE, axis=-1)
        sorted_idx = mx.argsort(-probs, axis=-1)
        sorted_p = mx.take_along_axis(probs, sorted_idx, axis=-1)
        cumsum = mx.cumsum(sorted_p, axis=-1)
        mask = (cumsum - sorted_p) <= TOP_P
        sorted_p = sorted_p * mask
        sorted_p = sorted_p / (sorted_p.sum(axis=-1, keepdims=True) + 1e-10)
        token = mx.random.categorical(mx.log(sorted_p + 1e-10))
        token = mx.take_along_axis(sorted_idx, token[:, None], axis=-1).squeeze(-1)
        mx.eval(token)
        return token.item()

    def generate_stream(self, messages):
        """Yield tokens one at a time for streaming display."""
        # Build prompt using chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        # Prefill
        logits = self.forward(input_ids)
        mx.eval(logits)

        generated = []
        for _ in range(MAX_TOKENS):
            token_id = self.sample(logits, generated)
            if token_id in (151645, 151643):  # EOS
                break
            generated.append(token_id)
            yield self.tokenizer.decode([token_id])

            logits = self.forward(mx.array([[token_id]]))
            mx.eval(logits)


# ── UI ────────────────────────────────────────────

class ThinkingDisplay:
    def __init__(self):
        self.frame = 0
        self.start = time.time()
        self.phase = "thinking"

    def render(self):
        self.frame += 1
        elapsed = time.time() - self.start
        cf = CREATURE[self.frame % len(CREATURE)]
        t = Text()
        t.append(f"  {cf}", style="bright_cyan")
        t.append(f"  {self.phase}", style="bold bright_cyan")
        t.append(f"  {elapsed:.0f}s", style="dim")
        return t


def print_banner(pinned_gb, io_mode):
    console.print()
    logo = Text()
    logo.append("  flash", style="bold bright_cyan")
    logo.append("-", style="dim")
    logo.append("stream", style="bold bright_yellow")
    if "direct" in io_mode:
        logo.append(" v2", style="bold bright_green")
    console.print(logo)

    sub = Text()
    sub.append("  18.4 GB model on 16 GB Mac — SSD streaming", style="dim italic")
    console.print(sub)
    console.print()

    rows = [
        ("model", "Qwen3-32B", "full 4-bit quality · 18.4 GB total"),
        ("pinned", f"{pinned_gb:.1f} GB", "attention + embeddings in RAM"),
        ("streamed", "14.2 GB", "FFN from SSD per token"),
        ("I/O", io_mode, ""),
        ("cost", "$0.00/hr", "Apple Silicon Metal · local"),
    ]
    for label, value, extra in rows:
        line = Text()
        line.append(f"  {label:8s} ", style="bold dim")
        line.append(value, style="bold white")
        if extra:
            line.append(f"  {extra}", style="dim")
        console.print(line)

    console.print()
    console.print(Rule(style="dim"))
    console.print()


def render_speed(tokens, elapsed):
    if elapsed <= 0 or tokens <= 0:
        return
    speed = tokens / elapsed
    clr = "bright_green" if speed > 1 else "yellow" if speed > 0.1 else "red"
    s = Text()
    s.append(f"  {speed:.2f} tok/s", style=f"bold {clr}")
    s.append(f"  ·  {tokens} tokens  ·  {elapsed:.1f}s", style="dim")
    console.print(s)


# ── Main ──────────────────────────────────────────

def main():
    console.clear()

    # Loading screen
    console.print()
    with console.status("[bold bright_cyan]  Loading Flash Stream engine...", spinner="dots"):
        engine = FlashStreamEngine()
        pinned_gb = engine.load()

    print_banner(pinned_gb, engine.io_mode)

    messages = []
    session_tokens = 0
    session_time = 0.0

    system_prompt = (
        "You are a helpful AI assistant running via Flash Stream inference — "
        "an 18.4 GB model streaming from SSD on a 16 GB Mac. "
        "Be concise and helpful."
    )
    messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            console.print("  [bold bright_yellow]>[/] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]goodbye.[/]\n")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip().lower()
        if cmd in ("/quit", "/exit", "/q"):
            console.print("  [dim]goodbye.[/]\n")
            break
        elif cmd == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            engine.reset_cache()
            console.clear()
            print_banner(pinned_gb, engine.io_mode)
            console.print("  [dim]cleared.[/]\n")
            continue
        elif cmd == "/stats":
            avg = session_tokens / session_time if session_time > 0 else 0
            mem = mx.get_active_memory() / 1e9
            t = Table(show_header=False, box=None, padding=(0, 1))
            t.add_column(style="bold bright_cyan", width=12)
            t.add_column()
            t.add_row("tokens", f"{session_tokens:,}")
            t.add_row("time", f"{session_time:.1f}s")
            t.add_row("avg speed", f"{avg:.2f} tok/s")
            t.add_row("GPU memory", f"{mem:.2f} GB")
            console.print(t)
            console.print()
            continue
        elif cmd in ("/help", "/?"):
            for c, d in [
                ("/clear", "Reset conversation"),
                ("/stats", "Session statistics"),
                ("/quit", "Exit"),
            ]:
                console.print(f"  [bold bright_cyan]{c:10s}[/] [dim]{d}[/]")
            console.print()
            continue

        messages.append({"role": "user", "content": user_input})
        engine.reset_cache()
        console.print()

        # Show thinking animation during prefill
        display = ThinkingDisplay()
        display.phase = "streaming from SSD"
        full = ""
        tokens = 0
        start = time.time()
        first_token = True

        with Live(display.render(), console=console, refresh_per_second=6, transient=True) as live:
            for chunk in engine.generate_stream(messages):
                if first_token:
                    first_token = False
                    live.stop()
                    console.print("  ", end="")
                console.print(chunk, end="", highlight=False)
                full += chunk
                tokens += 1
                display.frame += 1

        elapsed = time.time() - start
        console.print("\n")
        render_speed(tokens, elapsed)
        console.print()

        messages.append({"role": "assistant", "content": full})
        session_tokens += tokens
        session_time += elapsed

        # Keep conversation manageable
        if len(messages) > 20:
            messages = [messages[0]] + messages[-10:]


if __name__ == "__main__":
    main()
