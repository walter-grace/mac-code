#!/usr/bin/env python3
"""
mac-tensor agent — Interactive agentic REPL backed by distributed expert nodes.

The model can call tools by emitting XML tags. We use STOP SEQUENCES so
generation halts the moment a closing tag appears, parse the tool call,
execute it, then feed the result back into the SAME KV cache (no reset).

Tools:
  <read>path</read>             — read a file
  <ls>path</ls>                 — list directory contents
  <shell>command</shell>        — run a read-only shell command
  <search>query</search>        — DuckDuckGo web search
  <python>expr</python>         — restricted python eval
  <write path="...">content</write>  — write a file (requires --write)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections import deque


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are a coding agent on an Apple Silicon cluster. You answer the user by either:
  (a) calling ONE tool and stopping, or
  (b) giving a short final answer with no tags.

Rules — these are critical:
1. NEVER write a <result> tag yourself. The system inserts results.
2. After emitting a tool call, STOP. Do not write anything else.
3. Use exactly ONE tool per step. The system runs it and shows you the result.
4. When you have enough information, give a SHORT final answer (1-3 sentences).
5. Do not repeat yourself. Do not loop.

Tools:
  <read>path</read>           Read a file (use ~ for home).
  <ls>path</ls>               List a directory.
  <shell>cmd</shell>          Run a read-only shell command.
  <search>query</search>      Web search (DuckDuckGo).
  <python>expr</python>       Eval a Python expression.

Example interaction:

User: What's in the README?
You: <read>README.md</read>
[system inserts: <result># My Project\\nA tool for X.</result>]
You: The README describes a tool for X.

Now the user asks:
"""


# ============================================================
# TOOLS
# ============================================================


def tool_read(arg):
    try:
        path = os.path.expanduser(arg.strip())
        with open(path) as f:
            content = f.read()
        if len(content) > 4000:
            content = content[:4000] + f"\n\n[truncated, {len(content) - 4000} more chars]"
        return content
    except Exception as e:
        return f"Error reading {arg}: {e}"


def tool_ls(arg):
    try:
        path = os.path.expanduser(arg.strip())
        items = sorted(os.listdir(path))
        out = []
        for item in items[:50]:
            full = os.path.join(path, item)
            out.append(f"  {item}{'/' if os.path.isdir(full) else ''}")
        if len(items) > 50:
            out.append(f"  ... ({len(items) - 50} more)")
        return "\n".join(out) if out else "(empty)"
    except Exception as e:
        return f"Error: {e}"


def tool_shell(arg, allow_write=False):
    cmd = arg.strip()
    if not allow_write:
        DANGEROUS = ["rm ", "mv ", "dd ", "mkfs", ">", ">>", "chmod",
                     "chown", "sudo", "kill", "shutdown", "format", "fdisk"]
        for d in DANGEROUS:
            if d in cmd.lower():
                return f"Refused (destructive: '{d}')"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                                text=True, timeout=15)
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if not out and not err:
            return f"(exit {result.returncode}, no output)"
        parts = []
        if out:
            parts.append(out[:2000] + ("..." if len(out) > 2000 else ""))
        if err:
            parts.append(f"[stderr] {err[:500]}")
        if result.returncode != 0:
            parts.append(f"[exit {result.returncode}]")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "Error: timeout (15s)"
    except Exception as e:
        return f"Error: {e}"


def tool_search(arg):
    query = arg.strip()
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                    "AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        results = []
        pattern = re.compile(
            r'class="result__a"[^>]*>([^<]+)</a>.*?'
            r'class="result__snippet"[^>]*>(.+?)</a>',
            re.DOTALL,
        )
        for m in list(pattern.finditer(html))[:4]:
            title = re.sub(r"\s+", " ", m.group(1).strip())
            snippet = re.sub(r"<[^>]+>", "", m.group(2))
            snippet = re.sub(r"\s+", " ", snippet).strip()[:160]
            results.append(f"• {title}\n  {snippet}")
        return "\n\n".join(results) if results else "No results"
    except Exception as e:
        return f"Search failed: {e}"


def tool_python(arg):
    try:
        safe = {
            "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr,
            "dict": dict, "divmod": divmod, "enumerate": enumerate, "filter": filter,
            "float": float, "hex": hex, "int": int, "len": len, "list": list,
            "map": map, "max": max, "min": min, "ord": ord, "pow": pow,
            "range": range, "reversed": reversed, "round": round, "set": set,
            "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
        }
        return str(eval(arg.strip(), {"__builtins__": safe}, {}))
    except Exception as e:
        return f"Error: {e}"


def tool_write(path_arg, content, allow_write=False):
    if not allow_write:
        return "Refused. Pass --write to enable."
    try:
        path = os.path.expanduser(path_arg.strip())
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# TOOL PARSER
# ============================================================

# Match opening + content + closing tag. Tools have NO attributes except <write>.
TOOL_REGEX = re.compile(
    r"<(read|ls|shell|search|python)>(.+?)</\1>",
    re.DOTALL,
)
WRITE_REGEX = re.compile(
    r'<write\s+path="([^"]+)">(.+?)</write>',
    re.DOTALL,
)

# Stop sequences for early generation halt
STOP_SEQUENCES = [
    "</read>", "</ls>", "</shell>", "</search>", "</python>", "</write>",
]


def parse_first_tool(text):
    """Find the first complete tool call. Returns dict or None."""
    # Check write first (it has attributes)
    m = WRITE_REGEX.search(text)
    if m:
        return {"tool": "write", "path": m.group(1), "content": m.group(2),
                "start": m.start(), "end": m.end()}
    m = TOOL_REGEX.search(text)
    if m:
        return {"tool": m.group(1), "content": m.group(2),
                "start": m.start(), "end": m.end()}
    return None


def execute_tool(call, allow_write=False):
    t = call["tool"]
    if t == "read":   return tool_read(call["content"])
    if t == "ls":     return tool_ls(call["content"])
    if t == "shell":  return tool_shell(call["content"], allow_write=allow_write)
    if t == "search": return tool_search(call["content"])
    if t == "python": return tool_python(call["content"])
    if t == "write":  return tool_write(call["path"], call["content"], allow_write=allow_write)
    return f"Unknown tool: {t}"


# ============================================================
# DISTRIBUTED LLM BACKEND WITH PERSISTENT CACHE
# ============================================================


class AgentBackend:
    """Wraps the distributed engine with token-level generation, stop sequences,
    and repetition penalty. Maintains a SINGLE KV cache across the whole turn —
    we never reset it between tool calls, we just append the result tokens.
    """

    def __init__(self, model_key, node_urls):
        self.model_key = model_key
        self.node_urls = node_urls
        self.engine = None

    def load(self):
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        sys.path.insert(0, script_dir)

        if self.model_key == "gemma4":
            cli_agent_src = os.path.expanduser("~/cli-agent/src")
            if os.path.exists(cli_agent_src):
                sys.path.insert(0, cli_agent_src)
            from gemma4_distributed import Gemma4DistributedEngine
            self.engine = Gemma4DistributedEngine(node_urls=self.node_urls)
        else:
            from distributed_interactive import InteractiveDistributedEngine
            self.engine = InteractiveDistributedEngine(node_urls=self.node_urls)

        self.engine.load()

    def reset(self):
        self.engine.reset_cache()

    def encode(self, text):
        if self.model_key == "gemma4":
            return self.engine.encode(text)
        # Qwen — use the underlying transformers tokenizer
        return self.engine.tokenizer.encode(text)

    def decode(self, ids):
        if self.model_key == "gemma4":
            return self.engine.decode(ids)
        return self.engine.tokenizer.decode(ids, skip_special_tokens=False)

    def encode_chat_prefix(self, system_and_user_text):
        """Wrap the prompt in the model's chat template."""
        if self.model_key == "gemma4":
            return self.engine.encode_chat(system_and_user_text)
        # Qwen — apply chat template via the HF tokenizer
        try:
            messages = [{"role": "user", "content": system_and_user_text}]
            text = self.engine.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            return self.engine.tokenizer.encode(text)
        except Exception:
            return self.encode(system_and_user_text)

    def feed_tokens(self, token_ids):
        """Run forward on a batch of new tokens. Returns final logits.

        Token IDs are appended to the existing KV cache — no reset.
        """
        import mlx.core as mx
        ids = mx.array([token_ids])
        logits = self.engine.forward(ids)
        mx.eval(logits)
        return logits

    def generate_until_stop(self, prompt_text, stop_sequences,
                             max_tokens=400, temperature=0.5,
                             repetition_penalty=1.15, repetition_window=64,
                             on_chunk=None):
        """Generate tokens until any stop sequence appears or max_tokens hit.

        Returns the generated text (excluding the stop sequence itself).

        Uses persistent KV cache: if you call this multiple times in a row,
        each call appends to the same cache.
        """
        import mlx.core as mx
        import numpy as np

        # Encode and feed the prompt
        prompt_ids = self.encode_chat_prefix(prompt_text)
        logits = self.feed_tokens(prompt_ids)

        generated_ids = []
        recent = deque(maxlen=repetition_window)
        decoded_buffer = ""

        # EOS tokens
        if self.model_key == "gemma4":
            eos_set = {1, 106}  # <eos>, <turn|>
        else:
            eos_set = {248044, 248046}

        for step in range(max_tokens):
            # Get last-token logits
            last_logits = logits[0, -1]

            # Apply repetition penalty
            if recent and repetition_penalty != 1.0:
                last_np = np.array(last_logits.astype(mx.float32))
                for tid in set(recent):
                    if last_np[tid] > 0:
                        last_np[tid] /= repetition_penalty
                    else:
                        last_np[tid] *= repetition_penalty
                last_logits = mx.array(last_np)

            # Sample
            if temperature <= 0:
                next_token = int(mx.argmax(last_logits).item())
            else:
                probs = mx.softmax(last_logits / temperature, axis=-1)
                next_token = int(mx.random.categorical(mx.log(probs + 1e-10)).item())

            generated_ids.append(next_token)
            recent.append(next_token)

            if next_token in eos_set:
                break

            # Update decoded buffer (decode all generated_ids fresh because of BPE)
            new_decoded = self.decode(generated_ids)
            new_chunk = new_decoded[len(decoded_buffer):]
            decoded_buffer = new_decoded

            if on_chunk and new_chunk:
                on_chunk(new_chunk)

            # Check stop sequences
            stop_hit = None
            for s in stop_sequences:
                if s in decoded_buffer:
                    stop_hit = s
                    break
            if stop_hit:
                break

            # Feed the new token to the cache for the next step
            logits = self.feed_tokens([next_token])

        return decoded_buffer


# ============================================================
# AGENT LOOP
# ============================================================


def run_agent_turn(backend, user_question, max_iterations=6, max_tokens=400,
                   temperature=0.5, allow_write=False, verbose=True):
    """Run one user turn — multi-step tool use loop."""
    backend.reset()  # Fresh KV cache per user turn

    # Initial prompt
    prompt = SYSTEM_PROMPT + " " + user_question

    final_answer = None

    for step in range(max_iterations):
        if verbose:
            print(f"\n  [Step {step + 1}/{max_iterations}]", end=" ", flush=True)

        def stream_chunk(chunk):
            if verbose:
                # Strip the model's attempts to invent <result> tags from display
                print(chunk, end="", flush=True)

        # First step uses the full prompt; subsequent steps just feed result tokens
        if step == 0:
            text = backend.generate_until_stop(
                prompt,
                stop_sequences=STOP_SEQUENCES + ["</result>"],
                max_tokens=max_tokens,
                temperature=temperature,
                on_chunk=stream_chunk,
            )
        else:
            # Continuation: feed only the new tokens (result + nudge)
            text = backend.generate_until_stop(
                continuation,
                stop_sequences=STOP_SEQUENCES + ["</result>"],
                max_tokens=max_tokens,
                temperature=temperature,
                on_chunk=stream_chunk,
            )

        # Strip any hallucinated <result>...</result> the model wrote
        text = re.sub(r"<result>.*?</result>", "", text, flags=re.DOTALL)

        # Parse first tool call
        call = parse_first_tool(text)

        if call is None:
            # No tool call → final answer
            final_answer = text.strip()
            if verbose:
                print()
            break

        # Execute the tool
        tool_name = call["tool"]
        tool_arg = call.get("content", "")[:60].replace("\n", " ")
        if verbose:
            print(f"\n  → tool: <{tool_name}> {tool_arg}...")

        result = execute_tool(call, allow_write=allow_write)
        if verbose:
            preview = result[:120].replace("\n", " | ")
            print(f"  ← {preview}{'...' if len(result) > 120 else ''}")

        # Build the continuation prompt for the next step.
        # We feed: the model's tool call + a real <result>...</result> + a nudge
        continuation = (
            f"<result>\n{result}\n</result>\n\n"
            f"Now either call another tool or give a final answer (no tags):"
        )

    if final_answer is None:
        final_answer = "(hit iteration limit without producing a final answer)"

    return final_answer


def agent_loop(backend, max_iterations=6, max_tokens=400, allow_write=False):
    print()
    print("=" * 60)
    print("  mac-tensor agent (v2 — stop sequences + repetition penalty)")
    print(f"  Model: {backend.model_key} | Nodes: {len(backend.node_urls)}")
    print(f"  Tools: read, ls, shell, search, python" + (", write" if allow_write else ""))
    print("=" * 60)
    print("\nType your question. 'reset' to clear, 'quit' to exit.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            return
        if user_input.lower() == "reset":
            backend.reset()
            print("Context cleared.")
            continue

        answer = run_agent_turn(
            backend, user_input,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            allow_write=allow_write,
        )

        print(f"\n\nAgent: {answer}\n")


def main(args):
    if not args.nodes:
        print("Error: --nodes is required")
        sys.exit(1)

    node_urls = [u.strip() for u in args.nodes.split(",")]
    model_key = args.model or "gemma4"

    print(f"Loading {model_key} distributed engine...")
    backend = AgentBackend(model_key=model_key, node_urls=node_urls)
    backend.load()

    agent_loop(
        backend,
        max_iterations=args.max_iterations or 6,
        max_tokens=args.max_tokens or 300,
        allow_write=args.write,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen35", "gemma4"], default="gemma4")
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    main(args)
