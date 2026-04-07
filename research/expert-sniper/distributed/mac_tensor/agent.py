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
# SYSTEM PROMPTS
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


VISION_SYSTEM_PROMPT = """You are a vision agent on an Apple Silicon cluster. You can see the image the user uploaded, and you have a precision instrument called Falcon Perception that finds exact pixel locations of objects.

Rules:
1. NEVER write a <result> tag yourself. The system inserts results.
2. After emitting a tool call, STOP. Wait for the result.
3. Use ONE tool per step.
4. When you have enough information, give a SHORT final answer (1-3 sentences) with no tags.
5. Trust the numbers the tools give you over your own visual estimates.

When to use tools vs answer directly:
- If the user just wants a description of the image, answer directly. No tools needed.
- If the user asks "how many" or "where exactly" or "which is largest/closest/leftmost", use grounding tools.
- If the user asks for spatial relationships (closest, biggest, leftmost), first ground the objects, then use a spatial tool.

Tools:

<ground>object name or short description</ground>
  Find all instances of an object in the image. Returns a list of masks
  with id, centroid (x, y in 0-1), area_fraction, region (top-left, center, etc.).
  Example: <ground>bird</ground>

<extreme slot="object_name" direction="bottommost"/>
  After grounding, find the most extreme mask in a slot.
  direction can be: topmost, bottommost, leftmost, rightmost, largest, smallest
  In a 2D photo, "closest to camera" usually means bottommost (lowest in frame).
  Example: <extreme slot="bird" direction="bottommost"/>

<count slot="object_name"/>
  Return the number of masks in a slot.
  Example: <count slot="bird"/>

Example interaction (single ground):
User: How many cats are in this image?
You: <ground>cat</ground>
[system: <result>{"slot":"cat","count":3,"masks":[{"id":1,...},{"id":2,...},{"id":3,...}]}</result>]
You: There are 3 cats in the image.

Example interaction (chained):
User: Which bird is biggest?
You: <ground>bird</ground>
[system: <result>{"slot":"bird","count":4,"masks":[...]}</result>]
You: <extreme slot="bird" direction="largest"/>
[system: <result>{"winner":{"id":3,"area_fraction":0.18,...}}</result>]
You: The largest bird is mask #3, taking up about 18% of the image.

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
    r"<(read|ls|shell|search|python|ground|count)>(.+?)</\1>",
    re.DOTALL,
)
WRITE_REGEX = re.compile(
    r'<write\s+path="([^"]+)">(.+?)</write>',
    re.DOTALL,
)
# Self-closing vision tools with attributes
EXTREME_REGEX = re.compile(
    r'<extreme\s+slot="([^"]+)"\s+direction="([^"]+)"\s*/>',
)
COUNT_ATTR_REGEX = re.compile(
    r'<count\s+slot="([^"]+)"\s*/>',
)
BBOX_REGEX = re.compile(
    r'<bbox\s+id="?(\d+)"?\s*/>',
)

# Stop sequences for early generation halt
STOP_SEQUENCES = [
    "</read>", "</ls>", "</shell>", "</search>", "</python>", "</write>",
    "</ground>", "</count>",
    # Self-closing vision tags
    "/>",
]


VISION_STOP_SEQUENCES = [
    "</ground>", "</count>",
    "/>",  # extreme, count attr, bbox
    "</result>",
]


def parse_first_tool(text):
    """Find the first complete tool call. Returns dict or None."""
    # Check write first (it has attributes)
    m = WRITE_REGEX.search(text)
    if m:
        return {"tool": "write", "path": m.group(1), "content": m.group(2),
                "start": m.start(), "end": m.end()}
    # Self-closing vision tags
    m = EXTREME_REGEX.search(text)
    if m:
        return {"tool": "extreme", "slot": m.group(1), "direction": m.group(2),
                "content": "", "start": m.start(), "end": m.end()}
    m = COUNT_ATTR_REGEX.search(text)
    if m:
        return {"tool": "count", "slot": m.group(1),
                "content": "", "start": m.start(), "end": m.end()}
    m = BBOX_REGEX.search(text)
    if m:
        return {"tool": "bbox", "mask_id": int(m.group(1)),
                "content": "", "start": m.start(), "end": m.end()}
    # Standard tool tags
    m = TOOL_REGEX.search(text)
    if m:
        return {"tool": m.group(1), "content": m.group(2),
                "start": m.start(), "end": m.end()}
    return None


def execute_tool(call, allow_write=False, falcon_tools=None):
    t = call["tool"]
    if t == "read":   return tool_read(call["content"])
    if t == "ls":     return tool_ls(call["content"])
    if t == "shell":  return tool_shell(call["content"], allow_write=allow_write)
    if t == "search": return tool_search(call["content"])
    if t == "python": return tool_python(call["content"])
    if t == "write":  return tool_write(call["path"], call["content"], allow_write=allow_write)
    # Vision tools (require falcon_tools)
    if falcon_tools is None:
        return f"Vision tool '{t}' requires Falcon Perception (--falcon)"
    if t == "ground":
        return falcon_tools.ground(call["content"].strip())
    if t == "extreme":
        return falcon_tools.extreme(call["slot"], call["direction"])
    if t == "count":
        # Could be <count>slot</count> or <count slot="X"/>
        slot = call.get("slot") or call.get("content", "").strip()
        return falcon_tools.count_slot(slot)
    if t == "bbox":
        return falcon_tools.bbox(call["mask_id"])
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
# VISION AGENT BACKEND (single-machine with Falcon)
# ============================================================


class VisionAgentBackend:
    """Wraps VisionGemma4Sniper + FalconPerceptionTools for chained tool use."""

    def __init__(self, vision_engine, falcon_tools=None):
        self.vision_engine = vision_engine
        self.falcon_tools = falcon_tools

    def reset(self):
        self.vision_engine.sniper.reset_cache()

    def encode(self, text):
        return self.vision_engine.sniper.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.vision_engine.sniper.tokenizer.decode(ids)

    def feed_tokens_text(self, token_ids):
        """Feed text-only tokens after the initial vision prefill."""
        import mlx.core as mx
        ids = mx.array([token_ids])
        logits = self.vision_engine.sniper.forward(ids)
        mx.eval(logits)
        return logits

    def run_vision_prefill(self, prompt_text, image_path):
        """Run the initial vision-aware forward pass and return logits."""
        if self.falcon_tools and image_path:
            self.falcon_tools.set_image(image_path)
        tokens, image_features, _ = self.vision_engine.encode_chat(prompt_text, image_path)
        logits = self.vision_engine._prefill_logits(tokens, image_features)
        return logits, len(tokens)


def _patch_vision_engine_prefill():
    """Add a _prefill_logits method to VisionGemma4Sniper.

    The original _prefill returns the first sampled token. We need a version
    that returns the full logits so the agent can apply repetition penalty.
    """
    import mlx.core as mx
    try:
        from .vision_engine import VisionGemma4Sniper, IMAGE_TOKEN_ID
    except Exception:
        return

    if hasattr(VisionGemma4Sniper, "_prefill_logits"):
        return

    def _prefill_logits(self, input_token_ids, image_features):
        from mlx_lm.models.base import create_attention_mask
        from mlx_vlm.models.gemma4.gemma4 import masked_scatter
        from moe_agent_gemma4 import run_expert_ffn

        self.sniper.reset_cache()
        input_ids = mx.array([input_token_ids])
        h = self.sniper.model.model.embed_tokens(input_ids)
        h = h * (self.sniper.model.args.hidden_size ** 0.5)

        if image_features is not None:
            image_mask = (input_ids == IMAGE_TOKEN_ID)
            image_feats_flat = image_features.reshape(-1, image_features.shape[-1])
            image_feats_flat = image_feats_flat.astype(h.dtype)
            image_mask_expanded = mx.expand_dims(image_mask, -1)
            image_mask_expanded = mx.broadcast_to(image_mask_expanded, h.shape)
            h = masked_scatter(h, image_mask_expanded, image_feats_flat)

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
        return logits

    VisionGemma4Sniper._prefill_logits = _prefill_logits


_patch_vision_engine_prefill()


def run_vision_agent_turn_stream(vision_backend, user_question, image_path,
                                  max_iterations=4, max_tokens=300,
                                  temperature=0.5, allow_write=False):
    """Run one user turn with vision + Falcon tool use."""
    import mlx.core as mx
    import json
    from collections import deque

    vision_backend.reset()

    prompt = VISION_SYSTEM_PROMPT + " " + user_question

    yield {"type": "step_start", "step": 1, "max": max_iterations}

    try:
        logits, _ = vision_backend.run_vision_prefill(prompt, image_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": f"vision prefill failed: {e}"}
        return

    last_logits = logits[0, -1]
    if temperature <= 0:
        next_token = int(mx.argmax(last_logits).item())
    else:
        probs = mx.softmax(last_logits / temperature, axis=-1)
        next_token = int(mx.random.categorical(mx.log(probs + 1e-10)).item())

    eos_set = {1, 106}

    for iteration in range(max_iterations):
        if iteration > 0:
            yield {"type": "step_start", "step": iteration + 1, "max": max_iterations}

        generated = [next_token]
        recent = deque([next_token], maxlen=64)
        decoded_buffer = vision_backend.decode([next_token])
        if decoded_buffer:
            yield {"type": "token", "text": decoded_buffer}

        for step in range(max_tokens):
            if next_token in eos_set:
                break

            stop_hit = any(s in decoded_buffer for s in VISION_STOP_SEQUENCES)
            if stop_hit:
                break

            try:
                logits = vision_backend.feed_tokens_text([next_token])
            except Exception as e:
                yield {"type": "error", "message": str(e)}
                return

            last_logits = logits[0, -1]

            if recent:
                last_np = np.array(last_logits.astype(mx.float32))
                for tid in set(recent):
                    if last_np[tid] > 0:
                        last_np[tid] /= 1.15
                    else:
                        last_np[tid] *= 1.15
                last_logits = mx.array(last_np)

            if temperature <= 0:
                next_token = int(mx.argmax(last_logits).item())
            else:
                probs = mx.softmax(last_logits / temperature, axis=-1)
                next_token = int(mx.random.categorical(mx.log(probs + 1e-10)).item())

            generated.append(next_token)
            recent.append(next_token)

            full = vision_backend.decode(generated)
            new_chunk = full[len(decoded_buffer):]
            decoded_buffer = full
            if new_chunk:
                yield {"type": "token", "text": new_chunk}

        text = re.sub(r"<result>.*?</result>", "", decoded_buffer, flags=re.DOTALL)
        call = parse_first_tool(text)

        if call is None:
            final_answer = text.strip()
            yield {"type": "final", "text": final_answer}
            yield {"type": "done"}
            return

        tool_name = call["tool"]
        tool_args = call.get("content") or call.get("slot") or str(call.get("mask_id", ""))
        yield {"type": "tool_call", "tool": tool_name, "args": str(tool_args)[:200]}

        result = execute_tool(
            call, allow_write=allow_write, falcon_tools=vision_backend.falcon_tools
        )

        if isinstance(result, dict):
            if "masks" in result:
                trimmed = []
                for m in result["masks"]:
                    trimmed.append({k: v for k, v in m.items() if not k.startswith("_")})
                result_for_llm = {**result, "masks": trimmed}
            else:
                result_for_llm = result
            result_str = json.dumps(result_for_llm)
            display_str = json.dumps(result_for_llm, indent=2)[:1500]
        else:
            result_str = str(result)
            display_str = result_str[:1500]

        yield {"type": "tool_result", "result": display_str}

        continuation = (
            f"<result>{result_str}</result>\n\n"
            f"Now either call another tool or give a final answer (no tags):"
        )

        try:
            cont_tokens = vision_backend.encode(continuation)
            logits = vision_backend.feed_tokens_text(cont_tokens)
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        last_logits = logits[0, -1]
        if temperature <= 0:
            next_token = int(mx.argmax(last_logits).item())
        else:
            probs = mx.softmax(last_logits / temperature, axis=-1)
            next_token = int(mx.random.categorical(mx.log(probs + 1e-10)).item())

    yield {"type": "final", "text": "(iteration limit reached)"}
    yield {"type": "done"}


# ============================================================
# AGENT LOOP
# ============================================================


def run_agent_turn_stream(backend, user_question, max_iterations=6,
                            max_tokens=300, temperature=0.5, allow_write=False):
    """Run one user turn as a generator yielding events.

    Each yield is a dict with a 'type' key. Types:
      - {"type": "step_start", "step": N, "max": M}
      - {"type": "token", "text": "..."}        — streaming tokens
      - {"type": "tool_call", "tool": "ls", "args": "."}
      - {"type": "tool_result", "result": "..."}
      - {"type": "final", "text": "..."}        — final answer
      - {"type": "done"}
      - {"type": "error", "message": "..."}
    """
    backend.reset()
    prompt = SYSTEM_PROMPT + " " + user_question
    final_answer = None
    continuation = None

    # We collect tokens via this list because the streaming callback runs
    # in a different scope. The generator body checks it after each generate call.
    chunk_buffer = []

    def on_chunk(text):
        chunk_buffer.append(text)

    for step in range(max_iterations):
        yield {"type": "step_start", "step": step + 1, "max": max_iterations}
        chunk_buffer.clear()

        try:
            text = backend.generate_until_stop(
                prompt if step == 0 else continuation,
                stop_sequences=STOP_SEQUENCES + ["</result>"],
                max_tokens=max_tokens,
                temperature=temperature,
                on_chunk=on_chunk,
            )
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        # Yield buffered tokens (they're already chunked, but the server's SSE
        # path consumes the buffer after generation completes — see server.py
        # for the live-streaming path which uses on_chunk directly)
        for c in chunk_buffer:
            yield {"type": "token", "text": c}

        # Strip hallucinated <result> blocks
        text = re.sub(r"<result>.*?</result>", "", text, flags=re.DOTALL)

        call = parse_first_tool(text)

        if call is None:
            final_answer = text.strip()
            yield {"type": "final", "text": final_answer}
            yield {"type": "done"}
            return

        # Execute tool
        tool_name = call["tool"]
        tool_arg = call.get("content", "")
        yield {"type": "tool_call", "tool": tool_name, "args": tool_arg[:200]}

        result = execute_tool(call, allow_write=allow_write)
        yield {"type": "tool_result", "result": result[:2000]}

        continuation = (
            f"<result>\n{result}\n</result>\n\n"
            f"Now either call another tool or give a final answer (no tags):"
        )

    yield {"type": "final", "text": "(iteration limit reached)"}
    yield {"type": "done"}


def run_agent_turn(backend, user_question, max_iterations=6, max_tokens=300,
                   temperature=0.5, allow_write=False, verbose=True):
    """Synchronous wrapper around run_agent_turn_stream — used by the REPL."""
    final = "(no answer)"
    for event in run_agent_turn_stream(
        backend, user_question,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        temperature=temperature,
        allow_write=allow_write,
    ):
        et = event["type"]
        if et == "step_start" and verbose:
            print(f"\n  [Step {event['step']}/{event['max']}]", end=" ", flush=True)
        elif et == "token" and verbose:
            print(event["text"], end="", flush=True)
        elif et == "tool_call" and verbose:
            preview = event["args"][:60].replace("\n", " ")
            print(f"\n  → tool: <{event['tool']}> {preview}...")
        elif et == "tool_result" and verbose:
            preview = event["result"][:120].replace("\n", " | ")
            print(f"  ← {preview}{'...' if len(event['result']) > 120 else ''}")
        elif et == "final":
            final = event["text"]
        elif et == "error" and verbose:
            print(f"\n  ERROR: {event['message']}")
    return final


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
