#!/usr/bin/env python3
"""
mac code — claude code for your Mac
"""

import json, sys, os, time, subprocess, re, threading, queue
import urllib.request, random
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.live import Live
from rich.padding import Padding
from rich.columns import Columns

SERVER = os.environ.get("LLAMA_URL", "http://localhost:8000")

# ── Self-improvement: failure logging ─────────────
LOGS_DIR = Path.home() / ".mac-code" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def log_interaction(query, intent, response, speed, grade=None, error=None):
    """Log every interaction for self-improvement training data."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "intent": intent,
        "response": response[:500] if response else None,
        "speed": speed,
        "grade": grade,  # "good", "bad", or None (ungraded)
        "error": error,
        "model": get_current_model() if "get_current_model" in dir() else "unknown",
    }
    log_file = LOGS_DIR / f"interactions-{datetime.now().strftime('%Y-%m-%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_failure_stats():
    """Show stats from logged interactions."""
    total = 0
    graded = {"good": 0, "bad": 0}
    intents = {"search": 0, "shell": 0, "chat": 0}
    errors = 0

    for log_file in LOGS_DIR.glob("interactions-*.jsonl"):
        for line in open(log_file):
            try:
                entry = json.loads(line)
                total += 1
                if entry.get("grade"):
                    graded[entry["grade"]] = graded.get(entry["grade"], 0) + 1
                if entry.get("intent"):
                    intents[entry["intent"]] = intents.get(entry["intent"], 0) + 1
                if entry.get("error"):
                    errors += 1
            except:
                pass

    return {"total": total, "graded": graded, "intents": intents, "errors": errors}


PICOCLAW = os.path.expanduser("~/Desktop/qwen/picoclaw/build/picoclaw-darwin-arm64")
console = Console()

# ── model configs ─────────────────────────────────
MODELS = {
    "9b": {
        "path": os.path.expanduser("~/models/Qwen3.5-9B-Q4_K_M.gguf"),
        "ctx": 32768,
        "flags": "--flash-attn on --n-gpu-layers 99 --reasoning off -t 4",
        "name": "Qwen3.5-9B",
        "detail": "8.95B dense · Q4_K_M · 32K ctx",
        "good_for": "tool calling, long conversations, agent tasks",
    },
    "35b": {
        "path": os.path.expanduser("~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"),
        "ctx": 8192,
        "flags": "--flash-attn on --n-gpu-layers 99 --reasoning off -np 1 -t 4",
        "name": "Qwen3.5-35B-A3B",
        "detail": "MoE 34.7B · 3B active · IQ2_M · 8K ctx",
        "good_for": "reasoning, math, knowledge, fast answers",
    },
}

# ── smart routing ─────────────────────────────────
TOOL_KEYWORDS = [
    "search",
    "find",
    "look up",
    "google",
    "what time",
    "when do",
    "when is",
    "when does",
    "when are",
    "who do",
    "who is playing",
    "who plays",
    "who won",
    "what happened",
    "what is the score",
    "weather",
    "news",
    "latest",
    "schedule",
    "score",
    "tonight",
    "today",
    "tomorrow",
    "yesterday",
    "this week",
    "next game",
    "play next",
    "playing next",
    "results",
    "standings",
    "price",
    "stock",
    "market",
    "crypto",
    "bitcoin",
    "fetch",
    "download",
    "read file",
    "write file",
    "create file",
    "run",
    "execute",
    "list files",
    "show me",
    "open",
    "browse",
    "url",
    "http",
    "website",
    "how much",
    "where is",
    "directions",
    "recipe",
    "explore",
    "repo",
    "repository",
    "github",
    "tell me more",
    "more about",
    "what else",
    "continue",
    "go deeper",
]


def classify_intent(message):
    """Ask LLM to classify: 'search', 'shell', or 'chat'. One fast call (~1s)."""
    try:
        result, _ = llm_call(
            [
                {
                    "role": "system",
                    "content": """Classify the user's request into exactly one category. Reply with ONLY the category word, nothing else.

Categories:
- search: needs web search (news, scores, weather, prices, current events, looking up info online)
- shell: needs filesystem or command execution (find files, list directories, read/write files, run commands, look at desktop, explore folders, check disk space, anything involving the local computer)
- chat: general conversation, reasoning, math, coding questions, explanations (no tools needed)

Reply with ONLY one word: search, shell, or chat""",
                },
                {"role": "user", "content": message},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        return result.strip().lower().split()[0]
    except Exception:
        return "chat"


def generate_shell_command(query, work_dir="."):
    """Ask LLM to generate the right shell command for a file/system task."""
    home = os.path.expanduser("~")
    result, _ = llm_call(
        [
            {
                "role": "system",
                "content": f"""You are a macOS shell command generator. The user's home directory is {home}. Current working directory is {work_dir}.

Generate a single shell command that accomplishes the user's request. Output ONLY the command, nothing else. No explanation, no markdown, no backticks.

Examples:
- "find videos on my desktop" → find {home}/Desktop -type f \\( -name "*.mp4" -o -name "*.mov" -o -name "*.avi" -o -name "*.mkv" -o -name "*.webm" \\)
- "what files are on my desktop" → ls -la {home}/Desktop
- "how much disk space do I have" → df -h /
- "show me python files in this project" → find . -name "*.py" -type f
- "read the readme" → cat README.md
- "what's running on port 8000" → lsof -i :8000
- "count lines of code" → find . -name "*.py" -exec wc -l {{}} +""",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=100,
        temperature=0.0,
    )
    return result.strip().strip("`").strip()


def run_smart_tool(query, work_dir="."):
    """Execute a shell command generated by the LLM, feed results back."""
    import subprocess as sp
    from datetime import datetime

    # Step 1: LLM generates the command (~1s)
    cmd = generate_shell_command(query, work_dir)

    # Step 2: Execute it
    try:
        result = sp.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=work_dir
        )
        output = result.stdout[:8000]
        if result.stderr:
            output += f"\n{result.stderr[:2000]}"
    except sp.TimeoutExpired:
        output = "Command timed out after 30 seconds"
    except Exception as e:
        output = f"Error: {e}"

    # Step 3: LLM summarizes results (~2-3s)
    today = datetime.now().strftime("%A, %B %d, %Y")
    content, timings = llm_call(
        [
            {
                "role": "system",
                "content": f"Today is {today}. You ran a shell command and got results. Present the results clearly to the user. If it's a file listing, format it nicely. If it's code, use formatting. Be helpful and concise.",
            },
            {
                "role": "user",
                "content": f"Command: {cmd}\nOutput:\n{output}\n\nOriginal question: {query}",
            },
        ],
        max_tokens=1000,
    )

    return content, timings.get("predicted_per_second", 0), cmd


def run_file_tool(query, work_dir="."):
    """Execute file/exec operations directly in Python, feed results to LLM."""
    import subprocess as sp
    from datetime import datetime

    lower = query.lower()
    tool_output = ""
    tool_name = ""

    try:
        # List directory
        if any(kw in lower for kw in ["list files", "list dir", "ls ", "what's in"]):
            # Extract path or use work_dir
            path = work_dir
            for token in query.split():
                expanded = os.path.expanduser(token)
                if os.path.isdir(expanded):
                    path = expanded
                    break
            entries = os.listdir(path)
            entries.sort()
            tool_name = f"list_dir({path})"
            tool_output = "\n".join(entries[:50])
            if len(entries) > 50:
                tool_output += f"\n... and {len(entries) - 50} more"

        # Read file
        elif any(
            kw in lower
            for kw in ["read file", "show me", "look at", "cat ", "what's in"]
        ):
            # Find file path in the query
            path = None
            for token in query.split():
                expanded = os.path.expanduser(token)
                if os.path.isfile(expanded):
                    path = expanded
                    break
                # Try with work_dir
                joined = os.path.join(work_dir, token)
                if os.path.isfile(joined):
                    path = joined
                    break
            if path:
                with open(path, "r", errors="ignore") as f:
                    content = f.read(10000)
                tool_name = f"read_file({path})"
                tool_output = content
            else:
                tool_output = f"Could not find file in query: {query}"
                tool_name = "read_file(not found)"

        # Write file
        elif any(
            kw in lower
            for kw in [
                "write file",
                "write a file",
                "create file",
                "create a file",
                "create a new",
                "save file",
                "save to",
                "save this",
            ]
        ):
            # LLM decides what to write
            content, _ = llm_call(
                [
                    {
                        "role": "system",
                        "content": "The user wants to create/write a file. Generate ONLY the file content. No explanations.",
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=2000,
            )

            # Extract filename from query
            filename = None
            for token in query.split():
                if "." in token and not token.startswith("http"):
                    filename = token
                    break
            if not filename:
                filename = "output.txt"

            filepath = os.path.join(work_dir, filename)
            with open(filepath, "w") as f:
                f.write(content)
            tool_name = f"write_file({filepath})"
            tool_output = f"Written {len(content)} bytes to {filepath}"

        # Execute command
        elif any(kw in lower for kw in ["execute", "run "]):
            # Extract command
            cmd = query
            for prefix in ["execute ", "run "]:
                if lower.startswith(prefix):
                    cmd = query[len(prefix) :]
                    break

            result = sp.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=work_dir,
            )
            tool_name = f"exec({cmd.strip()[:40]})"
            tool_output = result.stdout[:5000]
            if result.stderr:
                tool_output += f"\nSTDERR: {result.stderr[:1000]}"

        else:
            return None

    except Exception as e:
        tool_output = f"Error: {e}"
        tool_name = "error"

    # Feed tool output to LLM for final answer
    today = datetime.now().strftime("%A, %B %d, %Y")
    content, timings = llm_call(
        [
            {
                "role": "system",
                "content": f"Today is {today}. You executed a tool and got results. Summarize the results clearly for the user. If it's code, format it nicely.",
            },
            {
                "role": "user",
                "content": f"Tool: {tool_name}\nResult:\n{tool_output}\n\nOriginal question: {query}",
            },
        ],
        max_tokens=1000,
    )

    return content, timings.get("predicted_per_second", 0), tool_name


def llm_call(messages, max_tokens=300, temperature=0.1):
    """Single LLM call, returns content + timings."""
    payload = json.dumps(
        {
            "model": "local",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return d["choices"][0]["message"]["content"], d.get("timings", {})


def quick_search(query):
    """LLM rewrites query → DuckDuckGo search → LLM answers. ~5-8s total."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None

    from datetime import datetime

    today = datetime.now().strftime("%A, %B %d, %Y")

    # Step 1: LLM rewrites query into optimal search terms (~1s)
    try:
        search_query, _ = llm_call(
            [
                {
                    "role": "system",
                    "content": f"Today is {today}. Rewrite the user's question into an optimal web search query that will find current, specific data (not articles about announcements). Include 'today' or 'tonight' and the full date for time-sensitive queries. Add words like 'scores', 'results', 'live', or 'now' when looking for current data. Output ONLY the search query string, nothing else.",
                },
                {"role": "user", "content": query},
            ],
            max_tokens=30,
            temperature=0.0,
        )
        search_query = search_query.strip().strip("\"'")
    except Exception:
        search_query = query

    # Step 2: DuckDuckGo search — text (15 results) + news (5 results)
    ddg = DDGS()
    all_results = []

    try:
        text_results = ddg.text(search_query, max_results=15)
        all_results.extend(text_results)
    except Exception:
        pass

    try:
        news_results = ddg.news(search_query, max_results=5)
        all_results.extend(news_results)
    except Exception:
        pass

    if not all_results:
        return None

    # Combine all snippets
    snippets = "\n".join(
        [f"- {r.get('title', '')}: {r.get('body', '')}" for r in all_results]
    )

    # Check if snippets actually contain useful data or just meta descriptions
    # If total snippet text is mostly generic, fetch the best page
    import re as _re

    page_content = ""
    snippet_words = len(snippets.split())

    # Heuristic: check if snippets have actual specific data
    # Numbers with context (times, scores, prices) count. Generic "live scores available" doesn't.
    specific_patterns = _re.findall(
        r"\d{1,2}:\d{2}\s*(?:p\.m\.|a\.m\.|ET|PT)|\$[\d,.]+|\d+-\d+(?:\s*(?:win|loss|final))",
        snippets.lower(),
    )
    has_specifics = len(specific_patterns) >= 2  # need at least 2 specific data points

    if not has_specifics and all_results:
        # Snippets are weak — use Jina Reader to fetch the best page
        # Jina reads JS-rendered pages (ESPN, etc.) that urllib can't
        for r in all_results[:3]:
            url = r.get("href") or r.get("link", "")
            if not url:
                continue
            try:
                jina_url = f"https://r.jina.ai/{url}"
                req = urllib.request.Request(
                    jina_url,
                    headers={
                        "User-Agent": "Mozilla/5.0",
                        "Accept": "text/plain",
                    },
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    text = resp.read(6000).decode("utf-8", errors="ignore")
                    if len(text) > 200:
                        page_content = text[:4000]
                        break
            except Exception:
                continue

    context = snippets
    if page_content:
        context += f"\n\nDetailed content from top result:\n{page_content}"

    # Step 3: LLM answers using results (~2-3s)
    content, timings = llm_call(
        [
            {
                "role": "system",
                "content": f"Today is {today}. Answer the user's question using the search results below. Be specific, direct, and detailed. Extract dates, times, scores, names, numbers, prices, and facts. Present them clearly.",
            },
            {
                "role": "user",
                "content": f"Search results:\n\n{context}\n\nQuestion: {query}",
            },
        ],
        max_tokens=1000,
    )

    return content, timings.get("predicted_per_second", 0)


def get_current_model():
    """Check which model the running server has loaded."""
    try:
        req = urllib.request.Request(f"{SERVER}/props")
        with urllib.request.urlopen(req, timeout=3) as r:
            d = json.loads(r.read())
        alias = d.get("model_alias", "") or d.get("model_path", "")
        if "35B-A3B" in alias:
            return "35b"
        elif "9B" in alias:
            return "9b"
    except Exception:
        pass
    return None


def swap_model(target_key):
    """Stop current server and start a new one with the target model."""
    cfg = MODELS[target_key]
    if not os.path.exists(cfg["path"]):
        return False, f"Model not found: {cfg['path']}"

    # Kill current server
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)

    # Start new server
    cmd_list = [
        "llama-server",
        "--model",
        cfg["path"],
        "--port",
        "8000",
        "--host",
        "127.0.0.1",
        "--ctx-size",
        str(cfg["ctx"]),
    ] + cfg["flags"].split()
    subprocess.Popen(cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for ready
    for i in range(30):
        time.sleep(2)
        try:
            req = urllib.request.Request(f"{SERVER}/health")
            with urllib.request.urlopen(req, timeout=2) as r:
                d = json.loads(r.read())
            if d.get("status") == "ok":
                return True, f"Switched to {cfg['name']} ({cfg['ctx']} ctx)"
        except Exception:
            pass

    return False, "Server failed to start"


# ── ANSI strip ─────────────────────────────────────
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\r")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


# ── live working display ──────────────────────────
DOTS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class WorkingDisplay:
    def __init__(self):
        self.events = []
        self.phase = "thinking"
        self.frame = 0
        self.start_time = time.time()
        self.logs = []

    def add_log(self, line):
        clean = strip_ansi(line).strip()
        if not clean:
            return

        lower = clean.lower()
        new_phase = None
        detail = ""

        if "processing message" in lower:
            new_phase = "reading your message"
        elif "llm_request" in lower:
            new_phase = "thinking"
        elif "tool_call" in lower or "web_search" in lower:
            if "web_search" in lower or "duckduckgo" in lower:
                new_phase = "searching the web"
            elif "web_fetch" in lower or "fetch" in lower:
                new_phase = "fetching page"
            elif "exec" in lower:
                new_phase = "running command"
            elif "read_file" in lower:
                new_phase = "reading file"
            elif "write_file" in lower:
                new_phase = "writing file"
            else:
                new_phase = "using tools"
        elif "context_compress" in lower:
            new_phase = "compressing context"
        elif "turn_end" in lower:
            new_phase = "finishing up"

        if new_phase:
            self.phase = new_phase
            self.events.append((time.time() - self.start_time, new_phase, detail))

        # Keep last few interesting log lines
        if any(
            k in lower
            for k in [
                "llm_request",
                "tool_call",
                "tool_result",
                "turn_end",
                "web_search",
                "fetch",
                "exec",
            ]
        ):
            short = clean
            if ">" in short:
                short = short.split(">", 1)[-1].strip()
            if len(short) > 70:
                short = short[:67] + "..."
            self.logs.append(short)
            if len(self.logs) > 3:
                self.logs.pop(0)

    def render(self):
        self.frame += 1
        elapsed = time.time() - self.start_time
        spinner = DOTS[self.frame % len(DOTS)]

        t = Text()
        t.append(f"  {spinner} ", style="bold bright_cyan")
        t.append(self.phase, style="bold bright_cyan")
        t.append(f"  {elapsed:.0f}s", style="dim")
        t.append("\n")

        for log in self.logs[-3:]:
            t.append(f"    {log}\n", style="dim italic")

        return t


# ── detect model ───────────────────────────────────
def detect_model():
    try:
        req = urllib.request.Request(f"{SERVER}/props")
        with urllib.request.urlopen(req, timeout=3) as r:
            d = json.loads(r.read())
        alias = d.get("model_alias", "") or d.get("model_path", "")
        if "35B-A3B" in alias:
            return "Qwen3.5-35B-A3B", "MoE 34.7B · 3B active · IQ2_M"
        elif "9B" in alias:
            return "Qwen3.5-9B", "8.95B dense · Q4_K_M"
        return alias.replace(".gguf", "").split("/")[-1], "local"
    except Exception:
        return "offline", ""


# ── streaming chat (raw mode) ─────────────────────
def stream_llm(messages):
    payload = json.dumps(
        {
            "model": "local",
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.7,
            "stream": True,
        }
    ).encode()

    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full = ""
    start = time.time()
    tokens = 0

    with urllib.request.urlopen(req, timeout=300) as resp:
        buf = ""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw == "[DONE]":
                    return full, tokens, time.time() - start
                try:
                    obj = json.loads(raw)
                    delta = obj["choices"][0].get("delta", {})
                    c = delta.get("content", "")
                    if c:
                        full += c
                        tokens += 1
                        yield c
                except Exception:
                    pass

    return full, tokens, time.time() - start


# ── picoclaw agent call with LIVE log streaming ───
def picoclaw_call_live(message, session="mac-code"):
    """Run picoclaw with real-time log streaming into animated display."""
    cmd = [PICOCLAW, "agent", "-m", message, "-s", session]
    display = WorkingDisplay()
    all_lines = []

    # Launch with Popen — picoclaw writes everything to stdout
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    # Read stdout line-by-line in a thread for real-time updates
    def read_output():
        try:
            for line in proc.stdout:
                all_lines.append(line)
                display.add_log(line)
        except Exception:
            pass

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()

    # Animate while process runs
    with Live(
        display.render(), console=console, refresh_per_second=8, transient=True
    ) as live:
        while proc.poll() is None:
            live.update(display.render())
            time.sleep(0.12)
        # Give reader a moment to finish
        time.sleep(0.3)
        live.update(display.render())

    reader.join(timeout=2)

    # Parse: strip ANSI, find lobster emoji, take text after it
    raw = "".join(all_lines)
    clean = strip_ansi(raw)

    idx = clean.rfind("\U0001f99e")  # last lobster emoji
    if idx >= 0:
        response = clean[idx:].lstrip("\U0001f99e").strip()
        # If it starts with "Error:" it's a picoclaw error, not a model response
        if response.startswith("Error:"):
            # Extract the useful part of the error
            response = f"[agent error] {response[:200]}"
    else:
        # No lobster — take non-banner lines
        lines = clean.split("\n")
        resp = []
        past = False
        for line in lines:
            s = line.strip()
            if not past:
                if not s or any(c in s for c in ["██", "╔", "╚", "╝", "║"]):
                    continue
                past = True
            if past and s:
                resp.append(s)
        response = "\n".join(resp).strip()

    return response, display.events


# ── banner ─────────────────────────────────────────
def print_banner(model_name, model_detail):
    console.print()
    logo = Text()
    logo.append("  \U0001f34e ", style="default")
    logo.append("mac", style="bold bright_cyan")
    logo.append(" ", style="default")
    logo.append("code", style="bold bright_yellow")
    console.print(logo)

    sub = Text()
    sub.append("  claude code, but it runs on your Mac for free", style="dim italic")
    console.print(sub)
    console.print()

    rows = [
        ("model", model_name, model_detail),
        ("tools", "search · fetch · exec · files", ""),
        ("cost", "$0.00/hr", "Apple M4 Metal · localhost:8000"),
    ]
    for label, value, extra in rows:
        line = Text()
        line.append(f"  {label:6s} ", style="bold dim")
        line.append(value, style="bold white")
        if extra:
            line.append(f"  {extra}", style="dim")
        console.print(line)

    console.print()
    console.print(Rule(style="dim"))
    console.print(
        "  [dim]type [bold bright_cyan]/[/bold bright_cyan] to see all commands[/]\n"
    )


# ── render helpers ─────────────────────────────────
def render_response(response):
    """Render a response — use Rich Markdown if it has formatting, plain text otherwise."""
    if any(c in response for c in ["##", "**", "```", "| ", "- ", "1. ", "* "]):
        console.print(Padding(Markdown(response), (0, 2)))
    else:
        for line in response.split("\n"):
            console.print(f"  {line}")


def render_speed(tokens, elapsed):
    if elapsed <= 0 or tokens <= 0:
        return
    speed = tokens / elapsed
    clr = "bright_green" if speed > 20 else "yellow" if speed > 10 else "red"
    s = Text()
    s.append(f"  {speed:.1f} tok/s", style=f"bold {clr}")
    s.append(f"  ·  {tokens} tokens  ·  {elapsed:.1f}s", style="dim")
    console.print(s)


def render_timeline(events):
    """Show a compact summary of what the agent did."""
    if not events:
        return
    summary = []
    last_phase = None
    for ts, phase, detail in events:
        if phase != last_phase:
            summary.append(phase)
            last_phase = phase

    if len(summary) <= 1:
        return

    t = Text()
    t.append("  ", style="dim")
    for i, phase in enumerate(summary):
        t.append(phase, style="dim italic")
        if i < len(summary) - 1:
            t.append(" → ", style="dim")
    console.print(t)


# ── commands ───────────────────────────────────────
COMMANDS = [
    ("/agent", "Switch to agent mode (tools + web search)"),
    ("/raw", "Switch to raw mode (direct streaming, no tools)"),
    ("/btw", "Ask a side question without adding to conversation history"),
    ("/loop", "Run a prompt on a recurring interval — /loop 5m <prompt>"),
    ("/branch", "Save conversation checkpoint you can restore later"),
    ("/restore", "Restore last saved conversation checkpoint"),
    ("/add-dir", "Set working directory — /add-dir <path>"),
    ("/save", "Save conversation to a file — /save <filename>"),
    ("/search", "Quick web search — /search <query>"),
    ("/bench", "Run a quick speed benchmark"),
    ("/clear", "Clear conversation and start fresh"),
    ("/stats", "Show session statistics"),
    ("/model", "Show or switch model — /model 9b or /model 35b"),
    ("/auto", "Toggle smart auto-routing between 9B and 35B"),
    ("/tools", "List available agent tools"),
    ("/system", "Set system prompt — /system <message>"),
    ("/compact", "Toggle compact output (no markdown rendering)"),
    ("/stop", "Stop a running /loop"),
    ("/cost", "Show estimated cost savings vs cloud APIs"),
    ("/good", "Grade last response as good (for self-improvement)"),
    ("/bad", "Grade last response as bad (for self-improvement)"),
    ("/improve", "Show self-improvement stats from logged interactions"),
    ("/quit", "Exit mac code"),
]


def show_slash_menu(filter_text=""):
    """Print slash commands inline — like Claude Code."""
    matches = COMMANDS
    if filter_text and filter_text != "/":
        matches = [(c, d) for c, d in COMMANDS if c.startswith(filter_text)]

    for cmd, desc in matches:
        line = Text()
        line.append(f"  {cmd}", style="bold bright_cyan")
        pad = " " * max(14 - len(cmd), 1)
        line.append(pad)
        line.append(desc, style="dim")
        console.print(line)


# ── main ───────────────────────────────────────────
def main():
    model_name, model_detail = detect_model()
    console.clear()
    print_banner(model_name, model_detail)

    messages = []
    session_tokens = 0
    session_time = 0.0
    session_turns = 0
    session_id = f"mc-{int(time.time())}"
    use_agent = True
    compact_mode = False
    auto_route = True  # smart routing between 9B and 35B
    work_dir = os.getcwd()
    branch_save = None
    loop_thread = None
    loop_running = False
    last_interaction = None  # for /good /bad grading

    while True:
        try:
            cur = get_current_model() or "?"
            tag = f"{'auto' if auto_route else 'agent'} {cur}" if use_agent else "raw"
            console.print(f"  [dim]{tag}[/] [bold bright_yellow]>[/] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip()
        cmd_lower = cmd.lower()

        # ── slash command handling ─────────────
        if cmd == "/":
            show_slash_menu()
            continue
        elif cmd_lower.startswith("/") and not cmd_lower.startswith("/system "):
            # Check for partial match — typing "/st" shows "/stats" and "/system"
            exact = cmd_lower.split()[0]

            if exact in ("/quit", "/exit", "/q"):
                break
            elif exact == "/clear":
                messages.clear()
                session_id = f"mc-{int(time.time())}"
                console.clear()
                print_banner(model_name, model_detail)
                console.print("  [dim]cleared.[/]\n")
                continue
            elif exact == "/stats":
                avg = session_tokens / session_time if session_time > 0 else 0
                t = Table(show_header=False, box=None, padding=(0, 1))
                t.add_column(style="bold bright_cyan", width=12)
                t.add_column()
                t.add_row("turns", str(session_turns))
                t.add_row("tokens", f"{session_tokens:,}")
                t.add_row("time", f"{session_time:.1f}s")
                t.add_row("avg speed", f"{avg:.1f} tok/s")
                t.add_row("mode", tag)
                console.print(t)
                console.print()
                continue
            elif exact == "/model":
                # Check if user passed an argument like "/model 9b"
                parts = cmd.split()
                if len(parts) >= 2:
                    target = parts[1].lower().replace("b", "b")
                    if target in MODELS:
                        console.print(
                            f"  [dim]swapping to {MODELS[target]['name']}...[/]"
                        )
                        display = WorkingDisplay()
                        display.phase = f"loading {MODELS[target]['name']}"
                        with Live(
                            display.render(),
                            console=console,
                            refresh_per_second=8,
                            transient=True,
                        ) as live:
                            ok, msg = swap_model(target)
                            while not ok and display.frame < 100:
                                display.frame += 1
                                live.update(display.render())
                                time.sleep(0.2)
                        if ok:
                            model_name = MODELS[target]["name"]
                            model_detail = MODELS[target]["detail"]
                            console.print(f"  [bold bright_green]{msg}[/]\n")
                        else:
                            console.print(f"  [bold red]{msg}[/]\n")
                    else:
                        console.print(f"  [dim]available: 9b, 35b[/]\n")
                else:
                    cur = get_current_model()
                    model_name, model_detail = detect_model()
                    console.print(
                        f"  [bold white]{model_name}[/]  [dim]{model_detail}[/]"
                    )
                    console.print(
                        f"  [dim]auto-routing: {'on' if auto_route else 'off'}[/]"
                    )
                    console.print(f"  [dim]switch: /model 9b  or  /model 35b[/]\n")
                continue

            elif exact == "/auto":
                auto_route = not auto_route
                state = "on" if auto_route else "off"
                console.print(f"  [dim]smart auto-routing {state}[/]")
                if auto_route:
                    console.print(f"  [dim]  tools/search → 9B (32K ctx, reliable)[/]")
                    console.print(f"  [dim]  reasoning     → 35B (faster, smarter)[/]")
                console.print()
                continue
            elif exact == "/tools":
                for name, desc in [
                    ("web_search", "DuckDuckGo"),
                    ("web_fetch", "read URLs"),
                    ("exec", "shell commands"),
                    ("read_file", "local files"),
                    ("write_file", "create files"),
                    ("edit_file", "modify files"),
                    ("list_dir", "browse dirs"),
                    ("subagent", "spawn tasks"),
                ]:
                    t = Text()
                    t.append("  ▸ ", style="bright_cyan")
                    t.append(name, style="bold bright_cyan")
                    t.append(f"  {desc}", style="dim")
                    console.print(t)
                console.print()
                continue
            elif exact == "/agent":
                use_agent = True
                console.print("  [dim]agent mode (tools enabled)[/]\n")
                continue
            elif exact == "/raw":
                use_agent = False
                console.print("  [dim]raw mode (streaming, no tools)[/]\n")
                continue
            elif exact == "/compact":
                compact_mode = not compact_mode
                state = "on" if compact_mode else "off"
                console.print(f"  [dim]compact mode {state}[/]\n")
                continue

            elif exact == "/branch":
                branch_save = [m.copy() for m in messages]
                console.print(
                    f"  [dim]conversation saved ({len(messages)} messages). use /restore to go back.[/]\n"
                )
                continue

            elif exact == "/restore":
                if branch_save is not None:
                    messages = [m.copy() for m in branch_save]
                    console.print(
                        f"  [dim]restored to checkpoint ({len(messages)} messages)[/]\n"
                    )
                else:
                    console.print("  [dim]no checkpoint saved. use /branch first.[/]\n")
                continue

            elif exact == "/bench":
                console.print("  [dim]running speed benchmark...[/]")
                try:
                    payload = json.dumps(
                        {
                            "model": "local",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": "Count from 1 to 50, one number per line.",
                                }
                            ],
                            "max_tokens": 300,
                            "temperature": 0.1,
                        }
                    ).encode()
                    req = urllib.request.Request(
                        f"{SERVER}/v1/chat/completions",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    bstart = time.time()
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        d = json.loads(resp.read())
                    belapsed = time.time() - bstart
                    t = d.get("timings", {})
                    u = d.get("usage", {})
                    gen_speed = t.get("predicted_per_second", 0)
                    prompt_speed = t.get("prompt_per_second", 0)
                    tokens = u.get("completion_tokens", 0)
                    console.print(
                        f"  [bold bright_green]{gen_speed:.1f} tok/s[/] generation"
                    )
                    console.print(
                        f"  [bold bright_green]{prompt_speed:.1f} tok/s[/] prompt processing"
                    )
                    console.print(f"  [dim]{tokens} tokens in {belapsed:.1f}s[/]\n")
                except Exception as e:
                    console.print(f"  [bold red]benchmark failed: {e}[/]\n")
                continue

            elif exact == "/cost":
                cloud_rate = 0.34  # $/hr RunPod equivalent
                hours = session_time / 3600 if session_time > 0 else 0
                saved = cloud_rate * max(hours, 1 / 60)
                console.print(f"  [bold bright_green]$0.00[/] spent locally")
                console.print(
                    f"  [dim]~${saved:.4f} would have cost on cloud GPU (${cloud_rate}/hr)[/]"
                )
                console.print(
                    f"  [dim]session: {session_time:.0f}s · {session_tokens:,} tokens[/]\n"
                )
                continue

            elif exact == "/good":
                # Grade last response as good
                if last_interaction:
                    last_interaction["grade"] = "good"
                    log_interaction(**last_interaction)
                    console.print("  [bright_green]marked good[/]\n")
                else:
                    console.print("  [dim]no response to grade[/]\n")
                continue

            elif exact == "/bad":
                # Grade last response as bad
                if last_interaction:
                    last_interaction["grade"] = "bad"
                    log_interaction(**last_interaction)
                    console.print(
                        "  [bright_red]marked bad — logged for improvement[/]\n"
                    )
                else:
                    console.print("  [dim]no response to grade[/]\n")
                continue

            elif exact == "/improve":
                stats = get_failure_stats()
                t = Table(show_header=False, box=None, padding=(0, 1))
                t.add_column(style="bold bright_cyan", width=14)
                t.add_column()
                t.add_row("total", str(stats["total"]))
                t.add_row("good", str(stats["graded"].get("good", 0)))
                t.add_row("bad", str(stats["graded"].get("bad", 0)))
                t.add_row("errors", str(stats["errors"]))
                t.add_row("searches", str(stats["intents"].get("search", 0)))
                t.add_row("shell", str(stats["intents"].get("shell", 0)))
                t.add_row("chat", str(stats["intents"].get("chat", 0)))
                t.add_row("logs", str(LOGS_DIR))
                console.print(t)
                console.print()
                continue

            elif exact in ("/help", "/?"):
                show_slash_menu()
                continue
            else:
                # Partial match — show filtered results
                show_slash_menu(exact)
                continue

        # ── commands with arguments ────────────
        elif cmd_lower.startswith("/system "):
            sys_msg = cmd[8:].strip()
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = sys_msg
            else:
                messages.insert(0, {"role": "system", "content": sys_msg})
            console.print(f"  [dim italic]system: {sys_msg[:80]}[/]\n")
            continue

        elif cmd_lower.startswith("/btw "):
            # Side question — don't add to conversation history
            side_q = cmd[5:].strip()
            if not side_q:
                console.print("  [dim]/btw <question>[/]\n")
                continue
            console.print()
            if use_agent:
                start = time.time()
                # Use a separate session so it doesn't pollute main conversation
                response, events = picoclaw_call_live(
                    side_q, session=f"btw-{int(time.time())}"
                )
                elapsed = time.time() - start
                if response:
                    console.print(f"  [dim italic](side answer)[/]")
                    render_response(response)
                    console.print()
                    tokens_est = len(response.split())
                    render_speed(tokens_est, elapsed)
                    session_tokens += tokens_est
                    session_time += elapsed
            else:
                side_msgs = [{"role": "user", "content": side_q}]
                try:
                    payload = json.dumps(
                        {
                            "model": "local",
                            "messages": side_msgs,
                            "max_tokens": 2000,
                            "temperature": 0.7,
                        }
                    ).encode()
                    req = urllib.request.Request(
                        f"{SERVER}/v1/chat/completions",
                        data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        d = json.loads(resp.read())
                    content = d["choices"][0]["message"]["content"]
                    console.print(f"  [dim italic](side answer)[/]")
                    for line in content.split("\n"):
                        console.print(f"  {line}")
                except Exception as e:
                    console.print(f"  [bold red]{e}[/]")
            console.print()
            continue

        elif cmd_lower.startswith("/add-dir "):
            new_dir = os.path.expanduser(cmd[9:].strip())
            if os.path.isdir(new_dir):
                work_dir = new_dir
                os.chdir(work_dir)
                console.print(f"  [dim]working directory: {work_dir}[/]\n")
            else:
                console.print(f"  [bold red]not a directory: {new_dir}[/]\n")
            continue

        elif cmd_lower.startswith("/save "):
            filename = cmd[6:].strip()
            if not filename:
                filename = f"conversation-{int(time.time())}.json"
            try:
                save_path = os.path.join(work_dir, filename)
                with open(save_path, "w") as f:
                    json.dump(
                        {
                            "messages": messages,
                            "session_id": session_id,
                            "tokens": session_tokens,
                            "time": session_time,
                            "turns": session_turns,
                        },
                        f,
                        indent=2,
                    )
                console.print(f"  [dim]saved to {save_path}[/]\n")
            except Exception as e:
                console.print(f"  [bold red]{e}[/]\n")
            continue

        elif cmd_lower.startswith("/search "):
            query = cmd[8:].strip()
            if not query:
                console.print("  [dim]/search <query>[/]\n")
                continue
            console.print()
            start = time.time()
            response, events = picoclaw_call_live(
                f"Search the web for: {query}. Give a brief summary of the top results.",
                session=f"search-{int(time.time())}",
            )
            elapsed = time.time() - start
            if response:
                for line in response.split("\n"):
                    console.print(f"  {line}")
                console.print()
                s = Text()
                s.append(f"  ▸ agent", style="bold bright_cyan")
                s.append(f"  {elapsed:.1f}s total (search + inference)", style="dim")
                console.print(s)
                session_tokens += len(response.split())
                session_time += elapsed
            console.print()
            continue

        elif cmd_lower.startswith("/loop "):
            # Parse: /loop 5m <prompt>
            parts = cmd[6:].strip().split(None, 1)
            if len(parts) < 2:
                console.print(
                    "  [dim]/loop <interval> <prompt>  — e.g. /loop 5m check server status[/]\n"
                )
                continue

            interval_str, loop_prompt = parts
            # Parse interval
            try:
                if interval_str.endswith("m"):
                    interval_sec = int(interval_str[:-1]) * 60
                elif interval_str.endswith("s"):
                    interval_sec = int(interval_str[:-1])
                elif interval_str.endswith("h"):
                    interval_sec = int(interval_str[:-1]) * 3600
                else:
                    interval_sec = int(interval_str) * 60  # default minutes
            except ValueError:
                console.print(f"  [bold red]invalid interval: {interval_str}[/]\n")
                continue

            if loop_running:
                loop_running = False
                console.print("  [dim]stopped previous loop[/]")
                time.sleep(1)

            loop_running = True
            console.print(f"  [dim]looping every {interval_sec}s: {loop_prompt}[/]")
            console.print(f"  [dim]type /stop to cancel[/]\n")

            def run_loop(prompt, interval, sid):
                nonlocal loop_running, session_tokens, session_time
                while loop_running:
                    time.sleep(interval)
                    if not loop_running:
                        break
                    console.print(
                        f"\n  [dim italic]loop: running '{prompt[:40]}...'[/]"
                    )
                    resp, _ = picoclaw_call_live(prompt, session=sid)
                    if resp:
                        for line in resp.split("\n"):
                            console.print(f"  {line}")
                    console.print()

            loop_thread = threading.Thread(
                target=run_loop,
                args=(loop_prompt, interval_sec, f"loop-{session_id}"),
                daemon=True,
            )
            loop_thread.start()
            continue

        elif cmd_lower == "/stop":
            if loop_running:
                loop_running = False
                console.print("  [dim]loop stopped[/]\n")
            else:
                console.print("  [dim]no loop running[/]\n")
            continue

        console.print()

        # ── agent mode ─────────────────────────────
        if use_agent:
            start = time.time()

            # LLM classifies intent: search, shell, or chat (~1s)
            display = WorkingDisplay()
            display.phase = "classifying"
            intent_result = [None]

            def do_classify():
                intent_result[0] = classify_intent(user_input)

            cls_thread = threading.Thread(target=do_classify, daemon=True)
            cls_thread.start()

            with Live(
                display.render(), console=console, refresh_per_second=8, transient=True
            ) as live:
                while cls_thread.is_alive():
                    display.frame += 1
                    live.update(display.render())
                    time.sleep(0.12)

            cls_thread.join(timeout=1)
            intent = intent_result[0] or "chat"

            # Route based on LLM classification
            if intent == "shell":
                # File/system operations → LLM generates shell command
                display = WorkingDisplay()
                display.phase = "running command"
                tool_result = [None]

                def do_tool():
                    try:
                        tool_result[0] = run_smart_tool(user_input, work_dir)
                    except Exception as e:
                        tool_result[0] = None

                tool_thread = threading.Thread(target=do_tool, daemon=True)
                tool_thread.start()

                with Live(
                    display.render(),
                    console=console,
                    refresh_per_second=8,
                    transient=True,
                ) as live:
                    while tool_thread.is_alive():
                        display.frame += 1
                        t = time.time() - start
                        if t < 2:
                            display.phase = "generating command"
                        elif t < 5:
                            display.phase = "executing"
                        else:
                            display.phase = "reading results"
                        live.update(display.render())
                        time.sleep(0.12)

                tool_thread.join(timeout=1)
                result = tool_result[0]
                elapsed = time.time() - start

                if result:
                    response, speed, cmd = result
                    console.print()
                    console.print(f"  [dim]$ {cmd}[/]")
                    console.print()
                    render_response(response)
                    console.print()
                    s = Text()
                    s.append(f"  ▸ shell", style="bold bright_cyan")
                    s.append(f"  {elapsed:.1f}s", style="dim")
                    if speed > 0:
                        s.append(f"  ·  {speed:.1f} tok/s", style="bright_green")
                    console.print(s)
                    session_tokens += len(response.split())
                    session_time += elapsed
                    session_turns += 1
                    last_interaction = {
                        "query": user_input,
                        "intent": "shell",
                        "response": response,
                        "speed": speed,
                    }
                    messages.append({"role": "user", "content": user_input})
                    messages.append({"role": "assistant", "content": response})
                else:
                    console.print(f"  [bold red]command failed[/]\n")

            elif intent == "search":
                # Web search → fast direct path (~3-5s)
                display = WorkingDisplay()
                display.phase = "searching the web"
                search_result = [None]

                def do_search():
                    try:
                        search_result[0] = quick_search(user_input)
                    except Exception:
                        search_result[0] = None

                search_thread = threading.Thread(target=do_search, daemon=True)
                search_thread.start()

                with Live(
                    display.render(),
                    console=console,
                    refresh_per_second=8,
                    transient=True,
                ) as live:
                    while search_thread.is_alive():
                        display.frame += 1
                        t = time.time() - start
                        if t < 2:
                            display.phase = "rewriting query"
                        elif t < 3:
                            display.phase = "searching the web"
                        else:
                            display.phase = "generating answer"
                        live.update(display.render())
                        time.sleep(0.12)

                search_thread.join(timeout=1)
                result = search_result[0]
                elapsed = time.time() - start

                if result:
                    response, speed = result
                    console.print()
                    render_response(response)
                    console.print()
                    s = Text()
                    s.append(f"  ▸ search", style="bold bright_cyan")
                    s.append(f"  {elapsed:.1f}s", style="dim")
                    if speed > 0:
                        s.append(f"  ·  {speed:.1f} tok/s", style="bright_green")
                    console.print(s)
                    session_tokens += len(response.split())
                    session_time += elapsed
                    session_turns += 1
                    messages.append({"role": "user", "content": user_input})
                    messages.append({"role": "assistant", "content": response})
                    last_interaction = {
                        "query": user_input,
                        "intent": "search",
                        "response": response,
                        "speed": speed,
                    }
                else:
                    # Search failed, fall back to direct LLM
                    console.print("  [dim]search failed, asking model directly...[/]")
                    messages.append({"role": "user", "content": user_input})
                    full = ""
                    tokens = 0
                    first_token = True
                    display.phase = "thinking"
                    with Live(
                        display.render(),
                        console=console,
                        refresh_per_second=8,
                        transient=True,
                    ) as live:
                        gen = stream_llm(messages)
                        for chunk in gen:
                            if isinstance(chunk, str):
                                if first_token:
                                    first_token = False
                                    live.stop()
                                    console.print("  ", end="")
                                console.print(chunk, end="", highlight=False)
                                full += chunk
                                tokens += 1
                    elapsed = time.time() - start
                    console.print("\n")
                    render_speed(tokens, elapsed)
                    session_tokens += tokens
                    session_time += elapsed
                    session_turns += 1
                    messages.append({"role": "assistant", "content": full})
            else:
                # Direct LLM streaming (no tools needed)
                messages.append({"role": "user", "content": user_input})
                full = ""
                tokens = 0
                first_token = True
                display = WorkingDisplay()
                display.phase = "thinking"
                try:
                    with Live(
                        display.render(),
                        console=console,
                        refresh_per_second=8,
                        transient=True,
                    ) as live:
                        gen = stream_llm(messages)
                        for chunk in gen:
                            if isinstance(chunk, str):
                                if first_token:
                                    first_token = False
                                    live.stop()
                                    console.print("  ", end="")
                                console.print(chunk, end="", highlight=False)
                                full += chunk
                                tokens += 1
                    elapsed = time.time() - start
                    console.print("\n")
                    render_speed(tokens, elapsed)
                    session_tokens += tokens
                    session_time += elapsed
                    session_turns += 1
                    messages.append({"role": "assistant", "content": full})
                except Exception as e:
                    console.print(f"\n  [bold red]{e}[/]")
                    if messages and messages[-1]["role"] == "user":
                        messages.pop()

        # ── raw streaming mode ─────────────────────
        else:
            messages.append({"role": "user", "content": user_input})
            full = ""
            tokens = 0
            start = time.time()

            try:
                display = WorkingDisplay()
                display.phase = "thinking"
                first_token = True

                with Live(
                    display.render(),
                    console=console,
                    refresh_per_second=8,
                    transient=True,
                ) as live:
                    gen = stream_llm(messages)
                    for chunk in gen:
                        if isinstance(chunk, str):
                            if first_token:
                                first_token = False
                                live.stop()
                                console.print("  ", end="")
                            console.print(chunk, end="", highlight=False)
                            full += chunk
                            tokens += 1

                elapsed = time.time() - start
                if not compact_mode and any(
                    c in full for c in ["##", "**", "```", "- ", "1. "]
                ):
                    console.print("\n")
                    console.print(Padding(Markdown(full), (0, 2)))
                else:
                    console.print("\n")
                render_speed(tokens, elapsed)
                session_tokens += tokens
                session_time += elapsed
                session_turns += 1
                messages.append({"role": "assistant", "content": full})

            except Exception as e:
                console.print(f"  [bold red]{e}[/]")
                if messages and messages[-1]["role"] == "user":
                    messages.pop()

        console.print()

    # ── exit ───────────────────────────────────────
    console.print()
    if session_turns > 0:
        avg = session_tokens / session_time if session_time > 0 else 0
        console.print(
            f"  \U0001f34e [bold bright_cyan]mac[/] [bold bright_yellow]code[/]"
            f"  [dim]{session_turns} turns · {session_tokens:,} tokens · {avg:.1f} tok/s[/]"
        )
    console.print()


if __name__ == "__main__":
    main()
