#!/usr/bin/env python3
"""
mac code — claude code for your Mac
"""

import json, sys, os, time, subprocess, re, threading, queue
import urllib.request, random

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
PICOCLAW = os.path.expanduser("~/Desktop/qwen/picoclaw/build/picoclaw-darwin-arm64")
console = Console()

# ── ANSI strip ─────────────────────────────────────
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m|\r')
def strip_ansi(text):
    return ANSI_RE.sub('', text)

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
        if any(k in lower for k in ["llm_request", "tool_call", "tool_result", "turn_end", "web_search", "fetch", "exec"]):
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
    payload = json.dumps({
        "model": "local",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
        "stream": True,
    }).encode()

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
            ch = resp.read(1)
            if not ch:
                break
            buf += ch.decode("utf-8", errors="replace")
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
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
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
    with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
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
    console.print("  [dim]type [bold bright_cyan]/[/bold bright_cyan] to see all commands[/]\n")

# ── render helpers ─────────────────────────────────
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
    ("/agent",       "Switch to agent mode (tools + web search)"),
    ("/raw",         "Switch to raw mode (direct streaming, no tools)"),
    ("/btw",         "Ask a side question without adding to conversation history"),
    ("/loop",        "Run a prompt on a recurring interval — /loop 5m <prompt>"),
    ("/branch",      "Save conversation checkpoint you can restore later"),
    ("/restore",     "Restore last saved conversation checkpoint"),
    ("/add-dir",     "Set working directory — /add-dir <path>"),
    ("/save",        "Save conversation to a file — /save <filename>"),
    ("/search",      "Quick web search — /search <query>"),
    ("/bench",       "Run a quick speed benchmark"),
    ("/clear",       "Clear conversation and start fresh"),
    ("/stats",       "Show session statistics"),
    ("/model",       "Show current model info"),
    ("/tools",       "List available agent tools"),
    ("/system",      "Set system prompt — /system <message>"),
    ("/compact",     "Toggle compact output (no markdown rendering)"),
    ("/stop",        "Stop a running /loop"),
    ("/cost",        "Show estimated cost savings vs cloud APIs"),
    ("/quit",        "Exit mac code"),
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
    work_dir = os.getcwd()
    branch_save = None  # saved conversation checkpoint
    loop_thread = None
    loop_running = False

    while True:
        try:
            tag = "agent" if use_agent else "raw"
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
                model_name, model_detail = detect_model()
                console.print(f"  [bold white]{model_name}[/]  [dim]{model_detail}[/]\n")
                continue
            elif exact == "/tools":
                for name, desc in [
                    ("web_search", "DuckDuckGo"), ("web_fetch", "read URLs"),
                    ("exec", "shell commands"), ("read_file", "local files"),
                    ("write_file", "create files"), ("edit_file", "modify files"),
                    ("list_dir", "browse dirs"), ("subagent", "spawn tasks"),
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
                console.print(f"  [dim]conversation saved ({len(messages)} messages). use /restore to go back.[/]\n")
                continue

            elif exact == "/restore":
                if branch_save is not None:
                    messages = [m.copy() for m in branch_save]
                    console.print(f"  [dim]restored to checkpoint ({len(messages)} messages)[/]\n")
                else:
                    console.print("  [dim]no checkpoint saved. use /branch first.[/]\n")
                continue

            elif exact == "/bench":
                console.print("  [dim]running speed benchmark...[/]")
                try:
                    payload = json.dumps({
                        "model": "local",
                        "messages": [{"role": "user", "content": "Count from 1 to 50, one number per line."}],
                        "max_tokens": 300, "temperature": 0.1,
                    }).encode()
                    req = urllib.request.Request(
                        f"{SERVER}/v1/chat/completions", data=payload,
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
                    console.print(f"  [bold bright_green]{gen_speed:.1f} tok/s[/] generation")
                    console.print(f"  [bold bright_green]{prompt_speed:.1f} tok/s[/] prompt processing")
                    console.print(f"  [dim]{tokens} tokens in {belapsed:.1f}s[/]\n")
                except Exception as e:
                    console.print(f"  [bold red]benchmark failed: {e}[/]\n")
                continue

            elif exact == "/cost":
                cloud_rate = 0.34  # $/hr RunPod equivalent
                hours = session_time / 3600 if session_time > 0 else 0
                saved = cloud_rate * max(hours, 1/60)
                console.print(f"  [bold bright_green]$0.00[/] spent locally")
                console.print(f"  [dim]~${saved:.4f} would have cost on cloud GPU (${cloud_rate}/hr)[/]")
                console.print(f"  [dim]session: {session_time:.0f}s · {session_tokens:,} tokens[/]\n")
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
                response, events = picoclaw_call_live(side_q, session=f"btw-{int(time.time())}")
                elapsed = time.time() - start
                if response:
                    console.print(f"  [dim italic](side answer)[/]")
                    for line in response.split("\n"):
                        console.print(f"  {line}")
                    console.print()
                    tokens_est = len(response.split())
                    render_speed(tokens_est, elapsed)
                    session_tokens += tokens_est
                    session_time += elapsed
            else:
                side_msgs = [{"role": "user", "content": side_q}]
                try:
                    payload = json.dumps({
                        "model": "local", "messages": side_msgs,
                        "max_tokens": 2000, "temperature": 0.7,
                    }).encode()
                    req = urllib.request.Request(
                        f"{SERVER}/v1/chat/completions", data=payload,
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
                    json.dump({
                        "messages": messages,
                        "session_id": session_id,
                        "tokens": session_tokens,
                        "time": session_time,
                        "turns": session_turns,
                    }, f, indent=2)
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
                session=f"search-{int(time.time())}"
            )
            elapsed = time.time() - start
            if response:
                for line in response.split("\n"):
                    console.print(f"  {line}")
                console.print()
                render_speed(len(response.split()), elapsed)
                session_tokens += len(response.split())
                session_time += elapsed
            console.print()
            continue

        elif cmd_lower.startswith("/loop "):
            # Parse: /loop 5m <prompt>
            parts = cmd[6:].strip().split(None, 1)
            if len(parts) < 2:
                console.print("  [dim]/loop <interval> <prompt>  — e.g. /loop 5m check server status[/]\n")
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
                    console.print(f"\n  [dim italic]loop: running '{prompt[:40]}...'[/]")
                    resp, _ = picoclaw_call_live(prompt, session=sid)
                    if resp:
                        for line in resp.split("\n"):
                            console.print(f"  {line}")
                    console.print()

            loop_thread = threading.Thread(
                target=run_loop,
                args=(loop_prompt, interval_sec, f"loop-{session_id}"),
                daemon=True
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
            response, events = picoclaw_call_live(user_input, session=session_id)
            elapsed = time.time() - start

            if response:
                # Show what the agent did
                render_timeline(events)
                console.print()

                # Render response
                if not compact_mode and any(c in response for c in ["##", "**", "```", "- ", "1. ", "* "]):
                    console.print(Padding(Markdown(response), (0, 2)))
                else:
                    for line in response.split("\n"):
                        console.print(f"  {line}")
                console.print()
                tokens_est = len(response.split())
                render_speed(tokens_est, elapsed)
                session_tokens += tokens_est
                session_time += elapsed
                session_turns += 1
            else:
                console.print("  [bold red]no response[/]")

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

                with Live(display.render(), console=console, refresh_per_second=8, transient=True) as live:
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
                if not compact_mode and any(c in full for c in ["##", "**", "```", "- ", "1. "]):
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
