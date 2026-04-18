#!/usr/bin/env python3
"""
mac code web backend — bridges the browser UI to PicoClaw + llama-server.
Run this, then open index.html. The browser talks to this server,
which routes to PicoClaw (for tool calls) or llama-server (for streaming).
"""

import json, os, subprocess, re, time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import urllib.request

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False

def quick_search_and_answer(query):
    """Fast path: DuckDuckGo search + single LLM call. ~3s total."""
    if not HAS_DDGS:
        return None

    # Search — append today's date for time-sensitive queries
    from datetime import datetime
    search_query = f"{query} today {datetime.now().strftime('%B %d, %Y')}"

    try:
        results = DDGS().text(search_query, max_results=5)
        search_text = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception:
        return None

    if not search_text.strip():
        return None

    from datetime import datetime
    today = datetime.now().strftime("%A, %B %d, %Y")

    # Single LLM call with search context
    payload = json.dumps({
        "model": "local",
        "messages": [
            {"role": "system", "content": f"Today is {today}. Answer using the search results below. Extract specific dates, teams, scores, names, and numbers. Even if data is partial or abbreviated (like 'Mar. 23 at DET'), expand it into a clear answer (e.g., 'March 23 vs Detroit Pistons'). Be direct and confident. Never say 'not fully provided' — use what you have."},
            {"role": "user", "content": f"Web search results:\n\n{search_text}\n\nBased on these results, answer: {query}"},
        ],
        "max_tokens": 1000,
        "temperature": 0.1,
    }).encode()

    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.loads(resp.read())

    content = d["choices"][0]["message"]["content"]
    timings = d.get("timings", {})
    return {
        "response": content,
        "speed": timings.get("predicted_per_second", 0),
        "search_results": len(results),
    }

PORT = 8080
LLM_URL = "http://localhost:8000/v1/chat/completions"
PICOCLAW = os.path.expanduser("~/Desktop/qwen/picoclaw/build/picoclaw-darwin-arm64")
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m|\r')

# Model configs (same as agent.py)
MODELS = {
    "9b": {
        "path": os.path.expanduser("~/models/Qwen3.5-9B-Q4_K_M.gguf"),
        "ctx": 32768,
        "flags": ["--flash-attn", "on", "--n-gpu-layers", "99", "--reasoning", "off", "-t", "4"],
        "name": "Qwen3.5-9B",
    },
    "35b": {
        "path": os.path.expanduser("~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"),
        "ctx": 8192,
        "flags": ["--flash-attn", "on", "--n-gpu-layers", "99", "--reasoning", "off", "-np", "1", "-t", "4"],
        "name": "Qwen3.5-35B-A3B",
    },
}

current_model = None  # detected on first request

def get_current_model():
    global current_model
    try:
        req = urllib.request.Request("http://localhost:8000/props")
        with urllib.request.urlopen(req, timeout=3) as r:
            d = json.loads(r.read())
        alias = d.get("model_alias", "") or d.get("model_path", "")
        if "35B-A3B" in alias:
            current_model = "35b"
        elif "9B" in alias:
            current_model = "9b"
    except Exception:
        pass
    return current_model

def swap_model(target):
    global current_model
    cfg = MODELS.get(target)
    if not cfg or not os.path.exists(cfg["path"]):
        return False, f"Model not found: {target}"

    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(3)

    cmd = [
        "llama-server",
        "--model", cfg["path"],
        "--port", "8000", "--host", "127.0.0.1",
        "--ctx-size", str(cfg["ctx"]),
    ] + cfg["flags"]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i in range(30):
        time.sleep(2)
        try:
            req = urllib.request.Request("http://localhost:8000/health")
            with urllib.request.urlopen(req, timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    current_model = target
                    return True, f"Switched to {cfg['name']}"
        except Exception:
            pass
    return False, "Server failed to start"

class Handler(SimpleHTTPRequestHandler):

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/chat":
            self._handle_chat()
        elif path == "/api/agent":
            self._handle_agent_fast()
        elif path == "/api/swap":
            self._handle_swap()
        elif path == "/api/status":
            self._handle_status()
        else:
            self.send_error(404)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _handle_chat(self):
        """Proxy streaming chat to llama-server."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            req = urllib.request.Request(
                LLM_URL, data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self._cors_headers()
                self.end_headers()

                while True:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_agent_fast(self):
        """Fast agent: direct DuckDuckGo search + LLM (~3s), fallback to PicoClaw."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        message = body.get("message", "")

        # Try fast path first
        try:
            result = quick_search_and_answer(message)
            if result and result.get("response"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "response": result["response"],
                    "speed": result.get("speed", 0),
                    "method": "fast",
                }).encode())
                return
        except Exception as e:
            print(f"  Fast search failed: {e}, falling back to PicoClaw")

        # Fallback to PicoClaw
        self._handle_agent_picoclaw(body)

    def _handle_agent_picoclaw(self, body):
        """Fallback: Route through PicoClaw agent."""
        message = body.get("message", "")
        session = body.get("session", f"web-{int(time.time())}")

        try:
            result = subprocess.run(
                [PICOCLAW, "agent", "-m", message, "-s", session],
                capture_output=True, text=True, timeout=120,
            )
            clean = ANSI_RE.sub('', result.stdout)
            parts = clean.split("\U0001f99e")
            response = parts[-1].strip() if len(parts) >= 2 else clean.strip()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"response": response, "method": "picoclaw"}).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_agent_old(self):
        """Route through PicoClaw agent with live log streaming via SSE."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        message = body.get("message", "")
        session = body.get("session", f"web-{int(time.time())}")

        # Log patterns to detect phases
        PHASE_MAP = {
            "processing message": "reading message",
            "llm_request": "thinking",
            "tool_call": "calling tool",
            "web_search": "searching the web",
            "duckduckgo": "searching DuckDuckGo",
            "web_fetch": "fetching page",
            "tool_result": "got results",
            "context_compress": "compressing context",
            "turn_end": "finishing up",
        }

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self._cors_headers()
            self.end_headers()

            # Run PicoClaw with debug flag for live logs
            proc = subprocess.Popen(
                [PICOCLAW, "agent", "-m", message, "-s", session, "-d"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )

            all_output = []
            for line in proc.stdout:
                all_output.append(line)
                clean = ANSI_RE.sub('', line).strip()

                # Detect phase from log line
                lower = clean.lower()
                for keyword, phase in PHASE_MAP.items():
                    if keyword in lower:
                        # Send log event to browser
                        event = json.dumps({"type": "log", "phase": phase, "detail": clean[:80]})
                        self.wfile.write(f"data: {event}\n\n".encode())
                        self.wfile.flush()
                        break

            proc.wait(timeout=120)

            # Parse final response
            full = "".join(all_output)
            clean = ANSI_RE.sub('', full)
            parts = clean.split("\U0001f99e")
            if len(parts) >= 2:
                response = parts[-1].strip()
            else:
                response = ""

            # Send final response
            event = json.dumps({"type": "response", "response": response, "session": session})
            self.wfile.write(f"data: {event}\n\n".encode())
            self.wfile.write("data: [DONE]\n\n".encode())
            self.wfile.flush()

        except subprocess.TimeoutExpired:
            proc.kill()
            event = json.dumps({"type": "error", "error": "Agent timeout"})
            self.wfile.write(f"data: {event}\n\n".encode())
            self.wfile.write("data: [DONE]\n\n".encode())
            self.wfile.flush()

        except Exception as e:
            try:
                event = json.dumps({"type": "error", "error": str(e)})
                self.wfile.write(f"data: {event}\n\n".encode())
                self.wfile.write("data: [DONE]\n\n".encode())
                self.wfile.flush()
            except Exception:
                pass

    def _handle_swap(self):
        """Swap to a different model."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        target = body.get("model", "")

        if target not in MODELS:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Unknown model: {target}. Use '9b' or '35b'"}).encode())
            return

        cur = get_current_model()
        if cur == target:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "message": f"Already on {MODELS[target]['name']}", "model": target}).encode())
            return

        print(f"  Swapping to {MODELS[target]['name']}...")
        ok, msg = swap_model(target)

        self.send_response(200 if ok else 500)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({"ok": ok, "message": msg, "model": target}).encode())

    def _handle_status(self):
        """Return current model and server status."""
        cur = get_current_model()
        cfg = MODELS.get(cur, {})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({
            "model": cur,
            "name": cfg.get("name", "unknown"),
            "ctx": cfg.get("ctx", 0),
            "models_available": {k: os.path.exists(v["path"]) for k, v in MODELS.items()},
        }).encode())

    def do_GET(self):
        """Serve static files from web/ directory."""
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        return SimpleHTTPRequestHandler.do_GET(self)

    def log_message(self, format, *args):
        """Minimal logging."""
        msg = format % args
        if "favicon" not in msg:
            print(f"  {msg}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"\n  🍎 mac code web server")
    print(f"  http://localhost:{PORT}")
    print(f"  LLM:   localhost:8000 (llama-server)")
    print(f"  Agent: PicoClaw ({PICOCLAW})")
    print()
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
