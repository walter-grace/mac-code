#!/usr/bin/env python3
"""
mac-tensor CLI — Distributed MoE inference across multiple Macs.

The idea: you run this from YOUR Mac (or laptop). It SSHes into
your remote Macs, sets everything up, and starts inference.

Workflow:
    1. mac-tensor init      → Save your cluster config (IPs, credentials)
    2. mac-tensor deploy     → Push code + download model on all nodes
    3. mac-tensor up         → Start expert nodes on all remotes
    4. mac-tensor chat       → Chat with the model
    5. mac-tensor down       → Stop all nodes
    6. mac-tensor status     → Check what's running

Or all-in-one:
    mac-tensor run --model qwen35   → deploy + up + chat in one command
"""

import argparse
import json
import os
import subprocess
import sys
import time
import shutil


# ============================================================
# CONFIG
# ============================================================

CONFIG_DIR = os.path.expanduser("~/.mac-tensor")
CONFIG_FILE = os.path.join(CONFIG_DIR, "cluster.json")
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

SUPPORTED_MODELS = {
    "qwen35": {
        "name": "Qwen 3.5-35B-A3B",
        "hf_id": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "num_experts": 256,
        "num_layers": 40,
        "node_script": "expert_node_fast.py",
        "coordinator_script": "distributed_interactive.py",
        "split_script": "split_qwen.py",
        "needs_split": True,
        "default_port": 8301,
        "model_dir": "~/models/qwen35-4bit",
        "stream_dir": "~/models/qwen35-stream",
        "deploy_files": [
            "expert_node_fast.py",
            "distributed_interactive.py",
            "distributed_reader_fast.py",
            "split_qwen.py",
        ],
    },
    "gemma4": {
        "name": "Gemma 4-26B-A4B",
        "hf_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "num_experts": 128,
        "num_layers": 30,
        "node_script": "gemma4_expert_node.py",
        "coordinator_script": "gemma4_distributed.py",
        "split_script": "split_gemma4.py",
        "needs_split": False,
        "default_port": 8401,
        "model_dir": "~/models/gemma4-4bit",
        "stream_dir": "~/models/gemma4-4bit",
        "deploy_files": [
            "gemma4_expert_node.py",
            "gemma4_distributed.py",
            "distributed_reader_fast.py",
            "models_gemma4.py",
        ],
    },
}

# Files the coordinator always needs
COORDINATOR_FILES = [
    "distributed_reader_fast.py",
    "distributed_interactive.py",
    "gemma4_distributed.py",
    "models_gemma4.py",
]


def get_model(name):
    if name not in SUPPORTED_MODELS:
        print(f"Error: Unknown model '{name}'")
        print(f"Supported: {', '.join(SUPPORTED_MODELS.keys())}")
        sys.exit(1)
    return SUPPORTED_MODELS[name]


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return None
    with open(CONFIG_FILE) as f:
        return json.load(f)


def save_config(cfg):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def ssh_cmd(host, user, password, cmd, timeout=300):
    """Run a command on a remote Mac via SSH."""
    if password:
        full = f"sshpass -p '{password}' ssh -o StrictHostKeyChecking=no {user}@{host} \"{cmd}\""
    else:
        full = f"ssh -o StrictHostKeyChecking=no {user}@{host} \"{cmd}\""
    result = subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def scp_file(host, user, password, local_path, remote_path):
    """Copy a file to a remote Mac."""
    if password:
        cmd = f"sshpass -p '{password}' scp -o StrictHostKeyChecking=no {local_path} {user}@{host}:{remote_path}"
    else:
        cmd = f"scp -o StrictHostKeyChecking=no {local_path} {user}@{host}:{remote_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    return result.returncode == 0


def require_config():
    cfg = load_config()
    if not cfg:
        print("No cluster configured. Run 'mac-tensor init' first.")
        sys.exit(1)
    return cfg


def print_step(step, msg):
    print(f"\n  [{step}] {msg}")


# ============================================================
# COMMANDS
# ============================================================


def cmd_init(args):
    """Configure your cluster — save node IPs and credentials."""
    print("mac-tensor cluster setup\n")
    print("You need 2-3 Macs: 1 coordinator (this Mac or a remote) + 1-2 expert nodes.")
    print("The expert nodes hold the model weights. The coordinator runs the chat.\n")

    # Check for existing config
    existing = load_config()
    if existing:
        print(f"Existing config found with {len(existing.get('nodes', []))} nodes.")
        resp = input("Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Keeping existing config.")
            return

    nodes = []
    print("Add your expert nodes (the Macs that will hold model weights):")
    print("Format: user@ip:password (or user@ip if using SSH keys)\n")

    while True:
        entry = input(f"  Node {len(nodes) + 1} (or 'done'): ").strip()
        if entry.lower() in ("done", "d", ""):
            if len(nodes) == 0:
                print("  Need at least 1 node.")
                continue
            break

        # Parse user@ip:password or user@ip
        password = None
        if ":" in entry and "@" in entry:
            userhost, password = entry.rsplit(":", 1)
        else:
            userhost = entry

        if "@" not in userhost:
            print("  Format: user@ip or user@ip:password")
            continue

        user, host = userhost.split("@", 1)

        # Test connection
        print(f"  Testing SSH to {user}@{host}...", end="", flush=True)
        out, err, rc = ssh_cmd(host, user, password, "echo ok", timeout=10)
        if rc == 0 and "ok" in out:
            print(" connected!")
            nodes.append({"host": host, "user": user, "password": password})
        else:
            print(f" FAILED: {err[:80]}")
            print("  Check IP, username, and password. Skipping.")

    # Coordinator config
    print("\nCoordinator (where you'll run 'mac-tensor chat'):")
    print("  [1] This Mac (default)")
    print("  [2] A remote Mac")
    coord_choice = input("  Choice [1]: ").strip()

    coordinator = None
    if coord_choice == "2":
        entry = input("  Coordinator user@ip:password: ").strip()
        password = None
        if ":" in entry and "@" in entry:
            userhost, password = entry.rsplit(":", 1)
        else:
            userhost = entry
        user, host = userhost.split("@", 1)
        coordinator = {"host": host, "user": user, "password": password}

    # Model selection
    print("\nModel to run:")
    for i, (key, m) in enumerate(SUPPORTED_MODELS.items()):
        print(f"  [{i+1}] {key} — {m['name']} ({m['num_experts']} experts, {m['num_layers']} layers)")
    model_choice = input("  Choice [1]: ").strip()
    model_keys = list(SUPPORTED_MODELS.keys())
    model_idx = int(model_choice) - 1 if model_choice.isdigit() else 0
    model_key = model_keys[min(model_idx, len(model_keys) - 1)]

    cfg = {
        "model": model_key,
        "nodes": nodes,
        "coordinator": coordinator,
        "created": time.strftime("%Y-%m-%d %H:%M"),
    }

    save_config(cfg)

    model = SUPPORTED_MODELS[model_key]
    print(f"\nCluster saved to {CONFIG_FILE}")
    print(f"  Model: {model['name']}")
    print(f"  Nodes: {len(nodes)} expert node(s)")
    if coordinator:
        print(f"  Coordinator: {coordinator['user']}@{coordinator['host']}")
    else:
        print(f"  Coordinator: this Mac")
    print(f"\nNext: run 'mac-tensor deploy' to push code and download the model.")


def cmd_deploy(args):
    """Deploy code + download model on all nodes."""
    cfg = require_config()
    model_key = args.model or cfg.get("model", "qwen35")
    model = get_model(model_key)
    nodes = cfg["nodes"]
    coordinator = cfg.get("coordinator")

    print(f"Deploying {model['name']} to {len(nodes)} node(s)...")

    all_targets = list(nodes)
    if coordinator:
        all_targets.append(coordinator)

    for i, node in enumerate(all_targets):
        role = "coordinator" if node == coordinator else f"node {i+1}"
        host, user, pw = node["host"], node["user"], node.get("password")

        print_step(f"{role}", f"Deploying to {user}@{host}")

        # Create remote directory
        ssh_cmd(host, user, pw, "mkdir -p ~/expert-sniper-mlx")

        # Copy scripts
        files_to_copy = model["deploy_files"]
        if node == coordinator:
            files_to_copy = list(set(files_to_copy + COORDINATOR_FILES))

        for fname in files_to_copy:
            local = os.path.join(SCRIPT_DIR, fname)
            if os.path.exists(local):
                ok = scp_file(host, user, pw, local, f"~/expert-sniper-mlx/{fname}")
                if ok:
                    print(f"    Copied {fname}")
                else:
                    print(f"    FAILED to copy {fname}")

        # Check if model already downloaded
        check_dir = model["model_dir"]
        out, _, rc = ssh_cmd(host, user, pw, f"ls {check_dir}/config.json 2>/dev/null && echo EXISTS")
        if "EXISTS" in out:
            print(f"    Model already downloaded at {check_dir}")
        else:
            print(f"    Downloading {model['hf_id']}... (this takes a few minutes)")
            dl_cmd = (
                f"python3 -c \""
                f"from huggingface_hub import snapshot_download; "
                f"snapshot_download('{model['hf_id']}', local_dir='{check_dir}')"
                f"\""
            )
            out, err, rc = ssh_cmd(host, user, pw, dl_cmd, timeout=600)
            if rc == 0:
                print(f"    Download complete!")
            else:
                print(f"    Download failed: {err[:120]}")

        # Split if needed (Qwen only)
        if model["needs_split"] and node != coordinator:
            stream_dir = model["stream_dir"]
            out, _, _ = ssh_cmd(host, user, pw, f"ls {stream_dir}/pinned.safetensors 2>/dev/null && echo EXISTS")
            if "EXISTS" in out:
                print(f"    Model already split at {stream_dir}")
            else:
                print(f"    Splitting model...")
                split_cmd = (
                    f"cd ~/expert-sniper-mlx && python3 {model['split_script']} "
                    f"--input {check_dir} --output {stream_dir}"
                )
                out, err, rc = ssh_cmd(host, user, pw, split_cmd, timeout=600)
                if rc == 0:
                    print(f"    Split complete!")
                else:
                    print(f"    Split failed: {err[:120]}")

    cfg["model"] = model_key
    cfg["deployed"] = True
    save_config(cfg)
    print(f"\nDeploy complete! Run 'mac-tensor up' to start expert nodes.")


def cmd_up(args):
    """Start expert nodes on all remote Macs."""
    cfg = require_config()
    model_key = args.model or cfg.get("model", "qwen35")
    model = get_model(model_key)
    nodes = cfg["nodes"]
    n_experts = model["num_experts"]
    port = model["default_port"]

    # Calculate partitions — split experts evenly across nodes
    per_node = n_experts // len(nodes)
    node_urls = []

    for i, node in enumerate(nodes):
        host, user, pw = node["host"], node["user"], node.get("password")

        p_start = i * per_node
        p_end = (i + 1) * per_node - 1 if i < len(nodes) - 1 else n_experts - 1
        partition = f"{p_start}-{p_end}"

        model_dir = model["stream_dir"]

        print_step(i + 1, f"Starting {user}@{host} — partition {partition}")

        # Kill any existing node on this port
        ssh_cmd(host, user, pw, f"lsof -i :{port} -t 2>/dev/null | xargs kill -9 2>/dev/null")
        time.sleep(1)

        # Start node in background
        start_cmd = (
            f"cd ~/expert-sniper-mlx && "
            f"nohup python3 {model['node_script']} "
            f"--partition {partition} --model-dir {model_dir} --port {port} "
            f"> /tmp/mac-tensor-node.log 2>&1 &"
        )
        ssh_cmd(host, user, pw, start_cmd)
        node_urls.append(f"http://{host}:{port}")
        print(f"    Started on port {port}")

    # Save node URLs for chat
    cfg["node_urls"] = node_urls
    cfg["model"] = model_key
    save_config(cfg)

    print(f"\nExpert nodes starting... they need ~90 seconds to load weights.")
    print(f"Run 'mac-tensor status' to check, then 'mac-tensor chat' when ready.")


def cmd_down(args):
    """Stop all expert nodes."""
    cfg = require_config()
    model = get_model(cfg.get("model", "qwen35"))
    port = model["default_port"]

    for i, node in enumerate(cfg["nodes"]):
        host, user, pw = node["host"], node["user"], node.get("password")
        print(f"  Stopping {user}@{host}...")
        ssh_cmd(host, user, pw, f"lsof -i :{port} -t 2>/dev/null | xargs kill -9 2>/dev/null")

    cfg.pop("node_urls", None)
    save_config(cfg)
    print("All nodes stopped.")


def cmd_status(args):
    """Check cluster status — are nodes loaded and ready?"""
    cfg = require_config()
    node_urls = cfg.get("node_urls", [])

    if not node_urls:
        print("No nodes running. Run 'mac-tensor up' first.")
        return

    import urllib.request

    print(f"Checking {len(node_urls)} node(s)...\n")

    all_ok = True
    total_experts = 0
    total_mem = 0

    for url in node_urls:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=5)
            data = json.loads(resp.read())
            status = data.get("status", "?")
            partition = data.get("partition", "?")
            mem = data.get("memory_gb", 0)
            experts = data.get("total_experts_loaded", data.get("total_layers_loaded", 0))
            reqs = data.get("compute_requests", 0)
            avg_ms = data.get("avg_compute_ms", 0)
            model_name = data.get("model", "")

            total_mem += mem
            total_experts += experts if isinstance(experts, int) else 0

            print(f"  [OK]   {url}")
            print(f"         Partition: {partition} | RAM: {mem} GB | Requests: {reqs}")
        except Exception as e:
            print(f"  [WAIT] {url}")
            print(f"         Not ready yet ({e.__class__.__name__})")
            all_ok = False

    print()
    if all_ok:
        print(f"All nodes ready! Total: {total_mem:.1f} GB RAM across {len(node_urls)} nodes.")
        model_key = cfg.get("model", "qwen35")
        print(f"\nRun: mac-tensor chat")
    else:
        print("Some nodes still loading. Wait ~90 seconds and try again.")


def cmd_chat(args):
    """Start interactive chat — connects to running expert nodes.

    Two modes:
      1. With --nodes: connect directly without needing a saved cluster
      2. Without --nodes: use the saved cluster config (init/up flow)
    """
    cfg = load_config() or {}

    # If user passed --nodes explicitly, run in standalone mode (no config required)
    if args.nodes:
        node_urls = args.nodes
        model_key = args.model or "qwen35"
        coordinator = None
    else:
        # Fall back to saved cluster config
        if not cfg:
            print("No cluster configured and no --nodes provided.")
            print("Either run 'mac-tensor init' first, or pass --nodes directly:")
            print("  mac-tensor chat --model gemma4 --nodes http://mac2:8401,http://mac3:8401")
            sys.exit(1)
        node_urls = ",".join(cfg.get("node_urls", []))
        if not node_urls:
            print("No nodes running. Run 'mac-tensor up' first.")
            sys.exit(1)
        model_key = args.model or cfg.get("model", "qwen35")
        coordinator = cfg.get("coordinator")

    model = get_model(model_key)
    script = model["coordinator_script"]
    max_tokens = args.max_tokens or 300
    temperature = args.temperature or 0.7

    if coordinator:
        # Run coordinator on remote Mac
        host, user, pw = coordinator["host"], coordinator["user"], coordinator.get("password")
        print(f"Starting chat on coordinator {user}@{host}...")
        cmd = (
            f"cd ~/expert-sniper-mlx && python3 {script} "
            f"--nodes {node_urls} "
            f"--max-tokens {max_tokens} --temperature {temperature}"
        )
        if pw:
            full = f"sshpass -p '{pw}' ssh -t -o StrictHostKeyChecking=no {user}@{host} \"{cmd}\""
        else:
            full = f"ssh -t -o StrictHostKeyChecking=no {user}@{host} \"{cmd}\""
        os.system(full)
    else:
        # Run coordinator locally
        local_script = os.path.join(SCRIPT_DIR, script)
        if not os.path.exists(local_script):
            print(f"Error: Coordinator script not found: {local_script}")
            sys.exit(1)

        cmd = [
            "python3", local_script,
            "--nodes", node_urls,
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
        ]
        if args.model_dir:
            cmd.extend(["--model-dir", os.path.expanduser(args.model_dir)])
        os.execvp("python3", cmd)


def cmd_run(args):
    """All-in-one: deploy + up + wait + chat."""
    cfg = require_config()
    model_key = args.model or cfg.get("model", "qwen35")

    # Deploy
    print("=== Step 1: Deploy ===")
    args.model = model_key
    cmd_deploy(args)

    # Up
    print("\n=== Step 2: Start Nodes ===")
    cmd_up(args)

    # Wait for nodes to load
    print("\n=== Step 3: Waiting for nodes to load ===")
    cfg = load_config()  # reload after up
    node_urls = cfg.get("node_urls", [])
    import urllib.request

    for attempt in range(30):  # 30 * 10s = 5 minutes max
        time.sleep(10)
        ready = 0
        for url in node_urls:
            try:
                resp = urllib.request.urlopen(f"{url}/health", timeout=5)
                data = json.loads(resp.read())
                if data.get("status") == "ok":
                    ready += 1
            except Exception:
                pass

        print(f"  {ready}/{len(node_urls)} nodes ready... ({(attempt+1)*10}s)")
        if ready == len(node_urls):
            break
    else:
        print("Warning: Not all nodes loaded in time. Trying anyway...")

    # Chat
    print("\n=== Step 4: Chat ===")
    args.nodes = None
    args.max_tokens = args.max_tokens or 300
    args.temperature = args.temperature or 0.7
    args.model_dir = None
    cmd_chat(args)


def cmd_info(args):
    """Show supported models and current config."""
    print("mac-tensor — Distributed MoE Inference\n")

    # Current config
    cfg = load_config()
    if cfg:
        model_key = cfg.get("model", "?")
        nodes = cfg.get("nodes", [])
        node_urls = cfg.get("node_urls", [])
        print(f"  Current cluster:")
        print(f"    Model:  {model_key}")
        print(f"    Nodes:  {len(nodes)}")
        for n in nodes:
            print(f"      {n['user']}@{n['host']}")
        if node_urls:
            print(f"    Status: nodes running ({', '.join(node_urls)})")
        else:
            print(f"    Status: not started")
        print()

    print("  Supported Models:\n")
    for key, m in SUPPORTED_MODELS.items():
        print(f"    {key:10s} {m['name']:30s} {m['num_experts']} experts, {m['num_layers']} layers")
    print()

    print("  Workflow:")
    print("    mac-tensor init      Configure cluster (IPs, credentials)")
    print("    mac-tensor deploy    Push code + download model on all nodes")
    print("    mac-tensor up        Start expert nodes")
    print("    mac-tensor status    Check if nodes are ready")
    print("    mac-tensor chat      Interactive chat")
    print("    mac-tensor down      Stop all nodes")
    print("    mac-tensor run       All-in-one (deploy + up + wait + chat)")


# ============================================================
# LOCAL COMMANDS (run on the Mac you're sitting at)
# ============================================================


def cmd_node_local(args):
    """Start an expert node locally (run this on the Mac itself)."""
    model = get_model(args.model)
    port = args.port or model["default_port"]
    partition = args.partition
    model_dir = os.path.expanduser(args.model_dir or model["stream_dir"])

    if not partition:
        half = model["num_experts"] // 2
        partition = f"0-{half - 1}"

    script = os.path.join(SCRIPT_DIR, model["node_script"])
    if not os.path.exists(script):
        print(f"Error: {script} not found")
        sys.exit(1)

    print(f"Starting {model['name']} expert node locally")
    print(f"  Partition: {partition} | Port: {port} | Model: {model_dir}")

    mem_limit = args.memory_limit or 14.0
    os.execvp("python3", [
        "python3", script,
        "--partition", partition,
        "--model-dir", model_dir,
        "--port", str(port),
        "--memory-limit-gb", str(mem_limit),
    ])


def cmd_download_local(args):
    """Download model locally."""
    model = get_model(args.model)
    output = os.path.expanduser(args.output or model["model_dir"])

    print(f"Downloading {model['name']} to {output}...")
    from huggingface_hub import snapshot_download
    snapshot_download(model["hf_id"], local_dir=output)
    print(f"Done!")


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        prog="mac-tensor",
        description="Distributed MoE inference across multiple Macs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow (remote cluster):
  mac-tensor init                          Save cluster config (IPs, creds)
  mac-tensor deploy                        Push code + model to all nodes
  mac-tensor up                            Start expert nodes
  mac-tensor status                        Check if nodes are loaded
  mac-tensor chat                          Start chatting
  mac-tensor down                          Stop all nodes

All-in-one:
  mac-tensor run --model qwen35            Deploy + start + wait + chat

Local commands (run on the Mac itself):
  mac-tensor node --model qwen35 -p 0-127  Start a local expert node
  mac-tensor download --model qwen35        Download model locally
""",
    )
    sub = parser.add_subparsers(dest="command")

    # Cluster orchestration
    sub.add_parser("init", help="Configure cluster (IPs, credentials)")

    p_dep = sub.add_parser("deploy", help="Push code + download model on all nodes")
    p_dep.add_argument("--model", choices=SUPPORTED_MODELS.keys())

    p_up = sub.add_parser("up", help="Start expert nodes on all remotes")
    p_up.add_argument("--model", choices=SUPPORTED_MODELS.keys())

    sub.add_parser("down", help="Stop all expert nodes")
    sub.add_parser("status", help="Check cluster status")

    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("--model", choices=SUPPORTED_MODELS.keys())
    p_chat.add_argument("--nodes", help="Override node URLs (comma-separated)")
    p_chat.add_argument("--max-tokens", type=int)
    p_chat.add_argument("--temperature", type=float)
    p_chat.add_argument("--model-dir", help="Override model directory")

    p_run = sub.add_parser("run", help="All-in-one: deploy + up + wait + chat")
    p_run.add_argument("--model", choices=SUPPORTED_MODELS.keys())
    p_run.add_argument("--max-tokens", type=int)
    p_run.add_argument("--temperature", type=float)

    sub.add_parser("info", help="Show models and cluster config")

    # Local commands
    p_nd = sub.add_parser("node", help="Start a local expert node")
    p_nd.add_argument("--model", required=True, choices=SUPPORTED_MODELS.keys())
    p_nd.add_argument("--partition", "-p")
    p_nd.add_argument("--port", type=int)
    p_nd.add_argument("--model-dir")
    p_nd.add_argument("--memory-limit", type=float)

    p_dl = sub.add_parser("download", help="Download model locally")
    p_dl.add_argument("--model", required=True, choices=SUPPORTED_MODELS.keys())
    p_dl.add_argument("--output", "-o")

    # Health (standalone, no config needed)
    p_hl = sub.add_parser("health", help="Check node health (standalone)")
    p_hl.add_argument("--nodes", required=True, help="Comma-separated URLs")

    # Agent (interactive agentic REPL with tool use)
    p_ag = sub.add_parser("agent", help="Interactive agentic REPL with tools")
    p_ag.add_argument("--model", choices=SUPPORTED_MODELS.keys(), default="gemma4")
    p_ag.add_argument("--nodes", required=True, help="Comma-separated expert node URLs")
    p_ag.add_argument("--max-iterations", type=int, default=8,
                      help="Max tool-call iterations per question (default 8)")
    p_ag.add_argument("--max-tokens", type=int, default=400,
                      help="Max tokens per model response (default 400)")
    p_ag.add_argument("--write", action="store_true",
                      help="Allow file writes and destructive shell commands")

    # UI (web chat interface)
    p_ui = sub.add_parser("ui", help="Web chat UI for the distributed agent")
    p_ui.add_argument("--model", choices=SUPPORTED_MODELS.keys(), default="gemma4")
    p_ui.add_argument("--nodes", help="Comma-separated expert node URLs (omit if --vision)")
    p_ui.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    p_ui.add_argument("--port", type=int, default=8500, help="Port (default 8500)")
    p_ui.add_argument("--write", action="store_true",
                      help="Allow file writes and destructive shell commands")
    p_ui.add_argument("--vision", action="store_true",
                      help="Single-machine vision mode (Gemma 4 + image upload)")
    p_ui.add_argument("--stream-dir", help="Vision: path to gemma4-stream split dir")
    p_ui.add_argument("--source-dir", help="Vision: path to gemma4-26b-4bit source dir")
    p_ui.add_argument("--falcon", action="store_true",
                      help="Vision: also load Falcon Perception for grounded segmentation")
    p_ui.add_argument("--falcon-model", help="Vision: path to Falcon Perception model dir")

    # Swarm: leader (peer registry + UI)
    p_ld = sub.add_parser("leader", help="Run as swarm leader (peer registry + chat UI)")
    p_ld.add_argument("--model", choices=SUPPORTED_MODELS.keys(), default="gemma4")
    p_ld.add_argument("--host", default="0.0.0.0")
    p_ld.add_argument("--port", type=int, default=8500)

    # Swarm: join (worker registers with leader, runs expert node)
    p_jn = sub.add_parser("join", help="Join a swarm as a worker peer")
    p_jn.add_argument("leader", help="Leader URL, e.g. http://leader-ip:8500")
    p_jn.add_argument("--model", choices=SUPPORTED_MODELS.keys(), default="gemma4")
    p_jn.add_argument("--model-dir", help="Path to streaming model dir (default ~/models/<model>-stream)")
    p_jn.add_argument("--port", type=int, default=9301, help="Port for the local expert node")

    # Swarm: peers (list peers in a swarm)
    p_pr = sub.add_parser("peers", help="List peers in a swarm")
    p_pr.add_argument("leader", help="Leader URL, e.g. http://leader-ip:8500")

    # Bench (throughput benchmark)
    p_bn = sub.add_parser("bench", help="Throughput benchmark — concurrent clients hitting a UI server")
    p_bn.add_argument("--server", default="http://localhost:8500", help="UI server URL")
    p_bn.add_argument("--concurrent", "-c", type=int, default=4, help="Concurrent clients")
    p_bn.add_argument("--requests", "-r", type=int, default=3, help="Requests per client")
    p_bn.add_argument("--max-tokens", type=int, default=80, help="Tokens per request")
    p_bn.add_argument("--hourly-cost", type=float, default=0.40,
                      help="Cluster $/hr for cost calc (default 0.40)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    def cmd_agent(args):
        from .agent import main as agent_main
        agent_main(args)

    def cmd_ui(args):
        from .server import main as ui_main
        ui_main(args)

    def cmd_bench(args):
        from .bench import main as bench_main
        bench_main(args)

    def cmd_leader(args):
        # Run the UI server in swarm leader mode
        from .server import run_server
        run_server(
            model_key=args.model,
            node_urls=None,
            host=args.host,
            port=args.port,
            swarm_leader=True,
        )

    def cmd_join(args):
        from .join import main as join_main
        join_main(args)

    def cmd_peers(args):
        import urllib.request
        url = args.leader.rstrip("/") + "/swarm/peers"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"Error fetching peers from {args.leader}: {e}")
            sys.exit(1)

        print(f"Swarm @ {args.leader}")
        print(f"  Model:    {data['model']}")
        print(f"  Peers:    {data['peer_count']}")
        print(f"  Version:  {data['partition_version']}")
        print()
        if not data["peers"]:
            print("  (no peers registered)")
            return
        for p in data["peers"]:
            status = "OK" if p["alive"] else "DEAD"
            print(f"  [{status}] {p['id']}  partition={p['partition']:>10s}  "
                  f"ram={p['mem_gb']}GB  uptime={p['uptime_s']:.0f}s  "
                  f"  hb={p['last_heartbeat_s_ago']:.0f}s  {p['url']}")

    commands = {
        "init": cmd_init,
        "deploy": cmd_deploy,
        "up": cmd_up,
        "down": cmd_down,
        "status": cmd_status,
        "chat": cmd_chat,
        "run": cmd_run,
        "info": cmd_info,
        "node": cmd_node_local,
        "download": cmd_download_local,
        "health": lambda a: cmd_health_standalone(a),
        "agent": cmd_agent,
        "ui": cmd_ui,
        "bench": cmd_bench,
        "leader": cmd_leader,
        "join": cmd_join,
        "peers": cmd_peers,
    }

    commands[args.command](args)


def cmd_health_standalone(args):
    """Standalone health check — no config needed."""
    import urllib.request
    nodes = [n.strip() for n in args.nodes.split(",")]
    print(f"Checking {len(nodes)} node(s)...\n")
    for url in nodes:
        try:
            resp = urllib.request.urlopen(f"{url}/health", timeout=5)
            data = json.loads(resp.read())
            partition = data.get("partition", "?")
            mem = data.get("memory_gb", "?")
            reqs = data.get("compute_requests", 0)
            print(f"  [OK]   {url} — partition {partition}, {mem} GB, {reqs} reqs")
        except Exception as e:
            print(f"  [FAIL] {url} — {e}")


if __name__ == "__main__":
    main()
