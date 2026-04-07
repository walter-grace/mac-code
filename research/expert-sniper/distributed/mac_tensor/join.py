#!/usr/bin/env python3
"""
mac-tensor join — register this Mac as a worker peer and start an expert node.

Usage:
    mac-tensor join http://leader-ip:8500 \\
        --model qwen35 \\
        --model-dir ~/models/qwen35-stream \\
        --port 9301

Workflow:
    1. POST /swarm/register to the leader → get assigned partition + peer_id
    2. Start the expert node with that partition
    3. Send heartbeats every 20s
    4. On Ctrl-C: POST /swarm/leave and shut down the node
    5. If leader changes our partition (other peer joined), restart node
       with new partition (NOT YET — phase 2 work)
"""

import argparse
import json
import os
import signal
import sys
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error


HEARTBEAT_INTERVAL = 20.0  # seconds


def get_local_ip():
    """Best-effort: figure out our outward-facing IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def get_mem_gb():
    """Get total system RAM in GB."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
        return round(int(out) / 1e9, 1)
    except Exception:
        return 16  # default guess


def post_json(url, body, timeout=10):
    """POST JSON, return parsed response."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def start_expert_node(model_key, partition, model_dir, port, scripts_dir):
    """Spawn the expert node subprocess. Returns the Popen handle."""
    from .cli import SUPPORTED_MODELS
    model = SUPPORTED_MODELS[model_key]
    script = os.path.join(scripts_dir, model["node_script"])
    if not os.path.exists(script):
        raise FileNotFoundError(f"Expert node script not found: {script}")

    cmd = [
        "python3", script,
        "--partition", partition,
        "--model-dir", os.path.expanduser(model_dir),
        "--port", str(port),
        "--memory-limit-gb", "14",
    ]
    print(f"  Spawning: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def wait_for_node_ready(url, timeout=180):
    """Poll /health until the expert node responds OK."""
    print(f"  Waiting for {url}/health to come up...", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                data = json.loads(r.read())
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def heartbeat_loop(leader_url, peer_id, stop_event):
    """Background thread: send heartbeats until told to stop."""
    while not stop_event.is_set():
        try:
            resp = post_json(
                f"{leader_url}/swarm/heartbeat",
                {"peer_id": peer_id},
                timeout=5,
            )
            # The leader may tell us our partition changed — for now we just log it
            # (Phase 2: trigger graceful re-partition by restarting the node)
        except Exception as e:
            print(f"  [heartbeat] failed: {e}", flush=True)
        stop_event.wait(HEARTBEAT_INTERVAL)


def main(args):
    leader_url = args.leader.rstrip("/")
    model_key = args.model
    model_dir = args.model_dir or f"~/models/{model_key}-stream"
    port = args.port

    print("=" * 60)
    print(f"mac-tensor swarm join")
    print(f"  Leader:    {leader_url}")
    print(f"  Model:     {model_key}")
    print(f"  Local IP:  {get_local_ip()}")
    print(f"  Port:      {port}")
    print(f"  RAM:       {get_mem_gb()} GB")
    print("=" * 60)

    # Step 1: Register with the leader
    print("\n[1/3] Registering with leader...")
    self_url = f"http://{get_local_ip()}:{port}"

    try:
        resp = post_json(f"{leader_url}/swarm/register", {
            "url": self_url,
            "mem_gb": get_mem_gb(),
            "meta": {
                "hostname": socket.gethostname(),
                "model": model_key,
            },
        }, timeout=10)
    except Exception as e:
        print(f"  ERROR: could not register: {e}")
        sys.exit(1)

    peer_id = resp["peer_id"]
    partition = resp["partition"]
    print(f"  Peer ID:   {peer_id}")
    print(f"  Partition: {partition}")

    # Step 2: Start the expert node with the assigned partition
    print(f"\n[2/3] Starting expert node...")
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    node_proc = start_expert_node(model_key, partition, model_dir, port, scripts_dir)

    if not wait_for_node_ready(self_url, timeout=300):
        print("  ERROR: expert node did not come up in time")
        node_proc.terminate()
        try:
            post_json(f"{leader_url}/swarm/leave", {"peer_id": peer_id}, timeout=5)
        except Exception:
            pass
        sys.exit(1)

    print(f"  Expert node ready at {self_url}")

    # Step 3: Heartbeat loop until interrupted
    print(f"\n[3/3] Sending heartbeats every {HEARTBEAT_INTERVAL}s. Ctrl-C to leave.")
    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=heartbeat_loop, args=(leader_url, peer_id, stop_event), daemon=True
    )
    hb_thread.start()

    def graceful_exit(signum=None, frame=None):
        print("\n  Sending leave signal to leader...")
        stop_event.set()
        try:
            post_json(f"{leader_url}/swarm/leave", {"peer_id": peer_id}, timeout=5)
        except Exception as e:
            print(f"  Warning: leave failed: {e}")
        print("  Stopping expert node...")
        node_proc.terminate()
        try:
            node_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            node_proc.kill()
        print("  Bye.")
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    # Block on the node process
    try:
        node_proc.wait()
    except KeyboardInterrupt:
        graceful_exit()
