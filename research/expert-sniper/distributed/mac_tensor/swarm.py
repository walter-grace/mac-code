#!/usr/bin/env python3
"""
mac-tensor swarm — dynamic peer registry for distributed inference.

Architecture:
  - One leader (mac-tensor leader) holds the peer registry
  - Workers join via mac-tensor join http://leader:port
  - Leader assigns each worker a partition slice based on current peer count
  - Workers send heartbeats; missed heartbeats trigger re-partitioning
  - Coordinator queries the registry to get the live peer list

Endpoints (added to the leader server):
  POST /swarm/register   {model, mem_gb, port} → {peer_id, partition, peers}
  POST /swarm/heartbeat  {peer_id} → {ok, repartition?}
  POST /swarm/leave      {peer_id} → {ok}
  GET  /swarm/peers      → list of peers with status
"""

import json
import os
import time
import threading
import uuid


HEARTBEAT_TIMEOUT = 60.0  # seconds — peer is dead if no heartbeat in this window


class SwarmRegistry:
    """In-memory peer registry. Thread-safe."""

    def __init__(self, model_key):
        self.model_key = model_key
        self.lock = threading.Lock()
        self.peers = {}  # peer_id → {url, partition, mem_gb, last_heartbeat, joined_at}
        self.partition_version = 0  # bumped on every re-partition

    def register(self, url, mem_gb, peer_meta=None):
        """Add a new peer and return their assigned partition."""
        with self.lock:
            peer_id = str(uuid.uuid4())[:8]
            self.peers[peer_id] = {
                "id": peer_id,
                "url": url,
                "mem_gb": mem_gb,
                "meta": peer_meta or {},
                "joined_at": time.time(),
                "last_heartbeat": time.time(),
                "partition": None,  # filled by repartition()
                "alive": True,
            }
            self._repartition()
            return peer_id, self.peers[peer_id]["partition"]

    def heartbeat(self, peer_id):
        """Mark peer as alive."""
        with self.lock:
            if peer_id not in self.peers:
                return False, None
            self.peers[peer_id]["last_heartbeat"] = time.time()
            self.peers[peer_id]["alive"] = True
            return True, self.partition_version

    def leave(self, peer_id):
        """Gracefully remove a peer."""
        with self.lock:
            if peer_id in self.peers:
                del self.peers[peer_id]
                self._repartition()
                return True
            return False

    def reap_dead(self):
        """Remove peers that haven't heartbeated recently. Returns set of dead IDs."""
        with self.lock:
            now = time.time()
            dead = [
                pid for pid, p in self.peers.items()
                if (now - p["last_heartbeat"]) > HEARTBEAT_TIMEOUT
            ]
            for pid in dead:
                del self.peers[pid]
            if dead:
                self._repartition()
            return dead

    def get_live_peers(self):
        """Return live peers sorted by join time (stable order)."""
        with self.lock:
            now = time.time()
            return sorted([
                {
                    **p,
                    "alive": (now - p["last_heartbeat"]) <= HEARTBEAT_TIMEOUT,
                }
                for p in self.peers.values()
            ], key=lambda p: p["joined_at"])

    def _repartition(self):
        """Re-assign partitions across all current peers.

        Stable: the first peer to join always gets the first slice, etc.
        This avoids unnecessarily moving partitions when one peer joins/leaves.
        """
        from .cli import SUPPORTED_MODELS

        model = SUPPORTED_MODELS.get(self.model_key)
        if model is None:
            return
        num_experts = model["num_experts"]

        sorted_peers = sorted(self.peers.values(), key=lambda p: p["joined_at"])
        n = len(sorted_peers)
        if n == 0:
            self.partition_version += 1
            return

        # Equal split with the remainder going to the last peer
        per_peer = num_experts // n
        cursor = 0
        for i, peer in enumerate(sorted_peers):
            if i == n - 1:
                end = num_experts - 1
            else:
                end = cursor + per_peer - 1
            peer["partition"] = f"{cursor}-{end}"
            cursor = end + 1

        self.partition_version += 1

    def status(self):
        """Return a summary dict."""
        with self.lock:
            now = time.time()
            return {
                "model": self.model_key,
                "peer_count": len(self.peers),
                "partition_version": self.partition_version,
                "peers": [
                    {
                        "id": p["id"],
                        "url": p["url"],
                        "partition": p["partition"],
                        "mem_gb": p["mem_gb"],
                        "alive": (now - p["last_heartbeat"]) <= HEARTBEAT_TIMEOUT,
                        "uptime_s": round(now - p["joined_at"], 1),
                        "last_heartbeat_s_ago": round(now - p["last_heartbeat"], 1),
                    }
                    for p in sorted(self.peers.values(), key=lambda p: p["joined_at"])
                ],
            }


def reaper_loop(registry, interval=15):
    """Background thread: reap dead peers periodically."""
    while True:
        time.sleep(interval)
        try:
            dead = registry.reap_dead()
            if dead:
                print(f"[swarm] reaped {len(dead)} dead peer(s): {dead}")
        except Exception as e:
            print(f"[swarm] reaper error: {e}")
