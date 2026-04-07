"use client";

import { useEffect, useState, useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text, Html } from "@react-three/drei";
import * as THREE from "three";
import Link from "next/link";

type Peer = {
  id: string;
  url: string;
  partition: string | null;
  mem_gb: number;
  alive: boolean;
  uptime_s: number;
  last_heartbeat_s_ago: number;
  hostname?: string;
  expert_layers?: number;
};

type SwarmStatus = {
  model: string;
  mode: "swarm" | "vision-single";
  peer_count: number;
  partition_version: number;
  peers: Peer[];
};

function PeerNode({
  peer,
  position,
  isSelected,
  onClick,
}: {
  peer: Peer;
  position: [number, number, number];
  isSelected: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null!);
  const ringRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    if (meshRef.current) {
      // Pulse the alive nodes
      const t = state.clock.getElapsedTime();
      const scale = peer.alive ? 1 + Math.sin(t * 2) * 0.05 : 0.7;
      meshRef.current.scale.setScalar(scale);
      meshRef.current.rotation.y += 0.002;
    }
    if (ringRef.current) {
      ringRef.current.rotation.z += 0.005;
    }
  });

  const color = peer.alive ? "#22d3ee" : "#64748b";
  const glow = peer.alive ? "#22d3ee" : "#334155";

  return (
    <group position={position}>
      {/* Main node sphere */}
      <mesh ref={meshRef} onClick={onClick}>
        <icosahedronGeometry args={[0.8, 1]} />
        <meshStandardMaterial
          color={color}
          emissive={glow}
          emissiveIntensity={peer.alive ? 0.5 : 0.1}
          metalness={0.7}
          roughness={0.2}
        />
      </mesh>

      {/* Selection ring */}
      {isSelected && (
        <mesh ref={ringRef} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[1.3, 0.05, 16, 64]} />
          <meshStandardMaterial
            color="#f472b6"
            emissive="#f472b6"
            emissiveIntensity={1}
          />
        </mesh>
      )}

      {/* Hostname label below */}
      <Text
        position={[0, -1.5, 0]}
        fontSize={0.3}
        color="#f1f5f9"
        anchorX="center"
        anchorY="middle"
      >
        {peer.hostname || peer.id}
      </Text>

      {/* Partition label above */}
      {peer.partition && (
        <Text
          position={[0, 1.4, 0]}
          fontSize={0.25}
          color="#22d3ee"
          anchorX="center"
          anchorY="middle"
        >
          experts {peer.partition}
        </Text>
      )}
    </group>
  );
}

function ConnectionLine({
  from,
  to,
  active,
}: {
  from: [number, number, number];
  to: [number, number, number];
  active: boolean;
}) {
  const lineRef = useRef<THREE.LineSegments>(null!);
  const points = useMemo(
    () => [new THREE.Vector3(...from), new THREE.Vector3(...to)],
    [from, to],
  );
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry().setFromPoints(points);
    return g;
  }, [points]);

  return (
    <lineSegments ref={lineRef} geometry={geometry}>
      <lineBasicMaterial
        color={active ? "#22d3ee" : "#1e293b"}
        opacity={active ? 0.8 : 0.3}
        transparent
      />
    </lineSegments>
  );
}

function CenterCoordinator() {
  const meshRef = useRef<THREE.Mesh>(null!);
  useFrame((state) => {
    if (meshRef.current) {
      const t = state.clock.getElapsedTime();
      meshRef.current.scale.setScalar(1 + Math.sin(t * 3) * 0.08);
      meshRef.current.rotation.y += 0.005;
    }
  });
  return (
    <group>
      <mesh ref={meshRef}>
        <octahedronGeometry args={[0.6, 0]} />
        <meshStandardMaterial
          color="#f472b6"
          emissive="#f472b6"
          emissiveIntensity={0.8}
          metalness={0.9}
          roughness={0.1}
        />
      </mesh>
      <Text
        position={[0, -1.1, 0]}
        fontSize={0.25}
        color="#f472b6"
        anchorX="center"
        anchorY="middle"
      >
        coordinator
      </Text>
    </group>
  );
}

function SwarmScene({
  peers,
  selectedId,
  onSelect,
}: {
  peers: Peer[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  // Layout: peers in a circle around the coordinator
  const positions: Record<string, [number, number, number]> = {};
  const radius = Math.max(3, peers.length * 0.8);
  peers.forEach((p, i) => {
    const angle = (i / Math.max(1, peers.length)) * Math.PI * 2;
    positions[p.id] = [
      Math.cos(angle) * radius,
      Math.sin(angle * 0.5) * 0.5,
      Math.sin(angle) * radius,
    ];
  });

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1.5} color="#6366f1" />
      <pointLight position={[-10, -10, -10]} intensity={1} color="#22d3ee" />

      <CenterCoordinator />

      {peers.map((p) => (
        <ConnectionLine
          key={`line-${p.id}`}
          from={[0, 0, 0]}
          to={positions[p.id]}
          active={p.alive}
        />
      ))}

      {peers.map((p) => (
        <PeerNode
          key={p.id}
          peer={p}
          position={positions[p.id]}
          isSelected={selectedId === p.id}
          onClick={() => onSelect(p.id === selectedId ? null : p.id)}
        />
      ))}

      <OrbitControls
        enablePan={false}
        autoRotate
        autoRotateSpeed={0.3}
        minDistance={5}
        maxDistance={30}
      />
    </>
  );
}

export default function DashboardPage() {
  const [status, setStatus] = useState<SwarmStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/swarm", { cache: "no-store" });
        if (!res.ok) {
          setError(`HTTP ${res.status}`);
          return;
        }
        const data = await res.json();
        if (data.error) {
          setError(data.error);
          return;
        }
        setStatus(data);
        setError(null);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      }
    };
    fetchStatus();
    const t = setInterval(fetchStatus, 5000); // refresh every 5s
    return () => clearInterval(t);
  }, []);

  const selectedPeer = status?.peers.find((p) => p.id === selectedId);

  return (
    <div className="flex h-screen flex-col bg-gradient-to-br from-[#0a0a0f] via-[#111128] to-[#0a0a0f] text-slate-100">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-cyan-500/20 bg-black/50 px-6 py-4 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <Link
            href="/"
            className="text-2xl font-extrabold tracking-tight"
            style={{
              background:
                "linear-gradient(135deg, #6366f1, #22d3ee, #f472b6)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            mac-tensor
          </Link>
          <span className="text-sm text-slate-400">Swarm Dashboard</span>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/chat"
            className="rounded-lg border border-indigo-500/30 bg-indigo-500/5 px-4 py-2 text-sm transition hover:bg-indigo-500/10"
          >
            💬 Chat
          </Link>
        </div>
      </header>

      {/* 3D scene + side panel */}
      <main className="relative flex flex-1 overflow-hidden">
        {/* 3D scene */}
        <div className="relative flex-1">
          {error && (
            <div className="absolute left-1/2 top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 rounded-xl border border-red-500/30 bg-red-500/10 p-6 text-center">
              <div className="text-lg text-red-400">Backend unreachable</div>
              <div className="mt-2 text-sm text-slate-400">{error}</div>
            </div>
          )}
          {!status && !error && (
            <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 text-slate-500">
              Loading swarm...
            </div>
          )}
          {status && (
            <Canvas camera={{ position: [8, 6, 8], fov: 50 }}>
              <color attach="background" args={["#0a0a0f"]} />
              <fog attach="fog" args={["#0a0a0f", 15, 40]} />
              <SwarmScene
                peers={status.peers}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            </Canvas>
          )}

          {/* Stats overlay */}
          {status && (
            <div className="pointer-events-none absolute left-6 top-6 space-y-2">
              <div className="rounded-lg border border-indigo-500/30 bg-black/50 px-4 py-3 backdrop-blur-md">
                <div className="text-xs uppercase tracking-wider text-slate-500">
                  Model
                </div>
                <div className="text-lg font-semibold text-cyan-400">
                  {status.model}
                </div>
              </div>
              <div className="rounded-lg border border-indigo-500/30 bg-black/50 px-4 py-3 backdrop-blur-md">
                <div className="text-xs uppercase tracking-wider text-slate-500">
                  Mode
                </div>
                <div className="text-lg font-semibold text-pink-400">
                  {status.mode}
                </div>
              </div>
              <div className="rounded-lg border border-indigo-500/30 bg-black/50 px-4 py-3 backdrop-blur-md">
                <div className="text-xs uppercase tracking-wider text-slate-500">
                  Peers
                </div>
                <div className="text-lg font-semibold text-green-400">
                  {status.peers.filter((p) => p.alive).length} alive /{" "}
                  {status.peer_count}
                </div>
              </div>
              <div className="rounded-lg border border-indigo-500/30 bg-black/50 px-4 py-3 backdrop-blur-md">
                <div className="text-xs uppercase tracking-wider text-slate-500">
                  Partition v
                </div>
                <div className="text-lg font-semibold text-orange-400">
                  {status.partition_version}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right side panel — peer details */}
        <aside className="w-96 overflow-y-auto border-l border-cyan-500/20 bg-black/40 px-6 py-6 backdrop-blur-md">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-400">
            Peers
          </h2>
          {!status?.peers.length && (
            <div className="text-sm text-slate-500">No peers yet.</div>
          )}
          <div className="space-y-3">
            {status?.peers.map((p) => {
              const isSel = p.id === selectedId;
              return (
                <button
                  key={p.id}
                  onClick={() =>
                    setSelectedId(p.id === selectedId ? null : p.id)
                  }
                  className={`w-full rounded-lg border px-4 py-3 text-left transition ${
                    isSel
                      ? "border-pink-400 bg-pink-500/10"
                      : p.alive
                        ? "border-cyan-500/30 bg-cyan-500/5 hover:border-cyan-400"
                        : "border-slate-700 bg-slate-800/30"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="font-mono text-xs text-slate-500">
                      {p.id}
                    </div>
                    <div
                      className={`h-2 w-2 rounded-full ${p.alive ? "bg-green-400" : "bg-red-400"}`}
                    />
                  </div>
                  <div className="mt-1 text-sm font-semibold">
                    {p.hostname || p.url}
                  </div>
                  <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-slate-400">
                    <div>
                      <span className="text-slate-500">RAM</span> {p.mem_gb} GB
                    </div>
                    <div>
                      <span className="text-slate-500">partition</span>{" "}
                      {p.partition ?? "?"}
                    </div>
                    {p.expert_layers !== undefined && (
                      <div>
                        <span className="text-slate-500">layers</span>{" "}
                        {p.expert_layers}
                      </div>
                    )}
                    <div>
                      <span className="text-slate-500">uptime</span>{" "}
                      {Math.round(p.uptime_s)}s
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          {selectedPeer && (
            <div className="mt-6 rounded-lg border border-pink-500/30 bg-pink-500/5 px-4 py-3">
              <div className="mb-2 text-xs uppercase tracking-wider text-pink-400">
                Selected
              </div>
              <pre className="overflow-auto text-xs text-slate-300">
                {JSON.stringify(selectedPeer, null, 2)}
              </pre>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
