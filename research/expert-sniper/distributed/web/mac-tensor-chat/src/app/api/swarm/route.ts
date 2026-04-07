// Returns swarm/system status for the dashboard.
//
// In real swarm mode, this proxies /swarm/peers from the leader.
// In single-machine vision mode (current), it returns a synthetic
// "single peer" payload so the dashboard still has something to render.

const MAC_TENSOR_URL =
  process.env.MAC_TENSOR_URL || "http://62.210.166.98:8500";

export const runtime = "nodejs";

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

export async function GET() {
  // First, ask the backend what mode it's in
  let info: {
    model?: string;
    vision?: boolean;
    falcon?: boolean;
    swarm_leader?: boolean;
    nodes?: string[] | null;
  } = {};
  try {
    const res = await fetch(`${MAC_TENSOR_URL}/api/info`, {
      cache: "no-store",
    });
    if (res.ok) {
      info = await res.json();
    }
  } catch {
    /* fall through to error response */
  }

  // Swarm leader mode → proxy real registry
  if (info.swarm_leader) {
    try {
      const res = await fetch(`${MAC_TENSOR_URL}/swarm/peers`, {
        cache: "no-store",
      });
      const data = await res.json();
      const status: SwarmStatus = { ...data, mode: "swarm" };
      return Response.json(status);
    } catch (e) {
      return Response.json({ error: String(e) }, { status: 502 });
    }
  }

  // Single-machine vision mode → synthesize a 1-peer view
  if (info.vision) {
    const synthetic: SwarmStatus = {
      model: info.model ?? "gemma4",
      mode: "vision-single",
      peer_count: 1,
      partition_version: 1,
      peers: [
        {
          id: "vision-host",
          url: MAC_TENSOR_URL,
          partition: "0-127",
          mem_gb: 16,
          alive: true,
          uptime_s: 0,
          last_heartbeat_s_ago: 0,
          hostname: "Gemma 4 Vision Sniper",
          expert_layers: 30,
        },
      ],
    };
    return Response.json(synthetic);
  }

  return Response.json(
    { error: "backend unreachable", url: MAC_TENSOR_URL },
    { status: 503 },
  );
}
