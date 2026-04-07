import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0a0a0f] via-[#111128] to-[#0a0a0f] text-slate-100">
      <div
        className="absolute inset-0 opacity-40 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(99,102,241,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.05) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      <main className="relative z-10 flex min-h-screen flex-col items-center justify-center px-6 py-24">
        <div className="text-center">
          <h1
            className="text-7xl font-extrabold tracking-tight md:text-8xl"
            style={{
              background:
                "linear-gradient(135deg, #6366f1, #22d3ee, #f472b6)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            mac-tensor
          </h1>
          <p className="mt-6 text-2xl text-slate-400">
            Distributed MoE inference on Apple Silicon
          </p>
          <p className="mt-2 text-base text-slate-500">
            Gemma 4 vision · Falcon Perception · Live swarm
          </p>
        </div>

        <div className="mt-16 grid w-full max-w-4xl grid-cols-1 gap-6 md:grid-cols-2">
          <Link
            href="/chat"
            className="group rounded-2xl border border-indigo-500/30 bg-indigo-500/5 p-8 transition-all hover:border-indigo-400 hover:bg-indigo-500/10 hover:shadow-2xl hover:shadow-indigo-500/20"
          >
            <div className="text-5xl">💬</div>
            <h2 className="mt-4 text-2xl font-semibold text-white">
              Vision Chat
            </h2>
            <p className="mt-2 text-sm text-slate-400">
              Drop in an image, ask anything. Gemma 4 sees the image, calls
              Falcon Perception when it needs precision, and answers in plain
              English.
            </p>
            <div className="mt-4 text-xs font-mono text-cyan-400">
              powered by Vercel AI SDK →
            </div>
          </Link>

          <Link
            href="/dashboard"
            className="group rounded-2xl border border-cyan-500/30 bg-cyan-500/5 p-8 transition-all hover:border-cyan-400 hover:bg-cyan-500/10 hover:shadow-2xl hover:shadow-cyan-500/20"
          >
            <div className="text-5xl">🌐</div>
            <h2 className="mt-4 text-2xl font-semibold text-white">
              Swarm Dashboard
            </h2>
            <p className="mt-2 text-sm text-slate-400">
              3D visualization of every Mac in the network. See which expert
              partitions live where, watch tokens flow between nodes in real
              time, and monitor health.
            </p>
            <div className="mt-4 text-xs font-mono text-cyan-400">
              powered by Three.js →
            </div>
          </Link>
        </div>

        <footer className="mt-20 text-xs text-slate-600">
          <a
            href="https://github.com/walter-grace/mac-code"
            className="hover:text-slate-400"
          >
            github.com/walter-grace/mac-code
          </a>
        </footer>
      </main>
    </div>
  );
}
