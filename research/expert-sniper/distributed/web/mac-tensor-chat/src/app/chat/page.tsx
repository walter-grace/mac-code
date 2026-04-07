"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";

type AgentEvent =
  | { type: "step_start"; step: number; max: number }
  | { type: "token"; text: string }
  | { type: "tool_call"; tool: string; args: string }
  | { type: "tool_result"; result: string }
  | { type: "final"; text: string }
  | { type: "done" }
  | { type: "error"; message: string };

type FalconMask = {
  id: number;
  slot: string;
  area_fraction: number;
  centroid_norm: { x: number; y: number };
  bbox_norm: { x1: number; y1: number; x2: number; y2: number };
  image_region: string;
};

type FalconResult = {
  query: string;
  count: number;
  mask_ids: number[];
  masks: FalconMask[];
  annotated_image: string;
  elapsed_seconds: number;
};

type Message = {
  id: string;
  role: "user" | "assistant" | "falcon";
  text: string;
  imageDataUrl?: string;
  toolCalls: { tool: string; args: string; result: string }[];
  falcon?: FalconResult;
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [attachedImage, setAttachedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [stepInfo, setStepInfo] = useState<string>("");
  const fileRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, isStreaming]);

  const handleFile = (f: File | null) => {
    if (!f) {
      setAttachedImage(null);
      setPreviewUrl("");
      return;
    }
    setAttachedImage(f);
    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const send = async () => {
    if (isStreaming || !input.trim()) return;
    const text = input.trim();
    const imageDataUrl = previewUrl || undefined;
    const imgFile = attachedImage;

    // Push user message + create empty assistant message
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text,
      imageDataUrl,
      toolCalls: [],
    };
    const asstMsg: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "",
      toolCalls: [],
    };
    setMessages((m) => [...m, userMsg, asstMsg]);
    setInput("");
    setAttachedImage(null);
    setPreviewUrl("");
    if (fileRef.current) fileRef.current.value = "";

    setIsStreaming(true);
    setStepInfo("Sending...");

    try {
      const fd = new FormData();
      fd.append("message", text);
      fd.append("max_tokens", "300");
      if (imgFile) fd.append("image", imgFile);

      const res = await fetch("/api/chat", { method: "POST", body: fd });
      if (!res.ok || !res.body) {
        throw new Error(`HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let pendingTool: { tool: string; args: string } | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let event: AgentEvent;
          try {
            event = JSON.parse(line.slice(6));
          } catch {
            continue;
          }
          switch (event.type) {
            case "step_start":
              setStepInfo(`Step ${event.step}/${event.max}`);
              break;
            case "token":
              setMessages((m) => {
                const next = [...m];
                const last = next[next.length - 1];
                if (last && last.role === "assistant") {
                  next[next.length - 1] = {
                    ...last,
                    text: last.text + event.text,
                  };
                }
                return next;
              });
              break;
            case "tool_call":
              pendingTool = { tool: event.tool, args: event.args };
              setStepInfo(`Calling ${event.tool}...`);
              break;
            case "tool_result":
              if (pendingTool) {
                const tc = { ...pendingTool, result: event.result };
                pendingTool = null;
                setMessages((m) => {
                  const next = [...m];
                  const last = next[next.length - 1];
                  if (last && last.role === "assistant") {
                    next[next.length - 1] = {
                      ...last,
                      toolCalls: [...last.toolCalls, tc],
                    };
                  }
                  return next;
                });
              }
              break;
            case "final":
              setMessages((m) => {
                const next = [...m];
                const last = next[next.length - 1];
                if (last && last.role === "assistant") {
                  next[next.length - 1] = { ...last, text: event.text };
                }
                return next;
              });
              break;
            case "error":
              setMessages((m) => {
                const next = [...m];
                const last = next[next.length - 1];
                if (last && last.role === "assistant") {
                  next[next.length - 1] = {
                    ...last,
                    text: `Error: ${event.message}`,
                  };
                }
                return next;
              });
              break;
            case "done":
              setStepInfo("");
              break;
          }
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setMessages((m) => {
        const next = [...m];
        const last = next[next.length - 1];
        if (last && last.role === "assistant") {
          next[next.length - 1] = { ...last, text: `Error: ${msg}` };
        }
        return next;
      });
    } finally {
      setIsStreaming(false);
      setStepInfo("");
    }
  };

  const ground = async () => {
    if (isStreaming) return;
    if (!attachedImage) {
      alert("Attach an image first (📷 button or drag-and-drop)");
      return;
    }
    const query = input.trim();
    if (!query) {
      alert('Type what to find (e.g. "bird", "car", "person")');
      return;
    }

    const imageDataUrl = previewUrl || undefined;
    const imgFile = attachedImage;
    setInput("");
    setAttachedImage(null);
    setPreviewUrl("");
    if (fileRef.current) fileRef.current.value = "";

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      text: `🎯 Find: "${query}"`,
      imageDataUrl,
      toolCalls: [],
    };
    const placeholderMsg: Message = {
      id: crypto.randomUUID(),
      role: "falcon",
      text: "Running Falcon Perception...",
      toolCalls: [],
    };
    setMessages((m) => [...m, userMsg, placeholderMsg]);
    setIsStreaming(true);

    try {
      const fd = new FormData();
      fd.append("query", query);
      fd.append("image", imgFile);

      const res = await fetch("/api/falcon", { method: "POST", body: fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${res.status}`);
      }
      const data: FalconResult = await res.json();
      setMessages((m) => {
        const next = [...m];
        const last = next[next.length - 1];
        if (last && last.role === "falcon") {
          next[next.length - 1] = {
            ...last,
            text: `Found ${data.count} ${data.count === 1 ? "instance" : "instances"} of "${query}" in ${data.elapsed_seconds}s`,
            falcon: data,
          };
        }
        return next;
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setMessages((m) => {
        const next = [...m];
        const last = next[next.length - 1];
        if (last && last.role === "falcon") {
          next[next.length - 1] = { ...last, text: `Error: ${msg}` };
        }
        return next;
      });
    } finally {
      setIsStreaming(false);
    }
  };

  const reset = async () => {
    await fetch("/api/chat", { method: "POST", body: new FormData() }).catch(
      () => null,
    );
    setMessages([]);
  };

  return (
    <div className="flex h-screen flex-col bg-gradient-to-br from-[#0a0a0f] via-[#111128] to-[#0a0a0f] text-slate-100">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-indigo-500/20 bg-black/50 px-6 py-4 backdrop-blur-md">
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
          <span className="text-sm text-slate-400">
            Gemma 4-26B Vision · Falcon Perception
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/dashboard"
            className="rounded-lg border border-cyan-500/30 bg-cyan-500/5 px-4 py-2 text-sm transition hover:bg-cyan-500/10"
          >
            🌐 Dashboard
          </Link>
          <button
            onClick={reset}
            className="rounded-lg border border-slate-700 bg-slate-800/50 px-4 py-2 text-sm transition hover:bg-slate-700/50"
          >
            Reset
          </button>
        </div>
      </header>

      {/* Messages */}
      <main ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
        <div className="mx-auto max-w-3xl space-y-6">
          {messages.length === 0 && (
            <div className="mt-20 text-center text-slate-500">
              <p className="mb-4 text-2xl text-slate-300">
                Drop an image and ask anything
              </p>
              <p className="text-sm">
                Try: <em className="text-slate-400">&ldquo;What do you see?&rdquo;</em>
                {" · "}
                <em className="text-slate-400">&ldquo;How many people?&rdquo;</em>
                {" · "}
                <em className="text-slate-400">&ldquo;Where exactly is the bird?&rdquo;</em>
              </p>
            </div>
          )}

          {messages.map((m) => {
            const labelMap = {
              user: "You",
              assistant: "Agent",
              falcon: "Falcon Perception",
            };
            const borderMap = {
              user: "border-cyan-500/30 bg-cyan-500/10",
              assistant: "border-indigo-500/20 bg-indigo-500/5",
              falcon: "border-orange-500/30 bg-orange-500/5",
            };
            const labelColorMap = {
              user: "text-cyan-400",
              assistant: "text-indigo-400",
              falcon: "text-orange-400",
            };
            return (
              <div
                key={m.id}
                className={`flex flex-col gap-2 ${m.role === "user" ? "items-end" : "items-start"}`}
              >
                <div
                  className={`text-xs font-semibold uppercase tracking-wider ${labelColorMap[m.role]}`}
                >
                  {labelMap[m.role]}
                </div>
                <div
                  className={`max-w-[85%] rounded-2xl border px-5 py-3 ${borderMap[m.role]}`}
                >
                  {m.imageDataUrl && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={m.imageDataUrl}
                      alt=""
                      className="mb-3 max-h-64 rounded-lg border border-slate-700"
                    />
                  )}
                  {m.toolCalls.map((tc, i) => (
                    <details
                      key={i}
                      className="mb-2 rounded-lg border border-orange-500/30 bg-orange-500/5 px-3 py-2 text-sm"
                    >
                      <summary className="cursor-pointer">
                        <span className="font-mono text-orange-400">
                          ⚙ &lt;{tc.tool}&gt;
                        </span>
                        <span className="ml-2 text-slate-400">
                          {tc.args.slice(0, 60)}
                        </span>
                      </summary>
                      <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap text-xs text-slate-400">
                        {tc.result}
                      </pre>
                    </details>
                  ))}
                  <div className="whitespace-pre-wrap text-base leading-relaxed">
                    {m.text || (
                      <span className="italic text-slate-500">
                        {m === messages[messages.length - 1] && stepInfo
                          ? stepInfo
                          : "..."}
                      </span>
                    )}
                  </div>
                  {m.falcon && (
                    <div className="mt-3">
                      <div className="relative inline-block">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={m.falcon.annotated_image}
                          alt="annotated"
                          className="max-w-full rounded-lg border border-orange-500/40"
                        />
                        <a
                          href={m.falcon.annotated_image}
                          download={`falcon-${m.falcon.query.replace(/\s+/g, "_")}.png`}
                          className="absolute right-2 top-2 rounded-md border border-white/30 bg-black/70 px-2 py-1 text-xs text-white hover:bg-black/90"
                        >
                          ⬇ Download
                        </a>
                      </div>
                      {m.falcon.masks.length > 0 && (
                        <div className="mt-3 space-y-1 text-xs text-slate-400">
                          {m.falcon.masks.map((mask) => {
                            const cx = (mask.centroid_norm.x * 100).toFixed(0);
                            const cy = (mask.centroid_norm.y * 100).toFixed(0);
                            const area = (mask.area_fraction * 100).toFixed(1);
                            return (
                              <div key={mask.id}>
                                <span className="text-orange-400">
                                  #{mask.id}
                                </span>{" "}
                                — {mask.image_region}, center ({cx}%, {cy}%),
                                area {area}%
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </main>

      {/* Input bar */}
      <footer className="border-t border-indigo-500/20 bg-black/50 px-6 py-4 backdrop-blur-md">
        <div className="mx-auto max-w-3xl">
          {previewUrl && (
            <div className="mb-3 flex items-center gap-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrl}
                alt=""
                className="h-20 rounded-lg border border-slate-700"
              />
              <button
                onClick={() => handleFile(null)}
                className="rounded-full border border-red-500/30 bg-red-500/10 px-3 py-1 text-xs text-red-400 hover:bg-red-500/20"
              >
                ✕ remove
              </button>
            </div>
          )}
          <div className="flex items-end gap-3">
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={isStreaming}
              className="h-12 w-12 rounded-xl border border-slate-700 bg-slate-800/50 text-2xl transition hover:bg-slate-700/50 disabled:opacity-40"
              title="Attach image"
            >
              📷
            </button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Ask the agent anything..."
              rows={1}
              disabled={isStreaming}
              className="flex-1 resize-none rounded-xl border border-slate-700 bg-slate-800/30 px-4 py-3 text-base text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none disabled:opacity-50"
            />
            <button
              onClick={ground}
              disabled={isStreaming || !attachedImage || !input.trim()}
              className="h-12 rounded-xl border border-orange-500/40 bg-orange-500/10 px-4 font-semibold text-orange-300 transition hover:bg-orange-500/20 disabled:opacity-40"
              title="Ground: directly call Falcon Perception with the typed query (skips agent reasoning). Needs an image attached."
            >
              🎯 Ground
            </button>
            <button
              onClick={send}
              disabled={isStreaming || !input.trim()}
              className="h-12 rounded-xl px-6 font-semibold text-white transition disabled:opacity-40"
              style={{
                background:
                  "linear-gradient(135deg, #6366f1, #22d3ee)",
              }}
            >
              {isStreaming ? "..." : "Send"}
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
