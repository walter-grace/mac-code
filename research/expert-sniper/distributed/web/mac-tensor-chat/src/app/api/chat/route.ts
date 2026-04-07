// Proxy route: Vercel AI SDK ↔ mac-tensor backend
//
// The Vercel AI SDK's useChat() hook expects a streaming endpoint.
// Our mac-tensor backend speaks Server-Sent Events with custom event
// types: step_start, token, tool_call, tool_result, final, done, error.
//
// This route accepts the AI SDK request format (messages + optional
// image), forwards it to mac-tensor's /api/chat_vision, and translates
// the SSE events into the AI SDK's expected text/data stream format.

const MAC_TENSOR_URL =
  process.env.MAC_TENSOR_URL || "http://62.210.166.98:8500";

export const runtime = "nodejs";
export const maxDuration = 600;

export async function POST(req: Request) {
  const formData = await req.formData();
  const message = (formData.get("message") as string) ?? "";
  const image = formData.get("image") as File | null;
  const maxTokens = Number(formData.get("max_tokens") ?? 250);

  if (!message) {
    return new Response(JSON.stringify({ error: "empty message" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Build the request to the mac-tensor backend
  const upstreamForm = new FormData();
  upstreamForm.append("message", message);
  upstreamForm.append("max_tokens", String(maxTokens));
  if (image && image.size > 0) {
    upstreamForm.append("image", image);
  }

  const upstream = await fetch(`${MAC_TENSOR_URL}/api/chat_vision`, {
    method: "POST",
    body: upstreamForm,
  });

  if (!upstream.ok || !upstream.body) {
    return new Response(
      JSON.stringify({
        error: `mac-tensor upstream ${upstream.status}`,
      }),
      {
        status: 502,
        headers: { "Content-Type": "application/json" },
      },
    );
  }

  // Pass through the SSE stream as-is. The frontend will parse the
  // mac-tensor event format directly (it knows our event types).
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
