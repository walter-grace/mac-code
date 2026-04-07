// Direct proxy to mac-tensor /api/falcon
//
// Used by the "Ground" button in the chat UI — bypasses the agent loop
// and calls Falcon Perception directly with a text query + image,
// returning JSON with mask metadata + a base64 annotated image.

const MAC_TENSOR_URL =
  process.env.MAC_TENSOR_URL || "http://62.210.166.98:8500";

export const runtime = "nodejs";
export const maxDuration = 120;

export async function POST(req: Request) {
  const formData = await req.formData();
  const query = (formData.get("query") as string) ?? "";
  const image = formData.get("image") as File | null;

  if (!query) {
    return new Response(JSON.stringify({ error: "query required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }
  if (!image || image.size === 0) {
    return new Response(JSON.stringify({ error: "image required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const upstreamForm = new FormData();
  upstreamForm.append("query", query);
  upstreamForm.append("image", image);

  const upstream = await fetch(`${MAC_TENSOR_URL}/api/falcon`, {
    method: "POST",
    body: upstreamForm,
  });

  if (!upstream.ok) {
    const text = await upstream.text();
    return new Response(
      JSON.stringify({
        error: `mac-tensor upstream ${upstream.status}: ${text.slice(0, 200)}`,
      }),
      {
        status: 502,
        headers: { "Content-Type": "application/json" },
      },
    );
  }

  const data = await upstream.json();
  return Response.json(data);
}
