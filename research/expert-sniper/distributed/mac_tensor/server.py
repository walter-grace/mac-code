#!/usr/bin/env python3
"""
mac-tensor ui — Web chat UI for the distributed agent.

Serves a single-page HTML chat interface and a Server-Sent Events
endpoint that streams agent events (steps, tool calls, results, final answer).

Usage:
    mac-tensor ui --model gemma4 --nodes http://mac2:8401,http://mac3:8401
    # Then open http://localhost:8500 in your browser
"""

import json
import os
import sys
import time
import threading
from queue import Queue, Empty


def run_server(model_key, node_urls=None, host="0.0.0.0", port=8500, allow_write=False,
               vision=False, stream_dir=None, source_dir=None,
               falcon=False, falcon_model=None,
               swarm_leader=False):
    """Start the FastAPI server with the agent backend pre-loaded.

    Modes:
      - Distributed text-only: pass node_urls
      - Single-machine vision (Gemma 4 only): vision=True
      - Vision + Falcon Perception (segmentation): vision=True, falcon=True
      - Swarm leader: swarm_leader=True (peer registry + dynamic coordinator)
    """
    from fastapi import FastAPI, Request, UploadFile, File, Form
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from .agent import AgentBackend, run_agent_turn_stream

    vision_engine = None
    falcon_tools = None
    swarm_registry = None

    if swarm_leader:
        from .swarm import SwarmRegistry, reaper_loop
        swarm_registry = SwarmRegistry(model_key=model_key)
        threading.Thread(target=reaper_loop, args=(swarm_registry,), daemon=True).start()
        print(f"Swarm registry started (model={model_key})")

    if vision:
        print(f"Loading vision Gemma 4 sniper (single-machine)...")
        from .vision_engine import VisionGemma4Sniper
        vision_engine = VisionGemma4Sniper(
            stream_dir=stream_dir or "~/models/gemma4-stream",
            source_dir=source_dir or "~/models/gemma4-26b-4bit",
        )
        vision_engine.load()
        print("Vision engine ready.")

        if falcon:
            print(f"Loading Falcon Perception...")
            from .falcon_perception import FalconPerceptionTools
            falcon_tools = FalconPerceptionTools.load(
                model_path=falcon_model or "/Users/bigneek/models/falcon-perception"
            )
            print("Falcon Perception ready.")

        backend = None  # Not used in vision mode
    else:
        print(f"Loading {model_key} distributed engine...")
        backend = AgentBackend(model_key=model_key, node_urls=node_urls)
        backend.load()
        print(f"Backend ready. Connected to {len(node_urls)} expert nodes.")

    app = FastAPI(title="mac-tensor agent UI")

    # Read the static HTML file shipped alongside this server
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    html_path = os.path.join(static_dir, "chat.html")
    with open(html_path) as f:
        chat_html = f.read()

    # Inject backend info into the HTML so the UI can show it
    if vision:
        model_label = "Gemma 4-26B-A4B (Vision)"
        node_count_label = "single Mac · vision enabled"
    else:
        model_label = {"gemma4": "Gemma 4-26B-A4B",
                       "qwen35": "Qwen 3.5-35B-A3B"}.get(model_key, model_key)
        node_count_label = f"{len(node_urls)} expert nodes"

    chat_html = chat_html.replace("{{MODEL_NAME}}", model_label) \
                          .replace("{{NODE_COUNT}}", node_count_label) \
                          .replace("{{VISION_ENABLED}}", "true" if vision else "false") \
                          .replace("{{FALCON_ENABLED}}", "true" if falcon_tools is not None else "false")

    # Lock so only one chat request runs at a time (single MoE engine)
    lock = threading.Lock()

    @app.get("/")
    async def index():
        return HTMLResponse(chat_html)

    @app.get("/api/info")
    async def info():
        return {
            "model": model_key,
            "nodes": node_urls,
            "allow_write": allow_write,
            "vision": vision,
            "falcon": falcon_tools is not None,
            "swarm_leader": swarm_registry is not None,
        }

    # ============================================================
    # Swarm endpoints (only when running as leader)
    # ============================================================

    if swarm_registry is not None:

        @app.post("/swarm/register")
        async def swarm_register(request: Request):
            body = await request.json()
            url = body.get("url")
            mem_gb = body.get("mem_gb", 0)
            meta = body.get("meta", {})
            if not url:
                return JSONResponse({"error": "url required"}, status_code=400)
            peer_id, partition = swarm_registry.register(url, mem_gb, meta)
            print(f"[swarm] +peer {peer_id} at {url} → partition {partition}")
            return {
                "peer_id": peer_id,
                "partition": partition,
                "model": model_key,
                "partition_version": swarm_registry.partition_version,
            }

        @app.post("/swarm/heartbeat")
        async def swarm_heartbeat(request: Request):
            body = await request.json()
            peer_id = body.get("peer_id")
            if not peer_id:
                return JSONResponse({"error": "peer_id required"}, status_code=400)
            ok, version = swarm_registry.heartbeat(peer_id)
            if not ok:
                return JSONResponse({"error": "unknown peer"}, status_code=404)

            # Tell the peer if its partition has been reassigned
            current_partition = None
            with swarm_registry.lock:
                if peer_id in swarm_registry.peers:
                    current_partition = swarm_registry.peers[peer_id]["partition"]

            return {
                "ok": True,
                "partition_version": version,
                "partition": current_partition,
            }

        @app.post("/swarm/leave")
        async def swarm_leave(request: Request):
            body = await request.json()
            peer_id = body.get("peer_id")
            if not peer_id:
                return JSONResponse({"error": "peer_id required"}, status_code=400)
            swarm_registry.leave(peer_id)
            print(f"[swarm] -peer {peer_id} (graceful leave)")
            return {"ok": True}

        @app.get("/swarm/peers")
        async def swarm_peers():
            return swarm_registry.status()

    @app.post("/api/reset")
    async def reset():
        with lock:
            if vision_engine:
                vision_engine.sniper.reset_cache()
            elif backend:
                backend.reset()
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(request: Request):
        body = await request.json()
        message = body.get("message", "").strip()
        if not message:
            return JSONResponse({"error": "empty message"}, status_code=400)

        max_iterations = int(body.get("max_iterations", 5))
        max_tokens = int(body.get("max_tokens", 300))

        def event_stream():
            with lock:
                try:
                    for event in run_agent_turn_stream(
                        backend, message,
                        max_iterations=max_iterations,
                        max_tokens=max_tokens,
                        allow_write=allow_write,
                    ):
                        yield f"data: {json.dumps(event)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )

    # Build a vision agent backend if Falcon is loaded
    vision_agent = None
    if vision_engine is not None and falcon_tools is not None:
        from .agent import VisionAgentBackend
        vision_agent = VisionAgentBackend(
            vision_engine=vision_engine, falcon_tools=falcon_tools
        )
        print("Vision agent ready (Gemma 4 + Falcon Perception chained).")

    @app.post("/api/chat_vision")
    async def chat_vision(
        message: str = Form(...),
        max_tokens: int = Form(300),
        image: UploadFile = File(None),
    ):
        """Vision chat endpoint — accepts an optional image upload.

        If Falcon Perception is loaded, uses the vision agent loop with
        tool calling. Otherwise falls back to plain Gemma 4 vision.
        """
        if vision_engine is None:
            return JSONResponse({"error": "vision mode not enabled"}, status_code=400)

        image_path = None
        if image is not None and image.filename:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix="_" + image.filename, delete=False)
            tmp.write(await image.read())
            tmp.close()
            image_path = tmp.name

        def event_stream():
            with lock:
                try:
                    if vision_agent is not None and image_path:
                        # Chained mode: Gemma 4 + Falcon tool calls
                        from .agent import run_vision_agent_turn_stream
                        for event in run_vision_agent_turn_stream(
                            vision_agent, message, image_path,
                            max_iterations=4,
                            max_tokens=max_tokens,
                        ):
                            yield f"data: {json.dumps(event)}\n\n"
                    else:
                        # Simple mode: just Gemma 4 vision (no tools)
                        yield f"data: {json.dumps({'type': 'step_start', 'step': 1, 'max': 1})}\n\n"
                        chunks = []
                        def on_chunk(text):
                            chunks.append(text)
                        output = vision_engine.generate(
                            message, image_path=image_path,
                            max_tokens=max_tokens, temperature=0.6,
                            on_chunk=on_chunk,
                        )
                        for chunk in chunks:
                            yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
                        yield f"data: {json.dumps({'type': 'final', 'text': output.strip()})}\n\n"
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                finally:
                    if image_path and os.path.exists(image_path):
                        try:
                            os.unlink(image_path)
                        except Exception:
                            pass

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )

    @app.post("/api/falcon")
    async def falcon_ground(
        query: str = Form(...),
        image: UploadFile = File(...),
    ):
        """Run Falcon Perception on uploaded image with text query.

        Returns JSON with detected masks (count, IDs, metadata) plus a
        base64-encoded annotated image showing bounding boxes + labels.
        """
        if falcon_tools is None:
            return JSONResponse({"error": "Falcon Perception not loaded"}, status_code=400)

        # Save uploaded image
        import tempfile, base64, io
        tmp = tempfile.NamedTemporaryFile(suffix="_" + image.filename, delete=False)
        tmp.write(await image.read())
        tmp.close()

        try:
            with lock:
                # Set image in Falcon session
                falcon_tools.set_image(tmp.name)
                # Run grounding
                t0 = time.time()
                result = falcon_tools.ground(query, slot=query.replace(" ", "_")[:32])
                elapsed = time.time() - t0

                if "error" in result:
                    return JSONResponse(result, status_code=500)

                # Annotate the image with bounding boxes
                annotated = falcon_tools.annotate_image(mask_ids=result["mask_ids"])

                # Encode annotated image as base64 PNG
                buf = io.BytesIO()
                annotated.save(buf, format="PNG")
                annotated_b64 = base64.b64encode(buf.getvalue()).decode()

                return JSONResponse({
                    "query": query,
                    "count": result["count"],
                    "mask_ids": result["mask_ids"],
                    "masks": result["masks"],
                    "annotated_image": f"data:image/png;base64,{annotated_b64}",
                    "elapsed_seconds": round(elapsed, 2),
                })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    print()
    print("=" * 60)
    print(f"  mac-tensor UI ready")
    print(f"  Open: http://localhost:{port}")
    print(f"        http://{_local_ip()}:{port}  (LAN access)")
    print("=" * 60)
    print()

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")


def _local_ip():
    """Best-effort detection of the LAN IP."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def main(args):
    vision = getattr(args, "vision", False)

    if vision:
        # Vision mode: single-machine, no distributed nodes needed
        run_server(
            model_key="gemma4",
            node_urls=None,
            host=args.host or "0.0.0.0",
            port=args.port or 8500,
            allow_write=getattr(args, "write", False),
            vision=True,
            stream_dir=getattr(args, "stream_dir", None),
            source_dir=getattr(args, "source_dir", None),
            falcon=getattr(args, "falcon", False),
            falcon_model=getattr(args, "falcon_model", None),
        )
    else:
        if not args.nodes:
            print("Error: --nodes is required (or pass --vision for single-machine mode)")
            sys.exit(1)
        node_urls = [u.strip() for u in args.nodes.split(",")]
        run_server(
            model_key=args.model or "gemma4",
            node_urls=node_urls,
            host=args.host or "0.0.0.0",
            port=args.port or 8500,
            allow_write=args.write,
        )
