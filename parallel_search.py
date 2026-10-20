"""Optional anonymous Parallel Search MCP context for the local agent."""

import asyncio
import json
import uuid
from urllib.parse import urlsplit

ENDPOINT = "https://search.parallel.ai/mcp"
USER_AGENT = "mac-code/parallel-search (+https://github.com/walter-grace/mac-code)"
REQUEST_TIMEOUT = 30


class SearchError(Exception):
    """A selected search provider failed; do not silently switch providers."""


class SearchConversation:
    """Owned by one chat, independent of MCP connection lifetimes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.session_id = str(uuid.uuid4())


def _payload(result):
    if result.isError:
        raise SearchError("Parallel returned a tool error. Please try again later.")
    data = result.structuredContent
    if data is None:
        try:
            data = json.loads("\n".join(
                block.text for block in result.content if block.type == "text"
            ))
        except (ValueError, TypeError):
            raise SearchError("Parallel returned an invalid response.") from None
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise SearchError("Parallel returned an invalid response.")
    if data.get("errors") and not data["results"]:
        raise SearchError("Parallel could not retrieve results. Please try again later.")
    return data["results"]


def _context(rows):
    parts = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        url = row.get("url", "")
        if not isinstance(url, str) or urlsplit(url).scheme not in ("http", "https"):
            continue
        excerpts = row.get("excerpts") or []
        if not isinstance(excerpts, list):
            continue
        text = "\n".join(x for x in excerpts if isinstance(x, str))
        parts.append(f"{row.get('title', '')}\nSource: {url}\n{text[:1500]}")
    return "\n\n".join(parts)


async def _retrieve(query, session_id):
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        import httpx
    except ImportError:
        raise SearchError("Install Parallel support: pip3 install -r requirements-search.txt") from None

    # No ambient API keys or saved account configuration are read. Each HTTP
    # client is local to this call; conversation identity is supplied separately.
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT,
    ) as http:
        async with streamable_http_client(ENDPOINT, http_client=http) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                if not {"web_search", "web_fetch"}.issubset(tools):
                    raise SearchError("Parallel Search MCP is missing search or fetch tools.")
                results = _payload(await session.call_tool("web_search", {
                    "objective": query, "search_queries": [query], "session_id": session_id,
                }))
                context = _context(results)
                if not context:
                    raise SearchError("Parallel returned no usable search results.")
                # Fetch one relevant page via the same MCP service, retaining the
                # source URLs in the context consumed by the local model.
                first = next((row["url"] for row in results if isinstance(row, dict)
                              and isinstance(row.get("url"), str)
                              and urlsplit(row["url"]).scheme in ("http", "https")), None)
                if first:
                    pages = _payload(await session.call_tool("web_fetch", {
                        "urls": [first], "objective": query[:200],
                        "search_queries": [query], "session_id": session_id,
                    }))
                    context += "\n\nPage excerpts:\n" + _context(pages)
                return context


async def _bounded_context(query, session_id, cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise SearchError("Search cancelled.")

    async def cancelled():
        while cancel_event is None or not cancel_event.is_set():
            await asyncio.sleep(0.05)

    work = asyncio.create_task(_retrieve(query, session_id))
    stop = asyncio.create_task(cancelled())
    try:
        done, _ = await asyncio.wait({work, stop}, timeout=REQUEST_TIMEOUT,
                                     return_when=asyncio.FIRST_COMPLETED)
        if stop in done:
            raise SearchError("Search cancelled.")
        if work not in done:
            raise SearchError("Parallel search timed out. Please try again later.")
        return await work
    finally:
        work.cancel()
        stop.cancel()
        await asyncio.gather(work, stop, return_exceptions=True)


def search_context(query, conversation, cancel_event=None):
    """Run Streamable HTTP MCP synchronously from the agent's search worker."""
    try:
        return asyncio.run(_bounded_context(query, conversation.session_id, cancel_event))
    except SearchError:
        raise
    except Exception:
        # SDK transport errors may contain nested task groups and HTTP details.
        raise SearchError("Parallel search failed. Check your connection and try again.") from None
