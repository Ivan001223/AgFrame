from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import BaseMessage

from app.infrastructure.utils.logging import bind_logger, get_logger
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import AgentState
from app.skills.research.enhanced_search import enhanced_web_search

_log = get_logger("workflow.web_search")


def _get_last_user_query(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role in ("human", "user") and content:
            return str(content)
    return ""


@register_node("web_search")
async def web_search_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctx = dict(state.get("context") or {})
    trace = dict(state.get("trace") or {})
    trace_id = trace.get("trace_id") or ctx.get("trace_id")
    user_id = state.get("user_id") or ctx.get("user_id") or "-"
    session_id = ctx.get("session_id") or "-"

    messages = list(state.get("messages") or [])
    query = str(ctx.get("search_query") or _get_last_user_query(messages) or "").strip()
    if not query:
        return {"context": ctx}

    try:
        result = await enhanced_web_search(
            query=query,
            provider=None,
            use_cache=True,
            max_results=5,
        )
        ctx["web_search"] = {"query": query, "result": str(result)}
        bind_logger(
            _log,
            trace_id=str(trace_id or "-"),
            user_id=str(user_id),
            session_id=str(session_id),
            node="web_search",
        ).info("searched query_len=%d cost_ms=%d", len(query), int((time.perf_counter() - t0) * 1000))
        return {"context": ctx}
    except Exception as e:
        errors = list(state.get("errors") or [])
        errors.append(f"web_search_error: {e}")
        ctx["web_search"] = {"query": query, "result": f"Search failed: {e}"}
        bind_logger(
            _log,
            trace_id=str(trace_id or "-"),
            user_id=str(user_id),
            session_id=str(session_id),
            node="web_search",
        ).info("search failed cost_ms=%d", int((time.perf_counter() - t0) * 1000))
        return {"context": ctx, "errors": errors}

