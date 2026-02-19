from __future__ import annotations

import time
import uuid
from typing import Any

import anyio

from app.infrastructure.utils.logging import bind_logger, get_logger
from app.runtime.graph.memory_router import route_memory
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import AgentState

_log = get_logger("workflow.router")


@register_node("router")
async def router_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctx = dict(state.get("context") or {})
    trace = dict(state.get("trace") or {})
    trace_id = trace.get("trace_id") or ctx.get("trace_id") or str(uuid.uuid4())
    trace["trace_id"] = trace_id
    ctx["trace_id"] = trace_id

    user_id = state.get("user_id") or ctx.get("user_id") or "-"
    session_id = ctx.get("session_id") or "-"
    log = bind_logger(_log, trace_id=trace_id, user_id=str(user_id), session_id=str(session_id), node="router")

    existing_route = state.get("route") or {}
    if "needs_docs" in existing_route or "needs_history" in existing_route:
        route = {
            "needs_docs": bool(existing_route.get("needs_docs")),
            "needs_history": bool(existing_route.get("needs_history")),
            "reasoning": str(existing_route.get("reasoning") or "Provided by state"),
        }
    else:
        decision = await anyio.to_thread.run_sync(lambda: route_memory(state))
        route = {
            "needs_docs": bool(decision.needs_docs),
            "needs_history": bool(decision.needs_history),
            "reasoning": str(decision.reasoning),
        }
    ctx["route"] = route
    log.info("routed needs_docs=%s needs_history=%s cost_ms=%d", route["needs_docs"], route["needs_history"], int((time.perf_counter() - t0) * 1000))
    return {"route": route, "context": ctx, "trace": trace}
