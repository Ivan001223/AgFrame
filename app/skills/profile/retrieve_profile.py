from __future__ import annotations

from typing import Any, Dict, List

import anyio
import time
from langchain_core.messages import BaseMessage

from app.core.database.schema import ensure_schema_if_possible
from app.core.services.user_memory_engine import UserMemoryEngine
from app.core.utils.logging import bind_logger, get_logger
from app.core.workflow.registry import register_node
from app.core.workflow.state import AgentState

_log = get_logger("workflow.retrieve_profile")
_memory_engine = UserMemoryEngine()


def _get_last_user_query(messages: List[BaseMessage]) -> str:
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role in ("human", "user") and content:
            return str(content)
    if not messages:
        return ""
    last = messages[-1]
    return str(getattr(last, "content", "") or "")


@register_node("retrieve_profile")
async def retrieve_profile_node(state: AgentState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    messages = list(state.get("messages") or [])
    query = _get_last_user_query(messages)
    ctx = dict(state.get("context") or {})
    user_id = state.get("user_id") or ctx.get("user_id") or "default"
    session_id = ctx.get("session_id") or "-"

    items = []
    if ensure_schema_if_possible():
        try:
            items = await anyio.to_thread.run_sync(
                lambda: _memory_engine.retrieve_profile_items(user_id=str(user_id), query=query, k=6, fetch_k=30)
            )
        except Exception:
            items = []

    ctx["retrieved_profile_items"] = items
    trace_id = (state.get("trace") or {}).get("trace_id") or ctx.get("trace_id")
    bind_logger(
        _log,
        trace_id=str(trace_id or "-"),
        user_id=str(user_id),
        session_id=str(session_id),
        node="retrieve_profile",
    ).info(
        "retrieved profile_items=%d cost_ms=%d", len(items), int((time.perf_counter() - t0) * 1000)
    )
    return {"context": ctx, "retrieved_profile_items": items}

