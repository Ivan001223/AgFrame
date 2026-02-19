from __future__ import annotations

import time
from typing import Any

import anyio
from langchain_core.messages import BaseMessage

from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.utils.logging import bind_logger, get_logger
from app.memory.long_term.user_memory_engine import UserMemoryEngine
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import AgentState

_log = get_logger("workflow.retrieve_memories")
_memory_engine = UserMemoryEngine()


def _get_last_user_query(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role in ("human", "user") and content:
            return str(content)
    if not messages:
        return ""
    last = messages[-1]
    return str(getattr(last, "content", "") or "")


@register_node("retrieve_memories")
async def retrieve_memories_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    messages = list(state.get("messages") or [])
    query = _get_last_user_query(messages)
    user_id = state.get("user_id") or (state.get("context") or {}).get("user_id") or "default"
    memories = []
    if ensure_schema_if_possible():
        try:
            memories = await anyio.to_thread.run_sync(
                lambda: _memory_engine.retrieve_chat_summaries(user_id=user_id, query=query, k=3, fetch_k=20)
            )
        except Exception:
            memories = []
    ctx = dict(state.get("context") or {})
    ctx["retrieved_memories"] = memories
    trace_id = (state.get("trace") or {}).get("trace_id") or ctx.get("trace_id")
    session_id = ctx.get("session_id") or "-"
    bind_logger(_log, trace_id=str(trace_id or "-"), user_id=str(user_id), session_id=str(session_id), node="retrieve_memories").info(
        "retrieved memories=%d cost_ms=%d", len(memories), int((time.perf_counter() - t0) * 1000)
    )
    return {"retrieved_memories": memories, "context": ctx}
