from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage
import anyio
import time

from app.core.services.chat_memory_engine import ChatSummaryIndex
from app.core.workflow.registry import register_node
from app.core.workflow.state import AgentState
from app.core.utils.logging import bind_logger, get_logger

_chat_summary_index = ChatSummaryIndex()
_log = get_logger("workflow.retrieve_memories")


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


@register_node("retrieve_memories")
async def retrieve_memories_node(state: AgentState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    messages = list(state.get("messages") or [])
    query = _get_last_user_query(messages)
    user_id = state.get("user_id") or (state.get("context") or {}).get("user_id") or "default"
    memories = await anyio.to_thread.run_sync(
        lambda: _chat_summary_index.retrieve(user_id=user_id, query=query, k=3, fetch_k=20)
    )
    ctx = dict(state.get("context") or {})
    ctx["retrieved_memories"] = memories
    trace_id = (state.get("trace") or {}).get("trace_id") or ctx.get("trace_id")
    session_id = ctx.get("session_id") or "-"
    bind_logger(_log, trace_id=str(trace_id or "-"), user_id=str(user_id), session_id=str(session_id), node="retrieve_memories").info(
        "retrieved memories=%d cost_ms=%d", len(memories), int((time.perf_counter() - t0) * 1000)
    )
    return {"retrieved_memories": memories, "context": ctx}
