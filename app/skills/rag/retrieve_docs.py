from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import BaseMessage
import anyio
import time

from app.infrastructure.config.config_manager import config_manager
from app.skills.rag.rag_engine import get_rag_engine
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import AgentState
from app.infrastructure.utils.logging import bind_logger, get_logger

_log = get_logger("workflow.retrieve_docs")


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


def _get_candidate_k() -> int:
    cfg = config_manager.get_config() or {}
    rag_cfg = (cfg.get("rag") or {}).get("retrieval") or {}
    val = rag_cfg.get("candidate_k")
    if val is None:
        return 20
    try:
        return max(1, int(val))
    except Exception:
        return 20


@register_node("retrieve_docs")
async def retrieve_docs_node(state: AgentState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    messages = list(state.get("messages") or [])
    query = _get_last_user_query(messages)
    fetch_k = _get_candidate_k()

    # 从 context 中获取 user_id (通常由 server 在 invoke 时传入 state)
    ctx = dict(state.get("context") or {})
    # 优先从 state 顶层取，其次 context
    user_id = state.get("user_id") or ctx.get("user_id")

    docs = await anyio.to_thread.run_sync(
        lambda: get_rag_engine().retrieve_candidates(
            query, fetch_k=fetch_k, user_id=user_id
        )
    )

    ctx["retrieved_docs_candidates"] = docs
    trace_id = (state.get("trace") or {}).get("trace_id") or ctx.get("trace_id")
    session_id = ctx.get("session_id") or "-"
    bind_logger(
        _log,
        trace_id=str(trace_id or "-"),
        user_id=str(user_id),
        session_id=str(session_id),
        node="retrieve_docs",
    ).info(
        "retrieved doc_candidates=%d cost_ms=%d",
        len(docs),
        int((time.perf_counter() - t0) * 1000),
    )
    return {"retrieved_docs_candidates": docs, "context": ctx}
