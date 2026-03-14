from __future__ import annotations

import time
from typing import Any

import anyio
from langchain_core.messages import BaseMessage

from app.infrastructure.config.settings import settings
from app.infrastructure.utils.logging import bind_logger, get_logger
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import AgentState
from app.skills.rag.rag_engine import get_rag_engine

_log = get_logger("workflow.rerank_docs")


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


def _get_final_k() -> int:
    retrieval_k = settings.rag.retrieval.final_k
    if retrieval_k:
        return retrieval_k
    return settings.prompt.budget.max_docs


@register_node("rerank_docs")
async def rerank_docs_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctx = dict(state.get("context") or {})
    candidates = state.get("retrieved_docs_candidates") or ctx.get("retrieved_docs_candidates") or []
    messages = list(state.get("messages") or [])
    query = _get_last_user_query(messages)
    final_k = _get_final_k()

    docs = await anyio.to_thread.run_sync(
        lambda: get_rag_engine().restore_parents(
            get_rag_engine().rerank_candidates(query, list(candidates), k=final_k),
            k=final_k,
        )
    )

    ctx["retrieved_docs"] = docs
    trace_id = (state.get("trace") or {}).get("trace_id") or ctx.get("trace_id")
    user_id = state.get("user_id") or ctx.get("user_id") or "-"
    session_id = ctx.get("session_id") or "-"
    bind_logger(
        _log,
        trace_id=str(trace_id or "-"),
        user_id=str(user_id),
        session_id=str(session_id),
        node="rerank_docs",
    ).info(
        "reranked docs=%d candidates=%d cost_ms=%d",
        len(docs),
        len(candidates),
        int((time.perf_counter() - t0) * 1000),
    )
    return {"retrieved_docs": docs, "context": ctx}

