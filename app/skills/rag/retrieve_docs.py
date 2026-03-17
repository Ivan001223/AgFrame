from __future__ import annotations

import time
from typing import Any

import anyio
from langchain_core.messages import BaseMessage

from app.infrastructure.config.settings import settings
from app.infrastructure.utils.logging import bind_logger, get_logger
from app.runtime.contracts.pruning import (
    build_chat_context_pruning_payload,
    build_retrieval_debug_payload,
)
from app.runtime.contracts.trace import build_agent_trace_payload
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import AgentState
from app.runtime.prompts.context_pruner import (
    build_candidate_pruning_trace,
    build_pruning_config,
    prune_documents,
)
from app.skills.rag.rag_engine import get_rag_engine

_log = get_logger("workflow.retrieve_docs")


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


def _get_candidate_k() -> int:
    return settings.rag.retrieval.candidate_k


def _build_focus_hint(state: AgentState, query: str) -> str:
    ctx = dict(state.get("context") or {})
    explicit_hint = str(ctx.get("context_focus_hint") or state.get("context_focus_hint") or "").strip()
    if explicit_hint:
        return explicit_hint
    route = state.get("route") or ctx.get("route") or {}
    reasoning = str(route.get("reasoning") or state.get("reasoning") or "").strip()
    if reasoning:
        return f"{query}\nFocus: {reasoning}"
    return query


@register_node("retrieve_docs")
async def retrieve_docs_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    messages = list(state.get("messages") or [])
    query = _get_last_user_query(messages)
    fetch_k = _get_candidate_k()
    focus_hint = _build_focus_hint(state, query)

    # 从 context 中获取 user_id (通常由 server 在 invoke 时传入 state)
    ctx = build_chat_context_pruning_payload(current=state.get("context"))
    # 优先从 state 顶层取，其次 context
    user_id = state.get("user_id") or ctx.get("user_id")

    docs = await anyio.to_thread.run_sync(
        lambda: get_rag_engine().retrieve_candidates(
            query, fetch_k=fetch_k, user_id=user_id
        )
    )
    pruning_cfg = settings.prompt.context_pruning
    pruned_docs, pruning_summary = prune_documents(
        docs,
        query=query,
        focus_hint=focus_hint,
        config=build_pruning_config(pruning_cfg),
    )

    retrieval_debug = build_retrieval_debug_payload(
        current=ctx.get("retrieval_debug"),
        candidate_pruning=pruning_summary,
    )
    trace = build_agent_trace_payload(
        current=state.get("trace"),
        candidate_pruning=build_candidate_pruning_trace(pruning_summary),
    )
    ctx["retrieved_docs_candidates_raw"] = docs
    ctx["retrieved_docs_candidates"] = pruned_docs
    ctx = build_chat_context_pruning_payload(
        current=ctx,
        focus_hint=focus_hint,
        retrieval_debug=retrieval_debug,
    )
    trace_id = trace.get("trace_id") or ctx.get("trace_id")
    session_id = ctx.get("session_id") or "-"
    bind_logger(
        _log,
        trace_id=str(trace_id or "-"),
        user_id=str(user_id),
        session_id=str(session_id),
        node="retrieve_docs",
    ).info(
        "retrieved doc_candidates=%d pruned=%d prune_ratio=%.4f cost_ms=%d",
        len(docs),
        int(pruning_summary.get("items_pruned") or 0),
        float(pruning_summary.get("ratio") or 1.0),
        int((time.perf_counter() - t0) * 1000),
    )
    return {
        "retrieved_docs_candidates_raw": docs,
        "retrieved_docs_candidates": pruned_docs,
        "context_focus_hint": focus_hint,
        "context": ctx,
        "retrieval_debug": retrieval_debug,
        "trace": trace,
    }
