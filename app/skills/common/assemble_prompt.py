from __future__ import annotations

import time
from typing import Any

import anyio

from app.infrastructure.config.settings import settings
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.stores import MySQLConversationStore
from app.infrastructure.utils.logging import bind_logger, get_logger
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import AgentState
from app.runtime.prompts.prompt_builder import PromptBudget, build_system_prompt
from app.skills.profile.profile_engine import UserProfileEngine

_profile_engine = UserProfileEngine()
_log = get_logger("workflow.assemble_prompt")


@register_node("assemble_prompt")
async def assemble_prompt_node(state: AgentState) -> dict[str, Any]:
    t0 = time.perf_counter()
    ctx = dict(state.get("context") or {})
    user_id = state.get("user_id") or ctx.get("user_id") or "default"
    session_id = ctx.get("session_id")

    recent_history_lines: list[str] = []
    if session_id and ensure_schema_if_possible():
        try:
            store = MySQLConversationStore()
            recent_msgs = await anyio.to_thread.run_sync(
                lambda: store.get_recent_messages(
                    user_id=user_id, session_id=str(session_id), limit_messages=10
                )
            )
            for m in recent_msgs:
                recent_history_lines.append(f"{m.get('role')}: {m.get('content')}")
        except Exception as e:
            _log.warning(f"Failed to get recent history: {e}")
            recent_history_lines = []

    try:
        profile = await anyio.to_thread.run_sync(lambda: _profile_engine.get_profile(user_id))
    except Exception as e:
        _log.warning(f"Failed to get profile: {e}")
        profile = {"basic_info": {}, "tech_profile": {}, "preferences": {}, "facts": []}

    retrieved_profile_items = (
        state.get("retrieved_profile_items")
        or ctx.get("retrieved_profile_items")
        or ctx.get("retrieved_profile")
        or []
    )
    prefs = profile.get("preferences") if isinstance(profile, dict) else {}
    if not isinstance(prefs, dict):
        prefs = {}
    pinned_prefs: dict[str, Any] = {}
    for key in ["language", "communication_style", "interaction_protocol", "tone_instruction"]:
        val = prefs.get(key)
        if val is not None:
            pinned_prefs[key] = val

    profile_view = {
        "basic_info": (profile.get("basic_info") if isinstance(profile, dict) else {}) or {},
        "tech_profile": (profile.get("tech_profile") if isinstance(profile, dict) else {}) or {},
        "preferences": pinned_prefs,
        "retrieved_profile_items": retrieved_profile_items,
    }

    docs = state.get("retrieved_docs") or ctx.get("retrieved_docs") or []
    memories = state.get("retrieved_memories") or ctx.get("retrieved_memories") or []

    prompt_cfg = settings.prompt
    budget_cfg = prompt_cfg.budget
    budget = PromptBudget(
        max_recent_history_lines=budget_cfg.max_recent_history_lines,
        max_docs=budget_cfg.max_docs,
        max_memories=budget_cfg.max_memories,
        max_doc_chars_total=budget_cfg.max_doc_chars_total,
        max_memory_chars_total=budget_cfg.max_memory_chars_total,
        max_profile_chars_total=budget_cfg.max_doc_chars_total,
        max_item_chars=budget_cfg.max_item_chars,
    )

    system_prompt, citations = build_system_prompt(
        profile=profile_view,
        recent_history_lines=recent_history_lines,
        docs=docs,
        memories=memories,
        web_search=ctx.get("web_search"),
        self_correction=ctx.get("self_correction"),
        budget=budget,
    )

    ctx["system_prompt"] = system_prompt
    ctx["citations"] = citations
    trace_id = (state.get("trace") or {}).get("trace_id") or ctx.get("trace_id")
    bind_logger(
        _log,
        trace_id=str(trace_id or "-"),
        user_id=str(user_id),
        session_id=str(session_id or "-"),
        node="assemble",
    ).info(
        "assembled docs=%d memories=%d cost_ms=%d",
        len(docs),
        len(memories),
        int((time.perf_counter() - t0) * 1000),
    )

    return {"context": ctx, "citations": citations}

