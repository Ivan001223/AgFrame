from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from typing_extensions import TypedDict

from app.runtime.prompts.context_pruner import AggregatePruningSummary, PromptPruningSummary


class RetrievalDebugPayload(TypedDict, total=False):
    candidate_pruning: AggregatePruningSummary


class ChatContextPruningPayload(TypedDict, total=False):
    session_id: str
    context_focus_hint: str
    context_pruning: PromptPruningSummary
    retrieval_debug: RetrievalDebugPayload
    retrieved_docs_candidates_raw: list[Any]
    retrieved_docs_candidates: list[Any]
    retrieved_docs: list[Any]
    retrieved_memories: list[Any]
    retrieved_profile_items: list[Any]
    system_prompt: str
    citations: list[Any]


def build_retrieval_debug_payload(
    *,
    current: Mapping[str, Any] | None = None,
    candidate_pruning: AggregatePruningSummary | None = None,
) -> RetrievalDebugPayload:
    payload: RetrievalDebugPayload = {}
    existing = current or {}
    existing_candidate = existing.get("candidate_pruning")
    if isinstance(existing_candidate, dict):
        payload["candidate_pruning"] = cast(AggregatePruningSummary, existing_candidate)
    if candidate_pruning is not None:
        payload["candidate_pruning"] = candidate_pruning
    return payload


def build_chat_context_pruning_payload(
    *,
    current: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    focus_hint: str | None = None,
    prompt_pruning: PromptPruningSummary | None = None,
    retrieval_debug: RetrievalDebugPayload | None = None,
) -> ChatContextPruningPayload:
    payload: ChatContextPruningPayload = cast(ChatContextPruningPayload, dict(current or {}))
    existing_session_id = payload.get("session_id")
    if isinstance(existing_session_id, str) and existing_session_id:
        payload["session_id"] = existing_session_id
    existing_focus_hint = payload.get("context_focus_hint")
    if isinstance(existing_focus_hint, str) and existing_focus_hint:
        payload["context_focus_hint"] = existing_focus_hint
    existing_prompt_pruning = payload.get("context_pruning")
    if isinstance(existing_prompt_pruning, dict):
        payload["context_pruning"] = existing_prompt_pruning
    existing_retrieval_debug = payload.get("retrieval_debug")
    if isinstance(existing_retrieval_debug, dict):
        payload["retrieval_debug"] = build_retrieval_debug_payload(current=existing_retrieval_debug)

    if session_id:
        payload["session_id"] = session_id
    if focus_hint:
        payload["context_focus_hint"] = focus_hint
    if prompt_pruning is not None:
        payload["context_pruning"] = prompt_pruning
    if retrieval_debug is not None:
        payload["retrieval_debug"] = retrieval_debug
    return payload
