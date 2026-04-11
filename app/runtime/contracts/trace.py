from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from typing_extensions import TypedDict

from app.runtime.prompts.context_pruner import CandidatePruningTrace, PromptPruningTrace


class AgentTracePayload(TypedDict, total=False):
    trace_id: str
    self_correction_attempts: int
    candidate_pruning: CandidatePruningTrace
    prompt_pruning: PromptPruningTrace


def build_agent_trace_payload(
    *,
    current: Mapping[str, Any] | None = None,
    trace_id: str | None = None,
    self_correction_attempts: int | None = None,
    candidate_pruning: CandidatePruningTrace | None = None,
    prompt_pruning: PromptPruningTrace | None = None,
) -> AgentTracePayload:
    payload: AgentTracePayload = cast(AgentTracePayload, dict(current or {}))

    existing_trace_id = payload.get("trace_id")
    if isinstance(existing_trace_id, str) and existing_trace_id:
        payload["trace_id"] = existing_trace_id

    existing_attempts = payload.get("self_correction_attempts")
    if isinstance(existing_attempts, int):
        payload["self_correction_attempts"] = existing_attempts

    existing_candidate = payload.get("candidate_pruning")
    if isinstance(existing_candidate, dict):
        payload["candidate_pruning"] = existing_candidate

    existing_prompt = payload.get("prompt_pruning")
    if isinstance(existing_prompt, dict):
        payload["prompt_pruning"] = existing_prompt

    if trace_id:
        payload["trace_id"] = trace_id
    if self_correction_attempts is not None:
        payload["self_correction_attempts"] = self_correction_attempts
    if candidate_pruning is not None:
        payload["candidate_pruning"] = candidate_pruning
    if prompt_pruning is not None:
        payload["prompt_pruning"] = prompt_pruning
    return payload
