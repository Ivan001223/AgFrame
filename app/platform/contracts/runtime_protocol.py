from __future__ import annotations

from pydantic import BaseModel

from app.platform.contracts.versioning import RUNTIME_COMMAND_VERSION, RUNTIME_RESULT_VERSION


class RuntimeCommandV1(BaseModel):
    version: str = RUNTIME_COMMAND_VERSION
    command_id: str
    run_id: str
    command_type: str
    payload: dict[str, object]
    interruption_policy: str | None = None
    timeout_seconds: int | None = None


class RuntimeResultV1(BaseModel):
    version: str = RUNTIME_RESULT_VERSION
    command_id: str
    run_id: str
    result_type: str
    payload: dict[str, object]
    error_type: str | None = None
    resumable: bool | None = None


class RuntimeInterruption(BaseModel):
    interrupt_type: str
    reason: str | None = None
    resumable: bool = True


class RuntimeResumePoint(BaseModel):
    next_step_index: int
    rollback_state: dict[str, object] | None = None
    continuation: dict[str, object] | None = None


def runtime_resume_point_to_payload(point: RuntimeResumePoint | dict[str, object]) -> dict[str, object]:
    resume_point = point if isinstance(point, RuntimeResumePoint) else RuntimeResumePoint.model_validate(point)
    payload: dict[str, object] = {"next_step_index": resume_point.next_step_index}
    if resume_point.rollback_state is not None:
        payload["rollback_state"] = dict(resume_point.rollback_state)
    if resume_point.continuation is not None:
        payload["continuation"] = dict(resume_point.continuation)
    return payload


def runtime_resume_point_from_payload(payload: dict[str, object] | None) -> RuntimeResumePoint:
    data = dict(payload or {})
    return RuntimeResumePoint(
        next_step_index=int(data.get("next_step_index") or 0),
        rollback_state=dict(data.get("rollback_state") or {}) or None,
        continuation=dict(data.get("continuation") or {}) or None,
    )


def normalize_review_rejection_resume_payload(
    payload: dict[str, object] | None,
    *,
    review_stage: str,
    continue_mode: str | None = None,
    review_step_index: int | None = None,
) -> tuple[dict[str, object], str | None]:
    normalized = dict(payload or {})
    rollback_state = dict(normalized.get("rollback_state") or {})
    next_step_index = int(normalized.get("next_step_index") or 0)
    review_stage_value = str(review_stage or "").strip()
    continue_mode_value = str(continue_mode or "").strip()
    recovery_mode: str | None = None

    if rollback_state:
        normalized["state"] = rollback_state

    if review_stage_value == "cluster_research" or continue_mode_value == "discard_research_evidence":
        normalized["next_step_index"] = max(next_step_index, 0)
        recovery_mode = "continue_without_research"
    elif review_stage_value == "agent_output_stream":
        normalized["next_step_index"] = max(int(review_step_index or next_step_index), 0)
        normalized.pop("continuation", None)
        recovery_mode = "continue_from_stream_block"
    else:
        normalized["next_step_index"] = max(next_step_index - 1, 0)

    normalized.pop("review_decision", None)
    normalized.pop("continue_mode", None)
    return normalized, recovery_mode


def normalize_review_approval_resume_payload(
    payload: dict[str, object] | None,
    *,
    review_stage: str,
) -> tuple[dict[str, object], str | None]:
    normalized = dict(payload or {})
    review_stage_value = str(review_stage or "").strip()
    recovery_mode: str | None = None

    if review_stage_value == "cluster_research":
        normalized["continue_mode"] = "accept_research_evidence"
        recovery_mode = "continue_with_research"
    elif review_stage_value == "agent_output_stream":
        normalized["continue_mode"] = "accept_partial_stream_output"
        recovery_mode = "continue_with_partial_stream_output"

    normalized["review_decision"] = "approved"
    return normalized, recovery_mode
