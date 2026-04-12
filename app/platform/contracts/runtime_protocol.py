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
