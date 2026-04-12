from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from app.platform.contracts.versioning import RUN_CONTRACT_VERSION


class RunLifecycleStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    RESUMED = "resumed"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"


class RunEnvelopeV1(BaseModel):
    version: str = RUN_CONTRACT_VERSION
    run_id: str
    task_type: str
    lifecycle_status: RunLifecycleStatus
    input: dict[str, object]
    metadata: dict[str, object] | None = None
    correlation_id: str | None = None
    causation_id: str | None = None
    trace_id: str | None = None
    runtime_state_handle: str | None = None
