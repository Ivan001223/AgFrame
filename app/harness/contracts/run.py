from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class HarnessRunStatus(str, Enum):
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


class HarnessRunCreate(BaseModel):
    task_type: str
    input: dict[str, object]
    session_id: str | None = None
    metadata: dict[str, object] | None = None
