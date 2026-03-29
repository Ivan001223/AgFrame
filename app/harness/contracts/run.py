from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class HarnessTaskType(str, Enum):
    DOCUMENT_INGEST = "document_ingest"
    SESSION_RESUME_APPROVAL = "session_resume_approval"


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
    task_type: HarnessTaskType
    input: dict[str, object]
    session_id: str | None = None
    metadata: dict[str, object] | None = None
