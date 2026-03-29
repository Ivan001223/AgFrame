from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class HarnessTaskType(str, Enum):
    DOCUMENT_INGEST = "document_ingest"
    SESSION_RESUME_APPROVAL = "session_resume_approval"
    AGENT_ORCHESTRATION = "agent_orchestration"


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


class HarnessReviewState(BaseModel):
    stage: str | None = None
    status: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    review_output: str | None = None
    check_count: int | None = None
    segment_index: int | None = None
    segment_count: int | None = None
    segment_start_char: int | None = None
    segment_end_char: int | None = None
    last_reviewed_char: int | None = None


class HarnessContinuationState(BaseModel):
    enabled: bool = False
    mode: str | None = None
    status: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    step_index: int | None = None
    prefix_length: int = 0
    resumed_at: int | None = None
    completed_at: int | None = None


class HarnessResearchState(BaseModel):
    enabled: bool = False
    mode: str | None = None
    paper_count: int = 0
    browser_preview_count: int = 0
    source_count: int = 0
    cluster_ids: list[str] = Field(default_factory=list)


class HarnessRuntimeState(BaseModel):
    review: HarnessReviewState = Field(default_factory=HarnessReviewState)
    continuation: HarnessContinuationState = Field(default_factory=HarnessContinuationState)
    research: HarnessResearchState = Field(default_factory=HarnessResearchState)
