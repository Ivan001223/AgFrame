from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from app.platform.contracts.run import RunEnvelopeV1
from app.platform.contracts.translators import legacy_run_to_run_envelope, run_envelope_to_legacy_payload


class HarnessTaskType(StrEnum):
    DOCUMENT_INGEST = "document_ingest"
    SESSION_RESUME_APPROVAL = "session_resume_approval"
    AGENT_ORCHESTRATION = "agent_orchestration"


class HarnessRunStatus(StrEnum):
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


class HarnessWorkflowStep(BaseModel):
    step_id: str
    step_index: int = Field(default=0, ge=0)
    loop_number: int = Field(default=1, ge=1)
    label: str
    execution_id: str | None = None
    node_id: str | None = None
    status: Literal["pending", "in_progress", "completed", "blocked"] = "pending"
    kind: Literal["agent", "cluster_member", "cluster_summary"] = "agent"


class HarnessWorkflowProgress(BaseModel):
    enabled: bool = False
    status: Literal["idle", "pending", "running", "blocked", "completed", "failed"] = "idle"
    total_steps: int = 0
    completed_steps: int = 0
    blocked_steps: int = 0
    review_enabled: bool = False
    current_step_index: int | None = None
    current_step_label: str | None = None
    blocking_step_index: int | None = None
    blocking_step_label: str | None = None
    blocking_stage: str | None = None
    blocking_reason: str | None = None
    steps: list[HarnessWorkflowStep] = Field(default_factory=list)


class HarnessRunChecklistItem(BaseModel):
    item_id: str
    content: str = Field(min_length=1, max_length=240)
    status: Literal["pending", "in_progress", "completed"] = "pending"
    active_form: str | None = Field(default=None, max_length=240)


class HarnessRunChecklistSnapshot(BaseModel):
    enabled: bool = False
    total_items: int = 0
    open_items: int = 0
    completed_items: int = 0
    items: list[HarnessRunChecklistItem] = Field(default_factory=list)


class HarnessRuntimeState(BaseModel):
    review: HarnessReviewState = Field(default_factory=HarnessReviewState)
    continuation: HarnessContinuationState = Field(default_factory=HarnessContinuationState)
    research: HarnessResearchState = Field(default_factory=HarnessResearchState)


def to_platform_run_envelope(payload: dict[str, object]) -> RunEnvelopeV1:
    return legacy_run_to_run_envelope(payload)


def from_platform_run_envelope(payload: RunEnvelopeV1 | dict[str, object]) -> dict[str, object]:
    return run_envelope_to_legacy_payload(payload)
