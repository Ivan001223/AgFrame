from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from app.platform.contracts.versioning import APPROVAL_CONTRACT_VERSION


class ApprovalDecisionState(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalRecordV1(BaseModel):
    version: str = APPROVAL_CONTRACT_VERSION
    approval_id: str
    target_run_id: str
    decision_state: ApprovalDecisionState
    requested_decision: str
    approver_identity: str | None = None
    expires_at: int | None = None
    escalation_policy: str | None = None
