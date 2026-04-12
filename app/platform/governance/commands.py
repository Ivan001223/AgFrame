from __future__ import annotations

from pydantic import BaseModel


class ApprovalResolutionCommand(BaseModel):
    version: str = "governance_command.v1"
    run_id: str
    approval_id: str
    approved: bool
    resolved_by: str
    comment: str | None = None


class VerificationRecordCommand(BaseModel):
    version: str = "governance_command.v1"
    run_id: str
    verification_profile: str
    result_status: str
    checks_run: list[str] = []
    artifacts: dict[str, object] | None = None
    summary: str | None = None
