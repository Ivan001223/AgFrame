from __future__ import annotations

from pydantic import BaseModel

from app.platform.contracts.versioning import VERIFICATION_CONTRACT_VERSION


class VerificationRecordV1(BaseModel):
    version: str = VERIFICATION_CONTRACT_VERSION
    verification_id: str
    profile: str
    subject_run_id: str
    result_status: str
    evidence: dict[str, object] | None = None
    summary: str | None = None
    replay_id: str | None = None
    audit_metadata: dict[str, object] | None = None
