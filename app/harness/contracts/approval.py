from __future__ import annotations

from pydantic import BaseModel

from app.platform.contracts.approval import ApprovalRecordV1
from app.platform.contracts.translators import approval_record_to_legacy_payload, legacy_approval_to_record


class HarnessApprovalDecision(BaseModel):
    approved: bool = True
    comment: str | None = None


def to_platform_approval_record(payload: dict[str, object]) -> ApprovalRecordV1:
    return legacy_approval_to_record(payload)


def from_platform_approval_record(payload: ApprovalRecordV1 | dict[str, object]) -> dict[str, object]:
    return approval_record_to_legacy_payload(payload)
