from __future__ import annotations

from pydantic import BaseModel

from app.platform.contracts.translators import legacy_verification_to_record
from app.platform.contracts.verification import VerificationRecordV1


class HarnessVerificationResult(BaseModel):
    status: str
    checks_run: list[str]
    artifacts: dict[str, object] | None = None
    summary: str | None = None


def to_platform_verification_record(payload: dict[str, object]) -> VerificationRecordV1:
    return legacy_verification_to_record(payload)
