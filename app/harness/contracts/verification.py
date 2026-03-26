from __future__ import annotations

from pydantic import BaseModel


class HarnessVerificationResult(BaseModel):
    status: str
    checks_run: list[str]
    artifacts: dict[str, object] | None = None
    summary: str | None = None
