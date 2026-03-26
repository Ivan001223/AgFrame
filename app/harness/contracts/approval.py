from __future__ import annotations

from pydantic import BaseModel


class HarnessApprovalDecision(BaseModel):
    approved: bool = True
    comment: str | None = None
