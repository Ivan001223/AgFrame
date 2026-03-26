from __future__ import annotations

from pydantic import BaseModel


class HarnessPolicy(BaseModel):
    policy_id: str
    task_type: str
    approval_required: bool
    allowed_tools: list[str]
    verification_profile: str
    retry_budget: int
