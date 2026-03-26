from __future__ import annotations

from app.harness.contracts.policy import HarnessPolicy


_POLICIES: dict[str, HarnessPolicy] = {
    "document_ingest": HarnessPolicy(
        policy_id="document_ingest:v1",
        task_type="document_ingest",
        approval_required=False,
        allowed_tools=["document_ingest"],
        verification_profile="document_ingest_basic",
        retry_budget=1,
    ),
    "session_resume_approval": HarnessPolicy(
        policy_id="session_resume_approval:v1",
        task_type="session_resume_approval",
        approval_required=True,
        allowed_tools=["checkpoint_resume"],
        verification_profile="approval_checkpoint_basic",
        retry_budget=0,
    ),
}


def get_policy(task_type: str) -> HarnessPolicy:
    return _POLICIES[task_type]
