from __future__ import annotations

from app.harness.contracts.policy import HarnessPolicy
from app.harness.contracts.run import HarnessTaskType


class UnknownHarnessTaskTypeError(ValueError):
    pass


_POLICIES: dict[str, HarnessPolicy] = {
    HarnessTaskType.DOCUMENT_INGEST.value: HarnessPolicy(
        policy_id="document_ingest:v1",
        task_type=HarnessTaskType.DOCUMENT_INGEST.value,
        approval_required=False,
        allowed_tools=["document_ingest"],
        verification_profile="document_ingest_basic",
        retry_budget=1,
    ),
    HarnessTaskType.SESSION_RESUME_APPROVAL.value: HarnessPolicy(
        policy_id="session_resume_approval:v1",
        task_type=HarnessTaskType.SESSION_RESUME_APPROVAL.value,
        approval_required=True,
        allowed_tools=["checkpoint_resume"],
        verification_profile="approval_checkpoint_basic",
        retry_budget=0,
    ),
}


def list_policies() -> list[HarnessPolicy]:
    return [policy.model_copy(deep=True) for policy in _POLICIES.values()]


def normalize_task_type(task_type: str | HarnessTaskType) -> HarnessTaskType:
    if isinstance(task_type, HarnessTaskType):
        return task_type
    try:
        return HarnessTaskType(str(task_type))
    except ValueError as exc:
        supported = ", ".join(sorted(_POLICIES))
        raise UnknownHarnessTaskTypeError(
            f"Unknown harness task_type '{task_type}'. Supported task types: {supported}."
        ) from exc


def get_policy(task_type: str | HarnessTaskType) -> HarnessPolicy:
    normalized_task_type = normalize_task_type(task_type)
    return _POLICIES[normalized_task_type.value]
