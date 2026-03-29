from app.harness.contracts.run import HarnessRunStatus
from app.harness.runtime.policy_registry import get_policy


def test_document_ingest_policy_exists():
    policy = get_policy("document_ingest")

    assert policy.policy_id == "document_ingest:v1"
    assert policy.task_type == "document_ingest"
    assert policy.approval_required is False
    assert policy.verification_profile == "document_ingest_basic"
    assert policy.retry_budget == 1
    assert HarnessRunStatus.QUEUED.value == "queued"


def test_agent_orchestration_policy_exists():
    policy = get_policy("agent_orchestration")

    assert policy.policy_id == "agent_orchestration:v1"
    assert policy.task_type == "agent_orchestration"
    assert policy.approval_required is False
    assert "agent_canvas" in policy.allowed_tools
    assert policy.retry_budget == 1
