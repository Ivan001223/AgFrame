import pytest

from app.platform.governance.lifecycle import GovernanceLifecycleManager


def test_governance_lifecycle_manager_enforces_authoritative_transition_path():
    manager = GovernanceLifecycleManager()

    allowed = manager.transition(
        current_status="created",
        target_status="queued",
        reason="api_enqueued",
    )

    assert allowed.to_status == "queued"
    assert allowed.reason == "api_enqueued"


def test_governance_lifecycle_manager_rejects_invalid_transition():
    manager = GovernanceLifecycleManager()

    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        manager.transition(
            current_status="created",
            target_status="completed",
            reason="skip_execution",
        )
