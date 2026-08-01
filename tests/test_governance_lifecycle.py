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


def test_failed_to_created_retry_is_allowed():
    manager = GovernanceLifecycleManager()

    decision = manager.transition(
        current_status="failed",
        target_status="created",
        reason="retry_after_failure",
    )

    assert decision.from_status == "failed"
    assert decision.to_status == "created"


def test_resumed_to_waiting_approval_is_allowed():
    manager = GovernanceLifecycleManager()

    decision = manager.transition(
        current_status="resumed",
        target_status="waiting_approval",
        reason="blocking_review_during_resume",
    )

    assert decision.from_status == "resumed"
    assert decision.to_status == "waiting_approval"


def test_completed_is_terminal():
    manager = GovernanceLifecycleManager()

    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        manager.transition(
            current_status="completed",
            target_status="running",
            reason="cannot_reopen_completed",
        )


def test_failed_to_running_is_rejected():
    manager = GovernanceLifecycleManager()

    with pytest.raises(ValueError, match="invalid lifecycle transition"):
        manager.transition(
            current_status="failed",
            target_status="running",
        )


def test_transition_propagates_audit_context():
    manager = GovernanceLifecycleManager()

    decision = manager.transition(
        current_status="running",
        target_status="waiting_approval",
        reason="awaiting_human_decision",
        actor="user_123",
        triggered_by="orchestration_review",
        correlation_id="corr_abc",
    )

    assert decision.actor == "user_123"
    assert decision.triggered_by == "orchestration_review"
    assert decision.correlation_id == "corr_abc"


def test_same_status_transition_is_idempotent():
    manager = GovernanceLifecycleManager()

    decision = manager.transition(
        current_status="running",
        target_status="running",
    )

    assert decision.from_status == "running"
    assert decision.to_status == "running"
