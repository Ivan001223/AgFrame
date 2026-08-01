from __future__ import annotations

from app.platform.governance.lifecycle import (
    GovernanceLifecycleManager,
    LifecycleTransitionDecision,
)


class GovernanceService:
    def __init__(self, *, lifecycle_manager: GovernanceLifecycleManager | None = None):
        self.lifecycle_manager = lifecycle_manager or GovernanceLifecycleManager()

    def authorize_transition(
        self,
        *,
        current_status: str,
        target_status: str,
        reason: str | None = None,
        actor: str | None = None,
        triggered_by: str | None = None,
        correlation_id: str | None = None,
    ) -> LifecycleTransitionDecision:
        return self.lifecycle_manager.transition(
            current_status=current_status,
            target_status=target_status,
            reason=reason,
            actor=actor,
            triggered_by=triggered_by,
            correlation_id=correlation_id,
        )
