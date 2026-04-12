from __future__ import annotations

from pydantic import BaseModel


class LifecycleTransitionDecision(BaseModel):
    from_status: str
    to_status: str
    reason: str | None = None


class GovernanceLifecycleManager:
    _allowed: dict[str, set[str]] = {
        "created": {"queued", "waiting_approval"},
        "queued": {"running", "failed"},
        "running": {"waiting_approval", "verifying", "failed"},
        "waiting_approval": {"approved", "rejected"},
        "approved": {"resumed", "running"},
        "rejected": {"created"},
        "resumed": {"running", "failed"},
        "verifying": {"completed", "failed"},
        "completed": set(),
        "failed": set(),
    }

    def transition(
        self,
        *,
        current_status: str,
        target_status: str,
        reason: str | None = None,
    ) -> LifecycleTransitionDecision:
        allowed_targets = self._allowed.get(current_status, set())
        if target_status not in allowed_targets and current_status != target_status:
            raise ValueError(f"invalid lifecycle transition: {current_status} -> {target_status}")
        return LifecycleTransitionDecision(
            from_status=current_status,
            to_status=target_status,
            reason=reason,
        )
