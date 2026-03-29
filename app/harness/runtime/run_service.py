from __future__ import annotations

import uuid
from typing import Any

from app.harness.persistence.stores import (
    HarnessApprovalStore,
    HarnessEventStore,
    HarnessRunStore,
    HarnessVerificationStore,
)
from app.harness.runtime.event_service import HarnessEventService
from app.harness.runtime.policy_registry import get_policy, list_policies
from app.harness.runtime.verification_service import VerificationService


class HarnessRetryNotAllowedError(ValueError):
    pass


class HarnessRunService:
    def __init__(
        self,
        *,
        run_store: HarnessRunStore,
        approval_store: HarnessApprovalStore | None = None,
        verification_store: HarnessVerificationStore | None = None,
        event_store: HarnessEventStore | None = None,
    ):
        self.run_store = run_store
        self.approval_store = approval_store
        self.verification_store = verification_store
        self.event_service = None if event_store is None else HarnessEventService(event_store=event_store)
        self.verification_service = VerificationService()

    @staticmethod
    def _normalize_metadata(value: dict[str, object] | None) -> dict[str, object] | None:
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _merge_metadata(
        current: dict[str, object] | None,
        updates: dict[str, object],
    ) -> dict[str, object]:
        merged: dict[str, object] = dict(current or {})
        merged.update(updates)
        return merged

    @staticmethod
    def _policy_summary(policy: Any) -> dict[str, object]:
        if hasattr(policy, "model_dump"):
            return dict(policy.model_dump())
        return {
            "policy_id": str(getattr(policy, "policy_id", "") or ""),
            "task_type": str(getattr(policy, "task_type", "") or ""),
            "approval_required": bool(getattr(policy, "approval_required", False)),
            "allowed_tools": list(getattr(policy, "allowed_tools", []) or []),
            "verification_profile": str(getattr(policy, "verification_profile", "") or ""),
            "retry_budget": int(getattr(policy, "retry_budget", 0) or 0),
        }

    def _record_event_for_run(
        self,
        run: dict[str, object] | None,
        *,
        event_type: str,
        actor: str | None = None,
        details: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if self.event_service is None or run is None:
            return None
        user_id = str(run.get("user_id") or "").strip()
        if not user_id:
            return None
        session_id = str(run.get("session_id") or "").strip() or None
        run_id = str(run.get("run_id") or "").strip() or None
        return self.event_service.record(
            event_type=event_type,
            event_source="harness",
            user_id=user_id,
            session_id=session_id,
            run_id=run_id,
            actor=actor,
            details=details,
        )

    def _transition_run_status(
        self,
        run_id: str,
        *,
        status: str,
        actor: str | None = None,
        details: dict[str, object] | None = None,
        **changes: object,
    ) -> dict[str, object] | None:
        existing = self.run_store.get_run(run_id)
        if existing is None:
            return None
        updated = self.run_store.update_run(run_id, status=status, **changes)
        if updated is None:
            return None
        event_details = {
            "from_status": str(existing.get("status") or "") or None,
            "to_status": status,
        }
        if details:
            event_details.update(details)
        self._record_event_for_run(
            updated,
            event_type="run.status_changed",
            actor=actor,
            details=event_details,
        )
        return updated

    def create_run(
        self,
        *,
        user_id: str,
        task_type: str,
        input_json: dict[str, object],
        session_id: str | None = None,
        metadata_json: dict[str, object] | None = None,
        retry_count: int = 0,
    ) -> dict[str, object]:
        policy = get_policy(task_type)
        normalized_task_type = policy.task_type
        run_id = f"hr_{uuid.uuid4()}"
        run = self.run_store.create_run(
            run_id=run_id,
            user_id=user_id,
            session_id=session_id,
            task_type=normalized_task_type,
            status="waiting_approval" if policy.approval_required else "created",
            policy_id=policy.policy_id,
            input_json=input_json,
            metadata_json=metadata_json,
            approval_required=policy.approval_required,
            retry_count=retry_count,
        )
        self._record_event_for_run(
            run,
            event_type="run.created",
            actor=user_id,
            details={
                "task_type": normalized_task_type,
                "policy_id": policy.policy_id,
                "approval_required": policy.approval_required,
                "initial_status": str(run.get("status") or ""),
            },
        )
        if policy.approval_required and self.approval_store is not None:
            approval = self.approval_store.create_approval(
                approval_id=f"ha_{uuid.uuid4()}",
                run_id=run_id,
                action_type="resume",
                reason="Awaiting human approval before resume",
                payload_json={"session_id": session_id, "task_type": normalized_task_type},
                status="pending",
                requested_by=user_id,
            )
            self._record_event_for_run(
                run,
                event_type="approval.requested",
                actor=user_id,
                details={
                    "approval_id": str(approval.get("approval_id") or ""),
                    "action_type": str(approval.get("action_type") or "resume"),
                    "reason": str(approval.get("reason") or ""),
                    "status": str(approval.get("status") or "pending"),
                },
            )
        return run

    def can_retry_run(self, run_id: str) -> bool:
        run = self.run_store.get_run(run_id)
        if run is None:
            return False
        if str(run.get("status") or "") != "failed":
            return False
        policy = get_policy(str(run.get("task_type") or ""))
        retry_count = int(run.get("retry_count") or 0)
        return retry_count < int(policy.retry_budget)

    def create_retry_run(self, run_id: str, *, requested_by: str) -> dict[str, object]:
        source_run = self.run_store.get_run(run_id)
        if source_run is None:
            raise HarnessRetryNotAllowedError("Run not found")

        status = str(source_run.get("status") or "")
        if status != "failed":
            raise HarnessRetryNotAllowedError("Only failed harness runs can be retried")

        policy = get_policy(str(source_run.get("task_type") or ""))
        retry_count = int(source_run.get("retry_count") or 0)
        if retry_count >= int(policy.retry_budget):
            raise HarnessRetryNotAllowedError("Retry budget exhausted for this harness run")

        metadata = self._merge_metadata(
            self._normalize_metadata(source_run.get("metadata_json") if isinstance(source_run, dict) else None),
            {
                "retried_from_run_id": run_id,
                "retry_requested_by": requested_by,
            },
        )
        retried_run = self.create_run(
            user_id=str(source_run.get("user_id") or requested_by),
            task_type=str(source_run.get("task_type") or ""),
            input_json=dict(source_run.get("input_json") or {}),
            session_id=str(source_run.get("session_id") or "") or None,
            metadata_json=metadata,
            retry_count=retry_count + 1,
        )
        self._record_event_for_run(
            source_run,
            event_type="run.retry_requested",
            actor=requested_by,
            details={
                "retried_to_run_id": str(retried_run.get("run_id") or ""),
                "retry_count": retry_count + 1,
                "retry_budget": int(policy.retry_budget),
            },
        )
        return retried_run

    def mark_approved(self, run_id: str) -> dict[str, object] | None:
        return self._transition_run_status(
            run_id,
            status="approved",
            details={"reason": "approval_granted"},
        )

    def mark_rejected(self, run_id: str) -> dict[str, object] | None:
        return self._transition_run_status(
            run_id,
            status="rejected",
            details={"reason": "approval_rejected"},
        )

    def mark_resumed(self, run_id: str) -> dict[str, object] | None:
        run = self.run_store.get_run(run_id)
        if run is None:
            return None
        resume_count = int(run.get("resume_count") or 0) + 1
        return self._transition_run_status(
            run_id,
            status="resumed",
            details={"resume_count": resume_count},
            resume_count=resume_count,
        )

    def mark_waiting_approval(self, run_id: str) -> dict[str, object] | None:
        return self._transition_run_status(run_id, status="waiting_approval")

    def mark_failed(self, run_id: str, *, verification_status: str = "fail") -> dict[str, object] | None:
        return self._transition_run_status(
            run_id,
            status="failed",
            details={"verification_status": verification_status},
            verification_status=verification_status,
        )

    def mark_completed(self, run_id: str, *, verification_status: str = "pass") -> dict[str, object] | None:
        return self._transition_run_status(
            run_id,
            status="completed",
            details={"verification_status": verification_status},
            verification_status=verification_status,
        )

    def mark_verifying(self, run_id: str) -> dict[str, object] | None:
        return self._transition_run_status(run_id, status="verifying")

    def mark_running(self, run_id: str) -> dict[str, object] | None:
        return self._transition_run_status(run_id, status="running")

    def mark_queued(self, run_id: str) -> dict[str, object] | None:
        return self._transition_run_status(run_id, status="queued")

    def get_latest_approval(self, run_id: str) -> dict[str, object] | None:
        if self.approval_store is None:
            return None
        return self.approval_store.get_latest_by_run(run_id)

    def get_latest_verification(self, run_id: str) -> dict[str, object] | None:
        if self.verification_store is None:
            return None
        return self.verification_store.get_latest_by_run(run_id)

    def update_approval(
        self,
        approval_id: str,
        *,
        status: str,
        resolved_by: str,
        comment: str | None,
    ) -> dict[str, object] | None:
        if self.approval_store is None:
            return None
        updated = self.approval_store.update_approval(
            approval_id,
            status=status,
            resolved_by=resolved_by,
            comment=comment,
        )
        run = None
        if isinstance(updated, dict):
            updated_run_id = str(updated.get("run_id") or "").strip()
            if updated_run_id:
                run = self.run_store.get_run(updated_run_id)
        self._record_event_for_run(
            run,
            event_type="approval.resolved",
            actor=resolved_by,
            details={
                "approval_id": approval_id,
                "status": status,
                "comment": comment,
            },
        )
        return updated

    def build_approval_verification(self, *, approved: bool) -> dict[str, object]:
        return {
            "status": "pass" if approved else "partial",
            "checks_run": ["approval_resolution"],
            "artifacts": {"approved": approved},
            "summary": "approval accepted" if approved else "approval rejected",
        }

    def create_verification(self, run_id: str, verification_result: dict[str, object]) -> dict[str, object] | None:
        if self.verification_store is None:
            return None
        verification = self.verification_store.create_verification(
            verification_id=f"hv_{uuid.uuid4()}",
            run_id=run_id,
            status=str(verification_result.get("status") or "fail"),
            checks_json={"checks_run": verification_result.get("checks_run") or []},
            artifacts_json=verification_result.get("artifacts"),
            summary=str(verification_result.get("summary") or ""),
        )
        self._record_event_for_run(
            self.run_store.get_run(run_id),
            event_type="verification.recorded",
            details={
                "verification_id": str(verification.get("verification_id") or ""),
                "status": str(verification.get("status") or ""),
                "summary": str(verification.get("summary") or ""),
            },
        )
        return verification

    def build_document_ingest_verification(
        self,
        *,
        ok: bool,
        stage: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> dict[str, object]:
        return self.verification_service.build_document_ingest_result(
            ok=ok,
            stage=stage,
            error_code=error_code,
            error_message=error_message,
        )

    def build_approval_resolution_verification(self, *, approved: bool) -> dict[str, object]:
        return self.build_approval_verification(approved=approved)

    def get_resume_session_id(self, run_id: str) -> str | None:
        run = self.run_store.get_run(run_id)
        if run is None:
            return None
        value = run.get("session_id")
        return str(value) if value else None

    def get_task_type(self, run_id: str) -> str | None:
        run = self.run_store.get_run(run_id)
        if run is None:
            return None
        value = run.get("task_type")
        return str(value) if value else None

    def get_input_json(self, run_id: str) -> dict[str, object] | None:
        run = self.run_store.get_run(run_id)
        if run is None:
            return None
        value = run.get("input_json")
        return dict(value) if isinstance(value, dict) else None

    def get_user_id(self, run_id: str) -> str | None:
        run = self.run_store.get_run(run_id)
        if run is None:
            return None
        value = run.get("user_id")
        return str(value) if value else None

    def set_current_step(self, run_id: str, step: str) -> dict[str, object] | None:
        existing = self.run_store.get_run(run_id)
        if existing is None:
            return None
        updated = self.run_store.update_run(run_id, current_step=step)
        self._record_event_for_run(
            updated,
            event_type="run.step_updated",
            details={
                "from_step": str(existing.get("current_step") or "") or None,
                "to_step": step,
            },
        )
        return updated

    def update_run_status(self, run_id: str, status: str) -> dict[str, object] | None:
        return self._transition_run_status(run_id, status=status)

    def persist_approval_resolution(self, run_id: str, *, approved: bool) -> dict[str, object] | None:
        return self.create_verification(run_id, self.build_approval_verification(approved=approved))

    def approval_required(self, run_id: str) -> bool:
        run = self.run_store.get_run(run_id)
        if run is None:
            return False
        return bool(run.get("approval_required"))

    def get_run_for_user(self, run_id: str, user_id: str) -> dict[str, object] | None:
        run = self.run_store.get_run(run_id)
        if run is None:
            return None
        if str(run.get("user_id") or "") != user_id:
            return None
        return run

    def list_runs_for_user(self, user_id: str, limit: int = 50) -> list[dict[str, object]]:
        return self.run_store.list_runs(user_id=user_id, limit=limit)

    def get_pending_approval_for_run(self, run_id: str) -> dict[str, object] | None:
        approval = self.get_latest_approval(run_id)
        if approval is None:
            return None
        if str(approval.get("status") or "") != "pending":
            return None
        return approval

    def complete_with_verification(
        self,
        run_id: str,
        verification_result: dict[str, object],
    ) -> dict[str, object] | None:
        self.mark_verifying(run_id)
        self.create_verification(run_id, verification_result)
        terminal_status = "completed" if verification_result.get("status") == "pass" else "failed"
        verification_status = str(verification_result.get("status") or "fail")
        return self._transition_run_status(
            run_id,
            status=terminal_status,
            details={"verification_status": verification_status},
            verification_status=verification_status,
        )

    def get_run(self, run_id: str) -> dict[str, object] | None:
        return self.run_store.get_run(run_id)

    def get_run_detail(self, run_id: str) -> dict[str, object] | None:
        run = self.run_store.get_run(run_id)
        if run is None:
            return None
        detail = dict(run)
        policy = get_policy(str(run.get("task_type") or ""))
        detail["policy"] = self._policy_summary(policy)
        detail["can_retry"] = self.can_retry_run(run_id)
        detail["latest_approval"] = self.get_latest_approval(run_id)
        detail["latest_verification"] = self.get_latest_verification(run_id)
        detail["events"] = self.list_run_events(run_id=run_id, limit=20)
        return detail

    def list_runs(self, *, user_id: str, limit: int = 50) -> list[dict[str, object]]:
        runs = self.run_store.list_runs(user_id=user_id, limit=limit)
        summaries: list[dict[str, object]] = []
        for run in runs:
            run_id = str(run.get("run_id") or "").strip()
            summary = dict(run)
            policy = get_policy(str(run.get("task_type") or ""))
            summary["policy"] = self._policy_summary(policy)
            summary["can_retry"] = self.can_retry_run(run_id) if run_id else False
            summary["latest_approval"] = self.get_latest_approval(run_id) if run_id else None
            summary["latest_verification"] = self.get_latest_verification(run_id) if run_id else None
            summaries.append(summary)
        return summaries

    def list_run_events(self, *, run_id: str, user_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        if self.event_service is None:
            return []
        return self.event_service.list_for_run(run_id=run_id, user_id=user_id, limit=limit)

    def list_policies(self) -> list[dict[str, object]]:
        return [self._policy_summary(policy) for policy in list_policies()]


def build_run_service() -> HarnessRunService:
    return HarnessRunService(
        run_store=HarnessRunStore(),
        approval_store=HarnessApprovalStore(),
        verification_store=HarnessVerificationStore(),
        event_store=HarnessEventStore(),
    )
