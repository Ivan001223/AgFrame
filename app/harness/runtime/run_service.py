from __future__ import annotations

import uuid

from app.harness.persistence.stores import (
    HarnessApprovalStore,
    HarnessRunStore,
    HarnessVerificationStore,
)
from app.harness.runtime.policy_registry import get_policy
from app.harness.runtime.verification_service import VerificationService


class HarnessRunService:
    def __init__(
        self,
        *,
        run_store: HarnessRunStore,
        approval_store: HarnessApprovalStore | None = None,
        verification_store: HarnessVerificationStore | None = None,
    ):
        self.run_store = run_store
        self.approval_store = approval_store
        self.verification_store = verification_store
        self.verification_service = VerificationService()

    def create_run(
        self,
        *,
        user_id: str,
        task_type: str,
        input_json: dict[str, object],
        session_id: str | None = None,
        metadata_json: dict[str, object] | None = None,
    ) -> dict[str, object]:
        policy = get_policy(task_type)
        run_id = f"hr_{uuid.uuid4()}"
        run = self.run_store.create_run(
            run_id=run_id,
            user_id=user_id,
            session_id=session_id,
            task_type=task_type,
            status="waiting_approval" if policy.approval_required else "created",
            policy_id=policy.policy_id,
            input_json=input_json,
            metadata_json=metadata_json,
            approval_required=policy.approval_required,
        )
        if policy.approval_required and self.approval_store is not None:
            self.approval_store.create_approval(
                approval_id=f"ha_{uuid.uuid4()}",
                run_id=run_id,
                action_type="resume",
                reason="Awaiting human approval before resume",
                payload_json={"session_id": session_id, "task_type": task_type},
                status="pending",
                requested_by=user_id,
            )
        return run

    def mark_approved(self, run_id: str) -> dict[str, object] | None:
        return self.run_store.update_run(run_id, status="approved")

    def mark_rejected(self, run_id: str) -> dict[str, object] | None:
        return self.run_store.update_run(run_id, status="rejected")

    def mark_resumed(self, run_id: str) -> dict[str, object] | None:
        run = self.run_store.get_run(run_id)
        if run is None:
            return None
        resume_count = int(run.get("resume_count") or 0) + 1
        return self.run_store.update_run(run_id, status="resumed", resume_count=resume_count)

    def mark_waiting_approval(self, run_id: str) -> dict[str, object] | None:
        return self.run_store.update_run(run_id, status="waiting_approval")

    def get_latest_approval(self, run_id: str) -> dict[str, object] | None:
        if self.approval_store is None:
            return None
        return self.approval_store.get_latest_by_run(run_id)

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
        return self.approval_store.update_approval(
            approval_id,
            status=status,
            resolved_by=resolved_by,
            comment=comment,
        )

    def mark_failed(self, run_id: str, *, verification_status: str = "fail") -> dict[str, object] | None:
        return self.run_store.update_run(run_id, status="failed", verification_status=verification_status)

    def mark_completed(self, run_id: str, *, verification_status: str = "pass") -> dict[str, object] | None:
        return self.run_store.update_run(run_id, status="completed", verification_status=verification_status)

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
        return self.verification_store.create_verification(
            verification_id=f"hv_{uuid.uuid4()}",
            run_id=run_id,
            status=str(verification_result.get("status") or "fail"),
            checks_json={"checks_run": verification_result.get("checks_run") or []},
            artifacts_json=verification_result.get("artifacts"),
            summary=str(verification_result.get("summary") or ""),
        )

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
        return self.run_store.update_run(run_id, current_step=step)

    def update_run_status(self, run_id: str, status: str) -> dict[str, object] | None:
        return self.run_store.update_run(run_id, status=status)

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
        if self.verification_store is not None:
            self.verification_store.create_verification(
                verification_id=f"hv_{uuid.uuid4()}",
                run_id=run_id,
                status=str(verification_result.get("status") or "fail"),
                checks_json={"checks_run": verification_result.get("checks_run") or []},
                artifacts_json=verification_result.get("artifacts"),
                summary=str(verification_result.get("summary") or ""),
            )
        terminal_status = "completed" if verification_result.get("status") == "pass" else "failed"
        verification_status = str(verification_result.get("status") or "fail")
        return self.run_store.update_run(
            run_id,
            status=terminal_status,
            verification_status=verification_status,
        )

    def get_run(self, run_id: str) -> dict[str, object] | None:
        return self.run_store.get_run(run_id)

    def list_runs(self, *, user_id: str, limit: int = 50) -> list[dict[str, object]]:
        return self.run_store.list_runs(user_id=user_id, limit=limit)

    def mark_running(self, run_id: str) -> dict[str, object] | None:
        return self.run_store.update_run(run_id, status="running")

    def mark_queued(self, run_id: str) -> dict[str, object] | None:
        return self.run_store.update_run(run_id, status="queued")

    def complete_with_verification(
        self,
        run_id: str,
        verification_result: dict[str, object],
    ) -> dict[str, object] | None:
        if self.verification_store is not None:
            self.verification_store.create_verification(
                verification_id=f"hv_{uuid.uuid4()}",
                run_id=run_id,
                status=str(verification_result.get("status") or "fail"),
                checks_json={"checks_run": verification_result.get("checks_run") or []},
                artifacts_json=verification_result.get("artifacts"),
                summary=str(verification_result.get("summary") or ""),
            )
        terminal_status = "completed" if verification_result.get("status") == "pass" else "failed"
        verification_status = str(verification_result.get("status") or "fail")
        return self.run_store.update_run(
            run_id,
            status=terminal_status,
            verification_status=verification_status,
        )

    def get_run(self, run_id: str) -> dict[str, object] | None:
        return self.run_store.get_run(run_id)

    def list_runs(self, *, user_id: str, limit: int = 50) -> list[dict[str, object]]:
        return self.run_store.list_runs(user_id=user_id, limit=limit)

    def mark_running(self, run_id: str) -> dict[str, object] | None:
        return self.run_store.update_run(run_id, status="running")

    def mark_queued(self, run_id: str) -> dict[str, object] | None:
        return self.run_store.update_run(run_id, status="queued")

    def complete_with_verification(
        self,
        run_id: str,
        verification_result: dict[str, object],
    ) -> dict[str, object] | None:
        if self.verification_store is not None:
            self.verification_store.create_verification(
                verification_id=f"hv_{uuid.uuid4()}",
                run_id=run_id,
                status=str(verification_result.get("status") or "fail"),
                checks_json={"checks_run": verification_result.get("checks_run") or []},
                artifacts_json=verification_result.get("artifacts"),
                summary=str(verification_result.get("summary") or ""),
            )
        terminal_status = "completed" if verification_result.get("status") == "pass" else "failed"
        verification_status = str(verification_result.get("status") or "fail")
        return self.run_store.update_run(
            run_id,
            status=terminal_status,
            verification_status=verification_status,
        )


def build_run_service() -> HarnessRunService:
    return HarnessRunService(
        run_store=HarnessRunStore(),
        approval_store=HarnessApprovalStore(),
        verification_store=HarnessVerificationStore(),
    )
