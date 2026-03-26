from __future__ import annotations

import time

from sqlalchemy import select, update

from app.infrastructure.database.models import HarnessApproval, HarnessRun, HarnessVerification
from app.infrastructure.database.orm import get_session


class HarnessRunStore:
    def list_runs(self, *, user_id: str, limit: int = 50) -> list[dict[str, object]]:
        with get_session() as session:
            rows = session.execute(
                select(HarnessRun)
                .where(HarnessRun.user_id == user_id)
                .order_by(HarnessRun.created_at.desc())
                .limit(int(limit))
            ).scalars().all()
            return [self._to_dict(row) for row in rows]

    def create_run(
        self,
        *,
        run_id: str,
        user_id: str,
        session_id: str | None,
        task_type: str,
        status: str,
        policy_id: str,
        input_json: dict[str, object],
        metadata_json: dict[str, object] | None,
        approval_required: bool,
    ) -> dict[str, object]:
        now = int(time.time())
        row = HarnessRun(
            run_id=run_id,
            user_id=user_id,
            session_id=session_id,
            task_type=task_type,
            status=status,
            policy_id=policy_id,
            input_json=input_json,
            metadata_json=metadata_json,
            current_step=None,
            retry_count=0,
            resume_count=0,
            approval_required=approval_required,
            verification_status=None,
            created_at=now,
            updated_at=now,
            finished_at=None,
        )
        with get_session() as session:
            session.add(row)
            session.flush()
        return self._to_dict(row)

    def get_run(self, run_id: str) -> dict[str, object] | None:
        with get_session() as session:
            row = session.execute(
                select(HarnessRun).where(HarnessRun.run_id == run_id)
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._to_dict(row)

    def update_run(self, run_id: str, **changes: object) -> dict[str, object] | None:
        if not changes:
            return self.get_run(run_id)
        payload = dict(changes)
        payload["updated_at"] = int(time.time())
        if payload.get("status") in {"completed", "failed"} and "finished_at" not in payload:
            payload["finished_at"] = int(time.time())
        with get_session() as session:
            session.execute(
                update(HarnessRun).where(HarnessRun.run_id == run_id).values(**payload)
            )
        return self.get_run(run_id)

    @staticmethod
    def _to_dict(row: HarnessRun) -> dict[str, object]:
        return {
            "run_id": row.run_id,
            "user_id": row.user_id,
            "session_id": row.session_id,
            "task_type": row.task_type,
            "status": row.status,
            "policy_id": row.policy_id,
            "input_json": row.input_json,
            "metadata_json": row.metadata_json,
            "current_step": row.current_step,
            "retry_count": row.retry_count,
            "resume_count": row.resume_count,
            "approval_required": row.approval_required,
            "verification_status": row.verification_status,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "finished_at": row.finished_at,
        }


class HarnessApprovalStore:
    def get_latest_by_run(self, run_id: str) -> dict[str, object] | None:
        with get_session() as session:
            row = session.execute(
                select(HarnessApproval)
                .where(HarnessApproval.run_id == run_id)
                .order_by(HarnessApproval.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._to_dict(row)

    def update_approval(
        self,
        approval_id: str,
        *,
        status: str,
        resolved_by: str,
        comment: str | None,
    ) -> dict[str, object] | None:
        payload = {
            "status": status,
            "resolved_by": resolved_by,
            "comment": comment,
            "resolved_at": int(time.time()),
        }
        with get_session() as session:
            session.execute(
                update(HarnessApproval)
                .where(HarnessApproval.approval_id == approval_id)
                .values(**payload)
            )
        with get_session() as session:
            row = session.execute(
                select(HarnessApproval).where(HarnessApproval.approval_id == approval_id)
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._to_dict(row)

    def create_approval(
        self,
        *,
        approval_id: str,
        run_id: str,
        action_type: str,
        reason: str | None,
        payload_json: dict[str, object],
        status: str,
        requested_by: str | None,
    ) -> dict[str, object]:
        now = int(time.time())
        row = HarnessApproval(
            approval_id=approval_id,
            run_id=run_id,
            action_type=action_type,
            reason=reason,
            payload_json=payload_json,
            status=status,
            requested_by=requested_by,
            resolved_by=None,
            comment=None,
            created_at=now,
            resolved_at=None,
        )
        with get_session() as session:
            session.add(row)
            session.flush()
        return self._to_dict(row)

    @staticmethod
    def _to_dict(row: HarnessApproval) -> dict[str, object]:
        return {
            "approval_id": row.approval_id,
            "run_id": row.run_id,
            "action_type": row.action_type,
            "reason": row.reason,
            "payload_json": row.payload_json,
            "status": row.status,
            "requested_by": row.requested_by,
            "resolved_by": row.resolved_by,
            "comment": row.comment,
            "created_at": row.created_at,
            "resolved_at": row.resolved_at,
        }


class HarnessVerificationStore:
    def create_verification(
        self,
        *,
        verification_id: str,
        run_id: str,
        status: str,
        checks_json: dict[str, object],
        artifacts_json: dict[str, object] | None,
        summary: str | None,
    ) -> dict[str, object]:
        now = int(time.time())
        row = HarnessVerification(
            verification_id=verification_id,
            run_id=run_id,
            status=status,
            checks_json=checks_json,
            artifacts_json=artifacts_json,
            summary=summary,
            created_at=now,
        )
        with get_session() as session:
            session.add(row)
            session.flush()
        return self._to_dict(row)

    @staticmethod
    def _to_dict(row: HarnessVerification) -> dict[str, object]:
        return {
            "verification_id": row.verification_id,
            "run_id": row.run_id,
            "status": row.status,
            "checks_json": row.checks_json,
            "artifacts_json": row.artifacts_json,
            "summary": row.summary,
            "created_at": row.created_at,
        }
