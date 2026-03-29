from __future__ import annotations

import time

from sqlalchemy import delete, select, update

from app.infrastructure.database.models import (
    HarnessAgentProject,
    HarnessApproval,
    HarnessEvent,
    HarnessModelProvider,
    HarnessRun,
    HarnessRunRuntimeState,
    HarnessRunRuntimeStateHistory,
    HarnessVerification,
)
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
        retry_count: int = 0,
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
            retry_count=retry_count,
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


class HarnessAgentProjectStore:
    def list_projects(self, *, user_id: str, limit: int = 20) -> list[dict[str, object]]:
        with get_session() as session:
            rows = session.execute(
                select(HarnessAgentProject)
                .where(HarnessAgentProject.user_id == user_id)
                .order_by(HarnessAgentProject.updated_at.desc())
                .limit(int(limit))
            ).scalars().all()
            return [self._to_dict(row) for row in rows]

    def create_project(
        self,
        *,
        project_id: str,
        user_id: str,
        name: str,
        description: str | None,
        graph_json: dict[str, object],
    ) -> dict[str, object]:
        now = int(time.time())
        row = HarnessAgentProject(
            project_id=project_id,
            user_id=user_id,
            name=name,
            description=description,
            graph_json=graph_json,
            created_at=now,
            updated_at=now,
        )
        with get_session() as session:
            session.add(row)
            session.flush()
        return self._to_dict(row)

    def get_project(self, project_id: str) -> dict[str, object] | None:
        with get_session() as session:
            row = session.execute(
                select(HarnessAgentProject).where(HarnessAgentProject.project_id == project_id)
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._to_dict(row)

    def get_latest_project_for_user(self, user_id: str) -> dict[str, object] | None:
        with get_session() as session:
            row = session.execute(
                select(HarnessAgentProject)
                .where(HarnessAgentProject.user_id == user_id)
                .order_by(HarnessAgentProject.updated_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._to_dict(row)

    def update_project(self, project_id: str, **changes: object) -> dict[str, object] | None:
        if not changes:
            return self.get_project(project_id)
        payload = dict(changes)
        payload["updated_at"] = int(time.time())
        with get_session() as session:
            session.execute(
                update(HarnessAgentProject).where(HarnessAgentProject.project_id == project_id).values(**payload)
            )
        return self.get_project(project_id)

    @staticmethod
    def _to_dict(row: HarnessAgentProject) -> dict[str, object]:
        return {
            "project_id": row.project_id,
            "user_id": row.user_id,
            "name": row.name,
            "description": row.description,
            "graph_json": row.graph_json,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }


class HarnessVerificationStore:
    def get_latest_by_run(self, run_id: str) -> dict[str, object] | None:
        with get_session() as session:
            row = session.execute(
                select(HarnessVerification)
                .where(HarnessVerification.run_id == run_id)
                .order_by(HarnessVerification.created_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._to_dict(row)

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


class HarnessEventStore:
    def create_event(
        self,
        *,
        event_id: str,
        event_type: str,
        event_source: str,
        user_id: str,
        session_id: str | None,
        run_id: str | None,
        actor: str | None,
        details_json: dict[str, object] | None,
    ) -> dict[str, object]:
        now = int(time.time() * 1000)
        row = HarnessEvent(
            event_id=event_id,
            event_type=event_type,
            event_source=event_source,
            user_id=user_id,
            session_id=session_id,
            run_id=run_id,
            actor=actor,
            details_json=details_json,
            created_at=now,
        )
        with get_session() as session:
            session.add(row)
            session.flush()
        return self._to_dict(row)

    def list_events(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        with get_session() as session:
            stmt = select(HarnessEvent).order_by(HarnessEvent.created_at.asc()).limit(int(limit))
            if user_id is not None:
                stmt = stmt.where(HarnessEvent.user_id == user_id)
            if session_id is not None:
                stmt = stmt.where(HarnessEvent.session_id == session_id)
            if run_id is not None:
                stmt = stmt.where(HarnessEvent.run_id == run_id)
            rows = session.execute(stmt).scalars().all()
            return [self._to_dict(row) for row in rows]

    @staticmethod
    def _to_dict(row: HarnessEvent) -> dict[str, object]:
        return {
            "event_id": row.event_id,
            "event_type": row.event_type,
            "event_source": row.event_source,
            "user_id": row.user_id,
            "session_id": row.session_id,
            "run_id": row.run_id,
            "actor": row.actor,
            "details_json": row.details_json,
            "created_at": row.created_at,
        }


class HarnessRuntimeStateStore:
    def get_by_run(self, run_id: str) -> dict[str, object] | None:
        with get_session() as session:
            row = session.execute(
                select(HarnessRunRuntimeState).where(HarnessRunRuntimeState.run_id == run_id)
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._to_dict(row)

    def list_by_run_ids(self, run_ids: list[str]) -> dict[str, dict[str, object]]:
        normalized = [str(run_id).strip() for run_id in run_ids if str(run_id).strip()]
        if not normalized:
            return {}
        with get_session() as session:
            rows = session.execute(
                select(HarnessRunRuntimeState).where(HarnessRunRuntimeState.run_id.in_(normalized))
            ).scalars().all()
            return {str(row.run_id): self._to_dict(row) for row in rows}

    def upsert_state(
        self,
        *,
        run_id: str,
        review_state_json: dict[str, object],
        continuation_state_json: dict[str, object],
        research_state_json: dict[str, object],
    ) -> dict[str, object]:
        now = int(time.time())
        with get_session() as session:
            row = session.execute(
                select(HarnessRunRuntimeState).where(HarnessRunRuntimeState.run_id == run_id)
            ).scalar_one_or_none()
            if row is None:
                row = HarnessRunRuntimeState(
                    run_id=run_id,
                    review_state_json=review_state_json,
                    continuation_state_json=continuation_state_json,
                    research_state_json=research_state_json,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
                session.flush()
                return self._to_dict(row)

            row.review_state_json = review_state_json
            row.continuation_state_json = continuation_state_json
            row.research_state_json = research_state_json
            row.updated_at = now
            session.flush()
            return self._to_dict(row)

    @staticmethod
    def _to_dict(row: HarnessRunRuntimeState) -> dict[str, object]:
        return {
            "run_id": row.run_id,
            "review_state_json": dict(row.review_state_json or {}),
            "continuation_state_json": dict(row.continuation_state_json or {}),
            "research_state_json": dict(row.research_state_json or {}),
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }


class HarnessRuntimeStateHistoryStore:
    def list_for_run(self, *, run_id: str, limit: int = 100) -> list[dict[str, object]]:
        with get_session() as session:
            rows = session.execute(
                select(HarnessRunRuntimeStateHistory)
                .where(HarnessRunRuntimeStateHistory.run_id == run_id)
                .order_by(HarnessRunRuntimeStateHistory.version.asc())
                .limit(int(limit))
            ).scalars().all()
            return [self._to_dict(row) for row in rows]

    def get_latest_for_run(self, run_id: str) -> dict[str, object] | None:
        with get_session() as session:
            row = session.execute(
                select(HarnessRunRuntimeStateHistory)
                .where(HarnessRunRuntimeStateHistory.run_id == run_id)
                .order_by(HarnessRunRuntimeStateHistory.version.desc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._to_dict(row)

    def append_history(
        self,
        *,
        run_id: str,
        version: int,
        transition_type: str,
        stage: str | None,
        runtime_state_json: dict[str, object],
    ) -> dict[str, object]:
        now = int(time.time() * 1000)
        row = HarnessRunRuntimeStateHistory(
            run_id=run_id,
            version=version,
            transition_type=transition_type,
            stage=stage,
            runtime_state_json=runtime_state_json,
            created_at=now,
        )
        with get_session() as session:
            session.add(row)
            session.flush()
            return self._to_dict(row)

    @staticmethod
    def _to_dict(row: HarnessRunRuntimeStateHistory) -> dict[str, object]:
        return {
            "history_id": row.history_id,
            "run_id": row.run_id,
            "version": row.version,
            "transition_type": row.transition_type,
            "stage": row.stage,
            "runtime_state_json": dict(row.runtime_state_json or {}),
            "created_at": row.created_at,
        }


class HarnessModelProviderStore:
    def list_providers(self, *, user_id: str | None = None, limit: int = 50) -> list[dict[str, object]]:
        with get_session() as session:
            stmt = (
                select(HarnessModelProvider)
                .order_by(HarnessModelProvider.updated_at.desc())
                .limit(int(limit))
            )
            if user_id is not None:
                stmt = stmt.where(HarnessModelProvider.user_id == user_id)
            rows = session.execute(stmt).scalars().all()
            return [self._to_dict(row) for row in rows]

    def get_provider(self, provider_id: str) -> dict[str, object] | None:
        with get_session() as session:
            row = session.execute(
                select(HarnessModelProvider).where(HarnessModelProvider.provider_id == provider_id)
            ).scalar_one_or_none()
            if row is None:
                return None
            return self._to_dict(row)

    def create_provider(
        self,
        *,
        provider_id: str,
        user_id: str,
        name: str,
        base_url: str,
        api_key_encrypted: str,
        models_json: list[str],
        is_default: bool = False,
        enabled: bool = True,
    ) -> dict[str, object]:
        now = int(time.time())
        row = HarnessModelProvider(
            provider_id=provider_id,
            user_id=user_id,
            name=name,
            base_url=base_url,
            api_key_encrypted=api_key_encrypted,
            models_json=models_json,
            is_default=is_default,
            enabled=enabled,
            created_at=now,
            updated_at=now,
        )
        with get_session() as session:
            session.add(row)
            session.flush()
        return self._to_dict(row)

    def update_provider(self, provider_id: str, **changes: object) -> dict[str, object] | None:
        if not changes:
            return self.get_provider(provider_id)
        payload = dict(changes)
        payload["updated_at"] = int(time.time())
        with get_session() as session:
            session.execute(
                update(HarnessModelProvider).where(HarnessModelProvider.provider_id == provider_id).values(**payload)
            )
        return self.get_provider(provider_id)

    def delete_provider(self, provider_id: str) -> bool:
        with get_session() as session:
            result = session.execute(
                delete(HarnessModelProvider).where(HarnessModelProvider.provider_id == provider_id)
            )
            return result.rowcount > 0

    @staticmethod
    def _to_dict(row: HarnessModelProvider) -> dict[str, object]:
        return {
            "provider_id": row.provider_id,
            "user_id": row.user_id,
            "name": row.name,
            "base_url": row.base_url,
            "api_key_encrypted": row.api_key_encrypted,
            "models_json": row.models_json,
            "is_default": row.is_default,
            "enabled": row.enabled,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
