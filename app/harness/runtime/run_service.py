from __future__ import annotations

import uuid
from typing import Any

from app.harness.contracts.run import HarnessRuntimeState
from app.harness.persistence.stores import (
    HarnessApprovalStore,
    HarnessEventStore,
    HarnessRunStore,
    HarnessRuntimeStateHistoryStore,
    HarnessRuntimeStateStore,
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
        runtime_state_store: HarnessRuntimeStateStore | None = None,
        runtime_state_history_store: HarnessRuntimeStateHistoryStore | None = None,
    ):
        self.run_store = run_store
        self.approval_store = approval_store
        self.verification_store = verification_store
        self.event_service = None if event_store is None else HarnessEventService(event_store=event_store)
        self.runtime_state_store = runtime_state_store
        self.runtime_state_history_store = runtime_state_history_store
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

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _build_runtime_state(
        self,
        *,
        run: dict[str, object],
        latest_approval: dict[str, object] | None,
        latest_verification: dict[str, object] | None,
        events: list[dict[str, object]],
    ) -> dict[str, object]:
        payload = dict(latest_approval.get("payload_json") or {}) if isinstance(latest_approval, dict) else {}
        metadata = dict(run.get("metadata_json") or {}) if isinstance(run, dict) else {}
        verification_artifacts = (
            dict(latest_verification.get("artifacts_json") or {})
            if isinstance(latest_verification, dict)
            else {}
        )
        output_artifacts = (
            dict(verification_artifacts.get("output_artifacts") or {})
            if isinstance(verification_artifacts, dict)
            else {}
        )

        resumed_event = next((event for event in reversed(events) if str(event.get("event_type") or "") == "orchestration.stream_continuation_resumed"), None)
        completed_event = next((event for event in reversed(events) if str(event.get("event_type") or "") == "orchestration.stream_continuation_completed"), None)

        review_state = {
            "stage": str(payload.get("review_stage") or "") or None,
            "status": str(latest_approval.get("status") or "") if isinstance(latest_approval, dict) else None,
            "agent_id": str(payload.get("agent_id") or "") or None,
            "agent_name": str(payload.get("agent_name") or "") or None,
            "review_output": str(payload.get("review_output") or "") or None,
            "check_count": self._coerce_int(payload.get("check_count")),
            "segment_index": self._coerce_int(payload.get("segment_index")),
            "segment_count": self._coerce_int(payload.get("segment_count")),
            "segment_start_char": self._coerce_int(payload.get("segment_start_char")),
            "segment_end_char": self._coerce_int(payload.get("segment_end_char")),
            "last_reviewed_char": self._coerce_int(
                payload.get("last_reviewed_char")
                or payload.get("review_cursor_char")
                or payload.get("segment_end_char")
            ),
        }

        continuation_mode = str(metadata.get("review_recovery_mode") or "") or None
        continuation_state = {
            "enabled": bool(
                review_state["stage"] == "agent_output_stream"
                or continuation_mode in {"continue_with_partial_stream_output", "continue_from_stream_block"}
                or resumed_event
                or completed_event
            ),
            "mode": continuation_mode,
            "status": (
                "completed"
                if completed_event
                else "resumed"
                if resumed_event
                else str(latest_approval.get("status") or "") if isinstance(latest_approval, dict) else None
            ),
            "agent_id": str(payload.get("agent_id") or (resumed_event or {}).get("details_json", {}).get("agent_id") or "") or None,
            "agent_name": str(payload.get("agent_name") or (resumed_event or {}).get("details_json", {}).get("agent_name") or "") or None,
            "step_index": self._coerce_int(payload.get("step_index") or (resumed_event or {}).get("details_json", {}).get("next_step_index")),
            "prefix_length": len(str(payload.get("partial_output") or "")) or self._coerce_int((resumed_event or {}).get("details_json", {}).get("partial_length")) or 0,
            "resumed_at": self._coerce_int((resumed_event or {}).get("created_at")),
            "completed_at": self._coerce_int((completed_event or {}).get("created_at")),
        }

        cluster_ids: list[str] = []
        paper_count = 0
        browser_preview_count = 0
        source_count = 0
        research_mode: str | None = None
        for cluster_id, artifact in output_artifacts.items():
            if not isinstance(artifact, dict):
                continue
            research = artifact.get("research")
            if not isinstance(research, dict):
                continue
            cluster_ids.append(str(cluster_id))
            paper_count += len(list(research.get("papers") or []))
            browser_preview_count += len(list(research.get("browser_previews") or []))
            source_count += len(list(research.get("sources") or []))
            if not research_mode:
                research_mode = str(research.get("research_mode") or "") or None

        runtime_state = HarnessRuntimeState(
            review=review_state,
            continuation=continuation_state,
            research={
                "enabled": bool(cluster_ids),
                "mode": research_mode,
                "paper_count": paper_count,
                "browser_preview_count": browser_preview_count,
                "source_count": source_count,
                "cluster_ids": cluster_ids,
            },
        )
        return runtime_state.model_dump()

    @staticmethod
    def _normalize_runtime_state_payload(value: dict[str, object] | None) -> dict[str, object]:
        return HarnessRuntimeState.model_validate(value or {}).model_dump()

    def _persist_runtime_state(self, run_id: str, runtime_state: dict[str, object]) -> dict[str, object]:
        normalized = self._normalize_runtime_state_payload(runtime_state)
        if self.runtime_state_store is not None:
            self.runtime_state_store.upsert_state(
                run_id=run_id,
                review_state_json=dict(normalized.get("review") or {}),
                continuation_state_json=dict(normalized.get("continuation") or {}),
                research_state_json=dict(normalized.get("research") or {}),
            )
        return normalized

    def _append_runtime_state_history(
        self,
        run_id: str,
        *,
        runtime_state: dict[str, object],
        transition_type: str,
    ) -> dict[str, object] | None:
        if self.runtime_state_history_store is None:
            return None
        latest = self.runtime_state_history_store.get_latest_for_run(run_id)
        next_version = int((latest or {}).get("version") or 0) + 1
        stage = str(((runtime_state.get("review") or {}) if isinstance(runtime_state, dict) else {}).get("stage") or "").strip() or None
        return self.runtime_state_history_store.append_history(
            run_id=run_id,
            version=next_version,
            transition_type=transition_type,
            stage=stage,
            runtime_state_json=runtime_state,
        )

    def _load_persisted_runtime_state(self, run_id: str) -> dict[str, object] | None:
        if self.runtime_state_store is None:
            return None
        stored = self.runtime_state_store.get_by_run(run_id)
        if not isinstance(stored, dict):
            return None
        return self._normalize_runtime_state_payload(
            {
                "review": stored.get("review_state_json") or {},
                "continuation": stored.get("continuation_state_json") or {},
                "research": stored.get("research_state_json") or {},
            }
        )

    def sync_runtime_state(
        self,
        run_id: str,
        *,
        run: dict[str, object] | None = None,
        latest_approval: dict[str, object] | None = None,
        latest_verification: dict[str, object] | None = None,
        events: list[dict[str, object]] | None = None,
        transition_type: str = "sync",
    ) -> dict[str, object] | None:
        resolved_run = run or self.run_store.get_run(run_id)
        if resolved_run is None:
            return None
        resolved_approval = latest_approval if latest_approval is not None else self.get_latest_approval(run_id)
        resolved_verification = (
            latest_verification if latest_verification is not None else self.get_latest_verification(run_id)
        )
        if events is not None:
            resolved_events = events
        else:
            try:
                resolved_events = self.list_run_events(run_id=run_id, limit=20)
            except AttributeError:
                resolved_events = []
        runtime_state = self._build_runtime_state(
            run=resolved_run,
            latest_approval=resolved_approval,
            latest_verification=resolved_verification,
            events=list(resolved_events or []),
        )
        persisted = self._persist_runtime_state(run_id, runtime_state)
        self._append_runtime_state_history(run_id, runtime_state=persisted, transition_type=transition_type)
        return persisted

    def patch_runtime_state(
        self,
        run_id: str,
        *,
        review: dict[str, object] | None = None,
        continuation: dict[str, object] | None = None,
        research: dict[str, object] | None = None,
        transition_type: str = "patch",
    ) -> dict[str, object] | None:
        if self.run_store.get_run(run_id) is None:
            return None
        current = self._load_persisted_runtime_state(run_id) or HarnessRuntimeState().model_dump()
        if review:
            merged = dict(current.get("review") or {})
            merged.update(review)
            current["review"] = merged
        if continuation:
            merged = dict(current.get("continuation") or {})
            merged.update(continuation)
            current["continuation"] = merged
        if research:
            merged = dict(current.get("research") or {})
            merged.update(research)
            current["research"] = merged
        persisted = self._persist_runtime_state(run_id, current)
        self._append_runtime_state_history(run_id, runtime_state=persisted, transition_type=transition_type)
        return persisted

    def list_runtime_state_history(self, *, run_id: str, limit: int = 100) -> list[dict[str, object]]:
        if self.runtime_state_history_store is None:
            return []
        return self.runtime_state_history_store.list_for_run(run_id=run_id, limit=limit)

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
        self.sync_runtime_state(run_id, run=updated, transition_type=f"status:{status}")
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
        initial_run_id = str(run.get("run_id") or "")
        initial_state = self._persist_runtime_state(initial_run_id, HarnessRuntimeState().model_dump())
        self._append_runtime_state_history(initial_run_id, runtime_state=initial_state, transition_type="run_created")
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
            self.sync_runtime_state(str(run.get("run_id") or ""), run=run, latest_approval=approval, transition_type="approval_requested")
        return run

    def can_retry_run(self, run_id: str) -> bool:
        run = self.run_store.get_run(run_id)
        if run is None:
            return False
        status = str(run.get("status") or "")
        task_type = str(run.get("task_type") or "")
        if status != "failed":
            approval = self.get_latest_approval(run_id)
            if not (
                status == "rejected"
                and task_type == "agent_orchestration"
                and isinstance(approval, dict)
                and str(approval.get("action_type") or "") == "orchestration_review"
                and str(approval.get("status") or "") == "rejected"
            ):
                return False
        policy = get_policy(str(run.get("task_type") or ""))
        retry_count = int(run.get("retry_count") or 0)
        return retry_count < int(policy.retry_budget)

    def create_retry_run(self, run_id: str, *, requested_by: str) -> dict[str, object]:
        source_run = self.run_store.get_run(run_id)
        if source_run is None:
            raise HarnessRetryNotAllowedError("Run not found")

        status = str(source_run.get("status") or "")
        source_task_type = str(source_run.get("task_type") or "")
        latest_approval = self.get_latest_approval(run_id)
        rejected_review = (
            status == "rejected"
            and source_task_type == "agent_orchestration"
            and isinstance(latest_approval, dict)
            and str(latest_approval.get("action_type") or "") == "orchestration_review"
            and str(latest_approval.get("status") or "") == "rejected"
        )
        if status != "failed" and not rejected_review:
            raise HarnessRetryNotAllowedError("Only failed or review-rejected orchestration runs can be retried")

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
        input_json = dict(source_run.get("input_json") or {})
        if rejected_review:
            orchestration_resume = dict(input_json.get("orchestration_resume") or {})
            rollback_state = dict(orchestration_resume.get("rollback_state") or {})
            payload_json = dict(latest_approval.get("payload_json") or {}) if isinstance(latest_approval, dict) else {}
            review_stage = str(payload_json.get("review_stage") or "").strip()
            continue_mode = str(orchestration_resume.get("continue_mode") or "").strip()
            if rollback_state:
                orchestration_resume["state"] = rollback_state
                if review_stage == "cluster_research" or continue_mode == "discard_research_evidence":
                    orchestration_resume["next_step_index"] = int(orchestration_resume.get("next_step_index") or 0)
                    metadata["review_recovery_mode"] = "continue_without_research"
                elif review_stage == "agent_output_stream":
                    orchestration_resume["next_step_index"] = int(
                        payload_json.get("step_index")
                        or orchestration_resume.get("next_step_index")
                        or 0
                    )
                    orchestration_resume.pop("continuation", None)
                    metadata["review_recovery_mode"] = "continue_from_stream_block"
                else:
                    orchestration_resume["next_step_index"] = int(orchestration_resume.get("next_step_index") or 0) - 1
                    if orchestration_resume["next_step_index"] < 0:
                        orchestration_resume["next_step_index"] = 0
            orchestration_resume.pop("review_decision", None)
            orchestration_resume.pop("continue_mode", None)
            input_json["orchestration_resume"] = orchestration_resume
            rejection_comment = str(latest_approval.get("comment") or "").strip() if isinstance(latest_approval, dict) else ""
            if rejection_comment:
                existing_task = str(input_json.get("task") or "").strip()
                suffix = (
                    f"User redirected after research evidence block: {rejection_comment}"
                    if review_stage == "cluster_research"
                    else f"User redirected after review block: {rejection_comment}"
                )
                input_json["task"] = f"{existing_task}\n\n{suffix}".strip() if existing_task else suffix
                metadata["review_rejection_comment"] = rejection_comment
        retried_run = self.create_run(
            user_id=str(source_run.get("user_id") or requested_by),
            task_type=source_task_type,
            input_json=input_json,
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
        if isinstance(updated, dict):
            updated_run_id = str(updated.get("run_id") or "").strip()
            if updated_run_id:
                self.sync_runtime_state(updated_run_id, run=run, latest_approval=updated, transition_type="approval_resolved")
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
        self.sync_runtime_state(run_id, latest_verification=verification, transition_type="verification_recorded")
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

    def update_run_input_json(self, run_id: str, input_json: dict[str, object]) -> dict[str, object] | None:
        existing = self.run_store.get_run(run_id)
        if existing is None:
            return None
        updated = self.run_store.update_run(run_id, input_json=input_json)
        self._record_event_for_run(
            updated,
            event_type="run.input_updated",
            details={"keys": sorted(input_json.keys())},
        )
        self.sync_runtime_state(run_id, run=updated, transition_type="input_updated")
        return updated

    def update_run_metadata_json(self, run_id: str, metadata_json: dict[str, object]) -> dict[str, object] | None:
        existing = self.run_store.get_run(run_id)
        if existing is None:
            return None
        updated = self.run_store.update_run(run_id, metadata_json=metadata_json)
        self._record_event_for_run(
            updated,
            event_type="run.metadata_updated",
            details={"keys": sorted(metadata_json.keys())},
        )
        self.sync_runtime_state(run_id, run=updated, transition_type="metadata_updated")
        return updated

    def create_approval_request(
        self,
        *,
        run_id: str,
        action_type: str,
        reason: str | None,
        payload_json: dict[str, object],
        requested_by: str | None,
    ) -> dict[str, object] | None:
        if self.approval_store is None:
            return None
        run = self.run_store.get_run(run_id)
        if run is None:
            return None
        approval = self.approval_store.create_approval(
            approval_id=f"ha_{uuid.uuid4()}",
            run_id=run_id,
            action_type=action_type,
            reason=reason,
            payload_json=payload_json,
            status="pending",
            requested_by=requested_by,
        )
        self.mark_waiting_approval(run_id)
        self._record_event_for_run(
            run,
            event_type="approval.requested",
            actor=requested_by,
            details={
                "approval_id": str(approval.get("approval_id") or ""),
                "action_type": action_type,
                "reason": reason,
            },
        )
        self.sync_runtime_state(run_id, run=run, latest_approval=approval, transition_type="approval_requested")
        return approval

    def record_event(
        self,
        run_id: str,
        *,
        event_type: str,
        actor: str | None = None,
        details: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        run = self.run_store.get_run(run_id)
        event = self._record_event_for_run(run, event_type=event_type, actor=actor, details=details)
        if run is not None:
            self.sync_runtime_state(run_id, run=run, transition_type=f"event:{event_type}")
        return event

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
        detail["runtime_state"] = self._load_persisted_runtime_state(run_id) or self.sync_runtime_state(
            run_id,
            run=detail,
            latest_approval=detail["latest_approval"],
            latest_verification=detail["latest_verification"],
            events=list(detail["events"] or []),
            transition_type="detail_sync",
        )
        return detail

    def list_runs(self, *, user_id: str, limit: int = 50) -> list[dict[str, object]]:
        runs = self.run_store.list_runs(user_id=user_id, limit=limit)
        persisted_states = (
            self.runtime_state_store.list_by_run_ids([str(run.get("run_id") or "") for run in runs])
            if self.runtime_state_store is not None
            else {}
        )
        summaries: list[dict[str, object]] = []
        for run in runs:
            run_id = str(run.get("run_id") or "").strip()
            summary = dict(run)
            policy = get_policy(str(run.get("task_type") or ""))
            summary["policy"] = self._policy_summary(policy)
            summary["can_retry"] = self.can_retry_run(run_id) if run_id else False
            summary["latest_approval"] = self.get_latest_approval(run_id) if run_id else None
            summary["latest_verification"] = self.get_latest_verification(run_id) if run_id else None
            stored = persisted_states.get(run_id)
            summary["runtime_state"] = (
                self._normalize_runtime_state_payload(
                    {
                        "review": stored.get("review_state_json") or {},
                        "continuation": stored.get("continuation_state_json") or {},
                        "research": stored.get("research_state_json") or {},
                    }
                )
                if isinstance(stored, dict)
                else self.sync_runtime_state(
                    run_id,
                    run=summary,
                    latest_approval=summary["latest_approval"],
                    latest_verification=summary["latest_verification"],
                    transition_type="list_sync",
                )
            )
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
        runtime_state_store=HarnessRuntimeStateStore(),
        runtime_state_history_store=HarnessRuntimeStateHistoryStore(),
    )
