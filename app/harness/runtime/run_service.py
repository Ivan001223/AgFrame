from __future__ import annotations

import uuid
from typing import Any, Literal, cast

from app.harness.contracts.run import (
    HarnessContinuationState,
    HarnessResearchState,
    HarnessReviewState,
    HarnessRunChecklistItem,
    HarnessRunChecklistSnapshot,
    HarnessRuntimeState,
    HarnessWorkflowProgress,
    HarnessWorkflowStep,
)
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
from app.platform.governance.audit import build_lifecycle_event_details
from app.platform.governance.commands import ApprovalResolutionCommand, VerificationRecordCommand
from app.platform.governance.service import GovernanceService
from app.runtime.graph.orchestration_graph import build_orchestration_execution_plan


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
        self.governance_service = GovernanceService()

    @staticmethod
    def _normalize_metadata(value: dict[str, object] | None) -> dict[str, object] | None:
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _as_object_dict(value: object) -> dict[str, object]:
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _as_object_list(value: object) -> list[object]:
        return list(value) if isinstance(value, list) else []

    @staticmethod
    def _as_dict_list(value: object) -> list[dict[str, object]]:
        return [cast(dict[str, object], item) for item in HarnessRunService._as_object_list(value) if isinstance(item, dict)]

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

    @staticmethod
    def _normalize_string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @staticmethod
    def _filter_graph_for_execution(
        graph_json: dict[str, object],
        *,
        selected_agent_ids: list[str],
    ) -> dict[str, object]:
        if not selected_agent_ids:
            return dict(graph_json)
        selected = {agent_id for agent_id in selected_agent_ids if agent_id}
        filtered = dict(graph_json)
        agents = HarnessRunService._as_object_list(graph_json.get("agents"))
        filtered["agents"] = [
            agent
            for agent in agents
            if isinstance(agent, dict) and str(agent.get("agent_id") or "") in selected
        ]
        edges = HarnessRunService._as_object_list(graph_json.get("edges"))
        filtered["edges"] = [
            edge
            for edge in edges
            if (
                isinstance(edge, dict)
                and str(edge.get("source_agent_id") or "") in selected
                and str(edge.get("target_agent_id") or "") in selected
            )
        ]
        selected_scope_orchestration_summary = graph_json.get("selected_scope_orchestration_summary")
        if isinstance(selected_scope_orchestration_summary, dict):
            filtered["orchestration_summary"] = dict(selected_scope_orchestration_summary)
        return filtered

    @staticmethod
    def _workflow_step_kind(agent_config: dict[str, object]) -> str:
        if bool(agent_config.get("cluster_summary")):
            return "cluster_summary"
        if str(agent_config.get("cluster_agent_id") or "").strip():
            return "cluster_member"
        return "agent"

    @staticmethod
    def _workflow_step_kind_literal(
        agent_config: dict[str, object],
    ) -> Literal["agent", "cluster_member", "cluster_summary"]:
        return cast(
            Literal["agent", "cluster_member", "cluster_summary"],
            HarnessRunService._workflow_step_kind(agent_config),
        )

    @staticmethod
    def _normalize_checklist_status(
        value: object,
    ) -> Literal["pending", "in_progress", "completed"]:
        normalized = str(value or "pending").strip()
        if normalized in {"pending", "in_progress", "completed"}:
            return cast(Literal["pending", "in_progress", "completed"], normalized)
        return "pending"

    @staticmethod
    def _normalize_workflow_status(
        value: object,
    ) -> Literal["idle", "pending", "running", "blocked", "completed", "failed"]:
        normalized = str(value or "idle").strip()
        if normalized in {"idle", "pending", "running", "blocked", "completed", "failed"}:
            return cast(Literal["idle", "pending", "running", "blocked", "completed", "failed"], normalized)
        return "idle"

    @staticmethod
    def _decorate_run_context(run: dict[str, object]) -> dict[str, object]:
        payload = dict(run)
        input_json = payload.get("input_json")
        metadata_json = payload.get("metadata_json")
        project_id = None
        project_name = None
        if isinstance(input_json, dict):
            project_id = str(input_json.get("project_id") or "").strip() or None
            project_name = str(input_json.get("project_name") or "").strip() or None
        if project_id is None and isinstance(metadata_json, dict):
            project_id = str(metadata_json.get("project_id") or "").strip() or None
        if project_name is None and isinstance(metadata_json, dict):
            project_name = str(metadata_json.get("project_name") or "").strip() or None
        payload["project_id"] = project_id
        payload["project_name"] = project_name
        return payload

    def _build_checklist_snapshot(self, *, run: dict[str, object]) -> dict[str, object] | None:
        if str(run.get("task_type") or "") != "agent_orchestration":
            return None
        input_json = run.get("input_json")
        if not isinstance(input_json, dict):
            return HarnessRunChecklistSnapshot().model_dump()

        items = input_json.get("task_checklist")
        if not isinstance(items, list):
            return HarnessRunChecklistSnapshot().model_dump()

        normalized_items: list[HarnessRunChecklistItem] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            active_form = str(item.get("active_form") or "").strip() or None
            normalized_items.append(
                HarnessRunChecklistItem(
                    item_id=str(item.get("item_id") or f"check_{index + 1}"),
                    content=content,
                    status=self._normalize_checklist_status(item.get("status")),
                    active_form=active_form,
                )
            )

        total_items = len(normalized_items)
        completed_items = sum(1 for item in normalized_items if item.status == "completed")

        return HarnessRunChecklistSnapshot(
            enabled=total_items > 0,
            total_items=total_items,
            open_items=total_items - completed_items,
            completed_items=completed_items,
            items=normalized_items,
        ).model_dump()

    @staticmethod
    def _first_incomplete_step_index(total_steps: int, completed_steps: set[int]) -> int | None:
        for index in range(total_steps):
            if index not in completed_steps:
                return index
        return None

    def _build_workflow_progress(
        self,
        *,
        run: dict[str, object],
        latest_approval: dict[str, object] | None,
        latest_verification: dict[str, object] | None,
        events: list[dict[str, object]] | None,
    ) -> dict[str, object] | None:
        if str(run.get("task_type") or "") != "agent_orchestration":
            return None
        input_json = run.get("input_json")
        if not isinstance(input_json, dict):
            return None
        graph_json = input_json.get("graph")
        if not isinstance(graph_json, dict):
            return None

        selected_agent_ids = self._normalize_string_list(input_json.get("selected_agent_ids"))
        filtered_graph = self._filter_graph_for_execution(
            graph_json,
            selected_agent_ids=selected_agent_ids,
        )
        try:
            ordered_agents, _, review_config = build_orchestration_execution_plan(filtered_graph)
        except Exception:
            return None

        loop_count = max(1, self._coerce_int(input_json.get("loop_count")) or 1)
        review_enabled = bool((review_config or {}).get("enabled", True))

        steps: list[HarnessWorkflowStep] = []
        step_index = 0
        for loop_number in range(1, loop_count + 1):
            for agent_config in ordered_agents:
                execution_id = str(agent_config.get("agent_id") or "").strip() or None
                node_id = (
                    str(agent_config.get("cluster_agent_id") or agent_config.get("agent_id") or "").strip() or None
                )
                label = str(agent_config.get("name") or node_id or f"step {step_index + 1}")
                steps.append(
                    HarnessWorkflowStep(
                        step_id=f"workflow_step_{step_index}",
                        step_index=step_index,
                        loop_number=loop_number,
                        label=label,
                        execution_id=execution_id,
                        node_id=node_id,
                        kind=self._workflow_step_kind_literal(agent_config),
                    )
                )
                step_index += 1

        total_steps = len(steps)
        if total_steps == 0:
            return HarnessWorkflowProgress(review_enabled=review_enabled).model_dump()

        run_status = str(run.get("status") or "").strip()
        verification_status = str((latest_verification or {}).get("status") or "").strip()
        approval_status = str((latest_approval or {}).get("status") or "").strip()
        approval_action_type = str((latest_approval or {}).get("action_type") or "").strip()
        approval_payload = self._as_object_dict(
            latest_approval.get("payload_json") if isinstance(latest_approval, dict) else None
        )

        completed_step_indices: set[int] = set()
        if run_status == "completed" or verification_status == "pass":
            completed_step_indices = set(range(total_steps))
        else:
            for event in events or []:
                if str(event.get("event_type") or "") != "orchestration.step_completed":
                    continue
                details = event.get("details_json")
                if not isinstance(details, dict):
                    continue
                completed_index = self._coerce_int(details.get("step_index"))
                if completed_index is None or completed_index < 0 or completed_index >= total_steps:
                    continue
                completed_step_indices.add(completed_index)

        blocked_step_index = None
        if approval_action_type == "orchestration_review":
            candidate = self._coerce_int(approval_payload.get("step_index"))
            if candidate is not None and 0 <= candidate < total_steps:
                blocked_step_index = candidate

        current_step_index = None
        workflow_status: Literal["idle", "pending", "running", "blocked", "completed", "failed"] = "idle"
        if run_status == "completed" or verification_status == "pass":
            workflow_status = "completed"
        elif approval_action_type == "orchestration_review" and approval_status in {"pending", "rejected"}:
            workflow_status = "blocked"
            current_step_index = blocked_step_index
        elif run_status in {"running", "approved", "resumed", "verifying"} or str(run.get("current_step") or "") == "executing_graph":
            workflow_status = "running"
            current_step_index = self._first_incomplete_step_index(total_steps, completed_step_indices)
        elif run_status == "failed":
            workflow_status = "blocked" if blocked_step_index is not None else "failed"
            current_step_index = (
                blocked_step_index
                if blocked_step_index is not None
                else self._first_incomplete_step_index(total_steps, completed_step_indices)
            )
        elif run_status in {"created", "queued", "waiting_approval"}:
            workflow_status = "pending"
            current_step_index = self._first_incomplete_step_index(total_steps, completed_step_indices)

        serialized_steps: list[dict[str, object]] = []
        for step in steps:
            status = "completed" if step.step_index in completed_step_indices else "pending"
            if status != "completed" and current_step_index == step.step_index:
                if workflow_status == "running":
                    status = "in_progress"
                elif workflow_status == "blocked":
                    status = "blocked"
            serialized_steps.append(step.model_copy(update={"status": status}).model_dump())

        current_step_label = None
        if current_step_index is not None and 0 <= current_step_index < len(serialized_steps):
            current_step_label = str(serialized_steps[current_step_index].get("label") or "") or None

        blocking_step_index = current_step_index if workflow_status == "blocked" else None
        blocking_step_label = current_step_label if workflow_status == "blocked" else None
        blocking_stage = (
            str(approval_payload.get("review_stage") or "").strip() or None
            if workflow_status == "blocked"
            else None
        )
        blocking_reason = None
        if workflow_status == "blocked":
            blocking_reason = (
                str((latest_approval or {}).get("comment") or "").strip()
                or str(approval_payload.get("review_output") or "").strip()
                or str((latest_approval or {}).get("reason") or "").strip()
                or None
            )

        return HarnessWorkflowProgress(
            enabled=True,
            status=self._normalize_workflow_status(workflow_status),
            total_steps=total_steps,
            completed_steps=sum(1 for step in serialized_steps if step.get("status") == "completed"),
            blocked_steps=sum(1 for step in serialized_steps if step.get("status") == "blocked"),
            review_enabled=review_enabled,
            current_step_index=current_step_index,
            current_step_label=current_step_label,
            blocking_step_index=blocking_step_index,
            blocking_step_label=blocking_step_label,
            blocking_stage=blocking_stage,
            blocking_reason=blocking_reason,
            steps=[HarnessWorkflowStep.model_validate(step) for step in serialized_steps],
        ).model_dump()

    def _build_runtime_state(
        self,
        *,
        run: dict[str, object],
        latest_approval: dict[str, object] | None,
        latest_verification: dict[str, object] | None,
        events: list[dict[str, object]],
    ) -> dict[str, object]:
        payload = self._as_object_dict(latest_approval.get("payload_json") if isinstance(latest_approval, dict) else None)
        metadata = self._as_object_dict(run.get("metadata_json") if isinstance(run, dict) else None)
        verification_artifacts = self._as_object_dict(
            latest_verification.get("artifacts_json") if isinstance(latest_verification, dict) else None
        )
        output_artifacts = self._as_object_dict(verification_artifacts.get("output_artifacts"))

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
            "agent_id": str(payload.get("agent_id") or self._as_object_dict((resumed_event or {}).get("details_json")).get("agent_id") or "") or None,
            "agent_name": str(payload.get("agent_name") or self._as_object_dict((resumed_event or {}).get("details_json")).get("agent_name") or "") or None,
            "step_index": self._coerce_int(payload.get("step_index") or self._as_object_dict((resumed_event or {}).get("details_json")).get("next_step_index")),
            "prefix_length": len(str(payload.get("partial_output") or "")) or self._coerce_int(self._as_object_dict((resumed_event or {}).get("details_json")).get("partial_length")) or 0,
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
            review=HarnessReviewState.model_validate(review_state),
            continuation=HarnessContinuationState.model_validate(continuation_state),
            research=HarnessResearchState(
                enabled=bool(cluster_ids),
                mode=research_mode,
                paper_count=paper_count,
                browser_preview_count=browser_preview_count,
                source_count=source_count,
                cluster_ids=cluster_ids,
            ),
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
                review_state_json=self._as_object_dict(normalized.get("review")),
                continuation_state_json=self._as_object_dict(normalized.get("continuation")),
                research_state_json=self._as_object_dict(normalized.get("research")),
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
        next_version = (self._coerce_int((latest or {}).get("version")) or 0) + 1
        review_payload = self._as_object_dict(runtime_state.get("review") if isinstance(runtime_state, dict) else None)
        stage = str(review_payload.get("stage") or "").strip() or None
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
                resolved_events = self.list_run_events(run_id=run_id, limit=500)
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
            merged = self._as_object_dict(current.get("review"))
            merged.update(review)
            current["review"] = merged
        if continuation:
            merged = self._as_object_dict(current.get("continuation"))
            merged.update(continuation)
            current["continuation"] = merged
        if research:
            merged = self._as_object_dict(current.get("research"))
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
        transition = self.governance_service.authorize_transition(
            current_status=str(existing.get("status") or ""),
            target_status=status,
            reason=str((details or {}).get("reason") or "") or None,
        )
        updated = self.run_store.update_run(run_id, status=status, **changes)
        if updated is None:
            return None
        event_details = build_lifecycle_event_details(
            run_id=run_id,
            actor=actor,
            contract_version="run.v1",
            from_status=transition.from_status or None,
            to_status=transition.to_status,
            triggered_by="run_service",
        )
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
        retry_count = self._coerce_int(run.get("retry_count")) or 0
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
        retry_count = self._coerce_int(source_run.get("retry_count")) or 0
        if retry_count >= int(policy.retry_budget):
            raise HarnessRetryNotAllowedError("Retry budget exhausted for this harness run")

        metadata = self._merge_metadata(
            self._normalize_metadata(
                self._as_object_dict(source_run.get("metadata_json") if isinstance(source_run, dict) else None)
            ),
            {
                "retried_from_run_id": run_id,
                "retry_requested_by": requested_by,
            },
        )
        input_json = self._as_object_dict(source_run.get("input_json"))
        if rejected_review:
            orchestration_resume = self._as_object_dict(input_json.get("orchestration_resume"))
            rollback_state = self._as_object_dict(orchestration_resume.get("rollback_state"))
            payload_json = self._as_object_dict(latest_approval.get("payload_json") if isinstance(latest_approval, dict) else None)
            review_stage = str(payload_json.get("review_stage") or "").strip()
            continue_mode = str(orchestration_resume.get("continue_mode") or "").strip()
            if rollback_state:
                orchestration_resume["state"] = rollback_state
            if review_stage == "cluster_research" or continue_mode == "discard_research_evidence":
                orchestration_resume["next_step_index"] = self._coerce_int(orchestration_resume.get("next_step_index")) or 0
                metadata["review_recovery_mode"] = "continue_without_research"
            elif review_stage == "agent_output_stream":
                orchestration_resume["next_step_index"] = (
                    self._coerce_int(payload_json.get("step_index"))
                    or self._coerce_int(orchestration_resume.get("next_step_index"))
                    or 0
                )
                orchestration_resume.pop("continuation", None)
                metadata["review_recovery_mode"] = "continue_from_stream_block"
            else:
                orchestration_resume["next_step_index"] = (
                    self._coerce_int(orchestration_resume.get("next_step_index")) or 0
                ) - 1
                if cast(int, orchestration_resume["next_step_index"]) < 0:
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
        resume_count = (self._coerce_int(run.get("resume_count")) or 0) + 1
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

    def resolve_approval_command(self, command: ApprovalResolutionCommand) -> dict[str, object] | None:
        approval_status = "approved" if command.approved else "rejected"
        return self.update_approval(
            command.approval_id,
            status=approval_status,
            resolved_by=command.resolved_by,
            comment=command.comment,
        )

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
            artifacts_json=self._as_object_dict(verification_result.get("artifacts")),
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

    def create_verification_command(self, command: VerificationRecordCommand) -> dict[str, object] | None:
        return self.create_verification(
            command.run_id,
            {
                "status": command.result_status,
                "checks_run": list(command.checks_run),
                "artifacts": dict(command.artifacts or {}),
                "summary": command.summary,
                "verification_profile": command.verification_profile,
            },
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

    def create_resolved_approval(
        self,
        *,
        run_id: str,
        action_type: str,
        reason: str | None,
        payload_json: dict[str, object],
        requested_by: str | None,
        status: str,
        resolved_by: str,
        comment: str | None,
    ) -> dict[str, object] | None:
        if self.approval_store is None:
            return None
        created = self.approval_store.create_approval(
            approval_id=f"ha_{uuid.uuid4()}",
            run_id=run_id,
            action_type=action_type,
            reason=reason,
            payload_json=payload_json,
            status=str(status or "approved"),
            requested_by=requested_by,
        )
        created_id = str(created.get("approval_id") or "").strip()
        if not created_id:
            return created
        return self.update_approval(
            created_id,
            status=str(status or "approved"),
            resolved_by=resolved_by,
            comment=comment,
        )

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
        detail = self._decorate_run_context(run)
        policy = get_policy(str(run.get("task_type") or ""))
        detail["policy"] = self._policy_summary(policy)
        detail["can_retry"] = self.can_retry_run(run_id)
        detail["latest_approval"] = self.get_latest_approval(run_id)
        detail["latest_verification"] = self.get_latest_verification(run_id)
        detail["events"] = self.list_run_events(run_id=run_id, limit=500)
        detail["runtime_state"] = self._load_persisted_runtime_state(run_id) or self.sync_runtime_state(
            run_id,
            run=detail,
            latest_approval=cast(dict[str, object] | None, detail["latest_approval"]),
            latest_verification=cast(dict[str, object] | None, detail["latest_verification"]),
            events=self._as_dict_list(detail["events"]),
            transition_type="detail_sync",
        )
        detail["workflow_progress"] = self._build_workflow_progress(
            run=detail,
            latest_approval=cast(dict[str, object] | None, detail["latest_approval"]),
            latest_verification=cast(dict[str, object] | None, detail["latest_verification"]),
            events=cast(list[dict[str, object]], self._as_object_list(detail["events"])),
        )
        detail["checklist_snapshot"] = self._build_checklist_snapshot(run=detail)
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
            summary = self._decorate_run_context(run)
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
                    latest_approval=cast(dict[str, object] | None, summary["latest_approval"]),
                    latest_verification=cast(dict[str, object] | None, summary["latest_verification"]),
                    transition_type="list_sync",
                )
            )
            summary_events = self.list_run_events(run_id=run_id, limit=500) if run_id else []
            summary["workflow_progress"] = self._build_workflow_progress(
                run=summary,
                latest_approval=cast(dict[str, object] | None, summary["latest_approval"]),
                latest_verification=cast(dict[str, object] | None, summary["latest_verification"]),
                events=summary_events,
            )
            summary["checklist_snapshot"] = self._build_checklist_snapshot(run=summary)
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
