import pytest

from app.harness.runtime.policy_registry import UnknownHarnessTaskTypeError
from app.harness.runtime.run_service import HarnessRunService


def test_create_document_ingest_run():
    created = {}

    class _Store:
        def create_run(self, **kwargs):
            created.update(kwargs)
            return {**kwargs, "created_at": 1, "updated_at": 1}

    service = HarnessRunService(run_store=_Store())
    run = service.create_run(
        user_id="u1",
        task_type="document_ingest",
        input_json={"file_path": "/tmp/a.pdf"},
        session_id=None,
        metadata_json=None,
    )

    assert run["task_type"] == "document_ingest"
    assert run["policy_id"] == "document_ingest:v1"
    assert run["status"] == "created"
    assert created["approval_required"] is False


def test_create_approval_required_run_creates_pending_approval():
    created = {}
    approvals = []
    events = []

    class _RunStore:
        def create_run(self, **kwargs):
            created.update(kwargs)
            return {**kwargs, "created_at": 1, "updated_at": 1}

    class _ApprovalStore:
        def create_approval(self, **kwargs):
            approvals.append(kwargs)
            return {**kwargs, "created_at": 1}

    class _EventStore:
        def create_event(self, **kwargs):
            events.append(kwargs)
            return {**kwargs, "created_at": 1}

    service = HarnessRunService(run_store=_RunStore(), approval_store=_ApprovalStore(), event_store=_EventStore())
    run = service.create_run(
        user_id="u1",
        task_type="session_resume_approval",
        input_json={"session_id": "s1"},
        session_id="s1",
        metadata_json=None,
    )

    assert run["task_type"] == "session_resume_approval"
    assert run["status"] == "waiting_approval"
    assert created["approval_required"] is True
    assert approvals[0]["run_id"] == run["run_id"]
    assert approvals[0]["status"] == "pending"
    assert approvals[0]["payload_json"]["session_id"] == "s1"
    assert [event["event_type"] for event in events] == ["run.created", "approval.requested"]


def test_status_transitions_and_verification_emit_events():
    state = {
        "run_id": "hr-1",
        "user_id": "u1",
        "session_id": "s1",
        "task_type": "session_resume_approval",
        "status": "created",
        "policy_id": "session_resume_approval:v1",
        "input_json": {"session_id": "s1"},
        "metadata_json": None,
        "current_step": None,
        "retry_count": 0,
        "resume_count": 0,
        "approval_required": True,
        "verification_status": None,
        "created_at": 1,
        "updated_at": 1,
        "finished_at": None,
    }
    events = []

    class _RunStore:
        def get_run(self, run_id: str):
            assert run_id == "hr-1"
            return dict(state)

        def update_run(self, run_id: str, **changes):
            assert run_id == "hr-1"
            state.update(changes)
            return dict(state)

    class _VerificationStore:
        def get_latest_by_run(self, run_id: str):
            return None

        def create_verification(self, **kwargs):
            return {**kwargs, "created_at": 2}

    class _EventStore:
        def create_event(self, **kwargs):
            events.append(kwargs)
            return {**kwargs, "created_at": 2}

    service = HarnessRunService(
        run_store=_RunStore(),
        verification_store=_VerificationStore(),
        event_store=_EventStore(),
    )

    service.mark_queued("hr-1")
    service.mark_running("hr-1")
    service.set_current_step("hr-1", "resume_graph")
    service.complete_with_verification(
        "hr-1",
        {
            "status": "pass",
            "checks_run": ["resume_execution"],
            "artifacts": {"session_id": "s1"},
            "summary": "resume ok",
        },
    )

    assert [event["event_type"] for event in events] == [
        "run.status_changed",
        "run.status_changed",
        "run.step_updated",
        "run.status_changed",
        "verification.recorded",
        "run.status_changed",
    ]
    assert events[-1]["details_json"]["to_status"] == "completed"


def test_get_run_detail_and_list_runs_include_latest_approval_and_verification():
    run = {
        "run_id": "hr-1",
        "user_id": "u1",
        "session_id": "s1",
        "task_type": "agent_orchestration",
        "status": "approved",
        "policy_id": "agent_orchestration:v1",
        "input_json": {
            "task": "Run a reviewed orchestration",
            "selected_agent_ids": ["agent-a", "agent-b"],
            "loop_count": 1,
            "task_checklist": [
                {"item_id": "check_1", "content": "Audit the current orchestration flow", "status": "completed"},
                {
                    "item_id": "check_2",
                    "content": "Implement the safer rollout path",
                    "status": "in_progress",
                    "active_form": "Implementing the safer rollout path",
                },
                {"item_id": "check_3", "content": "Verify the final behavior", "status": "pending"},
            ],
            "graph": {
                "agents": [
                    {"agent_id": "agent-a", "name": "Agent A", "model": "dev-stub"},
                    {"agent_id": "agent-b", "name": "Agent B", "model": "dev-stub"},
                ],
                "edges": [
                    {
                        "source_agent_id": "agent-a",
                        "target_agent_id": "agent-b",
                        "interaction": "handoff",
                    }
                ],
                "review_agent": {"enabled": True, "name": "Reviewer"},
            },
        },
        "metadata_json": {"review_recovery_mode": "continue_with_partial_stream_output"},
        "current_step": "resume_graph",
        "retry_count": 0,
        "resume_count": 0,
        "approval_required": False,
        "verification_status": None,
        "created_at": 1,
        "updated_at": 1,
        "finished_at": None,
    }

    class _RunStore:
        def get_run(self, run_id: str):
            assert run_id == "hr-1"
            return dict(run)

        def list_runs(self, *, user_id: str, limit: int = 50):
            assert user_id == "u1"
            return [dict(run)]

    class _ApprovalStore:
        def get_latest_by_run(self, run_id: str):
            assert run_id == "hr-1"
            return {
                "approval_id": "ha-1",
                "run_id": run_id,
                "status": "approved",
                "action_type": "orchestration_review",
                "payload_json": {
                    "review_stage": "agent_output_stream",
                    "agent_id": "agent-a",
                    "agent_name": "Agent A",
                    "review_output": "Safe to continue.",
                    "segment_index": 1,
                    "segment_count": 3,
                    "segment_start_char": 0,
                    "segment_end_char": 120,
                    "step_index": 2,
                    "partial_output": "Approved prefix",
                },
            }

    class _VerificationStore:
        def get_latest_by_run(self, run_id: str):
            assert run_id == "hr-1"
            return {
                "verification_id": "hv-1",
                "run_id": run_id,
                "status": "pass",
                "artifacts_json": {
                    "output_artifacts": {
                        "cluster-1": {
                            "research": {
                                "research_mode": "paper_first",
                                "papers": [{"title": "Paper 1"}, {"title": "Paper 2"}],
                                "browser_previews": [{"url": "https://example.com"}],
                                "sources": [{"url": "https://example.com"}, {"url": "https://arxiv.org/abs/1"}],
                            }
                        }
                    }
                },
            }

    class _EventStore:
        def list_events(self, *, user_id: str | None = None, session_id: str | None = None, run_id: str | None = None, limit: int = 100):
            assert run_id == "hr-1"
            return [
                {"event_id": "he-1", "run_id": run_id, "event_type": "run.created", "created_at": 1},
                {
                    "event_id": "he-2",
                    "run_id": run_id,
                    "event_type": "orchestration.stream_continuation_resumed",
                    "created_at": 2,
                    "details_json": {
                        "agent_id": "agent-a",
                        "agent_name": "Agent A",
                        "next_step_index": 2,
                        "partial_length": 15,
                    },
                },
                {
                    "event_id": "he-3",
                    "run_id": run_id,
                    "event_type": "orchestration.stream_continuation_completed",
                    "created_at": 3,
                    "details_json": {
                        "agent_id": "agent-a",
                    },
                },
            ]

    service = HarnessRunService(
        run_store=_RunStore(),
        approval_store=_ApprovalStore(),
        verification_store=_VerificationStore(),
        event_store=_EventStore(),
    )

    detail = service.get_run_detail("hr-1")
    listing = service.list_runs(user_id="u1")

    assert detail is not None
    assert detail["latest_approval"]["status"] == "approved"
    assert detail["latest_verification"]["status"] == "pass"
    assert detail["events"][0]["event_type"] == "run.created"
    assert detail["policy"]["retry_budget"] == 1
    assert detail["can_retry"] is False
    assert detail["runtime_state"]["review"]["stage"] == "agent_output_stream"
    assert detail["runtime_state"]["review"]["agent_name"] == "Agent A"
    assert detail["runtime_state"]["review"]["segment_count"] == 3
    assert detail["runtime_state"]["continuation"]["enabled"] is True
    assert detail["runtime_state"]["continuation"]["status"] == "completed"
    assert detail["runtime_state"]["continuation"]["step_index"] == 2
    assert detail["runtime_state"]["continuation"]["prefix_length"] == len("Approved prefix")
    assert detail["runtime_state"]["continuation"]["resumed_at"] == 2
    assert detail["runtime_state"]["continuation"]["completed_at"] == 3
    assert detail["runtime_state"]["research"]["enabled"] is True
    assert detail["runtime_state"]["research"]["mode"] == "paper_first"
    assert detail["runtime_state"]["research"]["paper_count"] == 2
    assert detail["runtime_state"]["research"]["browser_preview_count"] == 1
    assert detail["runtime_state"]["research"]["source_count"] == 2
    assert detail["runtime_state"]["research"]["cluster_ids"] == ["cluster-1"]
    assert listing[0]["latest_approval"]["approval_id"] == "ha-1"
    assert listing[0]["latest_verification"]["verification_id"] == "hv-1"
    assert listing[0]["runtime_state"]["review"]["stage"] == "agent_output_stream"
    assert listing[0]["runtime_state"]["continuation"]["enabled"] is True
    assert listing[0]["runtime_state"]["continuation"]["status"] == "completed"
    assert listing[0]["runtime_state"]["continuation"]["resumed_at"] == 2
    assert listing[0]["runtime_state"]["research"]["paper_count"] == 2
    assert detail["workflow_progress"]["enabled"] is True
    assert detail["workflow_progress"]["status"] == "completed"
    assert detail["workflow_progress"]["total_steps"] == 2
    assert detail["workflow_progress"]["completed_steps"] == 2
    assert detail["workflow_progress"]["steps"][0]["label"] == "Agent A"
    assert detail["workflow_progress"]["steps"][1]["label"] == "Agent B"
    assert detail["checklist_snapshot"]["enabled"] is True
    assert detail["checklist_snapshot"]["total_items"] == 3
    assert detail["checklist_snapshot"]["open_items"] == 2
    assert detail["checklist_snapshot"]["completed_items"] == 1
    assert detail["checklist_snapshot"]["items"][1]["active_form"] == "Implementing the safer rollout path"
    assert listing[0]["workflow_progress"]["status"] == "completed"
    assert listing[0]["workflow_progress"]["completed_steps"] == 2
    assert listing[0]["checklist_snapshot"]["enabled"] is True
    assert listing[0]["checklist_snapshot"]["total_items"] == 3
    assert listing[0]["checklist_snapshot"]["open_items"] == 2
    assert listing[0]["checklist_snapshot"]["items"][2]["content"] == "Verify the final behavior"


def test_get_run_detail_marks_blocked_workflow_step_from_review_approval():
    run = {
        "run_id": "hr-1",
        "user_id": "u1",
        "session_id": None,
        "task_type": "agent_orchestration",
        "status": "waiting_approval",
        "policy_id": "agent_orchestration:v1",
        "input_json": {
            "task": "Coordinate work",
            "selected_agent_ids": ["agent_a", "agent_b"],
            "loop_count": 1,
            "graph": {
                "agents": [
                    {"agent_id": "agent_a", "name": "Agent A", "model": "dev-stub"},
                    {"agent_id": "agent_b", "name": "Agent B", "model": "dev-stub"},
                ],
                "edges": [
                    {
                        "source_agent_id": "agent_a",
                        "target_agent_id": "agent_b",
                        "interaction": "handoff",
                    }
                ],
                "review_agent": {"enabled": True, "name": "Reviewer"},
            },
        },
        "metadata_json": None,
        "current_step": "executing_graph",
        "retry_count": 0,
        "resume_count": 0,
        "approval_required": False,
        "verification_status": None,
        "created_at": 1,
        "updated_at": 1,
        "finished_at": None,
    }

    class _RunStore:
        def get_run(self, run_id: str):
            return dict(run) if run_id == "hr-1" else None

    class _ApprovalStore:
        def get_latest_by_run(self, run_id: str):
            return {
                "approval_id": "ha-1",
                "run_id": run_id,
                "status": "pending",
                "action_type": "orchestration_review",
                "reason": "Awaiting review decision",
                "comment": "Please inspect this node output.",
                "payload_json": {
                    "agent_id": "agent_b",
                    "agent_name": "Agent B",
                    "review_stage": "agent_output_stream",
                    "step_index": 1,
                    "review_output": "Human confirmation required.",
                },
            }

    class _EventStore:
        def list_events(
            self,
            *,
            user_id: str | None = None,
            session_id: str | None = None,
            run_id: str | None = None,
            limit: int = 100,
        ):
            return [
                {
                    "event_id": "he-1",
                    "run_id": run_id,
                    "event_type": "orchestration.step_completed",
                    "details_json": {"step_index": 0, "agent_id": "agent_a", "agent_name": "Agent A"},
                    "created_at": 1,
                }
            ]

    service = HarnessRunService(
        run_store=_RunStore(),
        approval_store=_ApprovalStore(),
        event_store=_EventStore(),
    )

    detail = service.get_run_detail("hr-1")

    assert detail is not None
    assert detail["workflow_progress"]["status"] == "blocked"
    assert detail["workflow_progress"]["completed_steps"] == 1
    assert detail["workflow_progress"]["blocking_step_index"] == 1
    assert detail["workflow_progress"]["blocking_stage"] == "agent_output_stream"
    assert detail["workflow_progress"]["steps"][0]["status"] == "completed"
    assert detail["workflow_progress"]["steps"][1]["status"] == "blocked"


def test_filter_graph_for_execution_uses_selected_scope_orchestration_summary():
    filtered = HarnessRunService._filter_graph_for_execution(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Agent A"},
                {"agent_id": "agent_b", "name": "Agent B"},
            ],
            "edges": [
                {"source_agent_id": "agent_a", "target_agent_id": "agent_b", "interaction": "handoff"},
            ],
            "orchestration_summary": {
                "total_agent_count": 2,
                "start_agents": [{"agent_id": "agent_a", "agent_name": "Agent A"}],
            },
            "selected_scope_orchestration_summary": {
                "total_agent_count": 1,
                "start_agents": [{"agent_id": "agent_b", "agent_name": "Agent B"}],
            },
        },
        selected_agent_ids=["agent_b"],
    )

    assert [agent["agent_id"] for agent in filtered["agents"]] == ["agent_b"]
    assert filtered["edges"] == []
    assert filtered["orchestration_summary"]["total_agent_count"] == 1
    assert filtered["orchestration_summary"]["start_agents"][0]["agent_id"] == "agent_b"


def test_sync_runtime_state_persists_first_class_runtime_model():
    run = {
        "run_id": "hr-1",
        "user_id": "u1",
        "session_id": None,
        "task_type": "agent_orchestration",
        "status": "approved",
        "policy_id": "agent_orchestration:v1",
        "input_json": {"task": "Coordinate work"},
        "metadata_json": {"review_recovery_mode": "continue_with_partial_stream_output"},
        "current_step": "executing_graph",
        "retry_count": 0,
        "resume_count": 0,
        "approval_required": False,
        "verification_status": None,
        "created_at": 1,
        "updated_at": 1,
        "finished_at": None,
    }
    persisted = {}

    class _RunStore:
        def get_run(self, run_id: str):
            return dict(run) if run_id == "hr-1" else None

        def list_runs(self, *, user_id: str, limit: int = 50):
            return [dict(run)]

    class _ApprovalStore:
        def get_latest_by_run(self, run_id: str):
            return {
                "approval_id": "ha-1",
                "run_id": run_id,
                "status": "approved",
                "action_type": "orchestration_review",
                "payload_json": {
                    "review_stage": "agent_output_stream",
                    "agent_id": "agent-a",
                    "agent_name": "Agent A",
                    "review_output": "safe",
                    "check_count": 4,
                    "segment_index": 3,
                    "segment_count": 4,
                    "segment_start_char": 24,
                    "segment_end_char": 32,
                    "last_reviewed_char": 32,
                    "step_index": 1,
                    "partial_output": "approved prefix",
                },
            }

    class _VerificationStore:
        def get_latest_by_run(self, run_id: str):
            return {
                "verification_id": "hv-1",
                "run_id": run_id,
                "status": "pass",
                "artifacts_json": {
                    "output_artifacts": {
                        "cluster-1": {
                            "research": {
                                "research_mode": "paper_first",
                                "papers": [{"title": "Paper 1"}],
                                "browser_previews": [{"url": "https://example.com"}],
                                "sources": [{"url": "https://example.com"}],
                            }
                        }
                    }
                },
            }

    class _EventStore:
        def list_events(self, *, user_id: str | None = None, session_id: str | None = None, run_id: str | None = None, limit: int = 100):
            return []

    class _RuntimeStateStore:
        def get_by_run(self, run_id: str):
            return persisted.get(run_id)

        def list_by_run_ids(self, run_ids: list[str]):
            return {run_id: persisted[run_id] for run_id in run_ids if run_id in persisted}

        def upsert_state(self, *, run_id: str, review_state_json: dict[str, object], continuation_state_json: dict[str, object], research_state_json: dict[str, object]):
            payload = {
                "run_id": run_id,
                "review_state_json": review_state_json,
                "continuation_state_json": continuation_state_json,
                "research_state_json": research_state_json,
                "created_at": 1,
                "updated_at": 2,
            }
            persisted[run_id] = payload
            return payload

    service = HarnessRunService(
        run_store=_RunStore(),
        approval_store=_ApprovalStore(),
        verification_store=_VerificationStore(),
        event_store=_EventStore(),
        runtime_state_store=_RuntimeStateStore(),
    )

    runtime_state = service.sync_runtime_state("hr-1")
    listed = service.list_runs(user_id="u1")

    assert runtime_state["review"]["check_count"] == 4
    assert runtime_state["review"]["last_reviewed_char"] == 32
    assert persisted["hr-1"]["continuation_state_json"]["enabled"] is True
    assert persisted["hr-1"]["research_state_json"]["paper_count"] == 1
    assert listed[0]["runtime_state"]["review"]["agent_name"] == "Agent A"


def test_patch_runtime_state_appends_queryable_history_entries():
    run = {
        "run_id": "hr-1",
        "user_id": "u1",
        "session_id": None,
        "task_type": "agent_orchestration",
        "status": "running",
        "policy_id": "agent_orchestration:v1",
        "input_json": {"task": "Coordinate work"},
        "metadata_json": None,
        "current_step": "executing_graph",
        "retry_count": 0,
        "resume_count": 0,
        "approval_required": False,
        "verification_status": None,
        "created_at": 1,
        "updated_at": 1,
        "finished_at": None,
    }
    persisted = {}
    history = []

    class _RunStore:
        def get_run(self, run_id: str):
            return dict(run) if run_id == "hr-1" else None

    class _RuntimeStateStore:
        def get_by_run(self, run_id: str):
            return persisted.get(run_id)

        def list_by_run_ids(self, run_ids: list[str]):
            return {}

        def upsert_state(self, *, run_id: str, review_state_json: dict[str, object], continuation_state_json: dict[str, object], research_state_json: dict[str, object]):
            payload = {
                "run_id": run_id,
                "review_state_json": review_state_json,
                "continuation_state_json": continuation_state_json,
                "research_state_json": research_state_json,
                "created_at": 1,
                "updated_at": 2,
            }
            persisted[run_id] = payload
            return payload

    class _RuntimeStateHistoryStore:
        def get_latest_for_run(self, run_id: str):
            return history[-1] if history else None

        def append_history(self, *, run_id: str, version: int, transition_type: str, stage: str | None, runtime_state_json: dict[str, object]):
            entry = {
                "history_id": len(history) + 1,
                "run_id": run_id,
                "version": version,
                "transition_type": transition_type,
                "stage": stage,
                "runtime_state_json": runtime_state_json,
                "created_at": 1000 + len(history),
            }
            history.append(entry)
            return entry

        def list_for_run(self, *, run_id: str, limit: int = 100):
            return [entry for entry in history if entry["run_id"] == run_id][:limit]

    service = HarnessRunService(
        run_store=_RunStore(),
        runtime_state_store=_RuntimeStateStore(),
        runtime_state_history_store=_RuntimeStateHistoryStore(),
    )

    service.patch_runtime_state("hr-1", review={"stage": "agent_output_stream"}, transition_type="stream_event")
    service.patch_runtime_state("hr-1", continuation={"enabled": True, "status": "pending"}, transition_type="stream_blocked")

    entries = service.list_runtime_state_history(run_id="hr-1")

    assert len(entries) == 2
    assert entries[0]["version"] == 1
    assert entries[0]["transition_type"] == "stream_event"
    assert entries[0]["stage"] == "agent_output_stream"
    assert entries[1]["version"] == 2
    assert entries[1]["transition_type"] == "stream_blocked"


def test_create_retry_run_respects_budget_and_records_event():
    runs: dict[str, dict[str, object]] = {
        "hr-1": {
            "run_id": "hr-1",
            "user_id": "u1",
            "session_id": None,
            "task_type": "document_ingest",
            "status": "failed",
            "policy_id": "document_ingest:v1",
            "input_json": {"file_path": "/tmp/a.pdf"},
            "metadata_json": {"source": "manual"},
            "current_step": "ingest_document",
            "retry_count": 0,
            "resume_count": 0,
            "approval_required": False,
            "verification_status": "fail",
            "created_at": 1,
            "updated_at": 1,
            "finished_at": 1,
        }
    }
    created: list[dict[str, object]] = []
    events: list[dict[str, object]] = []

    class _RunStore:
        def get_run(self, run_id: str):
            return dict(runs[run_id]) if run_id in runs else None

        def create_run(self, **kwargs):
            created.append(kwargs)
            run = {**kwargs, "created_at": 2, "updated_at": 2, "finished_at": None}
            runs[str(kwargs["run_id"])] = run
            return run

        def update_run(self, run_id: str, **changes):
            runs[run_id].update(changes)
            return dict(runs[run_id])

    class _EventStore:
        def create_event(self, **kwargs):
            events.append(kwargs)
            return {**kwargs, "created_at": 2}

    service = HarnessRunService(run_store=_RunStore(), event_store=_EventStore())

    retried = service.create_retry_run("hr-1", requested_by="u1")

    assert retried["retry_count"] == 1
    assert created[0]["retry_count"] == 1
    assert created[0]["metadata_json"]["retried_from_run_id"] == "hr-1"
    assert created[0]["metadata_json"]["retry_requested_by"] == "u1"
    assert events[-1]["event_type"] == "run.retry_requested"
    assert service.can_retry_run("hr-1") is True
    assert service.can_retry_run(str(retried["run_id"])) is False


def test_create_retry_run_allows_rejected_orchestration_review_resume():
    runs: dict[str, dict[str, object]] = {
        "hr-1": {
            "run_id": "hr-1",
            "user_id": "u1",
            "session_id": None,
            "task_type": "agent_orchestration",
            "status": "rejected",
            "policy_id": "agent_orchestration:v1",
            "input_json": {
                "task": "Initial task",
                "orchestration_resume": {
                    "next_step_index": 2,
                    "state": {"task": "Initial task", "agent_outputs": {"agent_a": "done"}},
                    "rollback_state": {"task": "Initial task", "agent_outputs": {"agent_a": "safe"}},
                },
            },
            "metadata_json": {"source": "studio"},
            "current_step": "executing_graph",
            "retry_count": 0,
            "resume_count": 0,
            "approval_required": False,
            "verification_status": None,
            "created_at": 1,
            "updated_at": 1,
            "finished_at": 1,
        }
    }
    created: list[dict[str, object]] = []

    class _RunStore:
        def get_run(self, run_id: str):
            return dict(runs[run_id]) if run_id in runs else None

        def create_run(self, **kwargs):
            created.append(kwargs)
            run = {**kwargs, "created_at": 2, "updated_at": 2, "finished_at": None}
            runs[str(kwargs["run_id"])] = run
            return run

        def update_run(self, run_id: str, **changes):
            runs[run_id].update(changes)
            return dict(runs[run_id])

    class _ApprovalStore:
        def get_latest_by_run(self, run_id: str):
            if run_id == "hr-1":
                return {
                    "approval_id": "ha-1",
                    "run_id": run_id,
                    "action_type": "orchestration_review",
                    "status": "rejected",
                    "comment": "Use a safer rollout path.",
                }
            return None

    service = HarnessRunService(run_store=_RunStore(), approval_store=_ApprovalStore())

    assert service.can_retry_run("hr-1") is True
    retried = service.create_retry_run("hr-1", requested_by="u1")

    assert retried["retry_count"] == 1
    assert created[0]["input_json"]["orchestration_resume"]["next_step_index"] == 1
    assert created[0]["input_json"]["orchestration_resume"]["state"]["agent_outputs"]["agent_a"] == "safe"
    assert "Use a safer rollout path." in created[0]["input_json"]["task"]


def test_create_retry_run_continues_without_research_for_rejected_cluster_research_review():
    runs: dict[str, dict[str, object]] = {
        "hr-1": {
            "run_id": "hr-1",
            "user_id": "u1",
            "session_id": None,
            "task_type": "agent_orchestration",
            "status": "rejected",
            "policy_id": "agent_orchestration:v1",
            "input_json": {
                "task": "Initial task",
                "orchestration_resume": {
                    "next_step_index": 2,
                    "continue_mode": "discard_research_evidence",
                    "state": {"task": "Initial task", "agent_outputs": {"cluster_a": "done with research"}},
                    "rollback_state": {"task": "Initial task", "agent_outputs": {"cluster_a": "safe summary only"}},
                },
            },
            "metadata_json": {"source": "studio"},
            "current_step": "executing_graph",
            "retry_count": 0,
            "resume_count": 0,
            "approval_required": False,
            "verification_status": None,
            "created_at": 1,
            "updated_at": 1,
            "finished_at": 1,
        }
    }
    created: list[dict[str, object]] = []

    class _RunStore:
        def get_run(self, run_id: str):
            return dict(runs[run_id]) if run_id in runs else None

        def create_run(self, **kwargs):
            created.append(kwargs)
            run = {**kwargs, "created_at": 2, "updated_at": 2, "finished_at": None}
            runs[str(kwargs["run_id"])] = run
            return run

        def update_run(self, run_id: str, **changes):
            runs[run_id].update(changes)
            return dict(runs[run_id])

    class _ApprovalStore:
        def get_latest_by_run(self, run_id: str):
            if run_id == "hr-1":
                return {
                    "approval_id": "ha-1",
                    "run_id": run_id,
                    "action_type": "orchestration_review",
                    "status": "rejected",
                    "comment": "Skip external evidence and continue.",
                    "payload_json": {"review_stage": "cluster_research"},
                }
            return None

    service = HarnessRunService(run_store=_RunStore(), approval_store=_ApprovalStore())

    retried = service.create_retry_run("hr-1", requested_by="u1")

    assert retried["retry_count"] == 1
    assert created[0]["input_json"]["orchestration_resume"]["next_step_index"] == 2
    assert created[0]["input_json"]["orchestration_resume"]["state"]["agent_outputs"]["cluster_a"] == "safe summary only"
    assert "Skip external evidence and continue." in created[0]["input_json"]["task"]
    assert created[0]["metadata_json"]["review_recovery_mode"] == "continue_without_research"


def test_create_retry_run_restarts_blocked_stream_step_from_safe_state():
    runs: dict[str, dict[str, object]] = {
        "hr-1": {
            "run_id": "hr-1",
            "user_id": "u1",
            "session_id": None,
            "task_type": "agent_orchestration",
            "status": "rejected",
            "policy_id": "agent_orchestration:v1",
            "input_json": {
                "task": "Initial task",
                "orchestration_resume": {
                    "next_step_index": 3,
                    "state": {"task": "Initial task", "agent_outputs": {"agent_b": "partial"}},
                    "rollback_state": {"task": "Initial task", "agent_outputs": {"agent_a": "safe"}},
                    "continuation": {"agent_id": "agent_b", "partial_output": "safe partial"},
                },
            },
            "metadata_json": {"source": "studio"},
            "current_step": "executing_graph",
            "retry_count": 0,
            "resume_count": 0,
            "approval_required": False,
            "verification_status": None,
            "created_at": 1,
            "updated_at": 1,
            "finished_at": 1,
        }
    }
    created: list[dict[str, object]] = []

    class _RunStore:
        def get_run(self, run_id: str):
            return dict(runs[run_id]) if run_id in runs else None

        def create_run(self, **kwargs):
            created.append(kwargs)
            run = {**kwargs, "created_at": 2, "updated_at": 2, "finished_at": None}
            runs[str(kwargs["run_id"])] = run
            return run

        def update_run(self, run_id: str, **changes):
            runs[run_id].update(changes)
            return dict(runs[run_id])

    class _ApprovalStore:
        def get_latest_by_run(self, run_id: str):
            if run_id == "hr-1":
                return {
                    "approval_id": "ha-1",
                    "run_id": run_id,
                    "action_type": "orchestration_review",
                    "status": "rejected",
                    "comment": "Take a safer continuation.",
                    "payload_json": {"review_stage": "agent_output_stream", "step_index": 2},
                }
            return None

    service = HarnessRunService(run_store=_RunStore(), approval_store=_ApprovalStore())

    retried = service.create_retry_run("hr-1", requested_by="u1")

    assert retried["retry_count"] == 1
    assert created[0]["input_json"]["orchestration_resume"]["next_step_index"] == 2
    assert created[0]["input_json"]["orchestration_resume"]["state"]["agent_outputs"]["agent_a"] == "safe"
    assert created[0]["metadata_json"]["review_recovery_mode"] == "continue_from_stream_block"
    assert "Take a safer continuation." in created[0]["input_json"]["task"]


def test_create_run_rejects_unknown_task_type():
    class _Store:
        def create_run(self, **kwargs):
            return kwargs

    service = HarnessRunService(run_store=_Store())

    with pytest.raises(UnknownHarnessTaskTypeError, match="Unknown harness task_type"):
        service.create_run(
            user_id="u1",
            task_type="unknown_task",
            input_json={},
            session_id=None,
            metadata_json=None,
        )
