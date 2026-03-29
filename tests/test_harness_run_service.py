import pytest

from app.harness.runtime.run_service import HarnessRunService
from app.harness.runtime.policy_registry import UnknownHarnessTaskTypeError


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
        "task_type": "session_resume_approval",
        "status": "approved",
        "policy_id": "session_resume_approval:v1",
        "input_json": {"session_id": "s1"},
        "metadata_json": None,
        "current_step": "resume_graph",
        "retry_count": 0,
        "resume_count": 0,
        "approval_required": True,
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
            return {"approval_id": "ha-1", "run_id": run_id, "status": "approved"}

    class _VerificationStore:
        def get_latest_by_run(self, run_id: str):
            assert run_id == "hr-1"
            return {"verification_id": "hv-1", "run_id": run_id, "status": "pass"}

    class _EventStore:
        def list_events(self, *, user_id: str | None = None, session_id: str | None = None, run_id: str | None = None, limit: int = 100):
            assert run_id == "hr-1"
            return [{"event_id": "he-1", "run_id": run_id, "event_type": "run.created"}]

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
    assert detail["policy"]["retry_budget"] == 0
    assert detail["can_retry"] is False
    assert listing[0]["latest_approval"]["approval_id"] == "ha-1"
    assert listing[0]["latest_verification"]["verification_id"] == "hv-1"


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
