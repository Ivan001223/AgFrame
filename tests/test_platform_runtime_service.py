from app.platform.contracts.runtime_protocol import RuntimeCommandV1
from app.platform.runtime.service import RuntimeApplicationService


class _FakeRunService:
    def __init__(self, run: dict | None = None):
        self._run = run
        self._status = run.get("status") if run else None
        self.mark_running_calls: list[str] = []
        self.mark_failed_calls: list[str] = []

    def get_run(self, run_id: str) -> dict | None:
        if self._run is None:
            return None
        return {**self._run, "status": self._status}

    def mark_running(self, run_id: str) -> dict | None:
        self.mark_running_calls.append(run_id)
        self._status = "running"
        return self.get_run(run_id)

    def mark_failed(self, run_id: str, **kwargs) -> dict | None:
        self.mark_failed_calls.append(run_id)
        self._status = "failed"
        return self.get_run(run_id)


def _make_run(run_id: str = "hr-1", *, status: str = "queued", task_type: str = "agent_orchestration") -> dict:
    return {
        "run_id": run_id,
        "user_id": "u1",
        "session_id": "s1",
        "task_type": task_type,
        "status": status,
        "policy_id": f"{task_type}:v1",
        "input_json": {"graph": {}, "task": "do work"},
        "metadata_json": None,
    }


def test_start_command_returns_execution_ready_with_plan():
    fake = _FakeRunService(_make_run(status="queued"))
    service = RuntimeApplicationService(run_service=fake)
    command = RuntimeCommandV1(
        command_id="cmd-start",
        run_id="hr-1",
        command_type="start",
        payload={"task_type": "agent_orchestration"},
    )

    result = service.accept(command)

    assert result.command_id == "cmd-start"
    assert result.result_type == "execution_ready"
    assert result.resumable is True
    plan = result.payload["plan"]
    assert plan["governance_phase"] == "load_run_and_authorize"
    assert plan["runtime_phase"] == "execute_runtime_command"
    assert result.payload["task_type"] == "agent_orchestration"


def test_start_command_rejects_terminal_run():
    fake = _FakeRunService(_make_run(status="completed"))
    service = RuntimeApplicationService(run_service=fake)
    command = RuntimeCommandV1(
        command_id="cmd-start",
        run_id="hr-1",
        command_type="start",
        payload={},
    )

    result = service.accept(command)

    assert result.result_type == "rejected"
    assert result.resumable is False
    assert "terminal state" in result.payload["reason"]


def test_resume_command_parses_resume_point():
    fake = _FakeRunService(_make_run(status="running"))
    service = RuntimeApplicationService(run_service=fake)
    command = RuntimeCommandV1(
        command_id="cmd-resume",
        run_id="hr-1",
        command_type="resume",
        payload={
            "resume_point": {
                "next_step_index": 3,
                "rollback_state": {"agent_outputs": {"a": "done"}},
            },
        },
    )

    result = service.accept(command)

    assert result.result_type == "resume_ready"
    assert result.payload["next_step_index"] == 3
    assert result.payload["has_rollback_state"] is True
    assert result.resumable is True


def test_resume_command_without_resume_point_defaults_to_zero():
    fake = _FakeRunService(_make_run(status="running"))
    service = RuntimeApplicationService(run_service=fake)
    command = RuntimeCommandV1(
        command_id="cmd-resume",
        run_id="hr-1",
        command_type="resume",
        payload={},
    )

    result = service.accept(command)

    assert result.result_type == "resume_ready"
    assert result.payload["next_step_index"] == 0
    assert result.payload["has_rollback_state"] is False


def test_step_command_returns_step_acknowledged():
    fake = _FakeRunService(_make_run(status="running"))
    service = RuntimeApplicationService(run_service=fake)
    command = RuntimeCommandV1(
        command_id="cmd-step",
        run_id="hr-1",
        command_type="step",
        payload={"step_index": 2},
    )

    result = service.accept(command)

    assert result.result_type == "step_acknowledged"
    assert result.payload["step_index"] == 2


def test_cancel_command_marks_run_failed():
    fake = _FakeRunService(_make_run(status="running"))
    service = RuntimeApplicationService(run_service=fake)
    command = RuntimeCommandV1(
        command_id="cmd-cancel",
        run_id="hr-1",
        command_type="cancel",
        payload={},
    )

    result = service.accept(command)

    assert result.result_type == "cancelled"
    assert result.resumable is False
    assert fake.mark_failed_calls == ["hr-1"]


def test_cancel_command_rejects_terminal_run():
    fake = _FakeRunService(_make_run(status="failed"))
    service = RuntimeApplicationService(run_service=fake)
    command = RuntimeCommandV1(
        command_id="cmd-cancel",
        run_id="hr-1",
        command_type="cancel",
        payload={},
    )

    result = service.accept(command)

    assert result.result_type == "rejected"
    assert fake.mark_failed_calls == []


def test_run_not_found_returns_error():
    fake = _FakeRunService(None)
    service = RuntimeApplicationService(run_service=fake)
    command = RuntimeCommandV1(
        command_id="cmd-start",
        run_id="hr-missing",
        command_type="start",
        payload={},
    )

    result = service.accept(command)

    assert result.result_type == "error"
    assert result.error_type == "run_not_found"
    assert result.resumable is False


def test_unsupported_command_type_returns_rejected():
    fake = _FakeRunService(_make_run(status="queued"))
    service = RuntimeApplicationService(run_service=fake)
    command = RuntimeCommandV1(
        command_id="cmd-bad",
        run_id="hr-1",
        command_type="invalid_type",
        payload={},
    )

    result = service.accept(command)

    assert result.result_type == "rejected"
    assert "unsupported command_type" in result.payload["reason"]
