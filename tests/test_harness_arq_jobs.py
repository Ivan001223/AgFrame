import pytest

from app.infrastructure.queue import arq_jobs


@pytest.mark.anyio
async def test_run_harness_task_completes_document_ingest(monkeypatch):
    events = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "document_ingest",
                "input_json": {"file_path": "/tmp/a.pdf"},
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", run_id, step))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"]))

    class _Rag:
        def add_knowledge_base(self, file_path: str, user_id: str | None = None):
            return {"ok": True, "stage": "done"}

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "get_rag_engine", lambda: _Rag())

    ok = await arq_jobs.run_harness_task({}, "hr-1")

    assert ok is True
    assert events[0] == ("running", "hr-1")
    assert ("step", "hr-1", "ingest_document") in events
    assert events[-1] == ("completed", "pass")


@pytest.mark.anyio
async def test_run_harness_task_completes_session_resume_approval(monkeypatch):
    events = []
    persisted = {}

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "session_resume_approval",
                "input_json": {"session_id": "s1"},
                "user_id": "u1",
                "session_id": "s1",
                "status": "queued",
            }

        def mark_resumed(self, run_id: str):
            events.append(("resumed", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", run_id, step))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"], verification_result["artifacts"]["interrupted"]))

    class _CheckpointAdapter:
        async def load(self, session_id: str):
            return {
                "checkpoint": {
                    "interrupted": False,
                    "action_required": {"approved": True},
                }
            }

    class _GraphResumeService:
        async def resume_approved_session(self, *, session_id: str, checkpoint: dict[str, object]):
            assert session_id == "s1"
            assert checkpoint["checkpoint"]["action_required"]["approved"] is True
            return {
                "ok": True,
                "interrupted": False,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "done"},
                ],
            }

    def _persist(*, user_id: str, session_id: str, messages: list[dict[str, object]], background_tasks=None, title=None):
        persisted["user_id"] = user_id
        persisted["session_id"] = session_id
        persisted["messages"] = messages

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "CheckpointAdapter", lambda: _CheckpointAdapter())
    monkeypatch.setattr(arq_jobs, "GraphResumeService", lambda: _GraphResumeService())
    monkeypatch.setattr(arq_jobs, "persist_session_messages", _persist)

    ok = await arq_jobs.run_harness_task({}, "hr-approved")

    assert ok is True
    assert ("step", "hr-approved", "load_checkpoint") in events
    assert ("step", "hr-approved", "resume_graph") in events
    assert ("resumed", "hr-approved") in events
    assert events[-1] == ("completed", "pass", False)
    assert persisted["session_id"] == "s1"
    assert persisted["messages"][-1]["content"] == "done"


@pytest.mark.anyio
async def test_run_harness_task_marks_failed_on_exception(monkeypatch):
    events = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "document_ingest",
                "input_json": {"file_path": "/tmp/a.pdf"},
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", run_id, step))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"], verification_result["artifacts"]["stage"]))

    class _Rag:
        def add_knowledge_base(self, file_path: str, user_id: str | None = None):
            raise RuntimeError("boom")

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "get_rag_engine", lambda: _Rag())

    ok = await arq_jobs.run_harness_task({}, "hr-exception")

    assert ok is False
    assert events[0] == ("running", "hr-exception")
    assert events[-1][0] == "completed"
    assert events[-1][1] == "fail"
    assert events[-1][2] == "exception"


@pytest.mark.anyio
async def test_run_harness_task_marks_failed_for_unsupported_task(monkeypatch):
    events = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "unknown_task",
                "input_json": {},
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"], verification_result["artifacts"]["stage"]))

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())

    ok = await arq_jobs.run_harness_task({}, "hr-unsupported")

    assert ok is False
    assert events[0] == ("running", "hr-unsupported")
    assert events[-1] == ("completed", "fail", "unsupported_task_type")


@pytest.mark.anyio
async def test_resume_harness_task_delegates_to_run(monkeypatch):
    called = {"run_id": None}

    async def _run(ctx, run_id: str):
        called["run_id"] = run_id
        return True

    monkeypatch.setattr(arq_jobs, "run_harness_task", _run)

    ok = await arq_jobs.resume_harness_task({}, "hr-2")

    assert ok is True
    assert called["run_id"] == "hr-2"
