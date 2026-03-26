from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import harness as harness_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


class _ServiceListGet:
    def list_runs(self, *, user_id: str, limit: int = 50):
        return [{"run_id": "hr-1", "user_id": user_id, "status": "queued"}]

    def get_run(self, run_id: str):
        return {"run_id": run_id, "user_id": "u1", "status": "queued"}


async def _noop_enqueue(run_id: str):
    return "job-1"


def test_list_harness_runs(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/runs")

    assert response.status_code == 200
    assert response.json()["runs"][0]["run_id"] == "hr-1"


def test_get_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-1")

    assert response.status_code == 200
    assert response.json()["run_id"] == "hr-1"


def test_get_harness_run_not_found(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _MissingService(_ServiceListGet):
        def get_run(self, run_id: str):
            return None

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _MissingService())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-missing")

    assert response.status_code == 404


def test_create_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    queued = {}

    class _Service:
        def create_run(self, **kwargs):
            return {
                "run_id": "hr-1",
                "task_type": kwargs["task_type"],
                "status": "created",
                "policy_id": "document_ingest:v1",
                "approval_required": False,
            }

        def mark_queued(self, run_id: str):
            queued["marked"] = run_id
            return {
                "run_id": run_id,
                "status": "queued",
            }

    async def _enqueue(run_id: str):
        queued["run_id"] = run_id
        return "job-1"

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _Service())
    monkeypatch.setattr(harness_api, "enqueue_harness_run", _enqueue)
    client = TestClient(app)

    response = client.post(
        "/harness/runs",
        json={"task_type": "document_ingest", "input": {"file_path": "/tmp/a.pdf"}},
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == "hr-1"
    assert queued["run_id"] == "hr-1"
    assert queued["marked"] == "hr-1"
    assert response.json()["status"] == "queued"


def test_create_harness_run_with_approval_does_not_enqueue(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    called = {"enqueue": False}

    class _ApprovalRunService:
        def create_run(self, **kwargs):
            return {
                "run_id": "hr-2",
                "task_type": kwargs["task_type"],
                "status": "created",
                "policy_id": "session_resume_approval:v1",
                "approval_required": True,
            }

        def mark_queued(self, run_id: str):
            raise AssertionError("mark_queued should not be called for approval-required runs")

    async def _enqueue(run_id: str):
        called["enqueue"] = True
        return "job-1"

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ApprovalRunService())
    monkeypatch.setattr(harness_api, "enqueue_harness_run", _enqueue)
    client = TestClient(app)

    response = client.post(
        "/harness/runs",
        json={"task_type": "session_resume_approval", "input": {"session_id": "s1"}},
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == "hr-2"
    assert response.json()["status"] == "created"
    assert called["enqueue"] is False
