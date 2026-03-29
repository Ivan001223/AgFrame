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
    def list_policies(self):
        return [
            {
                "policy_id": "document_ingest:v1",
                "task_type": "document_ingest",
                "approval_required": False,
                "allowed_tools": ["document_ingest"],
                "verification_profile": "document_ingest_basic",
                "retry_budget": 1,
            }
        ]

    def list_runs(self, *, user_id: str, limit: int = 50):
        return [
            {
                "run_id": "hr-1",
                "user_id": user_id,
                "status": "queued",
                "task_type": "document_ingest",
                "policy": {"retry_budget": 1},
                "can_retry": False,
                "latest_approval": {"approval_id": "ha-1", "status": "pending"},
                "latest_verification": None,
            }
        ]

    def get_run(self, run_id: str):
        return {"run_id": run_id, "user_id": "u1", "status": "queued", "task_type": "document_ingest"}

    def get_run_detail(self, run_id: str):
        return {
            "run_id": run_id,
            "user_id": "u1",
            "status": "queued",
            "task_type": "document_ingest",
            "policy": {"retry_budget": 1},
            "can_retry": False,
            "latest_approval": {"approval_id": "ha-1", "status": "pending"},
            "latest_verification": {"verification_id": "hv-1", "status": "pass"},
            "events": [{"event_id": "he-1", "event_type": "run.created"}],
        }

    def get_latest_verification(self, run_id: str):
        return {"verification_id": "hv-1", "run_id": run_id, "status": "pass"}

    def list_run_events(self, *, run_id: str, user_id: str | None = None, limit: int = 100):
        return [
            {
                "event_id": "he-1",
                "run_id": run_id,
                "user_id": user_id or "u1",
                "event_type": "run.created",
            }
        ]


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
    assert response.json()["runs"][0]["policy"]["retry_budget"] == 1


def test_get_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-1")

    assert response.status_code == 200
    assert response.json()["run_id"] == "hr-1"
    assert response.json()["latest_approval"]["status"] == "pending"
    assert response.json()["policy"]["retry_budget"] == 1


def test_list_harness_policies(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/policies")

    assert response.status_code == 200
    assert response.json()["policies"][0]["task_type"] == "document_ingest"


def test_get_harness_verification(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-1/verification")

    assert response.status_code == 200
    assert response.json()["verification_id"] == "hv-1"


def test_retry_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    queued = {}

    class _RetryService(_ServiceListGet):
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u1", "status": "failed", "task_type": "document_ingest"}

        def create_retry_run(self, run_id: str, *, requested_by: str):
            queued["created"] = (run_id, requested_by)
            return {
                "run_id": "hr-2",
                "user_id": "u1",
                "task_type": "document_ingest",
                "status": "created",
                "approval_required": False,
            }

        def mark_queued(self, run_id: str):
            queued["marked"] = run_id
            return {"run_id": run_id, "status": "queued", "task_type": "document_ingest"}

        def get_run_detail(self, run_id: str):
            return {
                "run_id": run_id,
                "user_id": "u1",
                "task_type": "document_ingest",
                "status": "queued",
                "policy": {"retry_budget": 1},
                "can_retry": False,
                "latest_approval": None,
                "latest_verification": None,
                "events": [],
            }

    async def _enqueue(run_id: str):
        queued["enqueued"] = run_id
        return "job-1"

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _RetryService())
    monkeypatch.setattr(harness_api, "enqueue_harness_run", _enqueue)
    client = TestClient(app)

    response = client.post("/harness/runs/hr-1/retry")

    assert response.status_code == 200
    assert response.json()["run_id"] == "hr-2"
    assert queued["created"] == ("hr-1", "u1")
    assert queued["marked"] == "hr-2"
    assert queued["enqueued"] == "hr-2"


def test_retry_harness_run_rejects_when_not_allowed(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _RetryService(_ServiceListGet):
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u1", "status": "completed", "task_type": "document_ingest"}

        def create_retry_run(self, run_id: str, *, requested_by: str):
            raise harness_api.HarnessRetryNotAllowedError("Retry budget exhausted for this harness run")

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _RetryService())
    client = TestClient(app)

    response = client.post("/harness/runs/hr-1/retry")

    assert response.status_code == 400


def test_list_harness_run_events(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-1/events")

    assert response.status_code == 200
    assert response.json()["events"][0]["event_type"] == "run.created"


def test_get_harness_run_forbidden(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    class _OtherUserService(_ServiceListGet):
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u2", "status": "queued"}

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _OtherUserService())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-2")

    assert response.status_code == 403


def test_get_harness_run_allows_admin(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U(username="admin", role="admin")

    class _OtherUserService(_ServiceListGet):
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u2", "status": "queued"}

        def get_run_detail(self, run_id: str):
            return {
                "run_id": run_id,
                "user_id": "u2",
                "status": "queued",
                "latest_approval": None,
                "latest_verification": None,
                "events": [],
            }

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _OtherUserService())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-2")

    assert response.status_code == 200
    assert response.json()["user_id"] == "u2"


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


def test_create_harness_run_rejects_unknown_task_type(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.post(
        "/harness/runs",
        json={"task_type": "unknown_task", "input": {}},
    )

    assert response.status_code == 422
