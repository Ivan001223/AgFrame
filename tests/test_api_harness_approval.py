from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.platform.governance.commands import ApprovalResolutionCommand
from app.server.api import harness as harness_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


def test_get_harness_approval(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _RunService:
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u1", "status": "waiting_approval"}

    class _ApprovalService:
        def get_pending_approval(self, run_id: str):
            return {"run_id": run_id, "status": "pending", "action_type": "resume"}

        async def resolve(self, command: ApprovalResolutionCommand):
            assert isinstance(command, ApprovalResolutionCommand)
            return {"run_id": command.run_id, "status": "approved", "resolved_by": command.resolved_by}

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _RunService())
    monkeypatch.setattr(harness_api, "get_approval_service", lambda: _ApprovalService())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-1/approval")

    assert response.status_code == 200
    assert response.json()["status"] == "pending"


def test_get_harness_approval_forbidden(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    class _RunService:
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u2", "status": "waiting_approval"}

    class _ApprovalService:
        def get_pending_approval(self, run_id: str):
            raise AssertionError("approval lookup should not happen for forbidden runs")

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _RunService())
    monkeypatch.setattr(harness_api, "get_approval_service", lambda: _ApprovalService())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-2/approval")

    assert response.status_code == 403


def test_approve_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    called = {"resolved": None}

    class _RunService:
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u1", "status": "waiting_approval"}

    class _ApprovalService:
        def get_pending_approval(self, run_id: str):
            return {"run_id": run_id, "status": "pending", "action_type": "resume"}

        async def resolve(self, command: ApprovalResolutionCommand):
            called["resolved"] = command
            return {"run_id": command.run_id, "status": "approved", "resolved_by": command.resolved_by}

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _RunService())
    monkeypatch.setattr(harness_api, "get_approval_service", lambda: _ApprovalService())
    client = TestClient(app)

    response = client.post(
        "/harness/runs/hr-1/approval",
        json={"approved": True, "comment": "ok"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert called["resolved"].run_id == "hr-1"
    assert called["resolved"].approved is True
    assert called["resolved"].resolved_by == "u1"
    assert called["resolved"].comment == "ok"


def test_approve_harness_run_forbidden(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    class _RunService:
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u2", "status": "waiting_approval"}

    class _ApprovalService:
        async def resolve(self, command: ApprovalResolutionCommand):
            raise AssertionError("approval resolution should not happen for forbidden runs")

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _RunService())
    monkeypatch.setattr(harness_api, "get_approval_service", lambda: _ApprovalService())
    client = TestClient(app)

    response = client.post(
        "/harness/runs/hr-2/approval",
        json={"approved": True, "comment": "ok"},
    )

    assert response.status_code == 403


def test_reject_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    called = {"resolved": None}

    class _RunService:
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u1", "status": "waiting_approval"}

    class _ApprovalService:
        def get_pending_approval(self, run_id: str):
            return {"run_id": run_id, "status": "pending", "action_type": "resume"}

        async def resolve(self, command: ApprovalResolutionCommand):
            called["resolved"] = command
            return {"run_id": command.run_id, "status": "rejected", "resolved_by": command.resolved_by}

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _RunService())
    monkeypatch.setattr(harness_api, "get_approval_service", lambda: _ApprovalService())
    client = TestClient(app)

    response = client.post(
        "/harness/runs/hr-1/approval",
        json={"approved": False, "comment": "stop"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert called["resolved"].run_id == "hr-1"
    assert called["resolved"].approved is False
    assert called["resolved"].resolved_by == "u1"
    assert called["resolved"].comment == "stop"
