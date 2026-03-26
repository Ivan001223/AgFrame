from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

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

    class _ApprovalService:
        def get_pending_approval(self, run_id: str):
            return {"run_id": run_id, "status": "pending", "action_type": "resume"}

        async def resolve(self, run_id: str, approved: bool, resolved_by: str, comment: str | None):
            return {"run_id": run_id, "status": "approved", "resolved_by": resolved_by}

    monkeypatch.setattr(harness_api, "get_approval_service", lambda: _ApprovalService())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-1/approval")

    assert response.status_code == 200
    assert response.json()["status"] == "pending"


def test_approve_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    called = {"resolved": None}

    class _ApprovalService:
        def get_pending_approval(self, run_id: str):
            return {"run_id": run_id, "status": "pending", "action_type": "resume"}

        async def resolve(self, run_id: str, approved: bool, resolved_by: str, comment: str | None):
            called["resolved"] = (run_id, approved, resolved_by, comment)
            return {"run_id": run_id, "status": "approved", "resolved_by": resolved_by}

    monkeypatch.setattr(harness_api, "get_approval_service", lambda: _ApprovalService())
    client = TestClient(app)

    response = client.post(
        "/harness/runs/hr-1/approval",
        json={"approved": True, "comment": "ok"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert called["resolved"] == ("hr-1", True, "u1", "ok")


def test_reject_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    called = {"resolved": None}

    class _ApprovalService:
        def get_pending_approval(self, run_id: str):
            return {"run_id": run_id, "status": "pending", "action_type": "resume"}

        async def resolve(self, run_id: str, approved: bool, resolved_by: str, comment: str | None):
            called["resolved"] = (run_id, approved, resolved_by, comment)
            return {"run_id": run_id, "status": "rejected", "resolved_by": resolved_by}

    monkeypatch.setattr(harness_api, "get_approval_service", lambda: _ApprovalService())
    client = TestClient(app)

    response = client.post(
        "/harness/runs/hr-1/approval",
        json={"approved": False, "comment": "stop"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert called["resolved"] == ("hr-1", False, "u1", "stop")
