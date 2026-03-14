from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import interrupt as interrupt_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


def test_interrupt_endpoints(monkeypatch: pytest.MonkeyPatch):
    store: dict[str, dict[str, Any]] = {}

    class _Store:
        async def load(self, session_id: str):
            return store.get(session_id)

        async def save(self, session_id: str, checkpoint: dict[str, Any]):
            store[session_id] = {"checkpoint": checkpoint, "updated_at": "t"}

    monkeypatch.setattr(interrupt_api, "checkpoint_store", _Store())
    app = FastAPI()
    app.include_router(interrupt_api.router)
    app.dependency_overrides[interrupt_api.get_current_active_user] = lambda: _U(username="u1")
    c = TestClient(app)

    r404 = c.get("/interrupt/s1")
    assert r404.status_code == 404

    store["s1"] = {
        "checkpoint": {"interrupted": True, "action_required": {"action_type": "x", "approved": False}},
        "updated_at": "t0",
    }
    r = c.get("/interrupt/s1")
    assert r.status_code == 200
    assert r.json()["interrupted"] is True

    bad = c.post("/interrupt/s1/approve", json={"approved": True})
    assert bad.status_code == 200
    assert bad.json()["approved"] is True

    resume_blocked = c.get("/interrupt/s1/resume")
    assert resume_blocked.status_code == 200

    store["s2"] = {"checkpoint": {"interrupted": True}, "updated_at": "t0"}
    r400 = c.post("/interrupt/s2/approve", json={"approved": True})
    assert r400.status_code == 400
