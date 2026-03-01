from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import BackgroundTasks, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database.models import UserProfile
from app.server.api import history as history_api
from app.server.api import interrupt as interrupt_api
from app.server.api import profile as profile_api
from app.server.api import settings as settings_api
from app.server.api import tasks as tasks_api
from app.server.api import vectorstore as vectorstore_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


def test_tasks_endpoint_isolation(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    async def _get_task(task_id: str):
        if task_id == "missing":
            return None
        if task_id == "owned":
            return {"task_id": task_id, "user_id": "u1"}
        if task_id == "other":
            return {"task_id": task_id, "user_id": "u2"}
        return {"task_id": task_id, "user_id": "unknown"}

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    c = TestClient(app)

    r404 = c.get("/tasks/missing")
    assert r404.status_code == 404

    ok = c.get("/tasks/owned")
    assert ok.status_code == 200
    assert ok.json()["user_id"] == "u1"

    deny = c.get("/tasks/other")
    assert deny.status_code == 403


def test_tasks_endpoint_allows_admin(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="admin", role="admin")

    async def _get_task(task_id: str):
        return {"task_id": task_id, "user_id": "u2"}

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    c = TestClient(app)
    r = c.get("/tasks/t1")
    assert r.status_code == 200


def test_vectorstore_clear(monkeypatch: pytest.MonkeyPatch):
    import sys
    import types

    cleared = {"v": False}

    class _E:
        def clear(self):
            cleared["v"] = True

    fake_mod = types.ModuleType("app.skills.rag.rag_engine")
    fake_mod.get_rag_engine = lambda: _E()
    monkeypatch.setitem(sys.modules, "app.skills.rag.rag_engine", fake_mod)

    app = FastAPI()
    app.include_router(vectorstore_api.router)
    c = TestClient(app)
    r = c.post("/vectorstore/docs/clear")
    assert r.status_code == 200
    assert r.json()["message"] == "cleared"
    assert cleared["v"] is True


def test_profile_returns_none_when_db_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(profile_api, "ensure_schema_if_possible", lambda: False)
    app = FastAPI()
    app.include_router(profile_api.router)
    c = TestClient(app)
    r = c.get("/profile/u1")
    assert r.status_code == 200
    assert r.json()["profile"] is None


def test_profile_returns_value_when_db_available(monkeypatch: pytest.MonkeyPatch):
    import sys
    import types

    monkeypatch.setattr(profile_api, "ensure_schema_if_possible", lambda: True)

    class _Eng:
        def get_profile(self, user_id: str):
            return {"k": user_id}

    fake_mod = types.ModuleType("app.skills.profile.profile_engine")
    fake_mod.UserProfileEngine = lambda: _Eng()
    monkeypatch.setitem(sys.modules, "app.skills.profile.profile_engine", fake_mod)

    app = FastAPI()
    app.include_router(profile_api.router)
    c = TestClient(app)
    r = c.get("/profile/u1")
    assert r.status_code == 200
    assert r.json()["profile"] == {"k": "u1"}


def test_history_endpoints_file_store(tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(history_api, "ensure_schema_if_possible", lambda: False)
    monkeypatch.setattr(
        history_api.history_manager.__class__,
        "_ensure_data_dir",
        lambda self: None,
    )
    from app.infrastructure.database import history_manager as hm

    monkeypatch.setattr(hm, "HISTORY_FILE", os.path.join(str(tmp_path), "chat_history.json"))

    app = FastAPI()
    app.include_router(history_api.router)
    app.dependency_overrides[history_api.get_current_active_user] = lambda: _U(username="u1")
    c = TestClient(app)

    save = c.post(
        "/history/u1/save",
        json={"messages": [{"role": "user", "content": "hi"}], "title": "t"},
    )
    assert save.status_code == 200
    session_id = save.json()["id"]

    lst = c.get("/history/u1")
    assert lst.status_code == 200
    assert lst.json()["history"][0]["id"] == session_id

    other = c.get("/history/u2")
    assert other.status_code == 403

    d = c.delete(f"/history/u1/{session_id}")
    assert d.status_code == 200


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


def test_settings_admin_and_user(monkeypatch: pytest.MonkeyPatch):
    class _ServerCfg:
        def __init__(self):
            self.port = 8000

        def model_dump(self):
            return {"port": self.port}

    class _FakeSettings:
        def __init__(self):
            self.server = _ServerCfg()

        def model_dump(self):
            return {"server": self.server.model_dump()}

    monkeypatch.setattr(settings_api, "settings", _FakeSettings())

    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    UserProfile.__table__.create(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True)

    @contextmanager
    def _get_session() -> Any:
        s: Session = SessionLocal()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    monkeypatch.setattr(settings_api, "get_session", _get_session)

    app = FastAPI()
    app.include_router(settings_api.router)
    app.dependency_overrides[settings_api.get_current_admin_user] = lambda: _U(username="admin", role="admin")
    app.dependency_overrides[settings_api.get_current_active_user] = lambda: _U(username="u1", role="user")
    c = TestClient(app)

    g = c.get("/settings")
    assert g.status_code == 200
    assert g.json()["server"]["port"] == 8000

    u = c.post("/settings", json={"server": {"port": 9000}})
    assert u.status_code == 200
    assert u.json()["server"]["port"] == 9000

    me0 = c.get("/settings/user")
    assert me0.status_code == 200
    assert me0.json() == {}

    upd = c.post("/settings/user", json={"theme": "dark"})
    assert upd.status_code == 200

    me1 = c.get("/settings/user")
    assert me1.status_code == 200
    assert me1.json()["theme"] == "dark"
