from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import history as history_api
from app.server.api import profile as profile_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


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

    detail = c.get(f"/history/u1/{session_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == session_id

    search = c.get("/history/u1", params={"q": "hi"})
    assert search.status_code == 200
    assert len(search.json()["history"]) == 1

    rename = c.patch(f"/history/u1/{session_id}", json={"title": "new title"})
    assert rename.status_code == 200
    assert rename.json()["title"] == "new title"

    other = c.get("/history/u2")
    assert other.status_code == 403

    d = c.delete(f"/history/u1/{session_id}")
    assert d.status_code == 200
