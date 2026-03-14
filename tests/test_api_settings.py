from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database.models import UserProfile
from app.server.api import settings as settings_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


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
