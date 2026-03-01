from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database.models import User
from app.server.api import auth as auth_api


@dataclass(frozen=True)
class _AuthCfg:
    secret_key: str = "test_secret_key_012345678901234567890123456789"
    algorithm: str = "HS256"


@pytest.fixture
def test_app(monkeypatch: pytest.MonkeyPatch) -> tuple[FastAPI, sessionmaker]:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    User.__table__.create(bind=engine)
    SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True
    )

    monkeypatch.setattr(auth_api, "decode_access_token", auth_api.decode_access_token)
    from app.infrastructure.utils import security

    monkeypatch.setattr(security, "get_auth_config", lambda: _AuthCfg())

    app = FastAPI()
    app.include_router(auth_api.router)

    def _override_get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[auth_api.get_db] = _override_get_db
    return app, SessionLocal


def test_register_and_login_flow(test_app: tuple[FastAPI, sessionmaker]):
    app, _ = test_app
    client = TestClient(app)

    r1 = client.post("/auth/register", json={"username": "u1", "password": "p1"})
    assert r1.status_code == 200
    assert r1.json()["role"] == "admin"

    dup = client.post("/auth/register", json={"username": "u1", "password": "p1"})
    assert dup.status_code == 400

    r2 = client.post("/auth/register", json={"username": "u2", "password": "p2"})
    assert r2.status_code == 200
    assert r2.json()["role"] == "user"

    tok = client.post("/auth/token", data={"username": "u2", "password": "p2"})
    assert tok.status_code == 200
    access_token = tok.json()["access_token"]
    assert access_token

    bad = client.post("/auth/token", data={"username": "u2", "password": "wrong"})
    assert bad.status_code == 401

    me = client.get("/auth/users/me", headers={"Authorization": f"Bearer {access_token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "u2"


def test_users_me_rejects_invalid_token(test_app: tuple[FastAPI, sessionmaker]):
    app, _ = test_app
    client = TestClient(app)
    r = client.get("/auth/users/me", headers={"Authorization": "Bearer invalid"})
    assert r.status_code == 401


def test_users_me_rejects_missing_sub(test_app: tuple[FastAPI, sessionmaker]):
    app, _ = test_app
    client = TestClient(app)
    from app.infrastructure.utils import security

    token = security.create_access_token({"role": "user"})
    r = client.get("/auth/users/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_users_me_rejects_unknown_user(test_app: tuple[FastAPI, sessionmaker]):
    app, _ = test_app
    client = TestClient(app)
    from app.infrastructure.utils import security

    token = security.create_access_token({"sub": "missing_user"})
    r = client.get("/auth/users/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_get_db_closes_session(monkeypatch: pytest.MonkeyPatch):
    closed = {"v": False}

    class _S:
        def close(self):
            closed["v"] = True

    monkeypatch.setattr(auth_api, "get_sessionmaker", lambda: (lambda: _S()))
    gen = auth_api.get_db()
    next(gen)
    gen.close()
    assert closed["v"] is True


@pytest.mark.anyio
async def test_active_and_admin_guards():
    class _U:
        def __init__(self, role: str, is_active: bool):
            self.role = role
            self.is_active = is_active
            self.username = "u"
            self.hashed_password = "x"
            self.created_at = int(time.time())

    with pytest.raises(Exception):
        await auth_api.get_current_active_user(current_user=_U(role="user", is_active=False))

    u = await auth_api.get_current_active_user(current_user=_U(role="user", is_active=True))
    assert u.is_active is True

    with pytest.raises(Exception):
        await auth_api.get_current_admin_user(current_user=u)

    admin = await auth_api.get_current_admin_user(current_user=_U(role="admin", is_active=True))
    assert admin.role == "admin"
