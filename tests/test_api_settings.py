from __future__ import annotations

import os
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
    class _ContextPruningCfg:
        def __init__(self):
            self.method = "heuristic"
            self.auto_reranker_min_lines = 40

        def model_dump(self):
            return {
                "method": self.method,
                "auto_reranker_min_lines": self.auto_reranker_min_lines,
            }

    class _PromptCfg:
        def __init__(self):
            self.context_pruning = _ContextPruningCfg()

        def model_dump(self):
            return {"context_pruning": self.context_pruning.model_dump()}

    class _ServerCfg:
        def __init__(self):
            self.port = 8000

        def model_dump(self):
            return {"port": self.port}

    class _FakeSettings:
        def __init__(self):
            self.server = _ServerCfg()
            self.prompt = _PromptCfg()
            self.reranker = type("_RerankerCfg", (), {"model_name": "", "env_var": "MODEL_PATH_RERANKER"})()
            self.local_models = type("_LocalModelsCfg", (), {"rerank_model": ""})()

        def model_dump(self):
            return {
                "server": self.server.model_dump(),
                "prompt": self.prompt.model_dump(),
                "reranker": {
                    "model_name": self.reranker.model_name,
                    "env_var": self.reranker.env_var,
                },
                "local_models": {
                    "rerank_model": self.local_models.rerank_model,
                },
            }

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
    assert g.json()["runtime_status"]["reranker"]["configured"] is False
    assert g.json()["runtime_status"]["reranker"]["pruning_scoring_source"] == "lightweight_ranker"

    u = c.post("/settings", json={"server": {"port": 9000}})
    assert u.status_code == 200
    assert u.json()["server"]["port"] == 9000
    assert u.json()["runtime_status"]["reranker"]["configured"] is False

    nested = c.post(
        "/settings",
        json={"prompt": {"context_pruning": {"method": "auto", "auto_reranker_min_lines": 55}}},
    )
    assert nested.status_code == 200
    assert nested.json()["prompt"]["context_pruning"]["method"] == "auto"
    assert nested.json()["prompt"]["context_pruning"]["auto_reranker_min_lines"] == 55
    assert nested.json()["runtime_status"]["reranker"]["configured"] is False

    me0 = c.get("/settings/user")
    assert me0.status_code == 200
    assert me0.json() == {}

    upd = c.post("/settings/user", json={"theme": "dark"})
    assert upd.status_code == 200

    me1 = c.get("/settings/user")
    assert me1.status_code == 200
    assert me1.json()["theme"] == "dark"


def test_settings_runtime_status_uses_reranker_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path):
    model_dir = tmp_path / "rr"
    model_dir.mkdir()

    class _FakeSettings:
        def model_dump(self):
            return {
                "server": {"port": 8000},
                "prompt": {"context_pruning": {"method": "heuristic", "auto_reranker_min_lines": 40}},
                "reranker": {"model_name": "", "env_var": "MODEL_PATH_RERANKER", "provider": "hf"},
                "local_models": {"rerank_model": ""},
                "model_manager": {"provider": "hf"},
            }

    monkeypatch.setattr(settings_api, "settings", _FakeSettings())
    monkeypatch.setenv("MODEL_PATH_RERANKER", os.fspath(model_dir))

    app = FastAPI()
    app.include_router(settings_api.router)
    app.dependency_overrides[settings_api.get_current_admin_user] = lambda: _U(username="admin", role="admin")
    c = TestClient(app)

    g = c.get("/settings")
    assert g.status_code == 200
    assert g.json()["runtime_status"]["reranker"]["configured"] is True
    assert g.json()["runtime_status"]["reranker"]["pruning_scoring_source"] == "lightweight_ranker"


def test_settings_admin_can_update_mcp_inventory(monkeypatch: pytest.MonkeyPatch):
    class _McpCfg:
        def __init__(self):
            self.servers = []

        def model_dump(self):
            return {"servers": list(self.servers)}

    class _FakeSettings:
        def __init__(self):
            self.server = type("_ServerCfg", (), {"port": 8000, "model_dump": lambda self: {"port": self.port}})()
            self.prompt = type(
                "_PromptCfg",
                (),
                {"model_dump": lambda self: {"context_pruning": {"method": "heuristic", "auto_reranker_min_lines": 40}}},
            )()
            self.reranker = type("_RerankerCfg", (), {"model_name": "", "env_var": "MODEL_PATH_RERANKER", "provider": "hf"})()
            self.local_models = type("_LocalModelsCfg", (), {"rerank_model": ""})()
            self.model_manager = type("_ModelManagerCfg", (), {"provider": "hf"})()
            self.mcp = _McpCfg()

        def model_dump(self):
            return {
                "server": self.server.model_dump(),
                "prompt": self.prompt.model_dump(),
                "reranker": {
                    "model_name": self.reranker.model_name,
                    "env_var": self.reranker.env_var,
                    "provider": self.reranker.provider,
                },
                "local_models": {"rerank_model": self.local_models.rerank_model},
                "model_manager": {"provider": self.model_manager.provider},
                "mcp": self.mcp.model_dump(),
            }

    monkeypatch.setattr(settings_api, "settings", _FakeSettings())

    app = FastAPI()
    app.include_router(settings_api.router)
    app.dependency_overrides[settings_api.get_current_admin_user] = lambda: _U(username="admin", role="admin")
    c = TestClient(app)

    response = c.post(
        "/settings",
        json={
            "mcp": {
                "servers": [
                    {
                        "server_id": "filesystem",
                        "title": "Filesystem",
                        "description": "Workspace access",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                        "enabled": True,
                    }
                ]
            }
        },
    )

    assert response.status_code == 200
    assert response.json()["mcp"]["servers"][0]["server_id"] == "filesystem"
    assert response.json()["mcp"]["servers"][0]["title"] == "Filesystem"
