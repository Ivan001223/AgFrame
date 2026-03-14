from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import memory as memory_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


def test_memory_profile_returns_none_when_db_unavailable(monkeypatch):
    monkeypatch.setattr(memory_api, "ensure_schema_if_possible", lambda: False)
    app = FastAPI()
    app.include_router(memory_api.router)
    app.dependency_overrides[memory_api.get_current_active_user] = lambda: _U()
    c = TestClient(app)

    r = c.get("/memory/profile")
    assert r.status_code == 200
    assert r.json()["profile"] is None


def test_memory_profile_returns_user_profile(monkeypatch):
    monkeypatch.setattr(memory_api, "ensure_schema_if_possible", lambda: True)

    class _Engine:
        def get_profile(self, user_id: str):
            return {"facts": [{"text": user_id}]}

    monkeypatch.setattr(memory_api, "UserProfileEngine", lambda: _Engine())
    app = FastAPI()
    app.include_router(memory_api.router)
    app.dependency_overrides[memory_api.get_current_active_user] = lambda: _U()
    c = TestClient(app)

    r = c.get("/memory/profile")
    assert r.status_code == 200
    assert r.json()["profile"]["facts"][0]["text"] == "u1"


def test_list_memory_items_supports_filters(monkeypatch):
    monkeypatch.setattr(memory_api, "ensure_schema_if_possible", lambda: True)

    captured = {}

    class _Store:
        def list_items(self, **kwargs):
            captured.update(kwargs)
            return [{"item_id": 1, "text": "记忆", "kind": "semantic"}]

    monkeypatch.setattr(memory_api, "PgUserMemoryStore", lambda: _Store())
    app = FastAPI()
    app.include_router(memory_api.router)
    app.dependency_overrides[memory_api.get_current_active_user] = lambda: _U()
    c = TestClient(app)

    r = c.get("/memory/items", params={"kind": "semantic", "limit": 5})
    assert r.status_code == 200
    assert r.json()["items"][0]["item_id"] == 1
    assert captured["user_id"] == "u1"
    assert captured["kind"] == "semantic"
    assert captured["limit"] == 5


def test_delete_memory_item_checks_ownership(monkeypatch):
    monkeypatch.setattr(memory_api, "ensure_schema_if_possible", lambda: True)

    class _Store:
        def __init__(self):
            self.deleted = []

        def get_item(self, *, user_id: str, item_id: int):
            if item_id == 1:
                return {"item_id": 1, "user_id": user_id}
            return None

        def delete_item(self, *, user_id: str, item_id: int):
            self.deleted.append((user_id, item_id))
            return True

    store = _Store()
    monkeypatch.setattr(memory_api, "PgUserMemoryStore", lambda: store)
    app = FastAPI()
    app.include_router(memory_api.router)
    app.dependency_overrides[memory_api.get_current_active_user] = lambda: _U()
    c = TestClient(app)

    missing = c.delete("/memory/items/999")
    assert missing.status_code == 404

    ok = c.delete("/memory/items/1")
    assert ok.status_code == 200
    assert ok.json()["item_id"] == 1
    assert store.deleted == [("u1", 1)]


def test_update_memory_profile(monkeypatch):
    monkeypatch.setattr(memory_api, "ensure_schema_if_possible", lambda: True)

    saved = {}

    class _ProfileEngine:
        def get_profile(self, user_id: str):
            return {"facts": [{"text": "old"}]}

        def upsert_profile(self, user_id: str, profile: dict, version: int):
            saved["user_id"] = user_id
            saved["profile"] = profile
            saved["version"] = version

    class _MemoryEngine:
        def replace_profile_semantic_memory(self, *, user_id: str, profile: dict):
            saved["synced_user_id"] = user_id
            saved["synced_profile"] = profile
            return 1

    monkeypatch.setattr(memory_api, "UserProfileEngine", lambda: _ProfileEngine())
    monkeypatch.setattr(memory_api, "UserMemoryEngine", lambda: _MemoryEngine())
    app = FastAPI()
    app.include_router(memory_api.router)
    app.dependency_overrides[memory_api.get_current_active_user] = lambda: _U()
    c = TestClient(app)

    r = c.put("/memory/profile", json={"profile": {"facts": [{"text": "likes python"}]}})
    assert r.status_code == 200
    assert saved["user_id"] == "u1"
    assert saved["synced_user_id"] == "u1"
    assert r.json()["profile"]["facts"][0]["text"] == "likes python"


def test_update_memory_profile_rolls_back_memory_when_profile_write_fails(monkeypatch):
    monkeypatch.setattr(memory_api, "ensure_schema_if_possible", lambda: True)

    calls = []

    class _ProfileEngine:
        def get_profile(self, user_id: str):
            return {"facts": [{"text": "old"}]}

        def upsert_profile(self, user_id: str, profile: dict, version: int):
            raise RuntimeError("db write failed")

    class _MemoryEngine:
        def replace_profile_semantic_memory(self, *, user_id: str, profile: dict):
            calls.append(profile)
            return 1

    monkeypatch.setattr(memory_api, "UserProfileEngine", lambda: _ProfileEngine())
    monkeypatch.setattr(memory_api, "UserMemoryEngine", lambda: _MemoryEngine())
    app = FastAPI()
    app.include_router(memory_api.router)
    app.dependency_overrides[memory_api.get_current_active_user] = lambda: _U()
    c = TestClient(app)

    r = c.put("/memory/profile", json={"profile": {"facts": [{"text": "new"}]}})
    assert r.status_code == 500
    assert calls[0]["facts"][0]["text"] == "new"
    assert calls[1]["facts"][0]["text"] == "old"


def test_create_memory_item(monkeypatch):
    monkeypatch.setattr(memory_api, "ensure_schema_if_possible", lambda: True)

    captured = {}

    class _Embeddings:
        def embed_documents(self, texts):
            captured["texts"] = texts
            return [[0.1, 0.2]]

    class _Store:
        def upsert_items(self, rows):
            captured["rows"] = rows
            return 1

        def get_item_by_hash(self, **kwargs):
            captured["lookup"] = kwargs
            return {"item_id": 7, "text": "manual note", "kind": "semantic"}

    monkeypatch.setattr(memory_api, "get_embeddings", lambda: _Embeddings())
    monkeypatch.setattr(memory_api, "PgUserMemoryStore", lambda: _Store())
    app = FastAPI()
    app.include_router(memory_api.router)
    app.dependency_overrides[memory_api.get_current_active_user] = lambda: _U()
    c = TestClient(app)

    r = c.post(
        "/memory/items",
        json={"kind": "semantic", "subkind": "manual_note", "text": "manual note"},
    )
    assert r.status_code == 200
    assert captured["texts"] == ["manual note"]
    assert captured["rows"][0]["kind"] == "semantic"
    assert captured["lookup"]["kind"] == "semantic"
    assert r.json()["item"]["item_id"] == 7
