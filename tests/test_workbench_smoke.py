from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import documents as documents_api
from app.server.api import history as history_api
from app.server.api import memory as memory_api
from app.server.api import tasks as tasks_api
from app.server.api import upload as upload_api


@dataclass(frozen=True)
class _User:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


def test_workbench_smoke_flow(tmp_path: Any, monkeypatch: Any):
    docs_root = tmp_path / "documents"
    uploads_root = tmp_path / "uploads"
    docs_root.mkdir()
    uploads_root.mkdir()

    state: dict[str, Any] = {
        "tasks": {},
        "documents": [],
        "history": {},
        "memory_items": [],
        "profile": {"facts": [{"text": "prefers concise answers"}]},
        "next_doc_id": 1,
    }
    user = _User()
    orig_join = os.path.join

    def _join(a: str, *p: str) -> str:
        if a == "data/documents":
            return orig_join(str(docs_root), *p)
        if a == "data/uploads":
            return orig_join(str(uploads_root), *p)
        return orig_join(a, *p)

    async def _init_task(task_id: str, payload: dict[str, Any]) -> None:
        state["tasks"][task_id] = {k: str(v) for k, v in payload.items()}

    async def _get_task(task_id: str) -> dict[str, str]:
        return dict(state["tasks"].get(task_id) or {})

    async def _claim_task_operation(operation_key: str, task_id: str, **kwargs: Any) -> str:
        return task_id

    async def _release_task_operation(
        operation_key: str,
        *,
        expected_task_id: str | None = None,
    ) -> None:
        return None

    async def _enqueue(task_id: str, file_path: str, user_id: str = None) -> None:
        task = state["tasks"][task_id]
        task.update({"status": "succeeded", "progress": "100", "step": "done"})
        doc_id = state["next_doc_id"]
        state["next_doc_id"] += 1
        state["documents"].append(
            {
                "doc_id": doc_id,
                "user_id": user_id,
                "source_path": file_path,
                "checksum": "checksum",
                "created_at": 1,
                "parent_chunk_count": 1,
                "embedding_count": 1,
                "preview": [{"parent_chunk_id": 1, "doc_id": doc_id, "page_num": 1, "content": "hello doc"}],
            }
        )

    class _DocStore:
        def find_by_checksum(self, *, user_id: str, checksum: str):
            return None

        def search_documents(self, user_id: str, *, include_all_users: bool = False, filename_query: str | None = None):
            q = str(filename_query or "").lower()
            docs = [
                d for d in state["documents"]
                if include_all_users or d["user_id"] == user_id
            ]
            if q:
                docs = [d for d in docs if q in d["source_path"].lower()]
            return [{k: v for k, v in d.items() if k != "preview"} for d in docs]

        def get_document(self, doc_id: int):
            for d in state["documents"]:
                if d["doc_id"] == doc_id:
                    return {k: v for k, v in d.items() if k != "preview"}
            return None

        def get_document_preview(self, doc_id: int, *, limit: int = 5):
            for d in state["documents"]:
                if d["doc_id"] == doc_id:
                    return list(d["preview"])[:limit]
            return []

        def delete_document(self, doc_id: int):
            before = len(state["documents"])
            state["documents"] = [d for d in state["documents"] if d["doc_id"] != doc_id]
            return len(state["documents"]) != before

    class _HistoryStore:
        def search_sessions(self, user_id: str, query: str):
            sessions = list(state["history"].get(user_id, {}).values())
            q = str(query or "").lower()
            if q:
                sessions = [s for s in sessions if q in s["title"].lower()]
            return sessions

        def save_session(self, user_id: str, session_id: str, messages: list[dict[str, Any]], title: str | None = None):
            payload = {
                "id": session_id,
                "title": title or "hello session",
                "created_at": 1,
                "updated_at": 1,
                "messages": messages,
            }
            state["history"].setdefault(user_id, {})[session_id] = payload
            return payload

        def get_session_detail(self, user_id: str, session_id: str):
            return state["history"].get(user_id, {}).get(session_id)

        def rename_session(self, user_id: str, session_id: str, title: str):
            session = state["history"].get(user_id, {}).get(session_id)
            if not session:
                return None
            session["title"] = title
            return session

        def delete_session(self, user_id: str, session_id: str):
            state["history"].get(user_id, {}).pop(session_id, None)
            return True

    class _ProfileEngine:
        def get_profile(self, user_id: str):
            return state["profile"]

        def upsert_profile(self, user_id: str, profile: dict[str, Any], version: int):
            state["profile"] = profile

    class _MemoryEngine:
        def replace_profile_semantic_memory(self, *, user_id: str, profile: dict[str, Any]):
            state["profile"] = profile
            return 1

    class _MemoryStore:
        def list_items(self, **kwargs):
            return list(state["memory_items"])

        def get_item(self, *, user_id: str, item_id: int):
            for item in state["memory_items"]:
                if item["item_id"] == item_id:
                    return item
            return None

        def delete_item(self, *, user_id: str, item_id: int):
            before = len(state["memory_items"])
            state["memory_items"] = [i for i in state["memory_items"] if i["item_id"] != item_id]
            return len(state["memory_items"]) != before

        def upsert_items(self, rows: list[dict[str, Any]]):
            next_id = len(state["memory_items"]) + 1
            for row in rows:
                state["memory_items"].append(
                    {
                        "item_id": next_id,
                        "user_id": row["user_id"],
                        "kind": row["kind"],
                        "subkind": row.get("subkind"),
                        "item_hash": row["item_hash"],
                        "text": row["text"],
                    }
                )
                next_id += 1
            return len(rows)

        def get_item_by_hash(self, *, user_id: str, kind: str, item_hash: str):
            for item in state["memory_items"]:
                if (
                    item["user_id"] == user_id
                    and item["kind"] == kind
                    and item["item_hash"] == item_hash
                ):
                    return item
            return None

    class _Embeddings:
        def embed_documents(self, texts: list[str]):
            return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(upload_api.os.path, "join", _join)
    monkeypatch.setattr(upload_api, "ensure_schema_if_possible", lambda: False)
    monkeypatch.setattr(upload_api, "init_task", _init_task)
    monkeypatch.setattr(upload_api, "claim_task_operation", _claim_task_operation)
    monkeypatch.setattr(upload_api, "get_task", _get_task)
    monkeypatch.setattr(upload_api, "release_task_operation", _release_task_operation)
    monkeypatch.setattr(upload_api, "enqueue_ingest_pdf", _enqueue)

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    monkeypatch.setattr(documents_api, "ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr(documents_api, "MySQLDocStore", lambda: _DocStore())

    monkeypatch.setattr(history_api, "ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr(history_api, "MySQLConversationStore", lambda: _HistoryStore())
    monkeypatch.setattr(history_api, "_update_memory_after_save", lambda *args, **kwargs: None)

    monkeypatch.setattr(memory_api, "ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr(memory_api, "PgUserMemoryStore", lambda: _MemoryStore())
    monkeypatch.setattr(memory_api, "UserProfileEngine", lambda: _ProfileEngine())
    monkeypatch.setattr(memory_api, "UserMemoryEngine", lambda: _MemoryEngine())
    monkeypatch.setattr(memory_api, "get_embeddings", lambda: _Embeddings())

    app = FastAPI()
    app.include_router(upload_api.router)
    app.include_router(tasks_api.router)
    app.include_router(documents_api.router)
    app.include_router(history_api.router)
    app.include_router(memory_api.router)
    for module in [upload_api, tasks_api, documents_api, history_api, memory_api]:
        app.dependency_overrides[module.get_current_active_user] = lambda u=user: u

    client = TestClient(app)

    upload = client.post(
        "/upload",
        files=[("files", ("guide.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    assert upload.status_code == 200
    task_id = upload.json()["results"][0]["task_id"]

    task = client.get(f"/tasks/{task_id}")
    assert task.status_code == 200
    assert task.json()["status"] == "succeeded"

    docs = client.get("/documents", params={"q": "guide"})
    assert docs.status_code == 200
    assert len(docs.json()["documents"]) == 1
    doc_id = docs.json()["documents"][0]["doc_id"]

    doc = client.get(f"/documents/{doc_id}")
    assert doc.status_code == 200
    assert doc.json()["preview"][0]["content"] == "hello doc"

    save = client.post(
        f"/history/{user.username}/save",
        json={"session_id": "s1", "title": "guide chat", "messages": [{"role": "user", "content": "question"}]},
    )
    assert save.status_code == 200

    history = client.get(f"/history/{user.username}", params={"q": "guide"})
    assert history.status_code == 200
    assert history.json()["history"][0]["id"] == "s1"

    profile = client.get("/memory/profile")
    assert profile.status_code == 200
    assert profile.json()["profile"]["facts"][0]["text"] == "prefers concise answers"

    create_memory = client.post(
        "/memory/items",
        json={"kind": "semantic", "subkind": "manual_note", "text": "remember this"},
    )
    assert create_memory.status_code == 200
    assert create_memory.json()["item"]["item_id"] == 1

    memory_items = client.get("/memory/items")
    assert memory_items.status_code == 200
    assert len(memory_items.json()["items"]) == 1
