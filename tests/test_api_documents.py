from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.infrastructure.database.models import Base, DocContent, DocEmbedding, Document
from app.server.api import documents as documents_api


@dataclass(frozen=True)
class _U:
    username: str
    role: str = "user"
    is_active: bool = True


def _build_client(tmp_path: Any, monkeypatch: Any, user: _U) -> TestClient:
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine, tables=[Document.__table__, DocContent.__table__, DocEmbedding.__table__])
    SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True
    )

    @contextmanager
    def _get_session() -> Any:
        s: Session = SessionLocal()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    monkeypatch.setattr(documents_api, "ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr("app.infrastructure.database.stores.get_session", _get_session)

    with _get_session() as session:
        own_file = tmp_path / "u1-a.pdf"
        own_file.write_bytes(b"pdf")
        other_file = tmp_path / "u2-b.pdf"
        other_file.write_bytes(b"pdf")

        doc1 = Document(doc_id=1, user_id="u1", source_path=str(own_file), checksum="c1", created_at=100)
        doc2 = Document(doc_id=2, user_id="u2", source_path=str(other_file), checksum="c2", created_at=200)
        session.add_all([doc1, doc2])
        session.flush()
        session.add(DocContent(parent_chunk_id=1, doc_id=doc1.doc_id, content="p1", page_num=1, created_at=101))
        session.add(DocEmbedding(id=1, doc_id=doc1.doc_id, parent_chunk_id=None, child_index=0, source_path=str(own_file), content="c", embedding=[0.1] * 1024, metadata_json={"user_id": "u1"}, created_at=102))

    app = FastAPI()
    app.include_router(documents_api.router)
    app.dependency_overrides[documents_api.get_current_active_user] = lambda: user
    return TestClient(app)


def test_list_documents_is_user_scoped(tmp_path: Any, monkeypatch: Any):
    client = _build_client(tmp_path, monkeypatch, _U(username="u1"))
    r = client.get("/documents")
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["user_id"] == "u1"
    assert docs[0]["filename"] == "u1-a.pdf"
    assert docs[0]["parent_chunk_count"] == 1
    assert docs[0]["embedding_count"] == 1


def test_list_documents_supports_filename_search(tmp_path: Any, monkeypatch: Any):
    client = _build_client(tmp_path, monkeypatch, _U(username="u1"))
    r = client.get("/documents", params={"q": "u1-a"})
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "u1-a.pdf"

    empty = client.get("/documents", params={"q": "missing"})
    assert empty.status_code == 200
    assert empty.json()["documents"] == []


def test_list_documents_allows_admin_to_see_all(tmp_path: Any, monkeypatch: Any):
    client = _build_client(tmp_path, monkeypatch, _U(username="admin", role="admin"))
    r = client.get("/documents")
    assert r.status_code == 200
    assert len(r.json()["documents"]) == 2


def test_get_document_enforces_ownership(tmp_path: Any, monkeypatch: Any):
    client = _build_client(tmp_path, monkeypatch, _U(username="u1"))
    own = client.get("/documents/1")
    assert own.status_code == 200
    assert len(own.json()["preview"]) == 1
    assert own.json()["preview"][0]["content"] == "p1"

    other = client.get("/documents/2")
    assert other.status_code == 403


def test_delete_document_removes_row_and_file(tmp_path: Any, monkeypatch: Any):
    client = _build_client(tmp_path, monkeypatch, _U(username="u1"))
    target = tmp_path / "u1-a.pdf"
    assert os.path.exists(target)

    r = client.delete("/documents/1")
    assert r.status_code == 200
    assert r.json()["doc_id"] == 1
    assert r.json()["file_deleted"] is True
    assert not os.path.exists(target)

    missing = client.get("/documents/1")
    assert missing.status_code == 404


def test_reindex_document_checks_ownership(tmp_path: Any, monkeypatch: Any):
    client = _build_client(tmp_path, monkeypatch, _U(username="u1"))
    other = client.post("/documents/2/reindex")
    assert other.status_code == 403


def test_reindex_document_calls_rag_engine(tmp_path: Any, monkeypatch: Any):
    called = {}

    async def _claim_task_operation(operation_key: str, task_id: str, **kwargs: Any):
        return task_id

    async def _get_task(task_id: str):
        return {}

    async def _init_task(task_id: str, payload: dict[str, Any]):
        called["task_id"] = task_id
        called["payload"] = payload

    async def _enqueue(task_id: str, file_path: str, user_id: str | None = None):
        called["enqueued"] = {
            "task_id": task_id,
            "file_path": file_path,
            "user_id": user_id,
        }

    monkeypatch.setattr(documents_api, "claim_task_operation", _claim_task_operation)
    monkeypatch.setattr(documents_api, "get_task", _get_task)
    monkeypatch.setattr(documents_api, "init_task", _init_task)
    monkeypatch.setattr(documents_api, "enqueue_ingest_pdf", _enqueue)
    client = _build_client(tmp_path, monkeypatch, _U(username="u1"))
    r = client.post("/documents/1/reindex")
    assert r.status_code == 200
    assert r.json()["message"] == "Reindex queued"
    assert r.json()["doc_id"] == 1
    assert called["payload"]["task_type"] == "reindex_document"
    assert called["enqueued"]["user_id"] == "u1"
    assert called["enqueued"]["file_path"].endswith("u1-a.pdf")


def test_reindex_document_returns_existing_inflight_task(tmp_path: Any, monkeypatch: Any):
    async def _claim_task_operation(operation_key: str, task_id: str, **kwargs: Any):
        return "existing-task"

    async def _get_task(task_id: str):
        return {"task_id": task_id, "status": "running"}

    async def _init_task(*args: Any, **kwargs: Any):
        raise AssertionError("should not init new task")

    async def _enqueue(*args: Any, **kwargs: Any):
        raise AssertionError("should not enqueue new task")

    monkeypatch.setattr(documents_api, "claim_task_operation", _claim_task_operation)
    monkeypatch.setattr(documents_api, "get_task", _get_task)
    monkeypatch.setattr(documents_api, "init_task", _init_task)
    monkeypatch.setattr(documents_api, "enqueue_ingest_pdf", _enqueue)
    client = _build_client(tmp_path, monkeypatch, _U(username="u1"))

    r = client.post("/documents/1/reindex")
    assert r.status_code == 200
    assert r.json()["message"] == "Reindex already queued"
    assert r.json()["task_id"] == "existing-task"
