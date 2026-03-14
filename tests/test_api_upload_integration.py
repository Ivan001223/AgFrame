from __future__ import annotations

import builtins
import os
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import upload as upload_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"


@pytest.fixture
def client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    docs_root = tmp_path / "documents"
    uploads_root = tmp_path / "uploads"
    orig_join = os.path.join

    def _join(a: str, *p: str) -> str:
        if a == "data/documents":
            return orig_join(str(docs_root), *p)
        if a == "data/uploads":
            return orig_join(str(uploads_root), *p)
        return orig_join(a, *p)

    async def _init_task(*args: Any, **kwargs: Any) -> None:
        return None

    async def _enqueue(*args: Any, **kwargs: Any) -> None:
        return None

    async def _claim_task_operation(operation_key: str, task_id: str, **kwargs: Any) -> str:
        return task_id

    async def _get_task(task_id: str) -> dict[str, str]:
        return {}

    monkeypatch.setattr(upload_api.os.path, "join", _join)
    monkeypatch.setattr(upload_api, "init_task", _init_task)
    monkeypatch.setattr(upload_api, "enqueue_ingest_pdf", _enqueue)
    monkeypatch.setattr(upload_api, "claim_task_operation", _claim_task_operation)
    monkeypatch.setattr(upload_api, "get_task", _get_task)
    monkeypatch.setattr(upload_api, "ensure_schema_if_possible", lambda: False)

    app = FastAPI()
    app.include_router(upload_api.router)
    app.dependency_overrides[upload_api.get_current_active_user] = lambda: _U()
    return TestClient(app)


def test_upload_documents_skips_non_pdf(client: TestClient):
    r = client.post(
        "/upload",
        files=[("files", ("a.txt", b"hi", "text/plain"))],
    )
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["status"] == "skipped"


def test_upload_documents_sanitizes_filename_and_queues_task(client: TestClient):
    r = client.post(
        "/upload",
        files=[("files", ("../evil.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["status"] == "queued"
    assert ".." not in res["filename"]
    assert res["filename"].endswith("_evil.pdf")
    assert "task_id" in res


def test_upload_documents_error_branch(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    def _boom(*args: Any, **kwargs: Any):
        raise OSError("disk error")

    monkeypatch.setattr(builtins, "open", _boom)
    r = client.post(
        "/upload",
        files=[("files", ("ok.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["status"] == "error"


def test_upload_image_returns_url(client: TestClient):
    r = client.post(
        "/upload/image",
        files={"file": ("a.png", b"png", "image/png")},
    )
    assert r.status_code == 200
    out = r.json()
    assert out["url"].startswith("/uploads/")


def test_upload_documents_marks_duplicates(tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    docs_root = tmp_path / "documents"
    uploads_root = tmp_path / "uploads"
    orig_join = os.path.join

    def _join(a: str, *p: str) -> str:
        if a == "data/documents":
            return orig_join(str(docs_root), *p)
        if a == "data/uploads":
            return orig_join(str(uploads_root), *p)
        return orig_join(a, *p)

    async def _init_task(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("should not enqueue duplicate")

    async def _enqueue(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("should not enqueue duplicate")

    class _Store:
        def find_by_checksum(self, *, user_id: str, checksum: str):
            return {"doc_id": 42, "user_id": user_id, "checksum": checksum}

    monkeypatch.setattr(upload_api.os.path, "join", _join)
    monkeypatch.setattr(upload_api, "init_task", _init_task)
    monkeypatch.setattr(upload_api, "enqueue_ingest_pdf", _enqueue)
    monkeypatch.setattr(upload_api, "ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr(upload_api, "MySQLDocStore", lambda: _Store())
    monkeypatch.setattr(upload_api, "sha256_file", lambda path: "same-file")

    app = FastAPI()
    app.include_router(upload_api.router)
    app.dependency_overrides[upload_api.get_current_active_user] = lambda: _U()
    client = TestClient(app)

    r = client.post(
        "/upload",
        files=[("files", ("dup.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["status"] == "duplicate"
    assert res["existing_doc_id"] == 42


def test_upload_documents_returns_existing_inflight_task(tmp_path: Any, monkeypatch: pytest.MonkeyPatch):
    docs_root = tmp_path / "documents"
    uploads_root = tmp_path / "uploads"
    orig_join = os.path.join

    def _join(a: str, *p: str) -> str:
        if a == "data/documents":
            return orig_join(str(docs_root), *p)
        if a == "data/uploads":
            return orig_join(str(uploads_root), *p)
        return orig_join(a, *p)

    async def _claim_task_operation(operation_key: str, task_id: str, **kwargs: Any) -> str:
        return "existing-task"

    async def _get_task(task_id: str) -> dict[str, str]:
        return {"task_id": task_id, "status": "queued"}

    async def _init_task(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("should not init duplicate inflight task")

    async def _enqueue(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("should not enqueue duplicate inflight task")

    monkeypatch.setattr(upload_api.os.path, "join", _join)
    monkeypatch.setattr(upload_api, "claim_task_operation", _claim_task_operation)
    monkeypatch.setattr(upload_api, "get_task", _get_task)
    monkeypatch.setattr(upload_api, "init_task", _init_task)
    monkeypatch.setattr(upload_api, "enqueue_ingest_pdf", _enqueue)
    monkeypatch.setattr(upload_api, "ensure_schema_if_possible", lambda: False)

    app = FastAPI()
    app.include_router(upload_api.router)
    app.dependency_overrides[upload_api.get_current_active_user] = lambda: _U()
    client = TestClient(app)

    r = client.post(
        "/upload",
        files=[("files", ("dup.pdf", b"%PDF-1.4", "application/pdf"))],
    )
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["status"] == "already_queued"
    assert res["task_id"] == "existing-task"
