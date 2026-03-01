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

    monkeypatch.setattr(upload_api.os.path, "join", _join)
    monkeypatch.setattr(upload_api, "init_task", _init_task)
    monkeypatch.setattr(upload_api, "enqueue_ingest_pdf", _enqueue)

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
