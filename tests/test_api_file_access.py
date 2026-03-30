from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import file_access as file_access_api


@dataclass(frozen=True)
class _U:
    username: str
    role: str = "user"
    is_active: bool = True


def _build_client(tmp_path: Any, monkeypatch: Any, user: _U) -> TestClient:
    docs_root = tmp_path / "documents"
    uploads_root = tmp_path / "uploads"
    (docs_root / "u1").mkdir(parents=True)
    (docs_root / "u2").mkdir(parents=True)
    (uploads_root / "u1").mkdir(parents=True)
    (uploads_root / "u2").mkdir(parents=True)

    (docs_root / "u1" / "guide.pdf").write_bytes(b"pdf-u1")
    (docs_root / "u2" / "secret.pdf").write_bytes(b"pdf-u2")
    (uploads_root / "u1" / "image.png").write_bytes(b"png-u1")
    (uploads_root / "u2" / "image.png").write_bytes(b"png-u2")

    monkeypatch.setattr(file_access_api.settings.storage_local, "documents_dir", str(docs_root))
    monkeypatch.setattr(file_access_api.settings.storage_local, "uploads_dir", str(uploads_root))

    app = FastAPI()
    app.include_router(file_access_api.router)
    app.dependency_overrides[file_access_api.get_current_active_user] = lambda: user
    return TestClient(app)


def test_upload_file_route_allows_owner_access(tmp_path: Any, monkeypatch: Any):
    client = _build_client(tmp_path, monkeypatch, _U(username="u1"))

    response = client.get("/uploads/u1/image.png")

    assert response.status_code == 200
    assert response.content == b"png-u1"


def test_upload_file_route_blocks_other_user(tmp_path: Any, monkeypatch: Any):
    client = _build_client(tmp_path, monkeypatch, _U(username="u1"))

    response = client.get("/uploads/u2/image.png")

    assert response.status_code == 403


def test_document_file_route_allows_admin_access(tmp_path: Any, monkeypatch: Any):
    client = _build_client(tmp_path, monkeypatch, _U(username="admin", role="admin"))

    response = client.get("/files/u2/secret.pdf")

    assert response.status_code == 200
    assert response.content == b"pdf-u2"


def test_document_file_route_rejects_path_traversal(tmp_path: Any, monkeypatch: Any):
    client = _build_client(tmp_path, monkeypatch, _U(username="u1"))

    response = client.get("/files/u1/../../u2/secret.pdf")

    assert response.status_code == 404
