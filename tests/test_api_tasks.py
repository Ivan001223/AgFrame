from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import tasks as tasks_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


def test_tasks_endpoint_isolation(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    async def _get_task(task_id: str):
        if task_id == "missing":
            return None
        if task_id == "owned":
            return {"task_id": task_id, "user_id": "u1"}
        if task_id == "other":
            return {"task_id": task_id, "user_id": "u2"}
        return {"task_id": task_id, "user_id": "unknown"}

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    c = TestClient(app)

    r404 = c.get("/tasks/missing")
    assert r404.status_code == 404

    ok = c.get("/tasks/owned")
    assert ok.status_code == 200
    assert ok.json()["user_id"] == "u1"

    deny = c.get("/tasks/other")
    assert deny.status_code == 403


def test_tasks_endpoint_allows_admin(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="admin", role="admin")

    async def _get_task(task_id: str):
        return {"task_id": task_id, "user_id": "u2"}

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    c = TestClient(app)
    r = c.get("/tasks/t1")
    assert r.status_code == 200


def test_tasks_endpoint_marks_failed_task_retryable(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    async def _get_task(task_id: str):
        return {"task_id": task_id, "user_id": "u1", "status": "failed"}

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    c = TestClient(app)
    r = c.get("/tasks/t1")
    assert r.status_code == 200
    assert r.json()["can_retry"] == "true"
    assert r.json()["diagnostics"]["status"] == "failed"


def test_tasks_endpoint_exposes_structured_diagnostics(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    async def _get_task(task_id: str):
        return {
            "task_id": task_id,
            "user_id": "u1",
            "status": "failed",
            "error": "embedding timeout",
            "error_code": "embedding_failed",
            "result_stage": "embedding",
            "retryable": "true",
        }

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    c = TestClient(app)
    r = c.get("/tasks/t1")
    assert r.status_code == 200
    body = r.json()
    assert body["diagnostics"]["error_code"] == "embedding_failed"
    assert body["diagnostics"]["stage"] == "embedding"
    assert body["diagnostics"]["retryable"] is True
    assert body["diagnostics"]["error_message"] == "embedding timeout"
    assert body["diagnostics"]["title"] == "向量化失败"
    assert body["diagnostics"]["user_message"] == "文档文本在生成 embedding 时失败。"
    assert body["diagnostics"]["suggested_action"] == "请检查 embedding 服务状态后重试。"


def test_tasks_endpoint_flags_suspected_timeout(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    now = 1_700_000_000

    async def _get_task(task_id: str):
        return {
            "task_id": task_id,
            "user_id": "u1",
            "status": "running",
            "step": "finalizing",
            "started_at": str(now - tasks_api.RUNNING_TIMEOUT_SECONDS - 5),
            "retryable": "false",
        }

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    monkeypatch.setattr(tasks_api.time, "time", lambda: now)
    c = TestClient(app)
    r = c.get("/tasks/t1")
    assert r.status_code == 200
    body = r.json()
    assert body["can_retry"] == "true"
    assert body["diagnostics"]["timeout_exceeded"] is True
    assert body["diagnostics"]["error_code"] == "task_timeout_suspected"
    assert body["diagnostics"]["title"] == "任务疑似超时"
    assert body["diagnostics"]["suggested_action"] == "请检查依赖服务状态，必要时重新入队。"


def test_tasks_summary_aggregates_visible_tasks(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    now = 1_700_000_000

    class _Redis:
        async def scan_iter(self, match: str):
            for key in ("task:a", "task:b", "task:c"):
                yield key

        async def hgetall(self, key: str):
            data = {
                "task:a": {
                    "task_id": "a",
                    "user_id": "u1",
                    "status": "failed",
                    "error_code": "embedding_failed",
                    "error": "embedding timeout",
                    "result_stage": "embedding",
                },
                "task:b": {
                    "task_id": "b",
                    "user_id": "u1",
                    "status": "running",
                    "step": "finalizing",
                    "started_at": str(now - tasks_api.RUNNING_TIMEOUT_SECONDS - 10),
                },
                "task:c": {
                    "task_id": "c",
                    "user_id": "u2",
                    "status": "failed",
                    "error_code": "vectorstore_write_failed",
                },
            }
            return data[key]

    async def _list_task_incidents(limit: int = 20):
        return [
            {"task_id": "a", "user_id": "u1", "error_code": "embedding_failed", "archived": False},
            {"task_id": "archived", "user_id": "u1", "error_code": "embedding_failed", "archived": True},
            {"task_id": "x", "user_id": "u2", "error_code": "vectorstore_write_failed"},
        ]

    monkeypatch.setattr(tasks_api, "get_redis", lambda: _Redis())
    monkeypatch.setattr(tasks_api, "list_task_incidents", _list_task_incidents)
    monkeypatch.setattr(tasks_api.time, "time", lambda: now)
    c = TestClient(app)
    r = c.get("/tasks/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["status_counts"]["failed"] == 1
    assert body["status_counts"]["running"] == 1
    assert body["top_errors"][0]["error_code"] == "embedding_failed"
    assert body["suspected_timeouts"][0]["task_id"] == "b"
    assert body["recent_incidents"][0]["task_id"] == "a"


def test_tasks_incidents_filters_by_user(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    async def _list_task_incidents(limit: int = 20):
        return [
            {"task_id": "a", "user_id": "u1", "error_code": "embedding_failed"},
            {"task_id": "b", "user_id": "u2", "error_code": "vectorstore_write_failed"},
        ]

    monkeypatch.setattr(tasks_api, "list_task_incidents", _list_task_incidents)
    c = TestClient(app)
    r = c.get("/tasks/incidents?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert len(body["incidents"]) == 1
    assert body["incidents"][0]["task_id"] == "a"


def test_tasks_incidents_filters_by_handled_and_archived(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    async def _list_task_incidents(limit: int = 20):
        return [
            {"incident_id": "inc-1", "task_id": "a", "user_id": "u1", "handled": False, "archived": False},
            {"incident_id": "inc-2", "task_id": "b", "user_id": "u1", "handled": True, "archived": False},
            {"incident_id": "inc-3", "task_id": "c", "user_id": "u1", "handled": True, "archived": True},
        ]

    monkeypatch.setattr(tasks_api, "list_task_incidents", _list_task_incidents)
    c = TestClient(app)
    r = c.get("/tasks/incidents?handled=true&archived=false")
    assert r.status_code == 200
    body = r.json()
    assert len(body["incidents"]) == 1
    assert body["incidents"][0]["incident_id"] == "inc-2"


def test_tasks_summary_excludes_archived_incidents(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    class _Redis:
        async def scan_iter(self, match: str):
            if False:
                yield ""
            return

        async def hgetall(self, key: str):
            return {}

    async def _list_task_incidents(limit: int = 20):
        return [
            {"incident_id": "inc-1", "task_id": "a", "user_id": "u1", "archived": False},
            {"incident_id": "inc-2", "task_id": "b", "user_id": "u1", "archived": True},
        ]

    monkeypatch.setattr(tasks_api, "get_redis", lambda: _Redis())
    monkeypatch.setattr(tasks_api, "list_task_incidents", _list_task_incidents)
    c = TestClient(app)
    r = c.get("/tasks/summary")
    assert r.status_code == 200
    body = r.json()
    assert len(body["recent_incidents"]) == 1
    assert body["recent_incidents"][0]["incident_id"] == "inc-1"


def test_tasks_incident_patch_marks_handled(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    incident = {
        "incident_id": "inc-1",
        "task_id": "a",
        "user_id": "u1",
        "error_code": "embedding_failed",
        "handled": False,
        "archived": False,
    }

    async def _list_task_incidents(limit: int = 20):
        return [incident]

    async def _update_task_incident(incident_id: str, updates: dict[str, object]):
        return {**incident, **updates}

    monkeypatch.setattr(tasks_api, "list_task_incidents", _list_task_incidents)
    monkeypatch.setattr(tasks_api, "update_task_incident", _update_task_incident)
    monkeypatch.setattr(tasks_api.time, "time", lambda: 1_700_000_000)
    c = TestClient(app)
    r = c.patch("/tasks/incidents/inc-1", json={"handled": True})
    assert r.status_code == 200
    body = r.json()
    assert body["incident"]["incident_id"] == "inc-1"
    assert body["incident"]["handled"] is True
    assert body["incident"]["handled_at"] == 1_700_000_000


def test_tasks_incident_patch_rejects_other_user(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    async def _list_task_incidents(limit: int = 20):
        return [{"incident_id": "inc-2", "task_id": "b", "user_id": "u2"}]

    monkeypatch.setattr(tasks_api, "list_task_incidents", _list_task_incidents)
    c = TestClient(app)
    r = c.patch("/tasks/incidents/inc-2", json={"archived": True})
    assert r.status_code == 403


def test_retry_failed_task(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    async def _get_task(task_id: str):
        return {
            "task_id": task_id,
            "user_id": "u1",
            "status": "failed",
            "file_path": "/tmp/a.pdf",
            "filename": "a.pdf",
            "retry_count": "1",
        }

    queued: dict[str, Any] = {}

    async def _init_task(task_id: str, payload: dict[str, Any]):
        queued["task_id"] = task_id
        queued["payload"] = payload

    async def _enqueue(task_id: str, file_path: str, user_id: str | None = None):
        queued["enqueued"] = {
            "task_id": task_id,
            "file_path": file_path,
            "user_id": user_id,
        }

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    monkeypatch.setattr(tasks_api, "init_task", _init_task)
    monkeypatch.setattr(tasks_api, "enqueue_ingest_pdf", _enqueue)
    c = TestClient(app)

    r = c.post("/tasks/t1/retry")
    assert r.status_code == 200
    body = r.json()
    assert body["retried_from_task_id"] == "t1"
    assert body["retry_count"] == 2
    assert queued["payload"]["retried_from_task_id"] == "t1"
    assert queued["enqueued"]["file_path"] == "/tmp/a.pdf"


def test_retry_task_rejects_non_failed(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(tasks_api.router)
    app.dependency_overrides[tasks_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    async def _get_task(task_id: str):
        return {"task_id": task_id, "user_id": "u1", "status": "running"}

    monkeypatch.setattr(tasks_api, "get_task", _get_task)
    c = TestClient(app)
    r = c.post("/tasks/t1/retry")
    assert r.status_code == 400
