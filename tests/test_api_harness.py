from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.server.api import harness as harness_api


@dataclass(frozen=True)
class _U:
    username: str = "u1"
    role: str = "user"
    is_active: bool = True


class _ServiceListGet:
    def list_policies(self):
        return [
            {
                "policy_id": "document_ingest:v1",
                "task_type": "document_ingest",
                "approval_required": False,
                "allowed_tools": ["document_ingest"],
                "verification_profile": "document_ingest_basic",
                "retry_budget": 1,
            }
        ]

    def list_runs(self, *, user_id: str, limit: int = 50):
        return [
            {
                "run_id": "hr-1",
                "user_id": user_id,
                "status": "queued",
                "task_type": "document_ingest",
                "policy": {"retry_budget": 1},
                "can_retry": False,
                "latest_approval": {"approval_id": "ha-1", "status": "pending"},
                "latest_verification": None,
            }
        ]

    def get_run(self, run_id: str):
        return {"run_id": run_id, "user_id": "u1", "status": "queued", "task_type": "document_ingest"}

    def get_run_detail(self, run_id: str):
        return {
            "run_id": run_id,
            "user_id": "u1",
            "status": "queued",
            "task_type": "document_ingest",
            "policy": {"retry_budget": 1},
            "can_retry": False,
            "latest_approval": {"approval_id": "ha-1", "status": "pending"},
            "latest_verification": {"verification_id": "hv-1", "status": "pass"},
            "events": [{"event_id": "he-1", "event_type": "run.created"}],
        }

    def get_latest_verification(self, run_id: str):
        return {"verification_id": "hv-1", "run_id": run_id, "status": "pass"}

    def list_run_events(self, *, run_id: str, user_id: str | None = None, limit: int = 100):
        return [
            {
                "event_id": "he-1",
                "run_id": run_id,
                "user_id": user_id or "u1",
                "event_type": "run.created",
            }
        ]


async def _noop_enqueue(run_id: str):
    return "job-1"


def test_list_harness_runs(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/runs")

    assert response.status_code == 200
    assert response.json()["runs"][0]["run_id"] == "hr-1"
    assert response.json()["runs"][0]["policy"]["retry_budget"] == 1


def test_get_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-1")

    assert response.status_code == 200
    assert response.json()["run_id"] == "hr-1"
    assert response.json()["latest_approval"]["status"] == "pending"
    assert response.json()["policy"]["retry_budget"] == 1


def test_list_harness_policies(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/policies")

    assert response.status_code == 200
    assert response.json()["policies"][0]["task_type"] == "document_ingest"


def test_get_harness_verification(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-1/verification")

    assert response.status_code == 200
    assert response.json()["verification_id"] == "hv-1"


def test_retry_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    queued = {}

    class _RetryService(_ServiceListGet):
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u1", "status": "failed", "task_type": "document_ingest"}

        def create_retry_run(self, run_id: str, *, requested_by: str):
            queued["created"] = (run_id, requested_by)
            return {
                "run_id": "hr-2",
                "user_id": "u1",
                "task_type": "document_ingest",
                "status": "created",
                "approval_required": False,
            }

        def mark_queued(self, run_id: str):
            queued["marked"] = run_id
            return {"run_id": run_id, "status": "queued", "task_type": "document_ingest"}

        def get_run_detail(self, run_id: str):
            return {
                "run_id": run_id,
                "user_id": "u1",
                "task_type": "document_ingest",
                "status": "queued",
                "policy": {"retry_budget": 1},
                "can_retry": False,
                "latest_approval": None,
                "latest_verification": None,
                "events": [],
            }

    async def _enqueue(run_id: str):
        queued["enqueued"] = run_id
        return "job-1"

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _RetryService())
    monkeypatch.setattr(harness_api, "enqueue_harness_run", _enqueue)
    client = TestClient(app)

    response = client.post("/harness/runs/hr-1/retry")

    assert response.status_code == 200
    assert response.json()["run_id"] == "hr-2"
    assert queued["created"] == ("hr-1", "u1")
    assert queued["marked"] == "hr-2"
    assert queued["enqueued"] == "hr-2"


def test_retry_harness_run_rejects_when_not_allowed(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _RetryService(_ServiceListGet):
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u1", "status": "completed", "task_type": "document_ingest"}

        def create_retry_run(self, run_id: str, *, requested_by: str):
            raise harness_api.HarnessRetryNotAllowedError("Retry budget exhausted for this harness run")

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _RetryService())
    client = TestClient(app)

    response = client.post("/harness/runs/hr-1/retry")

    assert response.status_code == 400


def test_list_harness_run_events(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-1/events")

    assert response.status_code == 200
    assert response.json()["events"][0]["event_type"] == "run.created"


def test_list_harness_run_runtime_state_history(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _Service(_ServiceListGet):
        def list_runtime_state_history(self, *, run_id: str, limit: int = 100):
            assert run_id == "hr-1"
            return [
                {
                    "history_id": 1,
                    "run_id": run_id,
                    "version": 1,
                    "transition_type": "run_created",
                    "stage": None,
                    "runtime_state_json": {
                        "review": {},
                        "continuation": {"enabled": False},
                        "research": {"enabled": False},
                    },
                    "created_at": 1000,
                }
            ]

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _Service())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-1/runtime-state/history")

    assert response.status_code == 200
    assert response.json()["history"][0]["transition_type"] == "run_created"


def test_get_harness_run_forbidden(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U(username="u1", role="user")

    class _OtherUserService(_ServiceListGet):
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u2", "status": "queued"}

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _OtherUserService())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-2")

    assert response.status_code == 403


def test_get_harness_run_allows_admin(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U(username="admin", role="admin")

    class _OtherUserService(_ServiceListGet):
        def get_run(self, run_id: str):
            return {"run_id": run_id, "user_id": "u2", "status": "queued"}

        def get_run_detail(self, run_id: str):
            return {
                "run_id": run_id,
                "user_id": "u2",
                "status": "queued",
                "latest_approval": None,
                "latest_verification": None,
                "events": [],
            }

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _OtherUserService())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-2")

    assert response.status_code == 200
    assert response.json()["user_id"] == "u2"


def test_get_harness_run_not_found(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _MissingService(_ServiceListGet):
        def get_run(self, run_id: str):
            return None

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _MissingService())
    client = TestClient(app)

    response = client.get("/harness/runs/hr-missing")

    assert response.status_code == 404


def test_create_harness_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    queued = {}

    class _Service:
        def create_run(self, **kwargs):
            return {
                "run_id": "hr-1",
                "task_type": kwargs["task_type"],
                "status": "created",
                "policy_id": "document_ingest:v1",
                "approval_required": False,
            }

        def mark_queued(self, run_id: str):
            queued["marked"] = run_id
            return {
                "run_id": run_id,
                "status": "queued",
            }

    async def _enqueue(run_id: str):
        queued["run_id"] = run_id
        return "job-1"

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _Service())
    monkeypatch.setattr(harness_api, "enqueue_harness_run", _enqueue)
    client = TestClient(app)

    response = client.post(
        "/harness/runs",
        json={"task_type": "document_ingest", "input": {"file_path": "/tmp/a.pdf"}},
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == "hr-1"
    assert queued["run_id"] == "hr-1"
    assert queued["marked"] == "hr-1"
    assert response.json()["status"] == "queued"


def test_create_harness_run_with_approval_does_not_enqueue(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    called = {"enqueue": False}

    class _ApprovalRunService:
        def create_run(self, **kwargs):
            return {
                "run_id": "hr-2",
                "task_type": kwargs["task_type"],
                "status": "created",
                "policy_id": "session_resume_approval:v1",
                "approval_required": True,
            }

        def mark_queued(self, run_id: str):
            raise AssertionError("mark_queued should not be called for approval-required runs")

    async def _enqueue(run_id: str):
        called["enqueue"] = True
        return "job-1"

    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ApprovalRunService())
    monkeypatch.setattr(harness_api, "enqueue_harness_run", _enqueue)
    client = TestClient(app)

    response = client.post(
        "/harness/runs",
        json={"task_type": "session_resume_approval", "input": {"session_id": "s1"}},
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == "hr-2"
    assert response.json()["status"] == "created"
    assert called["enqueue"] is False


def test_create_harness_run_rejects_unknown_task_type(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _ServiceListGet())
    client = TestClient(app)

    response = client.post(
        "/harness/runs",
        json={"task_type": "unknown_task", "input": {}},
    )

    assert response.status_code == 422


def test_list_harness_studio_projects(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _StudioService:
        def list_projects(self, *, user_id: str):
            assert user_id == "u1"
            return [{"project_id": "hp-1", "name": "Studio", "agent_count": 2}]

    monkeypatch.setattr(harness_api, "get_studio_service", lambda: _StudioService())
    client = TestClient(app)

    response = client.get("/harness/studio/projects")

    assert response.status_code == 200
    assert response.json()["projects"][0]["project_id"] == "hp-1"


def test_get_current_harness_studio_project(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _StudioService:
        def get_current_project(self, *, user_id: str):
            assert user_id == "u1"
            return {"project_id": "hp-1", "name": "Studio", "graph_json": {"agents": []}}

    monkeypatch.setattr(harness_api, "get_studio_service", lambda: _StudioService())
    client = TestClient(app)

    response = client.get("/harness/studio/projects/current")

    assert response.status_code == 200
    assert response.json()["project_id"] == "hp-1"


def test_request_harness_studio_skill(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _StudioService:
        def request_skills(self, *, project_id: str, user_id: str, agent_id: str, requested_skills: list[str]):
            assert project_id == "hp-1"
            assert user_id == "u1"
            assert agent_id == "agent_a"
            assert requested_skills == ["research"]
            return {"project_id": project_id, "skill_request_result": {"created_requests": [{"skill_id": "research"}]}}

    monkeypatch.setattr(harness_api, "get_studio_service", lambda: _StudioService())
    client = TestClient(app)

    response = client.post(
        "/harness/studio/projects/hp-1/skill-requests",
        json={"agent_id": "agent_a", "requested_skills": ["research"]},
    )

    assert response.status_code == 200
    assert response.json()["skill_request_result"]["created_requests"][0]["skill_id"] == "research"


def test_create_harness_studio_run(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()
    queued = {}

    class _StudioService:
        def create_orchestration_run(
            self,
            *,
            project_id: str,
            user_id: str,
            run_scope: str,
            agent_ids: list[str],
            loop_count: int,
            task: str = "",
            timeout_seconds: int | None = None,
        ):
            queued["created"] = {
                "project_id": project_id,
                "user_id": user_id,
                "run_scope": run_scope,
                "agent_ids": agent_ids,
                "loop_count": loop_count,
                "task": task,
                "timeout_seconds": timeout_seconds,
            }
            return {
                "run_id": "hr-9",
                "task_type": "agent_orchestration",
                "status": "created",
                "approval_required": False,
            }

    class _RunService:
        def mark_queued(self, run_id: str):
            queued["marked"] = run_id
            return {"run_id": run_id, "status": "queued"}

        def get_run_detail(self, run_id: str):
            return {"run_id": run_id, "status": "queued", "task_type": "agent_orchestration"}

    async def _enqueue(run_id: str):
        queued["enqueued"] = run_id
        return "job-1"

    monkeypatch.setattr(harness_api, "get_studio_service", lambda: _StudioService())
    monkeypatch.setattr(harness_api, "get_run_service", lambda: _RunService())
    monkeypatch.setattr(harness_api, "enqueue_harness_run", _enqueue)
    client = TestClient(app)

    response = client.post(
        "/harness/studio/projects/hp-1/run",
        json={"run_scope": "selected", "agent_ids": ["agent_a"], "loop_count": 2, "task": "Do it", "timeout_seconds": 45},
    )

    assert response.status_code == 200
    assert response.json()["run_id"] == "hr-9"
    assert queued["created"]["agent_ids"] == ["agent_a"]
    assert queued["created"]["task"] == "Do it"
    assert queued["created"]["timeout_seconds"] == 45
    assert queued["enqueued"] == "hr-9"
    assert queued["marked"] == "hr-9"


def test_list_harness_model_providers_serializes_models_and_hides_api_key(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _ProviderStore:
        def list_providers(self, *, user_id=None, limit: int = 50):
            assert user_id == "u1"
            return [
                {
                    "provider_id": "provider_1",
                    "user_id": "u1",
                    "name": "Primary",
                    "base_url": "https://provider.test",
                    "api_key_encrypted": "secret",
                    "models_json": ["gpt-5.2", "gpt-5.1-codex-mini"],
                    "is_default": True,
                    "enabled": True,
                }
            ]

    monkeypatch.setattr(harness_api, "get_provider_store", lambda: _ProviderStore())
    client = TestClient(app)

    response = client.get("/harness/model-providers")

    assert response.status_code == 200
    provider = response.json()["providers"][0]
    assert provider["models"] == ["gpt-5.2", "gpt-5.1-codex-mini"]
    assert "models_json" not in provider
    assert "api_key_encrypted" not in provider


def test_update_harness_model_provider_serializes_models(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _ProviderStore:
        def get_provider(self, provider_id: str):
            return {
                "provider_id": provider_id,
                "user_id": "u1",
                "name": "Primary",
                "base_url": "https://provider.test",
                "api_key_encrypted": "secret",
                "models_json": ["gpt-5.2"],
                "is_default": False,
                "enabled": True,
            }

        def update_provider(self, provider_id: str, **changes):
            return {
                "provider_id": provider_id,
                "user_id": "u1",
                "name": changes.get("name", "Updated"),
                "base_url": "https://provider.test",
                "api_key_encrypted": "secret",
                "models_json": changes.get("models_json", ["gpt-5.1-codex-mini"]),
                "is_default": False,
                "enabled": True,
            }

    monkeypatch.setattr(harness_api, "get_provider_store", lambda: _ProviderStore())
    client = TestClient(app)

    response = client.put(
        "/harness/model-providers/provider_1",
        json={"models": ["gpt-5.1-codex-mini"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["models"] == ["gpt-5.1-codex-mini"]
    assert "models_json" not in body
    assert "api_key_encrypted" not in body


def test_delete_harness_model_provider_forbidden_for_other_user(monkeypatch):
    app = FastAPI()
    app.include_router(harness_api.router)
    app.dependency_overrides[harness_api.get_current_active_user] = lambda: _U()

    class _ProviderStore:
        def get_provider(self, provider_id: str):
            return {
                "provider_id": provider_id,
                "user_id": "u2",
                "name": "Foreign",
                "base_url": "https://provider.test",
                "api_key_encrypted": "secret",
                "models_json": ["gpt-5.2"],
                "is_default": False,
                "enabled": True,
            }

    monkeypatch.setattr(harness_api, "get_provider_store", lambda: _ProviderStore())
    client = TestClient(app)

    response = client.delete("/harness/model-providers/provider_1")

    assert response.status_code == 403
