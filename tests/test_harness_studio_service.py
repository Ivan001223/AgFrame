from app.harness.runtime.studio_service import HarnessStudioService


class _RunService:
    pass


def test_get_current_project_creates_default_studio():
    created = {}

    class _ProjectStore:
        def get_latest_project_for_user(self, user_id: str):
            assert user_id == "u1"
            return None

        def create_project(self, **kwargs):
            created.update(kwargs)
            return {
                **kwargs,
                "created_at": 1,
                "updated_at": 1,
            }

    service = HarnessStudioService(project_store=_ProjectStore(), run_service=_RunService())
    project = service.get_current_project(user_id="u1")

    assert project["name"] == "Default agent studio"
    assert project["graph_json"]["review_agent"]["hidden"] is True
    assert project["agent_count"] >= 1
    assert created["user_id"] == "u1"


def test_request_skills_creates_pending_requests_and_reuses_loaded_pool():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [{"agent_id": "agent_a", "name": "Agent A"}],
            "edges": [],
            "skill_pool": [{"skill_id": "research", "title": "Research", "source": "app/skills/research"}],
            "pending_skill_requests": [],
            "review_agent": {},
        },
        "created_at": 1,
        "updated_at": 1,
    }

    class _ProjectStore:
        def get_project(self, project_id: str):
            assert project_id == "hp-1"
            return dict(state)

        def update_project(self, project_id: str, **changes):
            state.update(changes)
            return dict(state)

    service = HarnessStudioService(project_store=_ProjectStore(), run_service=_RunService())
    updated = service.request_skills(
        project_id="hp-1",
        user_id="u1",
        agent_id="agent_a",
        requested_skills=["research", "rag"],
    )

    assert updated["skill_request_result"]["available_skill_ids"] == ["research"]
    assert len(updated["skill_request_result"]["created_requests"]) == 1
    assert updated["graph_json"]["pending_skill_requests"][0]["skill_id"] == "rag"


def test_resolve_skill_request_moves_skill_into_pool():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [{"agent_id": "agent_a", "name": "Agent A"}],
            "edges": [],
            "skill_pool": [],
            "pending_skill_requests": [
                {
                    "request_id": "hsr-1",
                    "agent_id": "agent_a",
                    "skill_id": "research",
                    "title": "Research",
                    "source": "app/skills/research",
                    "status": "pending",
                    "discovered_at": 1,
                }
            ],
            "review_agent": {},
        },
        "created_at": 1,
        "updated_at": 1,
    }

    class _ProjectStore:
        def get_project(self, project_id: str):
            return dict(state)

        def update_project(self, project_id: str, **changes):
            state.update(changes)
            return dict(state)

    service = HarnessStudioService(project_store=_ProjectStore(), run_service=_RunService())
    updated = service.resolve_skill_request(
        project_id="hp-1",
        user_id="u1",
        request_id="hsr-1",
        approved=True,
    )

    assert updated["resolved_skill_request"]["status"] == "approved"
    assert updated["graph_json"]["skill_pool"][0]["skill_id"] == "research"


def test_create_orchestration_run_uses_selected_agents():
    project = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {"agent_id": "agent_a", "name": "Agent A"},
                {"agent_id": "agent_b", "name": "Agent B"},
            ],
            "edges": [],
            "skill_pool": [],
            "pending_skill_requests": [],
            "review_agent": {"enabled": True},
        },
        "created_at": 1,
        "updated_at": 1,
    }
    created = {}

    class _ProjectStore:
        def get_project(self, project_id: str):
            return dict(project)

    class _RunService:
        def create_run(self, **kwargs):
            created.update(kwargs)
            return {"run_id": "hr-1", **kwargs}

    service = HarnessStudioService(project_store=_ProjectStore(), run_service=_RunService())
    run = service.create_orchestration_run(
        project_id="hp-1",
        user_id="u1",
        run_scope="selected",
        agent_ids=["agent_b"],
        loop_count=2,
    )

    assert run["task_type"] == "agent_orchestration"
    assert created["input_json"]["selected_agent_ids"] == ["agent_b"]
    assert created["metadata_json"]["review_agent_enabled"] is True


def test_update_project_preserves_review_provider_configuration():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [],
            "edges": [],
            "skill_pool": [],
            "pending_skill_requests": [],
            "review_agent": {
                "enabled": True,
                "preferred_provider_id": "provider_a",
                "fallback_provider_id": "provider_b",
            },
        },
        "created_at": 1,
        "updated_at": 1,
    }

    class _ProjectStore:
        def get_project(self, project_id: str):
            return dict(state)

        def update_project(self, project_id: str, **changes):
            state.update(changes)
            return dict(state)

    service = HarnessStudioService(project_store=_ProjectStore(), run_service=_RunService())
    updated = service.update_project(
        project_id="hp-1",
        user_id="u1",
        graph_json=state["graph_json"],
    )

    assert updated["graph_json"]["review_agent"]["preferred_provider_id"] == "provider_a"
    assert updated["graph_json"]["review_agent"]["fallback_provider_id"] == "provider_b"


def test_update_project_preserves_cluster_nodes():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "cluster_a",
                    "name": "Brainstorm Cluster",
                    "node_kind": "cluster",
                    "cluster_strategy": "brainstorm",
                    "cluster_members": [
                        {"member_id": "m1", "name": "Lead", "role": "chair", "model": "gpt-5.2"},
                        {"member_id": "m2", "name": "Critic", "role": "critic", "model": "gpt-5.1-codex-mini"},
                    ],
                    "brainstorm_rounds": 3,
                    "cluster_auto_research": True,
                    "cluster_auto_review": True,
                }
            ],
            "edges": [],
            "skill_pool": [],
            "pending_skill_requests": [],
            "review_agent": {},
        },
        "created_at": 1,
        "updated_at": 1,
    }

    class _ProjectStore:
        def get_project(self, project_id: str):
            return dict(state)

        def update_project(self, project_id: str, **changes):
            state.update(changes)
            return dict(state)

    service = HarnessStudioService(project_store=_ProjectStore(), run_service=_RunService())
    updated = service.update_project(project_id="hp-1", user_id="u1", graph_json=state["graph_json"])

    cluster = updated["graph_json"]["agents"][0]
    assert cluster["node_kind"] == "cluster"
    assert cluster["cluster_strategy"] == "brainstorm"
    assert len(cluster["cluster_members"]) == 2
    assert cluster["cluster_auto_research"] is True
