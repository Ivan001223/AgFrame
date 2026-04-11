from app.harness.runtime.studio_service import HarnessStudioService
from app.infrastructure.config.settings import settings


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
    assert project["graph_json"]["execution_checklist"] == []
    assert project["agent_count"] >= 1
    assert project["graph_json"]["skill_pool"] == []
    assert len(project["graph_json"]["skill_catalog"]) >= 1
    assert created["user_id"] == "u1"


def test_get_current_project_exposes_tool_catalog_and_agent_capabilities(monkeypatch):
    created = {}

    monkeypatch.setattr(settings.feature_flags, "enable_tools_python_repl", False)
    monkeypatch.setattr(settings.feature_flags, "enable_tools_write_file", False)
    monkeypatch.setattr(settings.feature_flags, "enable_tools_python_executor", False)

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

    tool_catalog = {item["tool_id"]: item for item in project["graph_json"]["tool_catalog"]}
    assert "web_search" in tool_catalog
    assert tool_catalog["calculator"]["status"] == "disabled"
    assert tool_catalog["write_file"]["status"] == "disabled"
    assert tool_catalog["python_executor"]["status"] == "disabled"

    skill_catalog = {item["skill_id"]: item for item in project["graph_json"]["skill_catalog"]}
    assert skill_catalog["research"]["prompt_hint"]
    assert "web_search" in skill_catalog["research"]["suggested_tool_ids"]

    summaries = {item["agent_id"]: item for item in project["graph_json"]["agent_capability_summaries"]}
    builder_summary = summaries["agent_builder"]
    assert "web_search" in builder_summary["enabled_tool_ids"]
    assert "calculator" in builder_summary["disabled_tool_ids"]
    assert builder_summary["mcp_server_ids"] == []
    assert builder_summary["execution_contract"]["skill_execution_mode"] == "guidance_only"
    assert builder_summary["execution_contract"]["tool_access_mode"] == "direct_execution"
    assert "web_search" in builder_summary["execution_contract"]["executable_tool_ids"]
    assert builder_summary["execution_contract"]["mcp_access_mode"] == "none"
    assert builder_summary["delegation_lane_ids"]
    assert builder_summary["delegation_focus"]
    assert builder_summary["recommended_collaborators"]
    assert "MCP servers: none configured in this project." in str(builder_summary["capability_brief"])
    assert "Delegation lanes:" in str(builder_summary["capability_brief"])
    assert "Delegation focus:" in str(builder_summary["capability_brief"])
    assert created["user_id"] == "u1"


def test_update_project_marks_tools_provider_limited_for_local_qwen():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_researcher",
                    "name": "Researcher",
                    "model": "local-qwen3-vl",
                    "skill_ids": ["research"],
                    "skill_intents": [],
                }
            ],
            "edges": [],
            "skill_pool": [
                {
                    "skill_id": "research",
                    "title": "Research",
                    "source": "app/skills/research",
                    "status": "loaded",
                }
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    summary = summaries["agent_researcher"]
    assert summary["enabled_tool_ids"] == []
    assert "web_search" in summary["provider_limited_tool_ids"]
    assert summary["tool_execution_support"] == "unsupported"
    assert "local Qwen" in str(summary["tool_execution_support_reason"])
    assert summary["execution_contract"]["tool_access_mode"] == "planning_only"
    assert summary["execution_contract"]["executable_tool_ids"] == []
    assert "web_search" in summary["execution_contract"]["planning_only_tool_ids"]
    assert "Provider-limited tools" in str(summary["capability_brief"])


def test_update_project_marks_agent_definition_unavailable_when_required_capabilities_missing(monkeypatch):
    monkeypatch.setattr(
        settings.mcp,
        "servers",
        [
            {
                "server_id": "github",
                "title": "GitHub",
                "description": "Repository and issue server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "enabled": False,
            },
        ],
    )
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "model": "local-qwen3-vl",
                    "required_skill_ids": ["research"],
                    "required_tool_ids": ["web_search"],
                    "allowed_tool_ids": ["web_search"],
                    "required_mcp_server_ids": ["GitHub"],
                    "requires_tool_calling": True,
                    "skill_ids": [],
                    "skill_intents": [],
                }
            ],
            "edges": [],
            "skill_pool": [],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    summary = summaries["agent_builder"]
    assert summary["required_skill_ids"] == ["research"]
    assert summary["missing_required_skill_ids"] == ["research"]
    assert summary["required_tool_ids"] == ["web_search"]
    assert summary["missing_required_tool_ids"] == ["web_search"]
    assert summary["required_mcp_server_ids"] == ["github"]
    assert summary["missing_required_mcp_server_ids"] == ["github"]
    assert summary["requires_tool_calling"] is True
    assert summary["availability_status"] == "unavailable"
    assert any("Definition requires approved skills" in item for item in summary["availability_blockers"])
    assert any("Definition requires tools that the current provider route" in item for item in summary["availability_blockers"])
    assert any("Definition requires enabled MCP servers" in item for item in summary["availability_blockers"])
    assert any("direct tool-calling support" in item for item in summary["availability_blockers"])
    assert "Availability: unavailable" in str(summary["capability_brief"])


def test_update_project_syncs_configured_mcp_inventory_and_agent_matches(monkeypatch):
    monkeypatch.setattr(
        settings.mcp,
        "servers",
        [
            {
                "server_id": "fetch",
                "title": "Fetch",
                "description": "Browser and HTTP retrieval server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-fetch"],
                "enabled": True,
            },
            {
                "server_id": "browser",
                "title": "Browser",
                "description": "Interactive browser automation server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-browser"],
                "enabled": True,
            },
            {
                "server_id": "github",
                "title": "GitHub",
                "description": "Repository and issue server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "enabled": False,
            },
        ],
    )
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_researcher",
                    "name": "Researcher",
                    "skill_ids": ["research"],
                    "skill_intents": [],
                }
            ],
            "edges": [],
            "skill_pool": [
                {
                    "skill_id": "research",
                    "title": "Research",
                    "source": "app/skills/research",
                    "status": "loaded",
                }
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    mcp_catalog = {item["server_id"]: item for item in updated["graph_json"]["mcp_server_catalog"]}
    assert mcp_catalog["fetch"]["status"] == "enabled"
    assert mcp_catalog["browser"]["command_preview"]
    assert mcp_catalog["github"]["status"] == "disabled"

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    summary = summaries["agent_researcher"]
    assert summary["mcp_server_ids"] == ["fetch", "browser"]
    assert summary["missing_mcp_server_ids"] == []
    assert summary["execution_contract"]["mcp_access_mode"] == "planning_only"
    assert summary["execution_contract"]["planning_only_mcp_server_ids"] == ["fetch", "browser"]
    assert "external research" in str(summary["delegation_focus"])
    assert "MCP servers: fetch, browser." in str(summary["capability_brief"])


def test_update_project_records_missing_relevant_mcp_inventory(monkeypatch):
    monkeypatch.setattr(
        settings.mcp,
        "servers",
        [
            {
                "server_id": "browser",
                "title": "Browser",
                "description": "Interactive browser automation server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-browser"],
                "enabled": True,
            },
            {
                "server_id": "github",
                "title": "GitHub",
                "description": "Repository and issue server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "enabled": False,
            },
        ],
    )
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "skill_ids": ["tools"],
                    "skill_intents": [],
                }
            ],
            "edges": [],
            "skill_pool": [
                {
                    "skill_id": "tools",
                    "title": "Tools",
                    "source": "app/skills/tools",
                    "status": "loaded",
                }
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    summary = summaries["agent_builder"]
    assert summary["mcp_server_ids"] == ["browser"]
    assert summary["missing_mcp_server_ids"] == ["filesystem", "github"]
    assert "repository workflows once GitHub MCP is enabled" in str(summary["delegation_focus"])
    assert "Relevant MCP servers not enabled in this project: filesystem, github." in str(summary["capability_brief"])


def test_request_skills_returns_directly_available_pool_skills():
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
    assert [item["skill_id"] for item in updated["skill_request_result"]["created_requests"]] == ["rag"]
    assert [item["skill_id"] for item in updated["graph_json"]["pending_skill_requests"]] == ["rag"]
    assert "rag" not in {item["skill_id"] for item in updated["graph_json"]["skill_pool"]}


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
    assert "research" in {item["skill_id"] for item in updated["graph_json"]["skill_pool"]}


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
            "execution_checklist": [
                {"item_id": "check_1", "content": "Investigate current behavior", "status": "completed"},
                {"item_id": "check_2", "content": "Implement the agreed change", "status": "pending"},
            ],
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
    assert created["input_json"]["graph"]["selected_scope_orchestration_summary"]["total_agent_count"] == 1
    assert created["input_json"]["graph"]["selected_scope_orchestration_summary"]["start_agents"][0]["agent_id"] == "agent_b"
    assert created["input_json"]["task_checklist"][1]["content"] == "Implement the agreed change"
    assert created["metadata_json"]["checklist_count"] == 2
    assert created["metadata_json"]["open_checklist_count"] == 1
    assert created["metadata_json"]["review_agent_enabled"] is True
    assert created["metadata_json"]["graph_weak_edge_count"] == 0
    assert created["metadata_json"]["graph_best_next_count"] == 0
    assert created["metadata_json"]["graph_total_agent_count"] == 2
    assert created["metadata_json"]["graph_ready_agent_count"] == 2
    assert created["metadata_json"]["scope_total_agent_count"] == 1
    assert created["metadata_json"]["scope_ready_agent_count"] == 1
    assert created["metadata_json"]["handoff_diagnostic_scope"] == "selected_agents"
    assert created["metadata_json"]["handoff_scope_weak_edge_count"] == 0
    assert created["metadata_json"]["handoff_scope_best_next_count"] == 0


def test_create_orchestration_run_scopes_handoff_diagnostics_for_selected_agents():
    project = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "created_at": 1,
        "updated_at": 1,
    }
    hydrated = {
        **project,
        "pending_skill_request_count": 0,
        "graph_json": {
            "agents": [
                {"agent_id": "agent_a", "name": "Agent A"},
                {"agent_id": "agent_b", "name": "Agent B"},
                {"agent_id": "agent_c", "name": "Agent C"},
            ],
            "edges": [],
            "knowledge_base_ids": [],
            "execution_checklist": [],
            "agent_capability_summaries": [
                {
                    "agent_id": "agent_a",
                    "delegation_contract": {
                        "primary_role_mode": "coordinator",
                        "work_strategy": "synthesize_and_route",
                        "should_coordinate_parallel_work": True,
                        "should_produce_final_output": False,
                    },
                    "execution_contract": {
                        "skill_execution_mode": "guidance_only",
                        "approved_skill_ids": ["research"],
                        "tool_access_mode": "direct_execution",
                        "executable_tool_ids": ["web_search"],
                        "planning_only_tool_ids": [],
                        "mcp_access_mode": "planning_only",
                        "planning_only_mcp_server_ids": ["fetch"],
                        "missing_mcp_server_ids": [],
                    },
                },
                {
                    "agent_id": "agent_b",
                    "delegation_contract": {
                        "primary_role_mode": "implementation",
                        "work_strategy": "self_contained_delivery",
                        "should_coordinate_parallel_work": False,
                        "should_produce_final_output": True,
                    },
                    "execution_contract": {
                        "skill_execution_mode": "guidance_only",
                        "approved_skill_ids": ["tools"],
                        "tool_access_mode": "planning_only",
                        "executable_tool_ids": [],
                        "planning_only_tool_ids": ["write_file"],
                        "mcp_access_mode": "none",
                        "planning_only_mcp_server_ids": [],
                        "missing_mcp_server_ids": [],
                    },
                },
                {
                    "agent_id": "agent_c",
                    "delegation_contract": {
                        "primary_role_mode": "verification",
                        "work_strategy": "verify_and_close",
                        "should_coordinate_parallel_work": False,
                        "should_produce_final_output": False,
                    },
                    "execution_contract": {
                        "skill_execution_mode": "guidance_only",
                        "approved_skill_ids": ["github"],
                        "tool_access_mode": "none",
                        "executable_tool_ids": [],
                        "planning_only_tool_ids": [],
                        "mcp_access_mode": "planning_only",
                        "planning_only_mcp_server_ids": ["github"],
                        "missing_mcp_server_ids": [],
                    },
                },
            ],
            "review_agent": {"enabled": True},
            "graph_diagnostics": {
                "weak_edge_count": 1,
                "best_next_count": 2,
                "weak_downstream_edges": [
                    {
                        "source_agent_id": "agent_a",
                        "source_agent_name": "Agent A",
                        "source_lane_ids": ["coordination"],
                        "delegation_focus": "Keep the implementation handoff tight.",
                        "target": {"agent_id": "agent_c", "agent_name": "Agent C"},
                        "suggested_replacements": [
                            {"agent_id": "agent_b", "agent_name": "Agent B", "score": 72, "fit": "good"}
                        ],
                    }
                ],
                "best_next_handoffs": [
                    {
                        "source_agent_id": "agent_a",
                        "source_agent_name": "Agent A",
                        "source_lane_ids": ["coordination"],
                        "delegation_focus": "Keep the implementation handoff tight.",
                        "target": {"agent_id": "agent_b", "agent_name": "Agent B"},
                    },
                    {
                        "source_agent_id": "agent_b",
                        "source_agent_name": "Agent B",
                        "source_lane_ids": ["implementation"],
                        "delegation_focus": "Close the loop with the finisher.",
                        "target": {"agent_id": "agent_c", "agent_name": "Agent C"},
                    },
                ],
            },
        },
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
    service._hydrate_project = lambda _: hydrated

    service.create_orchestration_run(
        project_id="hp-1",
        user_id="u1",
        run_scope="selected",
        agent_ids=["agent_a", "agent_b"],
        loop_count=1,
    )

    assert created["metadata_json"]["graph_weak_edge_count"] == 1
    assert created["metadata_json"]["graph_best_next_count"] == 2
    assert created["metadata_json"]["handoff_diagnostic_scope"] == "selected_agents"
    assert created["metadata_json"]["handoff_scope_weak_edge_count"] == 0
    assert created["metadata_json"]["handoff_scope_best_next_count"] == 1
    assert created["metadata_json"]["graph_direct_execution_agent_count"] == 1
    assert created["metadata_json"]["graph_planning_only_tool_agent_count"] == 1
    assert created["metadata_json"]["graph_planning_only_mcp_agent_count"] == 2
    assert created["metadata_json"]["graph_coordinator_agent_count"] == 1
    assert created["metadata_json"]["graph_parallel_coordinator_agent_count"] == 1
    assert created["metadata_json"]["graph_final_output_agent_count"] == 1
    assert created["metadata_json"]["graph_verification_agent_count"] == 1
    assert created["metadata_json"]["scope_direct_execution_agent_count"] == 1
    assert created["metadata_json"]["scope_planning_only_tool_agent_count"] == 1
    assert created["metadata_json"]["scope_planning_only_mcp_agent_count"] == 1
    assert created["metadata_json"]["scope_coordinator_agent_count"] == 1
    assert created["metadata_json"]["scope_parallel_coordinator_agent_count"] == 1
    assert created["metadata_json"]["scope_final_output_agent_count"] == 1
    assert created["metadata_json"]["scope_verification_agent_count"] == 0


def test_update_project_builds_graph_diagnostics_for_weak_edges():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_planner",
                    "name": "Planner",
                    "role": "coordinator",
                    "skill_ids": ["memory"],
                    "skill_intents": [],
                    "allowed_tool_ids": ["web_search"],
                    "allowed_mcp_server_ids": ["fetch"],
                },
                {
                    "agent_id": "agent_peer",
                    "name": "Peer",
                    "role": "coordinator",
                    "skill_ids": ["memory"],
                    "skill_intents": [],
                },
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "role": "implementation",
                    "skill_ids": ["tools"],
                    "skill_intents": [],
                },
            ],
            "edges": [
                {
                    "edge_id": "edge_plan_peer",
                    "source_agent_id": "agent_planner",
                    "target_agent_id": "agent_peer",
                    "interaction": "handoff",
                }
            ],
            "skill_pool": [
                {"skill_id": "memory", "title": "Memory", "source": "app/skills/memory", "status": "loaded"},
                {"skill_id": "tools", "title": "Tools", "source": "app/skills/tools", "status": "loaded"},
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    class _RunService:
        def create_run(self, **kwargs):
            return {"run_id": "hr-1", **kwargs}

    service = HarnessStudioService(project_store=_ProjectStore(), run_service=_RunService())
    updated = service.update_project(project_id="hp-1", user_id="u1", graph_json=state["graph_json"])

    diagnostics = updated["graph_json"]["graph_diagnostics"]
    assert diagnostics["weak_edge_count"] == 1
    assert diagnostics["best_next_count"] >= 1
    assert diagnostics["weak_downstream_edges"][0]["source_agent_id"] == "agent_planner"
    assert diagnostics["weak_downstream_edges"][0]["target"]["agent_id"] == "agent_peer"
    assert diagnostics["weak_downstream_edges"][0]["suggested_replacements"][0]["agent_id"] == "agent_builder"
    assert diagnostics["best_next_handoffs"][0]["target"]["agent_id"] == "agent_builder"


def test_update_project_builds_orchestration_summary_for_phase_routing_and_repairs(monkeypatch):
    monkeypatch.setattr(
        settings.mcp,
        "servers",
        [
            {
                "server_id": "fetch",
                "title": "Fetch",
                "description": "Browser and HTTP retrieval server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-fetch"],
                "enabled": True,
            },
            {
                "server_id": "browser",
                "title": "Browser",
                "description": "Interactive browser automation server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-browser"],
                "enabled": True,
            },
            {
                "server_id": "github",
                "title": "GitHub",
                "description": "Repository and issue server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "enabled": False,
            },
        ],
    )
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_planner",
                    "name": "Planner",
                    "role": "coordinator",
                    "skill_ids": ["memory"],
                    "skill_intents": [],
                    "allowed_tool_ids": ["web_search"],
                    "allowed_mcp_server_ids": ["fetch"],
                },
                {
                    "agent_id": "agent_researcher",
                    "name": "Researcher",
                    "role": "research",
                    "skill_ids": ["research"],
                    "skill_intents": [],
                },
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "role": "implementation",
                    "skill_ids": ["tools"],
                    "skill_intents": [],
                },
            ],
            "edges": [
                {
                    "edge_id": "edge_plan_research",
                    "source_agent_id": "agent_planner",
                    "target_agent_id": "agent_researcher",
                    "interaction": "delegate",
                },
                {
                    "edge_id": "edge_research_build",
                    "source_agent_id": "agent_researcher",
                    "target_agent_id": "agent_builder",
                    "interaction": "handoff",
                },
            ],
            "execution_checklist": [
                {
                    "item_id": "check_1",
                    "content": "Coordinate research, implementation, and verification lanes",
                    "status": "pending",
                }
            ],
            "skill_pool": [
                {"skill_id": "memory", "title": "Memory", "source": "app/skills/memory", "status": "loaded"},
                {"skill_id": "research", "title": "Research", "source": "app/skills/research", "status": "loaded"},
                {"skill_id": "tools", "title": "Tools", "source": "app/skills/tools", "status": "loaded"},
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summary = updated["graph_json"]["orchestration_summary"]
    assert summary["total_agent_count"] == 3
    assert summary["execution_step_count"] == 1
    assert summary["review_enabled"] is False
    assert summary["readiness"] == "repair"
    assert [item["agent_id"] for item in summary["start_agents"]] == ["agent_planner"]
    assert [item["agent_id"] for item in summary["terminal_agents"]] == ["agent_builder"]
    assert summary["single_owner_capability_count"] > 0
    assert summary["capability_gap_count"] > 0
    assert summary["policy_repair_agent_count"] >= 1

    phases = {item["phase_id"]: item for item in summary["phases"]}
    assert "agent_researcher" in {item["agent_id"] for item in phases["research"]["agents"]}
    assert "agent_planner" in {item["agent_id"] for item in phases["synthesis"]["agents"]}
    assert "agent_builder" in {item["agent_id"] for item in phases["implementation"]["agents"]}
    assert phases["verification"]["agent_count"] >= 1

    routing = summary["agent_routing"]
    assert routing["coordinator_anchors"][0]["agent_id"] == "agent_planner"
    assert "agent_researcher" in {item["agent_id"] for item in routing["research_anchors"]}
    assert routing["implementation_anchors"][0]["agent_id"] == "agent_builder"
    assert "agent_builder" in {item["agent_id"] for item in routing["tool_capable_anchors"]}
    assert "agent_planner" in {item["agent_id"] for item in routing["tool_capable_anchors"]}
    assert "agent_planner" in {item["agent_id"] for item in routing["mcp_capable_anchors"]}

    repair_priority_ids = [item["priority_id"] for item in summary["repair_priorities"]]
    assert "capability_gaps" in repair_priority_ids
    assert "policy_repair" in repair_priority_ids
    assert "role_profile_alignment" in repair_priority_ids
    assert "single_owner_capabilities" in repair_priority_ids
    assert "review_path" in repair_priority_ids
    assert summary["role_profile_drift_agent_count"] >= 1


def test_update_project_builds_orchestration_summary_role_profile_overlap_risk():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_research_a",
                    "name": "Research A",
                    "role": "research",
                    "skill_ids": ["research"],
                    "skill_intents": [],
                },
                {
                    "agent_id": "agent_research_b",
                    "name": "Research B",
                    "role": "research",
                    "skill_ids": ["research"],
                    "skill_intents": [],
                },
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "role": "implementation",
                    "skill_ids": ["tools"],
                    "skill_intents": [],
                },
            ],
            "edges": [
                {
                    "edge_id": "edge_research_build",
                    "source_agent_id": "agent_research_a",
                    "target_agent_id": "agent_builder",
                    "interaction": "handoff",
                },
                {
                    "edge_id": "edge_research_b_build",
                    "source_agent_id": "agent_research_b",
                    "target_agent_id": "agent_builder",
                    "interaction": "handoff",
                },
            ],
            "skill_pool": [
                {
                    "skill_id": "research",
                    "title": "Research",
                    "source": "app/skills/research",
                    "status": "loaded",
                },
                {
                    "skill_id": "tools",
                    "title": "Tools",
                    "source": "app/skills/tools",
                    "status": "loaded",
                },
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": True},
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

    summary = updated["graph_json"]["orchestration_summary"]
    assert summary["role_profile_overlap_risk_count"] >= 1
    repair_priority_ids = [item["priority_id"] for item in summary["repair_priorities"]]
    assert "role_profile_alignment" in repair_priority_ids


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


def test_update_project_preserves_execution_checklist_items():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [],
            "edges": [],
            "execution_checklist": [
                {
                    "item_id": "check_1",
                    "content": "Audit the current orchestration flow",
                    "status": "in_progress",
                    "active_form": "Auditing the current orchestration flow",
                },
                {
                    "item_id": "check_2",
                    "content": "Ship the revised harness UX",
                    "status": "pending",
                },
            ],
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

    checklist = updated["graph_json"]["execution_checklist"]
    assert len(checklist) == 2
    assert checklist[0]["status"] == "in_progress"
    assert checklist[0]["active_form"] == "Auditing the current orchestration flow"
    assert updated["open_checklist_count"] == 2


def test_update_project_builds_capability_summary_for_loaded_and_missing_skills():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "skill_ids": ["research", "rag"],
                    "skill_intents": ["tools"],
                    "fallback_provider_id": "backup_provider",
                }
            ],
            "edges": [],
            "skill_pool": [
                {
                    "skill_id": "research",
                    "title": "Research",
                    "source": "app/skills/research",
                    "status": "loaded",
                }
            ],
            "pending_skill_requests": [],
            "provider_config": {"preferred_provider_id": "project_provider"},
            "review_agent": {"enabled": True},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    summary = summaries["agent_builder"]
    assert summary["loaded_skill_ids"] == ["research"]
    assert summary["missing_skill_ids"] == ["rag"]
    assert summary["missing_skill_details"][0]["skill_id"] == "rag"
    assert summary["missing_skill_details"][0]["title"] == "RAG"
    assert "knowledge_retriever" in summary["missing_skill_details"][0]["suggested_tool_ids"]
    assert "filesystem" in summary["missing_skill_details"][0]["suggested_mcp_server_ids"]
    assert summary["readiness_status"] == "blocked"
    assert "Missing approved skills" in summary["readiness_blockers"][0]
    assert summary["loaded_skill_hints"]
    assert "external evidence" in summary["loaded_skill_hints"][0]
    assert "web_search" in summary["enabled_tool_ids"]
    assert "knowledge_retriever" in summary["enabled_tool_ids"]
    assert summary["provider_route"] == "project_provider -> backup_provider"
    assert summary["review_mode"] == "team review agent"


def test_update_project_advertises_tools_relevant_to_skills_instead_of_full_catalog():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_reader",
                    "name": "Reader",
                    "skill_ids": ["rag"],
                    "skill_intents": [],
                }
            ],
            "edges": [],
            "skill_pool": [
                {
                    "skill_id": "rag",
                    "title": "RAG",
                    "source": "app/skills/rag",
                    "status": "loaded",
                }
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    summary = summaries["agent_reader"]
    assert summary["enabled_tool_ids"] == ["knowledge_retriever", "read_document"]
    assert "web_search" not in summary["enabled_tool_ids"]
    assert "calculator" not in summary["disabled_tool_ids"]


def test_update_project_builds_collaborator_recommendations_from_lane_complementarity():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_planner",
                    "name": "Planner",
                    "role": "coordinator",
                    "skill_ids": ["memory"],
                    "skill_intents": [],
                },
                {
                    "agent_id": "agent_researcher",
                    "name": "Researcher",
                    "role": "research",
                    "skill_ids": ["research"],
                    "skill_intents": [],
                },
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "role": "implementation",
                    "skill_ids": ["tools"],
                    "skill_intents": [],
                },
            ],
            "edges": [
                {"edge_id": "edge_plan_research", "source_agent_id": "agent_planner", "target_agent_id": "agent_researcher", "interaction": "delegate"}
            ],
            "skill_pool": [
                {"skill_id": "memory", "title": "Memory", "source": "app/skills/memory", "status": "loaded"},
                {"skill_id": "research", "title": "Research", "source": "app/skills/research", "status": "loaded"},
                {"skill_id": "tools", "title": "Tools", "source": "app/skills/tools", "status": "loaded"},
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    planner_summary = summaries["agent_planner"]
    builder_summary = summaries["agent_builder"]
    assert planner_summary["delegation_lane_ids"] == ["coordination", "memory"]
    assert {item["agent_id"] for item in planner_summary["recommended_collaborators"]} >= {
        "agent_builder",
        "agent_researcher",
    }
    assert planner_summary["recommended_collaborators"][0]["fit"] in {"strong", "good"}
    assert planner_summary["downstream_handoff_scores"][0]["agent_id"] == "agent_researcher"
    assert planner_summary["downstream_handoff_scores"][0]["edge_present"] is True
    assert planner_summary["delegation_contract"]["primary_role_mode"] == "coordinator"
    assert planner_summary["delegation_contract"]["work_strategy"] == "synthesize_and_route"
    assert planner_summary["delegation_contract"]["should_coordinate_parallel_work"] is True
    assert planner_summary["delegation_contract"]["should_produce_final_output"] is False
    assert {item["agent_id"] for item in planner_summary["delegation_contract"]["downstream_agents"]} == {
        "agent_researcher"
    }
    assert {item["agent_id"] for item in planner_summary["delegation_contract"]["preferred_collaborators"]} >= {
        "agent_builder",
        "agent_researcher",
    }
    assert builder_summary["delegation_contract"]["primary_role_mode"] == "implementation"
    assert builder_summary["delegation_contract"]["work_strategy"] == "self_contained_delivery"
    assert builder_summary["delegation_contract"]["should_produce_final_output"] is True
    assert builder_summary["delegation_contract"]["upstream_agents"] == []
    assert "Recommended collaborators:" in str(planner_summary["capability_brief"])


def test_update_project_penalizes_same_role_profile_overlap_in_collaborator_recommendations(monkeypatch):
    monkeypatch.setattr(settings.feature_flags, "enable_tools_write_file", True)
    monkeypatch.setattr(settings.feature_flags, "enable_tools_python_executor", True)

    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_research_lead",
                    "name": "Lead Researcher",
                    "role": "research",
                    "skill_ids": ["research"],
                    "skill_intents": [],
                },
                {
                    "agent_id": "agent_research_peer",
                    "name": "Peer Researcher",
                    "role": "research",
                    "skill_ids": ["research"],
                    "skill_intents": [],
                },
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "role": "implementation",
                    "skill_ids": ["tools"],
                    "skill_intents": [],
                },
            ],
            "edges": [
                {
                    "edge_id": "edge_research_peer",
                    "source_agent_id": "agent_research_lead",
                    "target_agent_id": "agent_research_peer",
                    "interaction": "handoff",
                }
            ],
            "skill_pool": [
                {"skill_id": "research", "title": "Research", "source": "app/skills/research", "status": "loaded"},
                {"skill_id": "tools", "title": "Tools", "source": "app/skills/tools", "status": "loaded"},
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    lead_summary = summaries["agent_research_lead"]
    collaborator_by_id = {
        item["agent_id"]: item for item in lead_summary["recommended_collaborators"]
    }

    assert lead_summary["recommended_collaborators"][0]["agent_id"] == "agent_builder"
    assert collaborator_by_id["agent_builder"]["same_role_profile"] is False
    assert collaborator_by_id["agent_builder"]["target_profile_id"] == "implementation"
    assert collaborator_by_id["agent_builder"]["new_skill_ids"] == ["tools"]
    assert "write_file" in collaborator_by_id["agent_builder"]["new_tool_ids"]

    peer_fit = collaborator_by_id["agent_research_peer"]
    assert peer_fit["source_profile_id"] == "research"
    assert peer_fit["target_profile_id"] == "research"
    assert peer_fit["same_role_profile"] is True
    assert peer_fit["same_role_profile_overlap_risk"] is True
    assert peer_fit["edge_present"] is True
    assert peer_fit["fit"] == "weak"
    assert "research" in peer_fit["overlap_lane_ids"]
    assert "shares the same Research profile" in str(peer_fit["rationale"])
    assert peer_fit["score"] < collaborator_by_id["agent_builder"]["score"]


def test_update_project_builds_coordinator_role_profile_suggestion(monkeypatch):
    monkeypatch.setattr(
        settings.mcp,
        "servers",
        [
            {
                "server_id": "fetch",
                "title": "Fetch",
                "description": "Browser and HTTP retrieval server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-fetch"],
                "enabled": True,
            },
            {
                "server_id": "browser",
                "title": "Browser",
                "description": "Interactive browser automation server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-browser"],
                "enabled": True,
            },
        ],
    )
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_planner",
                    "name": "Planner",
                    "role": "coordinator",
                    "skill_ids": ["memory", "tools"],
                    "allowed_mcp_server_ids": ["fetch", "browser"],
                    "skill_intents": [],
                },
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "role": "implementation",
                    "skill_ids": ["tools"],
                    "skill_intents": [],
                },
            ],
            "edges": [
                {
                    "edge_id": "edge_plan_build",
                    "source_agent_id": "agent_planner",
                    "target_agent_id": "agent_builder",
                    "interaction": "delegate",
                }
            ],
            "skill_pool": [
                {"skill_id": "memory", "title": "Memory", "source": "app/skills/memory", "status": "loaded"},
                {"skill_id": "tools", "title": "Tools", "source": "app/skills/tools", "status": "loaded"},
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    planner_profile = summaries["agent_planner"]["role_profile_suggestion"]
    assert planner_profile["profile_id"] == "coordinator"
    assert planner_profile["available_skill_ids"] == ["memory"]
    assert planner_profile["missing_skill_ids"] == []
    assert planner_profile["suggested_tool_ids"] == ["get_current_time"]
    assert "get_current_time" not in planner_profile["restrictive_tool_ids"]
    assert "web_search" in planner_profile["restrictive_tool_ids"]
    assert "read_document" in planner_profile["restrictive_tool_ids"]
    assert set(planner_profile["restrictive_mcp_server_ids"]) == {"fetch", "browser"}


def test_update_project_builds_research_role_profile_with_knowledge_base_grounding(monkeypatch):
    monkeypatch.setattr(
        settings.mcp,
        "servers",
        [
            {
                "server_id": "fetch",
                "title": "Fetch",
                "description": "Browser and HTTP retrieval server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-fetch"],
                "enabled": True,
            },
            {
                "server_id": "browser",
                "title": "Browser",
                "description": "Interactive browser automation server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-browser"],
                "enabled": True,
            },
            {
                "server_id": "filesystem",
                "title": "Filesystem",
                "description": "Filesystem access server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "enabled": True,
            },
        ],
    )
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_researcher",
                    "name": "Researcher",
                    "role": "research",
                    "skill_ids": ["research"],
                    "skill_intents": [],
                }
            ],
            "edges": [],
            "knowledge_base_ids": ["kb-1"],
            "skill_pool": [
                {
                    "skill_id": "research",
                    "title": "Research",
                    "source": "app/skills/research",
                    "status": "loaded",
                }
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    researcher_profile = summaries["agent_researcher"]["role_profile_suggestion"]
    assert researcher_profile["profile_id"] == "research"
    assert researcher_profile["available_skill_ids"] == ["research"]
    assert researcher_profile["missing_skill_ids"] == ["rag"]
    assert "web_search" in researcher_profile["suggested_tool_ids"]
    assert "knowledge_retriever" in researcher_profile["suggested_tool_ids"]
    assert set(researcher_profile["suggested_mcp_server_ids"]) == {"fetch", "browser", "filesystem"}


def test_update_project_builds_implementation_role_profile_suggestion(monkeypatch):
    monkeypatch.setattr(
        settings.mcp,
        "servers",
        [
            {
                "server_id": "filesystem",
                "title": "Filesystem",
                "description": "Filesystem access server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem"],
                "enabled": True,
            },
            {
                "server_id": "github",
                "title": "GitHub",
                "description": "Repository and issue server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "enabled": True,
            },
        ],
    )
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_planner",
                    "name": "Planner",
                    "role": "coordinator",
                    "skill_ids": ["memory"],
                    "skill_intents": [],
                },
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "role": "implementation",
                    "skill_ids": ["tools"],
                    "skill_intents": [],
                }
            ],
            "edges": [
                {
                    "edge_id": "edge_plan_build",
                    "source_agent_id": "agent_planner",
                    "target_agent_id": "agent_builder",
                    "interaction": "handoff",
                }
            ],
            "skill_pool": [
                {"skill_id": "memory", "title": "Memory", "source": "app/skills/memory", "status": "loaded"},
                {"skill_id": "tools", "title": "Tools", "source": "app/skills/tools", "status": "loaded"}
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    builder_profile = summaries["agent_builder"]["role_profile_suggestion"]
    assert builder_profile["profile_id"] == "implementation"
    assert builder_profile["available_skill_ids"] == ["tools"]
    assert "read_document" in builder_profile["suggested_tool_ids"]
    assert "get_current_time" in builder_profile["suggested_tool_ids"]
    assert set(builder_profile["suggested_mcp_server_ids"]) == {"filesystem", "github"}


def test_update_project_applies_agent_tool_policy_overlay():
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_researcher",
                    "name": "Researcher",
                    "skill_ids": ["research"],
                    "skill_intents": [],
                    "allowed_tool_ids": ["knowledge_retriever", "web_search", "not_a_tool"],
                    "denied_tool_ids": ["web_search"],
                }
            ],
            "edges": [],
            "skill_pool": [
                {
                    "skill_id": "research",
                    "title": "Research",
                    "source": "app/skills/research",
                    "status": "loaded",
                }
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    summary = summaries["agent_researcher"]
    assert summary["configured_allowed_tool_ids"] == ["knowledge_retriever", "web_search"]
    assert summary["configured_denied_tool_ids"] == ["web_search"]
    assert summary["enabled_tool_ids"] == ["knowledge_retriever"]
    assert summary["policy_added_tool_ids"] == ["knowledge_retriever"]
    assert summary["policy_blocked_tool_ids"] == ["get_current_time", "read_document", "web_search"]
    assert summary["unknown_allowed_tool_ids"] == ["not_a_tool"]
    assert "Tool policy:" in str(summary["capability_brief"])
    assert "Tools blocked by node policy:" in str(summary["capability_brief"])
    assert "Unknown tool ids in node policy:" in str(summary["capability_brief"])


def test_update_project_applies_agent_mcp_policy_overlay(monkeypatch):
    monkeypatch.setattr(
        settings.mcp,
        "servers",
        [
            {
                "server_id": "browser",
                "title": "Browser",
                "description": "Interactive browser automation server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-browser"],
                "enabled": True,
            },
            {
                "server_id": "github",
                "title": "GitHub",
                "description": "Repository and issue server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "enabled": False,
            },
        ],
    )
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "skill_ids": ["tools"],
                    "skill_intents": [],
                    "allowed_mcp_server_ids": ["github", "fetch", "unknown_server"],
                    "denied_mcp_server_ids": ["fetch"],
                }
            ],
            "edges": [],
            "skill_pool": [
                {
                    "skill_id": "tools",
                    "title": "Tools",
                    "source": "app/skills/tools",
                    "status": "loaded",
                }
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    summary = summaries["agent_builder"]
    assert summary["configured_allowed_mcp_server_ids"] == ["github"]
    assert summary["configured_denied_mcp_server_ids"] == []
    assert summary["mcp_server_ids"] == []
    assert summary["missing_mcp_server_ids"] == ["github", "unknown_server"]
    assert summary["missing_mcp_server_details"][0]["server_id"] == "github"
    assert summary["missing_mcp_server_details"][0]["status"] == "disabled"
    assert summary["missing_mcp_server_details"][1]["server_id"] == "unknown_server"
    assert "No project MCP server inventory entry" in str(summary["missing_mcp_server_details"][1]["description"])
    assert summary["readiness_status"] == "limited"
    assert "Relevant MCP servers are not enabled" in summary["readiness_warnings"][0]
    assert summary["policy_added_mcp_server_ids"] == ["fetch", "unknown_server"]
    assert summary["policy_blocked_mcp_server_ids"] == ["filesystem", "browser", "fetch"]
    assert summary["unknown_allowed_mcp_server_ids"] == ["fetch", "unknown_server"]
    assert "MCP policy:" in str(summary["capability_brief"])
    assert "MCP servers blocked by node policy:" in str(summary["capability_brief"])
    assert "Unknown MCP server ids in node policy:" in str(summary["capability_brief"])


def test_update_project_matches_mcp_aliases_to_inventory_entries(monkeypatch):
    monkeypatch.setattr(
        settings.mcp,
        "servers",
        [
            {
                "server_id": "github_official",
                "title": "GitHub",
                "description": "Repository and issue server.",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "enabled": True,
            },
        ],
    )
    state = {
        "project_id": "hp-1",
        "user_id": "u1",
        "name": "Studio",
        "description": None,
        "graph_json": {
            "agents": [
                {
                    "agent_id": "agent_builder",
                    "name": "Builder",
                    "skill_ids": ["tools"],
                    "skill_intents": [],
                    "allowed_mcp_server_ids": ["GitHub", "@modelcontextprotocol/server-github"],
                }
            ],
            "edges": [],
            "skill_pool": [
                {
                    "skill_id": "tools",
                    "title": "Tools",
                    "source": "app/skills/tools",
                    "status": "loaded",
                }
            ],
            "pending_skill_requests": [],
            "provider_config": {},
            "review_agent": {"enabled": False},
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

    summaries = {item["agent_id"]: item for item in updated["graph_json"]["agent_capability_summaries"]}
    summary = summaries["agent_builder"]
    assert summary["configured_allowed_mcp_server_ids"] == ["github_official"]
    assert summary["mcp_server_ids"] == ["github_official"]
    assert summary["missing_mcp_server_ids"] == []
    assert summary["unknown_allowed_mcp_server_ids"] == []
    assert summary["policy_added_mcp_server_ids"] == []
    assert summary["policy_blocked_mcp_server_ids"] == ["filesystem", "browser"]
