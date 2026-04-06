from app.harness.runtime.verification_service import VerificationService


def test_build_agent_orchestration_result_includes_handoff_diagnostics():
    service = VerificationService()

    result = service.build_agent_orchestration_result(
        ok=True,
        active_agent_ids=["agent_planner", "agent_builder", "agent_researcher"],
        blocked_agents=[],
        loop_count=1,
        review_agent_enabled=True,
        error_code=None,
        error_message=None,
        capability_snapshot={
            "agent_capabilities": [
                {
                    "agent_id": "agent_planner",
                    "agent_name": "Planner",
                    "delegation_lane_ids": ["coordination", "memory"],
                    "delegation_focus": "Break work into reliable handoffs.",
                    "recommended_collaborators": [
                        {
                            "agent_id": "agent_builder",
                            "agent_name": "Builder",
                            "score": 84,
                            "fit": "strong",
                            "rationale": "adds implementation lane coverage",
                            "new_skill_ids": ["tools"],
                            "complementary_lane_ids": ["implementation"],
                            "new_tool_ids": ["python_executor"],
                            "new_mcp_server_ids": ["github"],
                            "gap_cover_mcp_server_ids": ["fetch"],
                            "source_profile_id": "coordinator",
                            "target_profile_id": "implementation",
                            "same_role_profile": False,
                            "same_role_profile_overlap_risk": False,
                        },
                        {
                            "agent_id": "agent_researcher",
                            "agent_name": "Researcher",
                            "score": 76,
                            "fit": "good",
                            "rationale": "covers missing MCP like fetch",
                            "complementary_lane_ids": ["research"],
                        },
                    ],
                    "downstream_handoff_scores": [
                        {
                            "agent_id": "agent_reviewer",
                            "agent_name": "Reviewer",
                            "score": 18,
                            "fit": "weak",
                            "rationale": "mostly overlaps the current node and adds limited new capacity",
                            "source_profile_id": "coordinator",
                            "target_profile_id": "coordinator",
                            "same_role_profile": True,
                            "same_role_profile_overlap_risk": True,
                            "edge_present": True,
                            "interaction": "handoff",
                        }
                    ],
                }
            ]
        },
    )

    assert result["status"] == "pass"
    assert "handoff_fit_diagnostics" in result["checks_run"]
    assert "1 weak downstream handoff(s) flagged" in result["summary"]

    diagnostics = result["artifacts"]["handoff_diagnostics"]
    assert diagnostics["weak_edge_count"] == 1
    assert diagnostics["best_next_count"] == 1
    assert diagnostics["weak_downstream_edges"][0]["source_agent_id"] == "agent_planner"
    assert diagnostics["weak_downstream_edges"][0]["target"]["agent_id"] == "agent_reviewer"
    assert diagnostics["weak_downstream_edges"][0]["target"]["source_profile_id"] == "coordinator"
    assert diagnostics["weak_downstream_edges"][0]["target"]["target_profile_id"] == "coordinator"
    assert diagnostics["weak_downstream_edges"][0]["target"]["same_role_profile"] is True
    assert diagnostics["weak_downstream_edges"][0]["target"]["same_role_profile_overlap_risk"] is True
    assert diagnostics["weak_downstream_edges"][0]["suggested_replacements"][0]["agent_id"] == "agent_builder"
    assert diagnostics["best_next_handoffs"][0]["target"]["agent_id"] == "agent_builder"
    assert diagnostics["best_next_handoffs"][0]["target"]["new_skill_ids"] == ["tools"]
    assert diagnostics["best_next_handoffs"][0]["target"]["new_tool_ids"] == ["python_executor"]
    assert diagnostics["best_next_handoffs"][0]["target"]["new_mcp_server_ids"] == ["github"]
    assert diagnostics["best_next_handoffs"][0]["target"]["gap_cover_mcp_server_ids"] == ["fetch"]
    assert diagnostics["best_next_handoffs"][0]["target"]["source_profile_id"] == "coordinator"
    assert diagnostics["best_next_handoffs"][0]["target"]["target_profile_id"] == "implementation"
    assert diagnostics["best_next_handoffs"][0]["target"]["same_role_profile"] is False
    assert diagnostics["best_next_handoffs"][0]["target"]["same_role_profile_overlap_risk"] is False


def test_build_agent_orchestration_result_skips_handoff_diagnostics_without_signal():
    service = VerificationService()

    result = service.build_agent_orchestration_result(
        ok=True,
        active_agent_ids=["agent_planner"],
        blocked_agents=[],
        loop_count=1,
        review_agent_enabled=False,
        error_code=None,
        error_message=None,
        capability_snapshot={"agent_capabilities": [{"agent_id": "agent_planner", "agent_name": "Planner"}]},
    )

    assert result["status"] == "pass"
    assert result["checks_run"] == ["agent_orchestration_cycle"]
    assert "handoff_diagnostics" not in result["artifacts"]
    assert result["summary"] == "agent orchestration completed"


def test_build_agent_orchestration_result_records_selected_handoff_scope():
    service = VerificationService()

    result = service.build_agent_orchestration_result(
        ok=True,
        active_agent_ids=["agent_a"],
        blocked_agents=[],
        loop_count=1,
        review_agent_enabled=False,
        error_code=None,
        error_message=None,
        capability_snapshot={
            "handoff_diagnostic_scope": "selected_agents",
            "agent_capabilities": [{"agent_id": "agent_a", "agent_name": "Planner"}],
        },
        selected_agent_ids=["agent_a"],
    )

    assert result["artifacts"]["handoff_diagnostics_scope"] == {
        "scope": "selected_agents",
        "active_agent_ids": ["agent_a"],
        "selected_agent_ids": ["agent_a"],
    }


def test_build_agent_orchestration_result_includes_capability_readiness_diagnostics():
    service = VerificationService()

    result = service.build_agent_orchestration_result(
        ok=True,
        active_agent_ids=["agent_a", "agent_b"],
        blocked_agents=[],
        loop_count=1,
        review_agent_enabled=False,
        error_code=None,
        error_message=None,
        capability_snapshot={
            "agent_capabilities": [
                {
                    "agent_id": "agent_a",
                    "agent_name": "Planner",
                    "readiness_status": "blocked",
                    "readiness_blockers": ["Missing approved skills before this node can run: research"],
                    "missing_skill_ids": ["research"],
                    "missing_skill_details": [{"skill_id": "research", "title": "Research"}],
                },
                {
                    "agent_id": "agent_b",
                    "agent_name": "Builder",
                    "readiness_status": "limited",
                    "readiness_warnings": ["Relevant MCP servers are not enabled in project inventory: github"],
                    "missing_mcp_server_ids": ["github"],
                    "provider_limited_tool_ids": ["web_search"],
                    "tool_execution_support": "unknown",
                    "requires_tool_calling": True,
                    "provider_route": "project default",
                },
            ]
        },
    )

    assert "capability_readiness" in result["checks_run"]
    assert "blocked capability profile(s) flagged" in result["summary"]
    readiness = result["artifacts"]["capability_readiness"]
    assert readiness["blocked_count"] == 1
    assert readiness["limited_count"] == 1
    assert readiness["agents"][0]["status"] == "blocked"
    assert readiness["agents"][0]["missing_skill_ids"] == ["research"]
    assert readiness["agents"][0]["recovery_actions"]["open_skill_pool"] is True
    assert readiness["agents"][1]["status"] == "limited"
    assert readiness["agents"][1]["missing_mcp_server_ids"] == ["github"]
    assert readiness["agents"][1]["recovery_actions"]["open_project_mcp_inventory"] is True
    assert readiness["agents"][1]["recovery_actions"]["open_project_providers"] is True


def test_build_agent_orchestration_result_includes_capability_availability_diagnostics():
    service = VerificationService()

    result = service.build_agent_orchestration_result(
        ok=True,
        active_agent_ids=["agent_a", "agent_b"],
        blocked_agents=[],
        loop_count=1,
        review_agent_enabled=False,
        error_code=None,
        error_message=None,
        capability_snapshot={
            "agent_capabilities": [
                {
                    "agent_id": "agent_a",
                    "agent_name": "Planner",
                    "availability_status": "unavailable",
                    "availability_blockers": ["Definition requires enabled MCP servers that are not currently available: github"],
                    "missing_required_mcp_server_ids": ["github"],
                },
                {
                    "agent_id": "agent_b",
                    "agent_name": "Builder",
                    "availability_status": "limited",
                    "availability_warnings": ["Definition expects direct tool-calling support, but the current provider route is not verified"],
                    "missing_required_skill_ids": ["research"],
                    "requires_tool_calling": True,
                    "tool_execution_support": "unknown",
                    "provider_route": "project default",
                },
            ]
        },
    )

    assert "capability_availability" in result["checks_run"]
    assert "unavailable capability definition(s) flagged" in result["summary"]
    availability = result["artifacts"]["capability_availability"]
    assert availability["unavailable_count"] == 1
    assert availability["limited_count"] == 1
    assert availability["agents"][0]["status"] == "unavailable"
    assert availability["agents"][0]["missing_required_mcp_server_ids"] == ["github"]
    assert availability["agents"][0]["recovery_actions"]["open_project_mcp_inventory"] is True
    assert availability["agents"][1]["status"] == "limited"
    assert availability["agents"][1]["missing_required_skill_ids"] == ["research"]
    assert availability["agents"][1]["recovery_actions"]["open_skill_pool"] is True
    assert availability["agents"][1]["recovery_actions"]["open_project_providers"] is True


def test_build_agent_orchestration_result_includes_execution_contract_diagnostics():
    service = VerificationService()

    result = service.build_agent_orchestration_result(
        ok=True,
        active_agent_ids=["agent_a", "agent_b"],
        blocked_agents=[],
        loop_count=1,
        review_agent_enabled=False,
        error_code=None,
        error_message=None,
        capability_snapshot={
            "agent_capabilities": [
                {
                    "agent_id": "agent_a",
                    "agent_name": "Researcher",
                    "loaded_skill_ids": ["research"],
                    "enabled_tool_ids": [],
                    "provider_limited_tool_ids": ["web_search"],
                    "mcp_server_ids": ["fetch"],
                    "missing_mcp_server_ids": ["github"],
                    "requires_tool_calling": True,
                    "tool_execution_support": "unsupported",
                    "tool_execution_support_reason": "This runtime adapter does not expose native tool binding.",
                    "provider_route": "project default",
                    "execution_contract": {
                        "skill_execution_mode": "guidance_only",
                        "approved_skill_ids": ["research"],
                        "tool_access_mode": "planning_only",
                        "executable_tool_ids": [],
                        "planning_only_tool_ids": ["web_search"],
                        "mcp_access_mode": "planning_only",
                        "planning_only_mcp_server_ids": ["fetch"],
                        "missing_mcp_server_ids": ["github"],
                    },
                },
                {
                    "agent_id": "agent_b",
                    "agent_name": "Builder",
                    "loaded_skill_ids": ["tools"],
                    "enabled_tool_ids": ["get_current_time"],
                    "execution_contract": {
                        "skill_execution_mode": "guidance_only",
                        "approved_skill_ids": ["tools"],
                        "tool_access_mode": "direct_execution",
                        "executable_tool_ids": ["get_current_time"],
                        "planning_only_tool_ids": [],
                        "mcp_access_mode": "none",
                        "planning_only_mcp_server_ids": [],
                    },
                },
            ]
        },
    )

    assert "capability_execution_contracts" in result["checks_run"]
    assert "planning-only tool contract(s) flagged" in result["summary"]
    contracts = result["artifacts"]["capability_execution_contracts"]
    assert contracts["agent_count"] == 2
    assert contracts["planning_only_tool_agent_count"] == 1
    assert contracts["planning_only_mcp_agent_count"] == 1
    assert contracts["direct_execution_agent_count"] == 1
    assert contracts["agents"][0]["tool_access_mode"] == "planning_only"
    assert contracts["agents"][0]["planning_only_tool_ids"] == ["web_search"]
    assert contracts["agents"][0]["planning_only_mcp_server_ids"] == ["fetch"]
    assert contracts["agents"][0]["recovery_actions"]["open_project_mcp_inventory"] is True
    assert contracts["agents"][0]["recovery_actions"]["open_project_providers"] is True
    assert contracts["agents"][1]["tool_access_mode"] == "direct_execution"
    assert contracts["agents"][1]["executable_tool_ids"] == ["get_current_time"]


def test_build_agent_orchestration_result_includes_collaboration_contract_diagnostics():
    service = VerificationService()

    result = service.build_agent_orchestration_result(
        ok=True,
        active_agent_ids=["agent_a", "agent_b"],
        blocked_agents=[],
        loop_count=1,
        review_agent_enabled=False,
        error_code=None,
        error_message=None,
        capability_snapshot={
            "agent_capabilities": [
                {
                    "agent_id": "agent_a",
                    "agent_name": "Researcher",
                    "delegation_contract": {
                        "primary_role_mode": "research",
                        "work_strategy": "gather_then_handoff",
                        "should_coordinate_parallel_work": False,
                        "should_produce_final_output": False,
                        "preferred_collaborators": [
                            {"agent_id": "agent_b", "agent_name": "Implementer"},
                        ],
                    },
                },
                {
                    "agent_id": "agent_b",
                    "agent_name": "Implementer",
                    "delegation_contract": {
                        "primary_role_mode": "implementation",
                        "work_strategy": "implement_then_handoff",
                        "should_coordinate_parallel_work": True,
                        "should_produce_final_output": False,
                        "downstream_agents": [],
                        "upstream_agents": [
                            {"agent_id": "agent_a", "agent_name": "Researcher"},
                        ],
                    },
                },
            ]
        },
    )

    assert "collaboration_contracts" in result["checks_run"]
    assert "4 collaboration contract risk(s) flagged" in result["summary"]
    diagnostics = result["artifacts"]["collaboration_contracts"]
    assert diagnostics["agent_count"] == 2
    assert diagnostics["coordinator_agent_count"] == 0
    assert diagnostics["parallel_coordinator_agent_count"] == 1
    assert diagnostics["final_output_agent_count"] == 0
    assert diagnostics["verification_agent_count"] == 0
    assert diagnostics["risk_count"] == 4
    assert diagnostics["parallel_coordinator_agents"] == [
        {"agent_id": "agent_b", "agent_name": "Implementer"}
    ]
    risk_ids = {risk["risk_id"] for risk in diagnostics["risks"]}
    assert risk_ids == {
        "missing_coordinator",
        "missing_final_output_owner",
        "missing_verification_owner",
        "coordinator_without_downstream",
    }


def test_build_agent_orchestration_result_flags_coordinator_execution_overlap():
    service = VerificationService()

    result = service.build_agent_orchestration_result(
        ok=True,
        active_agent_ids=["agent_planner", "agent_builder"],
        blocked_agents=[],
        loop_count=1,
        review_agent_enabled=True,
        error_code=None,
        error_message=None,
        capability_snapshot={
            "agent_capabilities": [
                {
                    "agent_id": "agent_planner",
                    "agent_name": "Planner",
                    "recommended_collaborators": [
                        {
                            "agent_id": "agent_builder",
                            "agent_name": "Builder",
                            "score": 82,
                            "fit": "strong",
                            "rationale": "adds implementation lane coverage; brings bash tool access",
                            "new_skill_ids": ["tools"],
                            "complementary_lane_ids": ["implementation"],
                            "new_tool_ids": ["bash"],
                            "new_mcp_server_ids": [],
                            "gap_cover_mcp_server_ids": ["github"],
                            "source_profile_id": "coordinator",
                            "target_profile_id": "implementation",
                            "same_role_profile": False,
                            "same_role_profile_overlap_risk": False,
                            "edge_present": True,
                            "interaction": "delegate",
                        }
                    ],
                    "execution_contract": {
                        "tool_access_mode": "direct_execution",
                        "executable_tool_ids": ["write_file"],
                        "planning_only_tool_ids": [],
                        "mcp_access_mode": "planning_only",
                        "planning_only_mcp_server_ids": ["github"],
                        "missing_mcp_server_ids": [],
                    },
                    "delegation_contract": {
                        "primary_role_mode": "coordinator",
                        "work_strategy": "synthesize_and_route",
                        "should_coordinate_parallel_work": True,
                        "should_produce_final_output": False,
                        "downstream_agents": [
                            {"agent_id": "agent_builder", "agent_name": "Builder"},
                        ],
                    },
                },
                {
                    "agent_id": "agent_builder",
                    "agent_name": "Builder",
                    "execution_contract": {
                        "tool_access_mode": "direct_execution",
                        "executable_tool_ids": ["bash"],
                        "planning_only_tool_ids": [],
                        "mcp_access_mode": "none",
                        "planning_only_mcp_server_ids": [],
                        "missing_mcp_server_ids": [],
                    },
                    "delegation_contract": {
                        "primary_role_mode": "implementation",
                        "work_strategy": "self_contained_delivery",
                        "should_coordinate_parallel_work": False,
                        "should_produce_final_output": True,
                        "upstream_agents": [
                            {"agent_id": "agent_planner", "agent_name": "Planner"},
                        ],
                    },
                },
            ]
        },
    )

    diagnostics = result["artifacts"]["collaboration_contracts"]
    assert diagnostics["risk_count"] == 2
    assert diagnostics["agents"][0]["executable_tool_ids"] == ["write_file"]
    assert diagnostics["agents"][0]["planning_only_mcp_server_ids"] == ["github"]
    assert diagnostics["agents"][0]["delegate_execution_candidates"][0]["agent_id"] == "agent_builder"
    assert diagnostics["agents"][0]["delegate_mcp_candidates"][0]["agent_id"] == "agent_builder"
    risk_ids = {risk["risk_id"] for risk in diagnostics["risks"]}
    assert risk_ids == {
        "coordinator_with_execution_tools",
        "coordinator_with_mcp_context",
    }
    execution_risk = next(
        risk for risk in diagnostics["risks"] if risk["risk_id"] == "coordinator_with_execution_tools"
    )
    assert execution_risk["source_agent_id"] == "agent_planner"
    assert execution_risk["delegate_tool_ids"] == ["write_file"]
    assert execution_risk["delegate_candidates"][0]["agent_id"] == "agent_builder"
    assert "Builder" in execution_risk["recommended_action"]
    mcp_risk = next(
        risk for risk in diagnostics["risks"] if risk["risk_id"] == "coordinator_with_mcp_context"
    )
    assert mcp_risk["source_agent_id"] == "agent_planner"
    assert mcp_risk["delegate_mcp_server_ids"] == ["github"]
    assert mcp_risk["delegate_candidates"][0]["gap_cover_mcp_server_ids"] == ["github"]


def test_build_agent_orchestration_result_flags_role_profile_drift_and_overlap():
    service = VerificationService()

    result = service.build_agent_orchestration_result(
        ok=True,
        active_agent_ids=["agent_planner", "agent_research_a", "agent_research_b", "agent_builder"],
        blocked_agents=[],
        loop_count=1,
        review_agent_enabled=True,
        error_code=None,
        error_message=None,
        capability_snapshot={
            "agent_capabilities": [
                {
                    "agent_id": "agent_planner",
                    "agent_name": "Planner",
                    "loaded_skill_ids": ["tools"],
                    "enabled_tool_ids": ["web_search"],
                    "mcp_server_ids": ["fetch"],
                    "configured_denied_tool_ids": [],
                    "configured_denied_mcp_server_ids": [],
                    "delegation_lane_ids": ["coordination", "research"],
                    "execution_contract": {
                        "tool_access_mode": "none",
                        "executable_tool_ids": [],
                        "planning_only_tool_ids": [],
                        "mcp_access_mode": "none",
                        "planning_only_mcp_server_ids": [],
                        "missing_mcp_server_ids": [],
                    },
                    "delegation_contract": {
                        "primary_role_mode": "coordinator",
                        "work_strategy": "synthesize_and_route",
                        "should_coordinate_parallel_work": True,
                        "should_produce_final_output": False,
                        "downstream_agents": [
                            {"agent_id": "agent_builder", "agent_name": "Builder"},
                        ],
                    },
                    "role_profile_suggestion": {
                        "profile_id": "coordinator",
                        "available_skill_ids": ["memory"],
                        "missing_skill_ids": [],
                        "suggested_tool_ids": ["get_current_time"],
                        "suggested_mcp_server_ids": [],
                        "restrictive_tool_ids": ["web_search"],
                        "restrictive_mcp_server_ids": ["fetch"],
                    },
                },
                {
                    "agent_id": "agent_research_a",
                    "agent_name": "Research A",
                    "loaded_skill_ids": ["research"],
                    "enabled_tool_ids": ["web_search"],
                    "mcp_server_ids": ["fetch"],
                    "configured_denied_tool_ids": [],
                    "configured_denied_mcp_server_ids": [],
                    "delegation_lane_ids": ["research"],
                    "execution_contract": {
                        "tool_access_mode": "direct_execution",
                        "executable_tool_ids": ["web_search"],
                        "planning_only_tool_ids": [],
                        "mcp_access_mode": "planning_only",
                        "planning_only_mcp_server_ids": ["fetch"],
                        "missing_mcp_server_ids": [],
                    },
                    "delegation_contract": {
                        "primary_role_mode": "research",
                        "work_strategy": "gather_then_handoff",
                        "should_coordinate_parallel_work": False,
                        "should_produce_final_output": False,
                    },
                    "role_profile_suggestion": {
                        "profile_id": "research",
                        "available_skill_ids": ["research"],
                        "missing_skill_ids": [],
                        "suggested_tool_ids": ["web_search"],
                        "suggested_mcp_server_ids": ["fetch"],
                        "restrictive_tool_ids": [],
                        "restrictive_mcp_server_ids": [],
                    },
                },
                {
                    "agent_id": "agent_research_b",
                    "agent_name": "Research B",
                    "loaded_skill_ids": ["research"],
                    "enabled_tool_ids": ["web_search"],
                    "mcp_server_ids": ["fetch"],
                    "configured_denied_tool_ids": [],
                    "configured_denied_mcp_server_ids": [],
                    "delegation_lane_ids": ["research"],
                    "execution_contract": {
                        "tool_access_mode": "direct_execution",
                        "executable_tool_ids": ["web_search"],
                        "planning_only_tool_ids": [],
                        "mcp_access_mode": "planning_only",
                        "planning_only_mcp_server_ids": ["fetch"],
                        "missing_mcp_server_ids": [],
                    },
                    "delegation_contract": {
                        "primary_role_mode": "research",
                        "work_strategy": "gather_then_handoff",
                        "should_coordinate_parallel_work": False,
                        "should_produce_final_output": False,
                    },
                    "role_profile_suggestion": {
                        "profile_id": "research",
                        "available_skill_ids": ["research"],
                        "missing_skill_ids": [],
                        "suggested_tool_ids": ["web_search"],
                        "suggested_mcp_server_ids": ["fetch"],
                        "restrictive_tool_ids": [],
                        "restrictive_mcp_server_ids": [],
                    },
                },
                {
                    "agent_id": "agent_builder",
                    "agent_name": "Builder",
                    "loaded_skill_ids": ["tools"],
                    "enabled_tool_ids": ["write_file"],
                    "mcp_server_ids": [],
                    "configured_denied_tool_ids": [],
                    "configured_denied_mcp_server_ids": [],
                    "delegation_lane_ids": ["implementation"],
                    "execution_contract": {
                        "tool_access_mode": "direct_execution",
                        "executable_tool_ids": ["write_file"],
                        "planning_only_tool_ids": [],
                        "mcp_access_mode": "none",
                        "planning_only_mcp_server_ids": [],
                        "missing_mcp_server_ids": [],
                    },
                    "delegation_contract": {
                        "primary_role_mode": "implementation",
                        "work_strategy": "self_contained_delivery",
                        "should_coordinate_parallel_work": False,
                        "should_produce_final_output": True,
                        "upstream_agents": [
                            {"agent_id": "agent_planner", "agent_name": "Planner"},
                        ],
                    },
                    "role_profile_suggestion": {
                        "profile_id": "implementation",
                        "available_skill_ids": ["tools"],
                        "missing_skill_ids": [],
                        "suggested_tool_ids": ["write_file"],
                        "suggested_mcp_server_ids": [],
                        "restrictive_tool_ids": [],
                        "restrictive_mcp_server_ids": [],
                    },
                },
            ]
        },
    )

    diagnostics = result["artifacts"]["collaboration_contracts"]
    assert diagnostics["role_profile_drift_agent_count"] == 1
    assert diagnostics["role_profile_overlap_risk_count"] == 1
    assert diagnostics["role_profile_drift_agents"][0]["agent_id"] == "agent_planner"
    assert diagnostics["role_profile_drift_agents"][0]["missing_skill_ids"] == ["memory"]
    assert diagnostics["role_profile_drift_agents"][0]["missing_tool_ids"] == ["get_current_time"]
    assert diagnostics["role_profile_drift_agents"][0]["outstanding_restrictive_tool_ids"] == ["web_search"]
    assert diagnostics["role_profile_drift_agents"][0]["outstanding_restrictive_mcp_server_ids"] == ["fetch"]
    assert diagnostics["role_profile_overlap_risks"][0]["profile_id"] == "research"
    assert {item["agent_id"] for item in diagnostics["role_profile_overlap_risks"][0]["agent_previews"]} == {
        "agent_research_a",
        "agent_research_b",
    }
    assert diagnostics["role_profile_overlap_risks"][0]["left_focus_lane_ids"] == ["research"]
    assert diagnostics["role_profile_overlap_risks"][0]["right_focus_lane_ids"] == ["grounding"]
    assert diagnostics["role_profile_overlap_risks"][0]["left_unique_tool_ids"] == []
    assert diagnostics["role_profile_overlap_risks"][0]["right_unique_mcp_server_ids"] == []
    risk_ids = {risk["risk_id"] for risk in diagnostics["risks"]}
    assert risk_ids == {"role_profile_drift", "role_profile_overlap"}
