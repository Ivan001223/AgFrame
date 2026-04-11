import time

import anyio
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.runtime.graph.orchestration_graph import (
    FIRST_PRINCIPLES_DIRECTIVE,
    GAME_THEORY_DIRECTIVE,
    _build_agent_collaboration_contract,
    _build_agent_output_artifact,
    _build_brainstorm_round_context,
    _build_brainstorm_summary_prompt,
    _build_cluster_output_artifact,
    _build_prior_research_context,
    _build_team_capability_roster,
    _build_upstream_handoff_context,
    _extract_brainstorm_votes,
    _format_capability_snapshot,
    _invoke_llm_with_streaming_fallback,
    _invoke_llm_with_tool_loop,
    _topological_sort,
    _winning_brainstorm_vote,
    build_orchestration_execution_plan,
    compile_orchestration_graph,
    review_orchestration_output,
)
from app.runtime.llm.provider_registry import ModelProviderRegistry, RegisteredProvider


def test_topological_sort_linear():
    agents = [{"agent_id": "A"}, {"agent_id": "B"}, {"agent_id": "C"}]
    edges = [
        {"source_agent_id": "A", "target_agent_id": "B"},
        {"source_agent_id": "B", "target_agent_id": "C"},
    ]
    ordered = _topological_sort(agents, edges)
    assert ordered == ["A", "B", "C"]

def test_topological_sort_branched():
    agents = [{"agent_id": "A"}, {"agent_id": "B"}, {"agent_id": "C"}, {"agent_id": "D"}]
    edges = [
        {"source_agent_id": "A", "target_agent_id": "B"},
        {"source_agent_id": "A", "target_agent_id": "C"},
        {"source_agent_id": "C", "target_agent_id": "D"},
    ]
    ordered = _topological_sort(agents, edges)
    assert ordered.index("A") < ordered.index("B")
    assert ordered.index("A") < ordered.index("C")
    assert ordered.index("C") < ordered.index("D")

def test_compile_orchestration_graph_empty():
    graph_app = compile_orchestration_graph({}, task="test")
    assert graph_app is not None


def test_build_skill_guidance_block_uses_catalog_prompt_hints_and_tools():
    from app.runtime.graph.orchestration_graph import _build_skill_guidance_block

    guidance = _build_skill_guidance_block(
        loaded_skill_ids=["research", "rag"],
        skill_catalog_by_id={
            "research": {
                "title": "Research",
                "prompt_hint": "Collect recent external evidence.",
                "suggested_tool_ids": ["web_search"],
            },
            "rag": {
                "title": "RAG",
                "prompt_hint": "Ground answers in project knowledge.",
                "suggested_tool_ids": ["knowledge_retriever", "read_document"],
            },
        },
    )

    assert "- Research: Collect recent external evidence." in guidance
    assert "Suggested tools when enabled: Web Search." in guidance
    assert "- RAG: Ground answers in project knowledge." in guidance


def test_team_capability_roster_uses_cluster_parent_summary_for_expanded_members():
    roster = _build_team_capability_roster(
        agents=[
            {
                "agent_id": "cluster_a__member_1",
                "cluster_agent_id": "cluster_a",
                "name": "Ideas / Lead",
                "role": "lead",
            }
        ],
        capability_summary_by_agent_id={
            "cluster_a": {
                "loaded_skill_ids": ["research"],
                "enabled_tool_ids": ["web_search"],
                "provider_route": "project default",
                "review_mode": "cluster summary plus team review agent",
            }
        },
    )

    assert "skills=research" in roster
    assert "tools=web_search" in roster
    assert "readiness=ready" in roster
    assert "review=cluster summary plus team review agent" in roster


def test_build_agent_collaboration_contract_lists_upstream_and_downstream_profiles():
    contract = _build_agent_collaboration_contract(
        agent_id="agent_builder",
        agents_by_id={
            "agent_planner": {"agent_id": "agent_planner", "name": "Planner"},
            "agent_builder": {"agent_id": "agent_builder", "name": "Builder"},
            "agent_reviewer": {"agent_id": "agent_reviewer", "name": "Reviewer"},
        },
        edges=[
            {"source_agent_id": "agent_planner", "target_agent_id": "agent_builder", "interaction": "delegate"},
            {"source_agent_id": "agent_builder", "target_agent_id": "agent_reviewer", "interaction": "review"},
        ],
        capability_summary_by_agent_id={
            "agent_planner": {
                "loaded_skill_ids": ["research"],
                "enabled_tool_ids": ["web_search"],
                "delegation_lane_ids": ["research"],
                "provider_route": "project default",
                "review_mode": "team review agent",
            },
            "agent_reviewer": {
                "loaded_skill_ids": ["policy"],
                "enabled_tool_ids": [],
                "delegation_lane_ids": ["generalist"],
                "provider_route": "review_provider",
                "review_mode": "direct handoff",
            },
            "agent_builder": {
                "recommended_collaborators": [
                    {
                        "agent_id": "agent_reviewer",
                        "agent_name": "Reviewer",
                        "fit": "good",
                        "score": 48,
                        "rationale": "adds policy review coverage",
                    }
                ],
                "downstream_handoff_scores": [
                    {
                        "agent_id": "agent_reviewer",
                        "agent_name": "Reviewer",
                        "fit": "good",
                        "score": 48,
                        "rationale": "adds policy review coverage",
                    }
                ],
            },
        },
    )

    assert "Planner -> you via delegate" in contract
    assert "you -> Reviewer via review" in contract
    assert "fit=good;" in contract
    assert "Best-fit collaborators if you need an extra handoff:" in contract
    assert "skills=research" in contract
    assert "readiness=ready" in contract
    assert "provider=review_provider" in contract


def test_build_agent_collaboration_contract_includes_explicit_tool_and_mcp_policy():
    contract = _build_agent_collaboration_contract(
        agent_id="agent_builder",
        agents_by_id={
            "agent_builder": {"agent_id": "agent_builder", "name": "Builder"},
            "agent_researcher": {"agent_id": "agent_researcher", "name": "Researcher"},
        },
        edges=[
            {"source_agent_id": "agent_builder", "target_agent_id": "agent_researcher", "interaction": "delegate"},
        ],
        capability_summary_by_agent_id={
            "agent_researcher": {
                "loaded_skill_ids": ["research"],
                "enabled_tool_ids": ["web_search"],
                "configured_allowed_tool_ids": ["web_search"],
                "configured_denied_mcp_server_ids": ["browser"],
                "mcp_server_ids": ["fetch"],
                "provider_route": "project default",
                "review_mode": "team review agent",
            }
        },
    )

    assert "tool_policy=allow web_search" in contract
    assert "mcp_policy=deny browser" in contract


def test_format_capability_snapshot_includes_availability_requirements():
    snapshot = _format_capability_snapshot(
        {
            "loaded_skill_ids": ["tools"],
            "required_skill_ids": ["research"],
            "missing_required_skill_ids": ["research"],
            "required_tool_ids": ["web_search"],
            "missing_required_tool_ids": ["web_search"],
            "requires_tool_calling": True,
            "required_mcp_server_ids": ["github"],
            "missing_required_mcp_server_ids": ["github"],
            "availability_status": "unavailable",
            "availability_blockers": ["Definition requires enabled MCP servers that are not currently available: github"],
            "tool_execution_support": "unsupported",
            "provider_route": "project default",
            "review_mode": "team review agent",
        }
    )

    assert "required_skills=research" in snapshot
    assert "required_tools=web_search" in snapshot
    assert "missing_required_tools=web_search" in snapshot
    assert "requires_tool_calling=yes" in snapshot
    assert "required_mcp=github" in snapshot
    assert "availability=unavailable" in snapshot
    assert "availability_blockers=Definition requires enabled MCP servers" in snapshot


def test_build_upstream_handoff_context_uses_explicit_edges_and_cluster_artifacts():
    context = _build_upstream_handoff_context(
        agent_id="agent_builder",
        agent_directory={
            "cluster_a__summary": {
                "agent_id": "cluster_a__summary",
                "cluster_summary": True,
                "cluster_agent_id": "cluster_a",
                "name": "Ideas / summary",
            },
            "agent_planner": {
                "agent_id": "agent_planner",
                "name": "Planner",
            },
        },
        incoming_edges=[
            {
                "source_agent_id": "cluster_a__summary",
                "target_agent_id": "agent_builder",
                "interaction": "handoff",
            },
            {
                "source_agent_id": "agent_planner",
                "target_agent_id": "agent_builder",
                "interaction": "delegate",
            },
        ],
        state={
            "messages": [],
            "task": "Ship safely",
            "agent_outputs": {
                "cluster_a": "Cluster synthesis output",
                "agent_planner": "Planner output",
            },
            "output_artifacts": {
                "cluster_a": {
                    "node_kind": "cluster",
                    "winning_strategy": "Pilot the rollout.",
                    "next_step": "Start with one cohort.",
                    "dominant_risks": "Overreach breaks trust.",
                }
            },
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        },
    )

    assert "From Ideas / summary via handoff." in context
    assert "Structured artifact: winning strategy=Pilot the rollout." in context
    assert "From Planner via delegate." in context
    assert "Completed upstream output:" in context


def test_build_agent_output_artifact_extracts_handoff_fields():
    artifact = _build_agent_output_artifact(
        agent_config={
            "agent_id": "agent_builder",
            "name": "Builder",
            "role": "implementation",
        },
        content=(
            "Summary: Implement the API integration and keep the retry path stable.\n"
            "- Update the run payload shape.\n"
            "- Add regression coverage for fallback routing.\n"
            "Question: Should the retry budget be surfaced in the UI?\n"
            "Risk: Provider mismatch could regress failover."
        ),
        incoming_edges=[
            {
                "source_agent_id": "agent_planner",
                "target_agent_id": "agent_builder",
                "interaction": "handoff",
            }
        ],
        outgoing_edges=[
            {
                "source_agent_id": "agent_builder",
                "target_agent_id": "agent_reviewer",
                "interaction": "review",
            }
        ],
        agent_directory={
            "agent_planner": {"agent_id": "agent_planner", "name": "Planner"},
            "agent_reviewer": {"agent_id": "agent_reviewer", "name": "Reviewer"},
        },
        state={
            "agent_outputs": {"agent_planner": "Plan the rollout before coding."},
            "output_artifacts": {
                "agent_planner": {
                    "node_kind": "agent",
                    "agent_id": "agent_planner",
                    "agent_name": "Planner",
                    "handoff_summary": "Summary: Sequence the rollout.",
                    "action_items": ["Confirm dependency order."],
                }
            },
        },
    )

    assert artifact["agent_id"] == "agent_builder"
    assert artifact["handoff_summary"].startswith("Summary:")
    assert artifact["action_items"] == [
        "Update the run payload shape.",
        "Add regression coverage for fallback routing.",
    ]
    assert artifact["open_questions"] == ["Should the retry budget be surfaced in the UI?"]
    assert artifact["risk_flags"] == ["Risk: Provider mismatch could regress failover."]
    assert artifact["consumed_handoffs"][0]["source_agent_name"] == "Planner"
    assert artifact["consumed_handoffs"][0]["artifact_summary"]
    assert artifact["downstream_handoffs"][0]["target_agent_name"] == "Reviewer"


def test_build_agent_output_artifact_includes_capability_policy_fields():
    artifact = _build_agent_output_artifact(
        agent_config={
            "agent_id": "agent_builder",
            "name": "Builder",
            "role": "implementation",
            "allowed_tool_ids": ["write_file"],
            "denied_tool_ids": ["web_search"],
            "allowed_mcp_server_ids": ["github"],
            "denied_mcp_server_ids": ["browser"],
        },
        content="Summary: Ship the change safely.",
        incoming_edges=[],
        outgoing_edges=[],
        agent_directory={},
        state={},
    )

    assert artifact["allowed_tool_ids"] == ["write_file"]
    assert artifact["denied_tool_ids"] == ["web_search"]
    assert artifact["allowed_mcp_server_ids"] == ["github"]
    assert artifact["denied_mcp_server_ids"] == ["browser"]


def test_build_orchestration_execution_plan_enriches_agents_with_collaboration_context():
    ordered_agents, provider_config, review_config = build_orchestration_execution_plan(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Planner", "model": "dev-stub"},
                {"agent_id": "agent_b", "name": "Builder", "model": "dev-stub"},
            ],
            "edges": [{"source_agent_id": "agent_a", "target_agent_id": "agent_b", "interaction": "handoff"}],
            "orchestration_summary": {
                "total_agent_count": 2,
                "execution_step_count": 1,
                "review_enabled": False,
                "readiness": "ready",
                "start_agents": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                "terminal_agents": [{"agent_id": "agent_b", "agent_name": "Builder"}],
                "phases": [
                    {"phase_id": "research", "agent_count": 1, "agents": [{"agent_id": "agent_a", "agent_name": "Planner"}]},
                    {"phase_id": "synthesis", "agent_count": 1, "agents": [{"agent_id": "agent_a", "agent_name": "Planner"}]},
                    {"phase_id": "implementation", "agent_count": 1, "agents": [{"agent_id": "agent_b", "agent_name": "Builder"}]},
                    {"phase_id": "verification", "agent_count": 1, "agents": [{"agent_id": "agent_b", "agent_name": "Builder"}]},
                ],
                "repair_priorities": [{"priority_id": "best_next_handoffs", "severity": "low", "count": 1}],
                "single_owner_capability_risks": [
                    {
                        "kind": "tool",
                        "capability_id": "web_search",
                        "owner_agents": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                    }
                ],
                "agent_routing": {
                    "coordinator_anchors": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                    "research_anchors": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                    "implementation_anchors": [{"agent_id": "agent_b", "agent_name": "Builder"}],
                    "verification_anchors": [{"agent_id": "agent_b", "agent_name": "Builder"}],
                    "skill_capable_anchors": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                    "tool_capable_anchors": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                    "mcp_capable_anchors": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                },
            },
            "mcp_server_catalog": [
                {"server_id": "fetch", "title": "Fetch", "description": "Browser and HTTP retrieval server."},
                {"server_id": "github", "title": "GitHub", "description": "Repository and issue server.", "status": "disabled"},
            ],
            "skill_catalog": [
                {
                    "skill_id": "research",
                    "title": "Research",
                    "prompt_hint": "Collect recent external evidence and separate facts from hypotheses.",
                    "suggested_tool_ids": ["web_search"],
                }
            ],
            "agent_capability_summaries": [
                {
                    "agent_id": "agent_a",
                    "loaded_skill_ids": ["research"],
                    "enabled_tool_ids": ["web_search"],
                    "delegation_lane_ids": ["research"],
                    "mcp_server_ids": ["fetch"],
                    "missing_mcp_server_ids": ["github"],
                    "delegation_focus": "external research, source comparison, and evidence gathering",
                    "delegation_contract": {
                        "primary_role_mode": "coordinator",
                        "supporting_role_modes": ["research"],
                        "work_strategy": "synthesize_and_route",
                        "should_coordinate_parallel_work": True,
                        "should_produce_final_output": False,
                        "primary_focus": "external research, source comparison, and evidence gathering",
                        "upstream_agents": [],
                        "downstream_agents": [{"agent_id": "agent_b", "agent_name": "Builder"}],
                        "preferred_collaborators": [{"agent_id": "agent_b", "agent_name": "Builder"}],
                        "weak_handoff_targets": [],
                        "watchouts": ["Keep the implementation handoff tight."],
                    },
                    "recommended_collaborators": [
                        {
                            "agent_id": "agent_b",
                            "agent_name": "Builder",
                            "score": 61,
                            "fit": "strong",
                            "rationale": "adds implementation lane coverage",
                            "complementary_lane_ids": ["implementation"],
                        }
                    ],
                    "provider_route": "project default",
                    "review_mode": "team review agent",
                    "capability_brief": "Approved skills: research. Enabled tools: web_search. MCP servers: fetch. Relevant MCP servers not enabled in this project: github.",
                }
            ],
            "provider_config": {"preferred_provider_id": "project_default"},
            "review_agent": {"enabled": False},
        }
    )

    assert provider_config["preferred_provider_id"] == "project_default"
    assert review_config["enabled"] is False
    planner = ordered_agents[0]
    builder = ordered_agents[1]
    assert "team_capability_roster" in planner
    assert "incoming_edges" in builder and builder["incoming_edges"][0]["source_agent_id"] == "agent_a"
    assert "outgoing_edges" in planner and planner["outgoing_edges"][0]["target_agent_id"] == "agent_b"
    assert planner["enabled_tool_ids"] == ["web_search"]
    assert planner["mcp_server_ids"] == ["fetch"]
    assert planner["missing_mcp_server_ids"] == ["github"]
    assert planner["delegation_lane_ids"] == ["research"]
    assert planner["delegation_focus"] == "external research, source comparison, and evidence gathering"
    assert planner["recommended_collaborators"][0]["agent_id"] == "agent_b"
    assert planner["delegation_contract"]["primary_role_mode"] == "coordinator"
    assert "Work strategy: synthesize_and_route." in planner["structured_delegation_contract"]
    assert "Web Search" in planner["tool_guidance"]
    assert "Configured MCP servers relevant to this node:" in planner["mcp_guidance"]
    assert "Relevant MCP servers not currently enabled in project inventory:" in planner["mcp_guidance"]
    assert "Collect recent external evidence" in planner["skill_guidance"]
    assert "Directly executable tools in this runtime: web_search." in planner["capability_execution_contract"]
    assert "MCP inventory is planning metadata only in this runtime: fetch." in planner["capability_execution_contract"]
    assert "Graph readiness: ready." in planner["orchestration_summary_brief"]
    assert "Routing anchors: coordinator=Planner" in planner["orchestration_summary_brief"]
    assert "Planner -> you via handoff" in builder["collaboration_contract"]


@pytest.mark.anyio
async def test_invoke_llm_with_tool_loop_parallelizes_read_only_batches():
    class _SlowReadOnlyTool:
        def __init__(self, name: str, delay_seconds: float):
            self.name = name
            self.delay_seconds = delay_seconds

        async def ainvoke(self, args):
            await anyio.sleep(self.delay_seconds)
            return f"{self.name}:{args!s}"

    class _FakeToolLLM:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            if any(getattr(message, "type", "") == "tool" for message in messages):
                return AIMessage(content="Final answer after read-only tools")
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "first"}, "id": "call_1", "type": "tool_call"},
                    {"name": "read_document", "args": {"file_path": "/tmp/example.txt"}, "id": "call_2", "type": "tool_call"},
                ],
            )

    started_at = time.perf_counter()
    content, tool_runs, mode = await _invoke_llm_with_tool_loop(
        _FakeToolLLM(),
        [HumanMessage(content="Use two read-only tools")],
        enabled_tools=[
            _SlowReadOnlyTool("web_search", 0.2),
            _SlowReadOnlyTool("read_document", 0.2),
        ],
        timeout_seconds=5,
    )
    elapsed = time.perf_counter() - started_at

    assert content == "Final answer after read-only tools"
    assert mode == "tool_loop"
    assert [run["tool_id"] for run in tool_runs] == ["web_search", "read_document"]
    assert elapsed < 0.35


@pytest.mark.anyio
async def test_compile_orchestration_graph_executes_enabled_tools_and_records_tool_runs(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )
    seen: dict[str, list[str]] = {"bound_tools": []}

    class _FakeToolLLM:
        def bind_tools(self, tools):
            seen["bound_tools"] = [getattr(tool, "name", "") for tool in tools]
            return self

        async def ainvoke(self, messages):
            if any(getattr(message, "type", "") == "tool" for message in messages):
                return AIMessage(content="Final answer after tool")
            return AIMessage(
                content="",
                tool_calls=[{"name": "get_current_time", "args": {}, "id": "call_1", "type": "tool_call"}],
            )

    monkeypatch.setattr(
        "app.runtime.graph.orchestration_graph.get_llm_for_provider",
        lambda **kwargs: _FakeToolLLM(),
    )

    graph_app = compile_orchestration_graph(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Builder", "model": "dev-stub"},
            ],
            "tool_catalog": [
                {"tool_id": "get_current_time", "title": "Get Current Time", "description": "Read the current system time."},
            ],
            "agent_capability_summaries": [
                {
                    "agent_id": "agent_a",
                    "enabled_tool_ids": ["get_current_time"],
                    "capability_brief": "Enabled tools: get_current_time.",
                }
            ],
            "review_agent": {"enabled": False},
        },
        task="Check the time",
        provider_registry=registry,
    )

    state = await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Check the time",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    assert seen["bound_tools"] == ["get_current_time"]
    assert state["agent_outputs"]["agent_a"] == "Final answer after tool"
    assert state["output_artifacts"]["agent_a"]["tool_runs"][0]["tool_id"] == "get_current_time"
    assert state["output_artifacts"]["agent_a"]["tool_runs"][0]["status"] == "success"


@pytest.mark.anyio
async def test_compile_orchestration_graph_skips_tools_for_runtime_without_native_tool_support(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["local-qwen3-vl"],
            is_default=True,
        )
    )
    seen = {"bind_tools_called": False}

    class _UnsupportedToolLLM:
        def bind_tools(self, tools):
            seen["bind_tools_called"] = True
            raise AssertionError("bind_tools should not be called for a tool-unsupported runtime")

        async def ainvoke(self, messages):
            return AIMessage(content="Local model response")

    monkeypatch.setattr(
        "app.runtime.graph.orchestration_graph.get_llm_for_provider",
        lambda **kwargs: _UnsupportedToolLLM(),
    )

    graph_app = compile_orchestration_graph(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Builder", "model": "local-qwen3-vl"},
            ],
            "tool_catalog": [
                {"tool_id": "get_current_time", "title": "Get Current Time", "description": "Read the current system time."},
            ],
            "agent_capability_summaries": [
                {
                    "agent_id": "agent_a",
                    "enabled_tool_ids": ["get_current_time"],
                    "provider_limited_tool_ids": ["get_current_time"],
                    "tool_execution_support": "unsupported",
                    "capability_brief": "Provider-limited tools: get_current_time.",
                }
            ],
            "review_agent": {"enabled": False},
        },
        task="Check the time",
        provider_registry=registry,
    )

    state = await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Check the time",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    assert seen["bind_tools_called"] is False
    assert state["agent_outputs"]["agent_a"] == "Local model response"
    assert "tool_runs" not in state["output_artifacts"]["agent_a"]


@pytest.mark.anyio
async def test_invoke_llm_with_streaming_fallback_prefers_astream():
    class _StreamingLLM:
        async def astream(self, messages):
            assert isinstance(messages[0], HumanMessage)
            yield AIMessageChunk(content="hello ")
            yield AIMessageChunk(content="world")

        async def ainvoke(self, messages):
            raise AssertionError("ainvoke should not be used when astream succeeds")

    content, mode = await _invoke_llm_with_streaming_fallback(_StreamingLLM(), [HumanMessage(content="test")])

    assert content == "hello world"
    assert mode == "astream"


@pytest.mark.anyio
async def test_invoke_llm_with_streaming_fallback_prefers_astream_events_and_emits_provider_events():
    provider_events = []

    class _EventStreamingLLM:
        async def astream_events(self, messages):
            assert isinstance(messages[0], HumanMessage)
            yield {"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content="hello ")}}
            yield {"event": "on_chat_model_stream", "data": {"chunk": AIMessageChunk(content="world")}}

        async def astream(self, messages):
            raise AssertionError("astream should not be used when astream_events exists")

        async def ainvoke(self, messages):
            raise AssertionError("ainvoke should not be used when astream_events succeeds")

    content, mode = await _invoke_llm_with_streaming_fallback(
        _EventStreamingLLM(),
        [HumanMessage(content="test")],
        on_stream_event=lambda event: provider_events.append(event),
    )

    assert content == "hello world"
    assert mode == "astream"
    assert any(event["type"] == "provider_event" for event in provider_events)
    assert any(event["type"] == "token" for event in provider_events)


@pytest.mark.anyio
async def test_invoke_llm_with_streaming_fallback_falls_back_to_ainvoke():
    class _InvokeOnlyLLM:
        async def ainvoke(self, messages):
            assert isinstance(messages[0], HumanMessage)
            return type("Resp", (), {"content": "fallback response"})()

    content, mode = await _invoke_llm_with_streaming_fallback(_InvokeOnlyLLM(), [HumanMessage(content="test")])

    assert content == "fallback response"
    assert mode == "ainvoke"


@pytest.mark.anyio
async def test_review_orchestration_output_blocks_invalid_decision_tokens(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )

    class _InvalidReviewLLM:
        async def ainvoke(self, messages):
            return type("Resp", (), {"content": "This looks okay to me."})()

    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _InvalidReviewLLM())

    result = await review_orchestration_output(
        review_config={"enabled": True, "model": "dev-stub"},
        provider_config={},
        default_timeout=5,
        provider_registry=registry,
        task="Ship safely",
        agent_id="agent_a",
        agent_name="Agent A",
        output="candidate output",
    )

    assert result["approved"] is False
    assert str(result["review_output"]).startswith("BLOCK:")


def test_brainstorm_vote_tally_and_winner():
    votes = _extract_brainstorm_votes(
        [
            "Some analysis\nVOTE: CONSERVATIVE",
            "Counterpoint\nVOTE: BALANCED",
            "More debate\nVOTE: BALANCED",
        ]
    )

    assert votes == {"conservative": 1, "balanced": 2, "aggressive": 0}
    assert _winning_brainstorm_vote(votes) == "BALANCED"


def test_brainstorm_summary_prompt_includes_vote_tally_and_sections():
    messages = _build_brainstorm_summary_prompt(
        cluster_name="Ideas",
        task="Design a rollout",
        prior_sections=[
            "Lead argues for caution\nVOTE: CONSERVATIVE",
            "Critic prefers speed\nVOTE: AGGRESSIVE",
            "Synthesizer prefers balance\nVOTE: BALANCED",
        ],
        rounds=3,
    )

    joined = "\n".join(str(message.content) for message in messages)
    assert "Observed vote tally" in joined
    assert "Winning Strategy:" in joined
    assert "Conservative Strategy:" in joined
    assert "Balanced Strategy:" in joined
    assert "Aggressive Strategy:" in joined
    assert "Game Theory Rationale:" in joined
    assert "Key Players:" in joined
    assert "Incentive Map:" in joined
    assert "Dominant Risks:" in joined
    assert "Expected Equilibrium:" in joined
    assert GAME_THEORY_DIRECTIVE in joined


def test_brainstorm_round_context_carries_forward_previous_winner():
    context = _build_brainstorm_round_context(
        agent_config={
            "cluster_strategy": "brainstorm",
            "cluster_round_index": 2,
            "cluster_agent_id": "cluster_a",
            "cluster_member_node_ids": [
                "cluster_a__round_1__member_1",
                "cluster_a__round_1__member_2",
                "cluster_a__round_2__member_1",
            ],
        },
        state={
            "messages": [],
            "task": "Design a launch",
            "agent_outputs": {
                "cluster_a__round_1__member_1": "Option one\nVOTE: CONSERVATIVE",
                "cluster_a__round_1__member_2": "Option two\nVOTE: BALANCED",
            },
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        },
    )

    assert "Previous round vote tally" in context
    assert "Current leading direction from the previous round: BALANCED" in context


def test_brainstorm_cluster_output_artifact_extracts_structured_sections():
    artifact = _build_cluster_output_artifact(
        cluster_agent_id="cluster_a",
        cluster_name="Ideas",
        cluster_strategy="brainstorm",
        member_node_ids=["cluster_a__round_1__member_1", "cluster_a__round_1__member_2"],
        outputs={
            "cluster_a__round_1__member_1": (
                "Lead proposal\n"
                "PLAYERS:\nProduct team and users.\n"
                "INCENTIVES:\nAdoption with trust.\n"
                "RISKS:\nOverreach breaks trust.\n"
                "EQUILIBRIUM:\nStaged release preserves cooperation.\n"
                "VOTE: BALANCED"
            ),
            "cluster_a__round_1__member_2": (
                "Critic rebuttal\n"
                "PLAYERS:\nCompetitors and users.\n"
                "INCENTIVES:\nCompetitors exploit instability.\n"
                "RISKS:\nAggressive launch invites backlash.\n"
                "EQUILIBRIUM:\nUsers defect if trust drops.\n"
                "VOTE: AGGRESSIVE"
            ),
        },
        rounds=2,
        summary=(
            "Conservative Strategy:\nImprove incrementally.\n\n"
            "Balanced Strategy:\nShip a staged rollout.\n\n"
            "Aggressive Strategy:\nLaunch globally.\n\n"
            "Winning Strategy:\nBalanced wins on execution risk.\n\n"
            "Game Theory Rationale:\nBalanced is the strongest equilibrium because it preserves cooperation while limiting defection incentives.\n\n"
            "Key Players:\nProduct team, users, and competitors.\n\n"
            "Incentive Map:\nThe team wants adoption, users want trust, competitors look for openings.\n\n"
            "Dominant Risks:\nAggressive release increases defection and trust loss.\n\n"
            "Expected Equilibrium:\nA staged rollout sustains cooperation while limiting opportunistic responses.\n\n"
            "Next Step:\nPilot with one customer cohort."
        ),
        prior_sections=[
            "Lead proposal\nVOTE: BALANCED",
            "Critic rebuttal\nVOTE: AGGRESSIVE",
            "Synthesis\nVOTE: BALANCED",
        ],
    )

    assert artifact["winning_vote"] == "BALANCED"
    assert artifact["vote_tally"] == {"conservative": 0, "balanced": 2, "aggressive": 1}
    assert artifact["strategies"]["balanced"] == "Ship a staged rollout."
    assert artifact["winning_strategy"] == "Balanced wins on execution risk."
    assert artifact["game_theory_rationale"] == "Balanced is the strongest equilibrium because it preserves cooperation while limiting defection incentives."
    assert artifact["key_players"] == "Product team, users, and competitors."
    assert artifact["incentive_map"] == "The team wants adoption, users want trust, competitors look for openings."
    assert artifact["dominant_risks"] == "Aggressive release increases defection and trust loss."
    assert artifact["expected_equilibrium"] == "A staged rollout sustains cooperation while limiting opportunistic responses."
    assert artifact["round_history"][0]["round_index"] == 1
    assert artifact["round_history"][0]["vote_tally"] == {"conservative": 0, "balanced": 1, "aggressive": 1}
    assert artifact["round_history"][0]["winning_vote"] == "BALANCED"
    assert artifact["round_history"][0]["equilibrium_shift"] == "Initial round baseline."
    assert artifact["round_history"][0]["equilibrium_shift_details"]["shift_type"] == "initial"
    assert artifact["round_history"][0]["equilibrium_shift_details"]["changed_fields"] == []
    assert artifact["round_history"][0]["key_players"] == "Product team and users. | Competitors and users."
    assert artifact["round_history"][0]["incentive_map"] == "Adoption with trust. | Competitors exploit instability."
    assert artifact["round_history"][0]["dominant_risks"] == "Overreach breaks trust. | Aggressive launch invites backlash."
    assert artifact["round_history"][0]["expected_equilibrium"] == "Staged release preserves cooperation. | Users defect if trust drops."
    assert artifact["next_step"] == "Pilot with one customer cohort."


def test_prior_research_context_includes_cluster_evidence():
    context = _build_prior_research_context(
        agent_id="agent_b",
        state={
            "messages": [],
            "task": "Ship safely",
            "agent_outputs": {},
            "output_artifacts": {
                "cluster_a": {
                    "cluster_name": "Ideas",
                    "winning_strategy": "Pilot the rollout.",
                    "game_theory_rationale": "This path keeps counterparties aligned while minimizing unilateral deviation incentives.",
                    "key_players": "Operator, users, and rivals.",
                    "incentive_map": "Each side prefers upside but reacts sharply to trust erosion.",
                    "dominant_risks": "Overreach triggers competitive or user backlash.",
                    "expected_equilibrium": "Controlled rollout preserves cooperation.",
                    "next_step": "Start with internal users.",
                    "research": {
                        "latest_progress": ["A recent benchmark shows staged releases reduce incident rate."],
                        "papers": [{"title": "Progressive Delivery at Scale"}],
                    },
                }
            },
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        },
    )

    assert "Evidence from prior cluster 'Ideas'" in context
    assert "Winning strategy: Pilot the rollout." in context
    assert "Game-theoretic rationale: This path keeps counterparties aligned while minimizing unilateral deviation incentives." in context
    assert "Key players: Operator, users, and rivals." in context
    assert "Expected equilibrium: Controlled rollout preserves cooperation." in context
    assert "Progressive Delivery at Scale" in context


@pytest.mark.anyio
async def test_compile_orchestration_graph_runs_selected_agents_and_review():
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )
    graph_app = compile_orchestration_graph(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Agent A", "model": "dev-stub"},
                {"agent_id": "agent_b", "name": "Agent B", "model": "dev-stub"},
            ],
            "edges": [{"source_agent_id": "agent_a", "target_agent_id": "agent_b"}],
            "review_agent": {"enabled": True, "name": "Reviewer", "model": "dev-stub"},
        },
        task="Prepare a plan",
        provider_registry=registry,
    )

    state = await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Prepare a plan",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    assert state["errors"] == []
    assert "agent_a" in state["agent_outputs"]
    assert "agent_b" in state["agent_outputs"]
    assert "review_agent" in state["agent_outputs"]


@pytest.mark.anyio
async def test_compile_orchestration_graph_injects_capabilities_and_collaboration_contract(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )
    seen_human_messages = []

    class _FakeLLM:
        async def ainvoke(self, messages):
            seen_human_messages.extend(
                message.content for message in messages if getattr(message, "type", "") == "human"
            )
            return type("Resp", (), {"content": "ok"})()

    monkeypatch.setattr(
        "app.runtime.graph.orchestration_graph.get_llm_for_provider",
        lambda **kwargs: _FakeLLM(),
    )

    graph_app = compile_orchestration_graph(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Planner", "model": "dev-stub"},
                {"agent_id": "agent_b", "name": "Builder", "model": "dev-stub"},
            ],
            "edges": [{"source_agent_id": "agent_a", "target_agent_id": "agent_b", "interaction": "handoff"}],
            "orchestration_summary": {
                "total_agent_count": 2,
                "execution_step_count": 1,
                "review_enabled": False,
                "readiness": "ready",
                "start_agents": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                "terminal_agents": [{"agent_id": "agent_b", "agent_name": "Builder"}],
                "phases": [
                    {"phase_id": "research", "agent_count": 1, "agents": [{"agent_id": "agent_a", "agent_name": "Planner"}]},
                    {"phase_id": "synthesis", "agent_count": 1, "agents": [{"agent_id": "agent_a", "agent_name": "Planner"}]},
                    {"phase_id": "implementation", "agent_count": 1, "agents": [{"agent_id": "agent_b", "agent_name": "Builder"}]},
                    {"phase_id": "verification", "agent_count": 1, "agents": [{"agent_id": "agent_b", "agent_name": "Builder"}]},
                ],
                "repair_priorities": [{"priority_id": "best_next_handoffs", "severity": "low", "count": 1}],
                "agent_routing": {
                    "coordinator_anchors": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                    "research_anchors": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                    "implementation_anchors": [{"agent_id": "agent_b", "agent_name": "Builder"}],
                    "verification_anchors": [{"agent_id": "agent_b", "agent_name": "Builder"}],
                    "skill_capable_anchors": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                    "tool_capable_anchors": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                    "mcp_capable_anchors": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                },
            },
            "skill_catalog": [
                {
                    "skill_id": "research",
                    "title": "Research",
                    "prompt_hint": "Collect recent external evidence and separate facts from hypotheses.",
                    "suggested_tool_ids": ["web_search"],
                },
                {
                    "skill_id": "rag",
                    "title": "RAG",
                    "prompt_hint": "Ground the answer in project knowledge.",
                    "suggested_tool_ids": ["knowledge_retriever"],
                },
            ],
            "mcp_server_catalog": [
                {"server_id": "fetch", "title": "Fetch", "description": "Browser and HTTP retrieval server."},
                {"server_id": "github", "title": "GitHub", "description": "Repository and issue server.", "status": "disabled"},
            ],
            "agent_capability_summaries": [
                {
                    "agent_id": "agent_a",
                    "loaded_skill_ids": ["research"],
                    "loaded_skill_hints": ["Collect recent external evidence and separate facts from hypotheses."],
                    "enabled_tool_ids": ["web_search"],
                    "delegation_lane_ids": ["research"],
                    "mcp_server_ids": ["fetch"],
                    "missing_mcp_server_ids": ["github"],
                    "delegation_focus": "external research, source comparison, and evidence gathering",
                    "delegation_contract": {
                        "primary_role_mode": "coordinator",
                        "supporting_role_modes": ["research"],
                        "work_strategy": "synthesize_and_route",
                        "should_coordinate_parallel_work": True,
                        "should_produce_final_output": False,
                        "primary_focus": "external research, source comparison, and evidence gathering",
                        "upstream_agents": [],
                        "downstream_agents": [{"agent_id": "agent_b", "agent_name": "Builder"}],
                        "preferred_collaborators": [{"agent_id": "agent_b", "agent_name": "Builder"}],
                        "weak_handoff_targets": [],
                        "watchouts": ["Keep the implementation handoff tight."],
                    },
                    "recommended_collaborators": [
                        {
                            "agent_id": "agent_b",
                            "agent_name": "Builder",
                            "fit": "strong",
                            "score": 61,
                            "rationale": "adds implementation lane coverage",
                        }
                    ],
                    "provider_route": "project default",
                    "review_mode": "team review agent",
                    "capability_brief": "Approved skills: research. Enabled tools: web_search. MCP servers: fetch. Relevant MCP servers not enabled in this project: github.",
                },
                {
                    "agent_id": "agent_b",
                    "loaded_skill_ids": ["rag"],
                    "loaded_skill_hints": ["Ground the answer in project knowledge."],
                    "enabled_tool_ids": ["read_document"],
                    "delegation_lane_ids": ["grounding"],
                    "delegation_contract": {
                        "primary_role_mode": "implementation",
                        "supporting_role_modes": ["verification"],
                        "work_strategy": "self_contained_delivery",
                        "should_coordinate_parallel_work": False,
                        "should_produce_final_output": True,
                        "primary_focus": "project grounding and file-backed analysis",
                        "upstream_agents": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                        "downstream_agents": [],
                        "preferred_collaborators": [{"agent_id": "agent_a", "agent_name": "Planner"}],
                        "weak_handoff_targets": [],
                        "watchouts": ["Close the loop with a self-contained result."],
                    },
                    "provider_route": "project default",
                    "review_mode": "team review agent",
                    "capability_brief": "Approved skills: rag. Enabled tools: read_document. MCP servers: none configured in this project.",
                },
            ],
            "review_agent": {"enabled": False},
        },
        task="Prepare a plan",
        provider_registry=registry,
    )

    await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Prepare a plan",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    joined = "\n\n".join(seen_human_messages)
    assert "Your approved collaboration capabilities are:" in joined
    assert "Structured delegation lanes for this node:" in joined
    assert "Preferred delegation lane for this node:" in joined
    assert "Best-fit collaborators already identified from the current canvas:" in joined
    assert "external research, source comparison, and evidence gathering" in joined
    assert "Approved skill pack guidance:" in joined
    assert "Execution contract for this node:" in joined
    assert "Structured delegation contract for this node:" in joined
    assert "Primary role mode: coordinator." in joined
    assert "Work strategy: synthesize_and_route." in joined
    assert "Treat this contract as the hard boundary between planning metadata and actually executable actions." in joined
    assert "Relevant project MCP inventory for this node:" in joined
    assert "Current graph orchestration brief:" in joined
    assert "Routing anchors: coordinator=Planner" in joined
    assert "Treat this MCP inventory as planning metadata only." in joined
    assert "Collect recent external evidence and separate facts from hypotheses." in joined
    assert "Your collaboration contract on this canvas is:" in joined
    assert "you -> Builder via handoff" in joined
    assert "Planner -> you via handoff" in joined


@pytest.mark.anyio
async def test_compile_orchestration_graph_limits_context_to_explicit_upstream_handoffs(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )
    seen_human_messages: dict[str, str] = {}

    class _FakeLLM:
        async def ainvoke(self, messages):
            system_message = next(
                (str(message.content) for message in messages if getattr(message, "type", "") == "system"),
                "",
            )
            human_message = next(
                (str(message.content) for message in messages if getattr(message, "type", "") == "human"),
                "",
            )
            if "You are Planner" in system_message:
                seen_human_messages["Planner"] = human_message
                return type("Resp", (), {"content": "Planner output"})()
            if "You are Sidecar" in system_message:
                seen_human_messages["Sidecar"] = human_message
                return type("Resp", (), {"content": "Sidecar output"})()
            if "You are Builder" in system_message:
                seen_human_messages["Builder"] = human_message
                return type("Resp", (), {"content": "Builder output"})()
            return type("Resp", (), {"content": "ok"})()

    monkeypatch.setattr(
        "app.runtime.graph.orchestration_graph.get_llm_for_provider",
        lambda **kwargs: _FakeLLM(),
    )

    graph_app = compile_orchestration_graph(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Planner", "model": "dev-stub"},
                {"agent_id": "agent_c", "name": "Sidecar", "model": "dev-stub"},
                {"agent_id": "agent_b", "name": "Builder", "model": "dev-stub"},
            ],
            "edges": [{"source_agent_id": "agent_a", "target_agent_id": "agent_b", "interaction": "handoff"}],
            "review_agent": {"enabled": False},
        },
        task="Prepare a plan",
        provider_registry=registry,
    )

    await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Prepare a plan",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    builder_prompt = seen_human_messages["Builder"]
    assert "Direct upstream handoffs already completed:" in builder_prompt
    assert "From Planner via handoff." in builder_prompt
    assert "Planner output" in builder_prompt
    assert "Sidecar output" not in builder_prompt


@pytest.mark.anyio
async def test_compile_orchestration_graph_records_agent_handoff_artifacts(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )

    class _FakeLLM:
        async def ainvoke(self, messages):
            system_text = "\n".join(str(message.content) for message in messages if getattr(message, "type", "") == "system")
            if "You are Planner" in system_text:
                return type(
                    "Resp",
                    (),
                    {
                        "content": (
                            "Summary: Produce the rollout plan.\n"
                            "- Confirm dependency order.\n"
                            "Question: Is a phased rollout required?\n"
                            "Risk: Tight timing may hide integration gaps."
                        )
                    },
                )()
            return type("Resp", (), {"content": "Builder complete."})()

    monkeypatch.setattr(
        "app.runtime.graph.orchestration_graph.get_llm_for_provider",
        lambda **kwargs: _FakeLLM(),
    )

    graph_app = compile_orchestration_graph(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Planner", "model": "dev-stub"},
                {"agent_id": "agent_b", "name": "Builder", "model": "dev-stub"},
            ],
            "edges": [{"source_agent_id": "agent_a", "target_agent_id": "agent_b", "interaction": "handoff"}],
            "review_agent": {"enabled": False},
        },
        task="Prepare a plan",
        provider_registry=registry,
    )

    state = await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Prepare a plan",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    planner_artifact = state["output_artifacts"]["agent_a"]
    assert planner_artifact["node_kind"] == "agent"
    assert planner_artifact["handoff_summary"].startswith("Summary:")
    assert planner_artifact["action_items"] == ["Confirm dependency order."]
    assert planner_artifact["open_questions"] == ["Is a phased rollout required?"]
    assert planner_artifact["risk_flags"] == ["Risk: Tight timing may hide integration gaps."]
    assert planner_artifact["downstream_handoffs"][0]["target_agent_id"] == "agent_b"
    builder_artifact = state["output_artifacts"]["agent_b"]
    assert builder_artifact["consumed_handoffs"][0]["source_agent_name"] == "Planner"
    assert builder_artifact["consumed_handoffs"][0]["output_preview"] == state["agent_outputs"]["agent_a"]


@pytest.mark.anyio
async def test_agent_provider_override_takes_precedence_over_project_provider(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="project_default",
            name="Project Default",
            base_url="https://project.test",
            api_key="project-key",
            models=["dev-stub"],
            is_default=True,
        )
    )
    registry.register(
        RegisteredProvider(
            provider_id="agent_provider",
            name="Agent Provider",
            base_url="https://agent.test",
            api_key="agent-key",
            models=["dev-stub"],
        )
    )
    seen = []

    class _FakeLLM:
        async def ainvoke(self, messages):
            return type("Resp", (), {"content": "ok"})()

    def _fake_get_llm_for_provider(**kwargs):
        seen.append(kwargs)
        return _FakeLLM()

    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", _fake_get_llm_for_provider)

    graph_app = compile_orchestration_graph(
        {
            "agents": [
                {
                    "agent_id": "agent_a",
                    "name": "Agent A",
                    "model": "dev-stub",
                    "preferred_provider_id": "agent_provider",
                }
            ],
            "provider_config": {"preferred_provider_id": "project_default"},
            "review_agent": {"enabled": False},
        },
        task="Prepare a plan",
        provider_registry=registry,
    )

    await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Prepare a plan",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    assert seen[0]["base_url"] == "https://agent.test"
    assert seen[0]["api_key"] == "agent-key"


@pytest.mark.anyio
async def test_review_agent_provider_override_is_preserved(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="project_default",
            name="Project Default",
            base_url="https://project.test",
            api_key="project-key",
            models=["dev-stub"],
            is_default=True,
        )
    )
    registry.register(
        RegisteredProvider(
            provider_id="review_provider",
            name="Review Provider",
            base_url="https://review.test",
            api_key="review-key",
            models=["dev-stub"],
        )
    )
    seen = []

    class _FakeLLM:
        async def ainvoke(self, messages):
            return type("Resp", (), {"content": "ok"})()

    def _fake_get_llm_for_provider(**kwargs):
        seen.append(kwargs)
        return _FakeLLM()

    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", _fake_get_llm_for_provider)

    graph_app = compile_orchestration_graph(
        {
            "agents": [{"agent_id": "agent_a", "name": "Agent A", "model": "dev-stub"}],
            "provider_config": {"preferred_provider_id": "project_default"},
            "review_agent": {
                "enabled": True,
                "name": "Reviewer",
                "model": "dev-stub",
                "preferred_provider_id": "review_provider",
            },
        },
        task="Prepare a plan",
        provider_registry=registry,
    )

    await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Prepare a plan",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    assert seen[0]["base_url"] == "https://project.test"
    assert seen[1]["base_url"] == "https://review.test"


@pytest.mark.anyio
async def test_first_principles_directive_is_injected_for_runtime_agents(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )
    seen_system_messages = []

    class _FakeLLM:
        async def ainvoke(self, messages):
            seen_system_messages.extend(message.content for message in messages if getattr(message, "type", "") == "system")
            return type("Resp", (), {"content": "ok"})()

    monkeypatch.setattr(
        "app.runtime.graph.orchestration_graph.get_llm_for_provider",
        lambda **kwargs: _FakeLLM(),
    )

    graph_app = compile_orchestration_graph(
        {
            "agents": [{"agent_id": "agent_a", "name": "Agent A", "model": "dev-stub", "system_prompt": "Do the task."}],
            "review_agent": {"enabled": True, "name": "Reviewer", "model": "dev-stub", "system_prompt": "Review carefully."},
        },
        task="Solve the problem",
        provider_registry=registry,
    )

    await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Solve the problem",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    assert any(FIRST_PRINCIPLES_DIRECTIVE in message for message in seen_system_messages)


@pytest.mark.anyio
async def test_cluster_nodes_expand_into_runtime_subgraph():
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )

    graph_app = compile_orchestration_graph(
        {
            "agents": [
                {
                    "agent_id": "cluster_a",
                    "name": "Brainstorm Cluster",
                    "node_kind": "cluster",
                    "cluster_strategy": "brainstorm",
                    "brainstorm_rounds": 2,
                    "cluster_members": [
                        {"member_id": "m1", "name": "Lead", "role": "chair", "model": "dev-stub"},
                        {"member_id": "m2", "name": "Critic", "role": "critic", "model": "dev-stub"},
                    ],
                },
                {"agent_id": "agent_b", "name": "Executor", "model": "dev-stub"},
            ],
            "edges": [{"source_agent_id": "cluster_a", "target_agent_id": "agent_b"}],
            "review_agent": {"enabled": False},
        },
        task="Plan a launch",
        provider_registry=registry,
    )

    state = await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Plan a launch",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    assert state["errors"] == []
    assert "cluster_a__round_1__member_1" in state["agent_outputs"]
    assert "cluster_a__round_1__member_2" in state["agent_outputs"]
    assert "cluster_a__round_2__member_1" in state["agent_outputs"]
    assert "cluster_a__round_2__member_2" in state["agent_outputs"]
    assert "cluster_a" in state["agent_outputs"]
    assert "agent_b" in state["agent_outputs"]


@pytest.mark.anyio
async def test_brainstorm_cluster_summary_populates_output_artifacts(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )

    class _FakeLLM:
        def __init__(self):
            self.member_calls = 0

        async def ainvoke(self, messages):
            system_text = "\n".join(str(message.content) for message in messages if getattr(message, "type", "") == "system")
            if "synthesis chair" in system_text:
                return type(
                    "Resp",
                    (),
                    {
                        "content": (
                            "Conservative Strategy:\nTighten the current approach.\n\n"
                            "Balanced Strategy:\nPilot the best candidate.\n\n"
                            "Aggressive Strategy:\nScale immediately.\n\n"
                            "Winning Strategy:\nBalanced offers the strongest tradeoff.\n\n"
                            "Game Theory Rationale:\nBalanced stabilizes cooperation and reduces incentives for adversarial counter-moves.\n\n"
                            "Key Players:\nPlatform, customers, and competitors.\n\n"
                            "Incentive Map:\nEach actor wants upside but hedges against trust and switching costs.\n\n"
                            "Dominant Risks:\nAggressive scaling invites retaliation and operational fragility.\n\n"
                            "Expected Equilibrium:\nPhased adoption keeps participation stable.\n\n"
                            "Next Step:\nRun a pilot."
                        )
                    },
                )()
            self.member_calls += 1
            if self.member_calls <= 2:
                return type(
                    "Resp",
                    (),
                    {
                        "content": (
                            "Proposal\n"
                            "PLAYERS: Platform and customers.\n"
                            "INCENTIVES: Ship quickly while keeping trust.\n"
                            "RISKS: Launch quality slips under pressure.\n"
                            "EQUILIBRIUM: Balanced pilot maintains alignment.\n"
                            "VOTE: BALANCED"
                        )
                    },
                )()
            return type(
                "Resp",
                (),
                {
                    "content": (
                        "Proposal\n"
                        "PLAYERS: Platform, customers, and regulators.\n"
                        "INCENTIVES: Preserve trust while absorbing compliance costs.\n"
                        "RISKS: Aggressive rollout triggers oversight and churn.\n"
                        "EQUILIBRIUM: Conservative sequencing stabilizes adoption.\n"
                        "VOTE: CONSERVATIVE"
                    )
                },
            )()

    fake_llm = _FakeLLM()
    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: fake_llm)

    graph_app = compile_orchestration_graph(
        {
            "agents": [
                {
                    "agent_id": "cluster_a",
                    "name": "Brainstorm Cluster",
                    "node_kind": "cluster",
                    "cluster_strategy": "brainstorm",
                    "brainstorm_rounds": 2,
                    "cluster_members": [
                        {"member_id": "m1", "name": "Lead", "role": "chair", "model": "dev-stub"},
                        {"member_id": "m2", "name": "Critic", "role": "critic", "model": "dev-stub"},
                    ],
                }
            ],
            "edges": [],
            "review_agent": {"enabled": False},
        },
        task="Plan a launch",
        provider_registry=registry,
    )

    state = await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Plan a launch",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    artifact = state["output_artifacts"]["cluster_a"]
    assert artifact["cluster_strategy"] == "brainstorm"
    assert artifact["winning_vote"] == "BALANCED"
    assert artifact["strategies"]["aggressive"] == "Scale immediately."
    assert artifact["game_theory_rationale"] == "Balanced stabilizes cooperation and reduces incentives for adversarial counter-moves."
    assert artifact["key_players"] == "Platform, customers, and competitors."
    assert artifact["dominant_risks"] == "Aggressive scaling invites retaliation and operational fragility."
    assert len(artifact["round_history"]) == 2
    assert artifact["round_history"][0]["round_index"] == 1
    assert artifact["round_history"][1]["round_index"] == 2
    assert artifact["round_history"][1]["equilibrium_shift_details"]["shift_type"] in {"state_change", "vote_change"}
    assert isinstance(artifact["round_history"][1]["equilibrium_shift_details"]["changed_fields"], list)
    assert artifact["round_history"][1]["equilibrium_shift"]
    assert artifact["next_step"] == "Run a pilot."


@pytest.mark.anyio
async def test_runtime_agents_receive_prior_cluster_research_context(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )
    seen_human_messages = []

    class _FakeLLM:
        async def ainvoke(self, messages):
            seen_human_messages.extend(str(message.content) for message in messages if getattr(message, "type", "") == "human")
            return type("Resp", (), {"content": "ok"})()

    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _FakeLLM())

    graph_app = compile_orchestration_graph(
        {
            "agents": [{"agent_id": "agent_b", "name": "Executor", "model": "dev-stub"}],
            "edges": [],
            "review_agent": {"enabled": False},
        },
        task="Ship safely",
        provider_registry=registry,
    )

    await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Ship safely",
            "agent_outputs": {"cluster_a": "Prior synthesis"},
            "output_artifacts": {
                "cluster_a": {
                    "cluster_name": "Ideas",
                    "winning_strategy": "Pilot the rollout.",
                    "next_step": "Start with internal users.",
                    "research": {
                        "latest_progress": ["A recent benchmark shows staged releases reduce incident rate."],
                        "papers": [{"title": "Progressive Delivery at Scale"}],
                    },
                }
            },
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    joined = "\n".join(seen_human_messages)
    assert "External research evidence from earlier clusters is available." in joined
    assert "Progressive Delivery at Scale" in joined


@pytest.mark.anyio
async def test_brainstorm_cluster_members_receive_game_theory_directive(monkeypatch):
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="default",
            name="Default",
            base_url="https://example.test",
            api_key="key",
            models=["dev-stub"],
            is_default=True,
        )
    )
    seen_system_messages = []

    class _FakeLLM:
        async def ainvoke(self, messages):
            seen_system_messages.extend(str(message.content) for message in messages if getattr(message, "type", "") == "system")
            return type("Resp", (), {"content": "Thoughts\nVOTE: BALANCED"})()

    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _FakeLLM())

    graph_app = compile_orchestration_graph(
        {
            "agents": [
                {
                    "agent_id": "cluster_a",
                    "name": "Brainstorm Cluster",
                    "node_kind": "cluster",
                    "cluster_strategy": "brainstorm",
                    "brainstorm_rounds": 1,
                    "cluster_members": [
                        {"member_id": "m1", "name": "Lead", "role": "chair", "model": "dev-stub"},
                    ],
                }
            ],
            "edges": [],
            "review_agent": {"enabled": False},
        },
        task="Plan a launch",
        provider_registry=registry,
    )

    await graph_app.ainvoke(
        {
            "messages": [],
            "task": "Plan a launch",
            "agent_outputs": {},
            "output_artifacts": {},
            "current_agent": "",
            "loop_index": 0,
            "errors": [],
        }
    )

    assert any(GAME_THEORY_DIRECTIVE in message for message in seen_system_messages)
    assert any("PLAYERS:" in message for message in seen_system_messages)
