import pytest

from app.runtime.graph.orchestration_graph import (
    FIRST_PRINCIPLES_DIRECTIVE,
    GAME_THEORY_DIRECTIVE,
    _build_cluster_output_artifact,
    _build_brainstorm_summary_prompt,
    _build_brainstorm_round_context,
    _invoke_llm_with_streaming_fallback,
    _build_prior_research_context,
    _extract_brainstorm_votes,
    _winning_brainstorm_vote,
    compile_orchestration_graph,
    _topological_sort,
)
from langchain_core.messages import AIMessageChunk, HumanMessage
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
