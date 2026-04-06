import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.infrastructure.queue import arq_jobs
from app.runtime.llm.provider_registry import ModelProviderRegistry, RegisteredProvider


def test_chunk_output_for_review_preserves_overlap():
    chunks = arq_jobs._chunk_output_for_review("abcdefghij", chunk_chars=4, overlap_chars=1)

    assert [chunk["content"] for chunk in chunks] == ["abcd", "defg", "ghij"]
    assert chunks[1]["start_char"] == 3
    assert chunks[2]["start_char"] == 6


def test_serialize_orchestration_resume_state_preserves_knowledge_context():
    payload = arq_jobs._serialize_orchestration_resume_state(
        {
            "task": "Coordinate work",
            "agent_outputs": {"agent_a": "done"},
            "output_artifacts": {},
            "current_agent": "agent_a",
            "loop_index": 1,
            "errors": [],
            "knowledge_base_ids": ["kb-1", "kb-1", "kb-2"],
            "knowledge_context": "Knowledge excerpt",
        },
        2,
    )

    restored, next_step_index = arq_jobs._restore_orchestration_state("fallback", payload)

    assert next_step_index == 2
    assert restored["knowledge_base_ids"] == ["kb-1", "kb-2"]
    assert restored["knowledge_context"] == "Knowledge excerpt"


def test_build_partial_output_artifact_snapshot_includes_mcp_inventory_and_policy():
    snapshot = arq_jobs._build_partial_output_artifact_snapshot(
        {
            "agent_id": "agent_a",
            "name": "Agent A",
            "allowed_tool_ids": ["write_file"],
            "denied_tool_ids": ["web_search"],
            "allowed_mcp_server_ids": ["github"],
            "denied_mcp_server_ids": ["browser"],
            "mcp_server_ids": ["fetch"],
            "missing_mcp_server_ids": ["github"],
        },
        "Summary\n- Next action",
    )

    assert snapshot["allowed_tool_ids"] == ["write_file"]
    assert snapshot["denied_tool_ids"] == ["web_search"]
    assert snapshot["allowed_mcp_server_ids"] == ["github"]
    assert snapshot["denied_mcp_server_ids"] == ["browser"]
    assert snapshot["mcp_server_ids"] == ["fetch"]
    assert snapshot["missing_mcp_server_ids"] == ["github"]


def test_build_capability_snapshot_records_agent_lanes_and_mcp_inventory():
    snapshot = arq_jobs._build_capability_snapshot(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Researcher", "role": "research"},
                {"agent_id": "agent_b", "name": "Builder", "role": "implementation"},
            ],
            "agent_capability_summaries": [
                {
                    "agent_id": "agent_a",
                    "delegation_focus": "external research, source comparison, and evidence gathering",
                    "delegation_lane_ids": ["research"],
                    "loaded_skill_ids": ["research"],
                    "required_skill_ids": ["research"],
                    "required_tool_ids": ["web_search"],
                    "missing_required_tool_ids": ["web_search"],
                    "requires_tool_calling": True,
                    "tool_execution_support": "unsupported",
                    "tool_execution_support_reason": "This runtime adapter does not expose native tool binding.",
                    "missing_skill_details": [
                        {
                            "skill_id": "rag",
                            "title": "RAG",
                            "source": "app/skills/rag",
                            "prompt_hint": "Ground answers in project evidence.",
                            "suggested_tool_ids": ["knowledge_retriever"],
                            "suggested_mcp_server_ids": ["filesystem"],
                        }
                    ],
                    "configured_allowed_tool_ids": ["web_search"],
                    "configured_denied_tool_ids": ["read_document"],
                    "enabled_tool_ids": ["web_search"],
                    "policy_added_tool_ids": ["web_search"],
                    "policy_blocked_tool_ids": ["read_document"],
                    "unknown_allowed_tool_ids": ["unknown_tool"],
                    "configured_allowed_mcp_server_ids": ["fetch"],
                    "configured_denied_mcp_server_ids": ["browser"],
                    "mcp_server_ids": ["fetch"],
                    "missing_mcp_server_ids": ["github"],
                    "missing_mcp_server_details": [
                        {
                            "server_id": "github",
                            "title": "GitHub",
                            "status": "disabled",
                            "command_preview": "npx -y @modelcontextprotocol/server-github",
                        }
                    ],
                    "policy_added_mcp_server_ids": ["fetch"],
                    "policy_blocked_mcp_server_ids": ["browser"],
                    "unknown_allowed_mcp_server_ids": ["unknown_server"],
                    "required_mcp_server_ids": ["github"],
                    "missing_required_mcp_server_ids": ["github"],
                    "availability_status": "unavailable",
                    "availability_blockers": [
                        "Definition requires enabled MCP servers that are not currently available: github"
                    ],
                    "delegation_contract": {
                        "primary_role_mode": "research",
                        "supporting_role_modes": ["implementation"],
                        "work_strategy": "gather_then_handoff",
                        "should_coordinate_parallel_work": False,
                        "should_produce_final_output": False,
                        "primary_focus": "external research, source comparison, and evidence gathering",
                        "upstream_agents": [],
                        "downstream_agents": [
                            {"agent_id": "agent_b", "agent_name": "Builder"},
                        ],
                        "preferred_collaborators": [
                            {"agent_id": "agent_b", "agent_name": "Builder"},
                        ],
                        "weak_handoff_targets": [],
                        "watchouts": ["Relevant MCP inventory is missing: github"],
                    },
                    "role_profile_suggestion": {
                        "profile_id": "research",
                        "suggested_skill_ids": ["research", "rag"],
                        "available_skill_ids": ["research"],
                        "missing_skill_ids": ["rag"],
                        "suggested_tool_ids": ["web_search", "knowledge_retriever"],
                        "suggested_mcp_server_ids": ["fetch", "filesystem"],
                        "restrictive_tool_ids": [],
                        "restrictive_mcp_server_ids": [],
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
                }
            ],
            "mcp_server_catalog": [
                {"server_id": "fetch", "title": "Fetch", "status": "enabled"},
                {"server_id": "github", "title": "GitHub", "status": "disabled"},
            ],
        },
        active_agent_ids=["agent_a", "agent_b"],
        handoff_scope="all_agents",
    )

    assert snapshot["active_agent_ids"] == ["agent_a", "agent_b"]
    assert snapshot["handoff_diagnostic_scope"] == "all_agents"
    assert snapshot["agent_capabilities"][0]["agent_name"] == "Researcher"
    assert snapshot["agent_capabilities"][0]["delegation_focus"] == "external research, source comparison, and evidence gathering"
    assert snapshot["agent_capabilities"][0]["delegation_lane_ids"] == ["research"]
    assert snapshot["agent_capabilities"][0]["configured_allowed_tool_ids"] == ["web_search"]
    assert snapshot["agent_capabilities"][0]["policy_blocked_tool_ids"] == ["read_document"]
    assert snapshot["agent_capabilities"][0]["unknown_allowed_tool_ids"] == ["unknown_tool"]
    assert snapshot["agent_capabilities"][0]["required_tool_ids"] == ["web_search"]
    assert snapshot["agent_capabilities"][0]["missing_required_tool_ids"] == ["web_search"]
    assert snapshot["agent_capabilities"][0]["requires_tool_calling"] is True
    assert snapshot["agent_capabilities"][0]["tool_execution_support"] == "unsupported"
    assert snapshot["agent_capabilities"][0]["missing_skill_details"][0]["skill_id"] == "rag"
    assert snapshot["agent_capabilities"][0]["missing_skill_details"][0]["suggested_tool_ids"] == ["knowledge_retriever"]
    assert snapshot["agent_capabilities"][0]["mcp_server_ids"] == ["fetch"]
    assert snapshot["agent_capabilities"][0]["configured_allowed_mcp_server_ids"] == ["fetch"]
    assert snapshot["agent_capabilities"][0]["missing_mcp_server_details"][0]["server_id"] == "github"
    assert snapshot["agent_capabilities"][0]["required_mcp_server_ids"] == ["github"]
    assert snapshot["agent_capabilities"][0]["missing_required_mcp_server_ids"] == ["github"]
    assert snapshot["agent_capabilities"][0]["availability_status"] == "unavailable"
    assert "Definition requires enabled MCP servers" in snapshot["agent_capabilities"][0]["availability_blockers"][0]
    assert snapshot["agent_capabilities"][0]["readiness_status"] == "limited"
    assert "Relevant MCP servers are not enabled" in snapshot["agent_capabilities"][0]["readiness_warnings"][0]
    assert snapshot["agent_capabilities"][0]["unknown_allowed_mcp_server_ids"] == ["unknown_server"]
    assert snapshot["agent_capabilities"][0]["recommended_collaborators"][0]["agent_id"] == "agent_b"
    assert snapshot["agent_capabilities"][0]["execution_contract"]["skill_execution_mode"] == "guidance_only"
    assert snapshot["agent_capabilities"][0]["execution_contract"]["tool_access_mode"] == "planning_only"
    assert snapshot["agent_capabilities"][0]["execution_contract"]["planning_only_tool_ids"] == ["web_search"]
    assert snapshot["agent_capabilities"][0]["execution_contract"]["mcp_access_mode"] == "planning_only"
    assert snapshot["agent_capabilities"][0]["execution_contract"]["planning_only_mcp_server_ids"] == ["fetch"]
    assert snapshot["agent_capabilities"][0]["delegation_contract"]["primary_role_mode"] == "research"
    assert snapshot["agent_capabilities"][0]["delegation_contract"]["work_strategy"] == "gather_then_handoff"
    assert snapshot["agent_capabilities"][0]["delegation_contract"]["downstream_agents"][0]["agent_id"] == "agent_b"
    assert snapshot["agent_capabilities"][0]["delegation_contract"]["watchouts"] == [
        "Relevant MCP inventory is missing: github"
    ]
    assert snapshot["agent_capabilities"][0]["role_profile_suggestion"]["profile_id"] == "research"
    assert snapshot["agent_capabilities"][0]["role_profile_suggestion"]["missing_skill_ids"] == ["rag"]
    assert snapshot["mcp_server_catalog"][1]["status"] == "disabled"


def test_build_capability_snapshot_filters_collaborators_for_selected_scope():
    snapshot = arq_jobs._build_capability_snapshot(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Researcher", "role": "research"},
                {"agent_id": "agent_b", "name": "Builder", "role": "implementation"},
            ],
            "agent_capability_summaries": [
                {
                    "agent_id": "agent_a",
                    "recommended_collaborators": [
                        {
                            "agent_id": "agent_b",
                            "agent_name": "Builder",
                            "score": 61,
                            "fit": "strong",
                            "rationale": "adds implementation lane coverage",
                        }
                    ],
                    "downstream_handoff_scores": [
                        {
                            "agent_id": "agent_b",
                            "agent_name": "Builder",
                            "score": 27,
                            "fit": "weak",
                            "rationale": "current handoff is underpowered",
                            "edge_present": True,
                            "interaction": "handoff",
                        }
                    ],
                }
            ],
        },
        active_agent_ids=["agent_a"],
        handoff_scope="selected_agents",
    )

    assert snapshot["handoff_diagnostic_scope"] == "selected_agents"
    assert snapshot["captured_from_selected_agents"] is True
    assert snapshot["agent_capabilities"][0]["recommended_collaborators"] == []
    assert snapshot["agent_capabilities"][0]["downstream_handoff_scores"] == []


def test_filter_graph_for_execution_uses_selected_scope_orchestration_summary():
    filtered = arq_jobs._filter_graph_for_execution(
        {
            "agents": [
                {"agent_id": "agent_a", "name": "Agent A"},
                {"agent_id": "agent_b", "name": "Agent B"},
            ],
            "edges": [
                {"source_agent_id": "agent_a", "target_agent_id": "agent_b", "interaction": "handoff"},
            ],
            "orchestration_summary": {
                "total_agent_count": 2,
                "start_agents": [{"agent_id": "agent_a", "agent_name": "Agent A"}],
            },
            "selected_scope_orchestration_summary": {
                "total_agent_count": 1,
                "start_agents": [{"agent_id": "agent_b", "agent_name": "Agent B"}],
            },
        },
        ["agent_b"],
    )

    assert [agent["agent_id"] for agent in filtered["agents"]] == ["agent_b"]
    assert filtered["edges"] == []
    assert filtered["orchestration_summary"]["total_agent_count"] == 1
    assert filtered["orchestration_summary"]["start_agents"][0]["agent_id"] == "agent_b"


@pytest.mark.anyio
async def test_run_harness_task_completes_document_ingest(monkeypatch):
    events = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "document_ingest",
                "input_json": {"file_path": "/tmp/a.pdf"},
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", run_id, step))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"]))

    class _Rag:
        def add_knowledge_base(self, file_path: str, user_id: str | None = None):
            return {"ok": True, "stage": "done"}

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "get_rag_engine", lambda: _Rag())

    ok = await arq_jobs.run_harness_task({}, "hr-1")

    assert ok is True
    assert events[0] == ("running", "hr-1")
    assert ("step", "hr-1", "ingest_document") in events
    assert events[-1] == ("completed", "pass")


@pytest.mark.anyio
async def test_run_harness_task_completes_session_resume_approval(monkeypatch):
    events = []
    persisted = {}

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "session_resume_approval",
                "input_json": {"session_id": "s1"},
                "user_id": "u1",
                "session_id": "s1",
                "status": "queued",
            }

        def mark_resumed(self, run_id: str):
            events.append(("resumed", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", run_id, step))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"], verification_result["artifacts"]["interrupted"]))

    class _CheckpointAdapter:
        async def load(self, session_id: str):
            return {
                "checkpoint": {
                    "interrupted": False,
                    "action_required": {"approved": True},
                }
            }

    class _GraphResumeService:
        async def resume_approved_session(self, *, session_id: str, checkpoint: dict[str, object]):
            assert session_id == "s1"
            assert checkpoint["checkpoint"]["action_required"]["approved"] is True
            return {
                "ok": True,
                "interrupted": False,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "done"},
                ],
            }

    def _persist(*, user_id: str, session_id: str, messages: list[dict[str, object]], background_tasks=None, title=None):
        persisted["user_id"] = user_id
        persisted["session_id"] = session_id
        persisted["messages"] = messages

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "CheckpointAdapter", lambda: _CheckpointAdapter())
    monkeypatch.setattr(arq_jobs, "GraphResumeService", lambda: _GraphResumeService())
    monkeypatch.setattr(arq_jobs, "persist_session_messages", _persist)

    ok = await arq_jobs.run_harness_task({}, "hr-approved")

    assert ok is True
    assert ("step", "hr-approved", "load_checkpoint") in events
    assert ("step", "hr-approved", "resume_graph") in events
    assert ("resumed", "hr-approved") in events
    assert events[-1] == ("completed", "pass", False)
    assert persisted["session_id"] == "s1"
    assert persisted["messages"][-1]["content"] == "done"


@pytest.mark.anyio
async def test_run_harness_task_marks_failed_on_exception(monkeypatch):
    events = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "document_ingest",
                "input_json": {"file_path": "/tmp/a.pdf"},
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", run_id, step))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"], verification_result["artifacts"]["stage"]))

    class _Rag:
        def add_knowledge_base(self, file_path: str, user_id: str | None = None):
            raise RuntimeError("boom")

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "get_rag_engine", lambda: _Rag())

    ok = await arq_jobs.run_harness_task({}, "hr-exception")

    assert ok is False
    assert events[0] == ("running", "hr-exception")
    assert events[-1][0] == "completed"
    assert events[-1][1] == "fail"
    assert events[-1][2] == "exception"


@pytest.mark.anyio
async def test_run_harness_task_marks_failed_for_unsupported_task(monkeypatch):
    events = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "unknown_task",
                "input_json": {},
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"], verification_result["artifacts"]["stage"]))

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())

    ok = await arq_jobs.run_harness_task({}, "hr-unsupported")

    assert ok is False
    assert events[0] == ("running", "hr-unsupported")
    assert events[-1] == ("completed", "fail", "unsupported_task_type")


@pytest.mark.anyio
async def test_run_harness_task_completes_agent_orchestration(monkeypatch):
    events = []
    verifications = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["agent_a"],
                    "loop_count": 2,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [
                            {"agent_id": "agent_a", "name": "Agent A", "skill_ids": ["research"], "model": "dev-stub"},
                            {"agent_id": "agent_b", "name": "Agent B", "skill_ids": ["research"], "model": "dev-stub"},
                        ],
                        "edges": [{"source_agent_id": "agent_a", "target_agent_id": "agent_b"}],
                        "skill_pool": [{"skill_id": "research"}],
                        "review_agent": {"enabled": True, "name": "Reviewer", "model": "dev-stub"},
                    },
                },
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            verifications.append(verification_result)
            events.append(("completed", verification_result["status"], verification_result["artifacts"]["loop_count"]))

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
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
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)

    async def _approved_review(**kwargs):
        return {"approved": True, "review_output": "PASS"}

    monkeypatch.setattr(arq_jobs, "review_orchestration_output", _approved_review)

    ok = await arq_jobs.run_harness_task({}, "hr-orch")

    assert ok is True
    assert events[0] == ("running", "hr-orch")
    assert ("step", "prepare_orchestration") in events
    assert any(event[1] == "orchestration.review_agent_attached" for event in events if event[0] == "event")
    assert any(event[1] == "orchestration.review_completed" for event in events if event[0] == "event")
    completed_event = next(event for event in events if event[0] == "event" and event[1] == "orchestration.completed")
    assert set(completed_event[2]["agent_outputs"]) == {"agent_a"}
    assert "agent_a" in completed_event[2]["output_artifacts"]
    assert completed_event[2]["output_artifacts"]["agent_a"]["node_kind"] == "agent"
    assert events[-1] == ("completed", "pass", 2)
    assert verifications[0]["artifacts"]["capability_snapshot"]["active_agent_ids"] == ["agent_a"]
    assert verifications[0]["artifacts"]["capability_snapshot"]["agent_capabilities"][0]["agent_id"] == "agent_a"


@pytest.mark.anyio
async def test_run_harness_task_injects_project_knowledge_context(monkeypatch):
    events = []
    rag_calls = []
    captured_messages = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["agent_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "knowledge_base_ids": ["kb-ops"],
                    "graph": {
                        "agents": [{"agent_id": "agent_a", "name": "Agent A", "skill_ids": [], "model": "dev-stub"}],
                        "edges": [],
                        "skill_pool": [],
                        "review_agent": {"enabled": False},
                    },
                },
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"]))

    class _Rag:
        def retrieve_context(self, query: str, k: int = 3, fetch_k: int = 20, user_id: str = None, knowledge_base_ids: list[str] | None = None):
            rag_calls.append(
                {
                    "query": query,
                    "user_id": user_id,
                    "knowledge_base_ids": list(knowledge_base_ids or []),
                    "k": k,
                    "fetch_k": fetch_k,
                }
            )
            return [
                Document(
                    page_content="Use the incident playbook before making infrastructure changes.",
                    metadata={
                        "source": "/tmp/incident-playbook.md",
                        "knowledge_base_id": "kb-ops",
                        "knowledge_base_name": "Ops KB",
                        "page_num": 2,
                    },
                )
            ]

    class _FakeLLM:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return type("Resp", (), {"content": "done"})()

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

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)
    monkeypatch.setattr(arq_jobs, "get_rag_engine", lambda: _Rag())
    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _FakeLLM())

    ok = await arq_jobs.run_harness_task({}, "hr-orch-kb")

    assert ok is True
    assert rag_calls[0]["query"] == "Coordinate work"
    assert rag_calls[0]["user_id"] == "u1"
    assert rag_calls[0]["knowledge_base_ids"] == ["kb-ops"]
    prompt = "\n\n".join(str(message.content) for message in captured_messages)
    assert "Project knowledge base context is available below." in prompt
    assert "Use the incident playbook before making infrastructure changes." in prompt
    assert "Ops KB" in prompt
    assert "/tmp/incident-playbook.md" in prompt
    knowledge_event = next(event for event in events if event[0] == "event" and event[1] == "orchestration.knowledge_context_loaded")
    assert knowledge_event[2]["knowledge_base_ids"] == ["kb-ops"]
    assert knowledge_event[2]["source_count"] == 1


@pytest.mark.anyio
async def test_run_harness_task_injects_execution_checklist_into_prompt(monkeypatch):
    events = []
    captured_messages = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["agent_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "task_checklist": [
                        {"item_id": "check_1", "content": "Audit the current orchestration flow", "status": "completed"},
                        {
                            "item_id": "check_2",
                            "content": "Implement the safer rollout path",
                            "status": "in_progress",
                            "active_form": "Implementing the safer rollout path",
                        },
                        {"item_id": "check_3", "content": "Verify the final behavior", "status": "pending"},
                    ],
                    "graph": {
                        "agents": [{"agent_id": "agent_a", "name": "Agent A", "skill_ids": [], "model": "dev-stub"}],
                        "edges": [],
                        "skill_pool": [],
                        "review_agent": {"enabled": False},
                    },
                },
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"]))

    class _FakeLLM:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return type("Resp", (), {"content": "done"})()

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

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)
    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _FakeLLM())

    ok = await arq_jobs.run_harness_task({}, "hr-orch-checklist")

    assert ok is True
    prompt = "\n\n".join(str(message.content) for message in captured_messages)
    assert "Execution checklist:" in prompt
    assert "[completed] Audit the current orchestration flow" in prompt
    assert "[in progress] Implementing the safer rollout path" in prompt
    assert "[pending] Verify the final behavior" in prompt
    checklist_event = next(event for event in events if event[0] == "event" and event[1] == "orchestration.checklist_loaded")
    assert checklist_event[2]["checklist_count"] == 3
    assert checklist_event[2]["open_item_count"] == 2
    assert checklist_event[2]["open_items_preview"] == [
        "Implementing the safer rollout path",
        "Verify the final behavior",
    ]


@pytest.mark.anyio
async def test_run_harness_task_records_cluster_output_artifacts(monkeypatch):
    events = []
    verifications = []
    memory_calls = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["cluster_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [
                            {
                                "agent_id": "cluster_a",
                                "name": "Brainstorm Cluster",
                                "node_kind": "cluster",
                                "cluster_strategy": "brainstorm",
                                "brainstorm_rounds": 1,
                                "cluster_auto_research": True,
                                "cluster_members": [
                                    {"member_id": "m1", "name": "Lead", "role": "chair", "model": "dev-stub"},
                                    {"member_id": "m2", "name": "Critic", "role": "critic", "model": "dev-stub"},
                                ],
                            }
                        ],
                        "edges": [],
                        "skill_pool": [],
                        "review_agent": {"enabled": False},
                    },
                },
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            verifications.append(verification_result)
            events.append(("completed", verification_result["status"]))

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
            joined = "\n".join(str(message.content) for message in messages)
            if "Respond using this exact section structure" in joined:
                return type(
                    "Resp",
                    (),
                    {
                        "content": (
                            "Conservative Strategy:\nImprove carefully.\n\n"
                            "Balanced Strategy:\nPilot the change.\n\n"
                            "Aggressive Strategy:\nRoll it out now.\n\n"
                            "Winning Strategy:\nBalanced is safest.\n\n"
                            "Next Step:\nPilot with internal users."
                        )
                    },
                )()
            return type("Resp", (), {"content": "Thoughts\nVOTE: BALANCED"})()

    class _MemoryEngine:
        def add_chat_summary(self, *, user_id: str, session_id: str, summary_text: str, start_msg_id=None, end_msg_id=None, created_at=None):
            memory_calls.append(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "summary_text": summary_text,
                }
            )

    async def _fake_search(**kwargs):
        return "1. [Paper A](https://example.test/paper-a)\n   Latest progress."

    async def _fake_paper_search(**kwargs):
        result = type(
            "SearchResult",
            (),
            {
                "title": "arXiv Paper",
                "url": "https://arxiv.org/abs/1234.5678",
                "snippet": "Paper abstract highlight.",
                "provider": "arxiv",
            },
        )()
        return type("SearchResponse", (), {"results": [result], "cached": False})()

    class _Preview:
        def __init__(self, url: str, title: str, description: str):
            self.url = url
            self.final_url = url
            self.title = title
            self.description = description
            self.status_code = 200
            self.content_type = "text/html"

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)
    monkeypatch.setattr(arq_jobs, "review_orchestration_output", lambda **kwargs: {"approved": True, "review_output": "PASS"})
    monkeypatch.setattr(arq_jobs, "enhanced_web_search", _fake_search)
    monkeypatch.setattr(arq_jobs, "enhanced_search_response", _fake_paper_search)
    monkeypatch.setattr(
        arq_jobs,
        "fetch_browser_previews",
        lambda urls, **kwargs: [_Preview(urls[0] if urls else "https://arxiv.org/abs/1234.5678", "Browser Checked Paper", "Validated preview from page fetch.")],
    )
    monkeypatch.setattr(arq_jobs, "ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr(arq_jobs, "UserMemoryEngine", lambda: _MemoryEngine())
    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _FakeLLM())

    ok = await arq_jobs.run_harness_task({}, "hr-orch-cluster")

    assert ok is True
    completed_event = next(event for event in events if event[0] == "event" and event[1] == "orchestration.completed")
    assert any(event[1] == "orchestration.cluster_research_completed" for event in events if event[0] == "event")
    artifact = completed_event[2]["output_artifacts"]["cluster_a"]
    assert artifact["winning_vote"] == "BALANCED"
    assert artifact["next_step"] == "Pilot with internal users."
    assert artifact["research"]["result_count"] == 4
    assert "arXiv Paper" in artifact["research"]["digest"]
    assert artifact["research"]["research_mode"] == "paper_first_browser_preview"
    assert artifact["research"]["paper_queries"]
    assert artifact["research"]["web_queries"]
    assert artifact["research"]["papers"][0]["title"] == "arXiv Paper"
    assert any(source["url"] == "https://example.test/paper-a" for source in artifact["research"]["sources"])
    assert any(citation["label"] == "Paper A" for citation in artifact["research"]["citations"])
    assert any(item == "Latest progress." for item in artifact["research"]["latest_progress"])
    assert artifact["research"]["browser_previews"][0]["title"] == "Browser Checked Paper"
    assert artifact["research"]["provider_runs"][0]["provider"] == "arxiv"
    assert artifact["research"]["memory"]["stored"] is True
    assert verifications[0]["artifacts"]["output_artifacts"]["cluster_a"]["strategies"]["balanced"] == "Pilot the change."
    assert "Research Digest:" in completed_event[2]["agent_outputs"]["cluster_a"]
    assert memory_calls[0]["user_id"] == "u1"


@pytest.mark.anyio
async def test_run_harness_task_records_recovery_mode_in_verification(monkeypatch):
    verifications = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["agent_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [{"agent_id": "agent_a", "name": "Agent A", "skill_ids": [], "model": "dev-stub"}],
                        "edges": [],
                        "skill_pool": [],
                        "review_agent": {"enabled": False},
                    },
                },
                "metadata_json": {"review_recovery_mode": "continue_without_research"},
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            return None

        def set_current_step(self, run_id: str, step: str):
            return None

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            return None

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            verifications.append(verification_result)

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

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)

    ok = await arq_jobs.run_harness_task({}, "hr-orch-recovery-mode")

    assert ok is True
    assert verifications[0]["artifacts"]["recovery_mode"] == "continue_without_research"


@pytest.mark.anyio
async def test_run_harness_task_pauses_agent_orchestration_on_review_block(monkeypatch):
    events = []
    saved_inputs = {}

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["agent_a", "agent_b"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [
                            {"agent_id": "agent_a", "name": "Planner", "skill_ids": ["research"], "model": "dev-stub"},
                            {"agent_id": "agent_b", "name": "Builder", "skill_ids": [], "model": "dev-stub"},
                        ],
                        "edges": [{"source_agent_id": "agent_a", "target_agent_id": "agent_b", "interaction": "handoff"}],
                        "tool_catalog": [
                            {"tool_id": "get_current_time", "title": "Get Current Time", "description": "Read the current system time."},
                        ],
                        "agent_capability_summaries": [
                            {
                                "agent_id": "agent_a",
                                "loaded_skill_ids": ["research"],
                                "enabled_tool_ids": [],
                                "capability_brief": "Approved skills: research.",
                            },
                            {
                                "agent_id": "agent_b",
                                "loaded_skill_ids": [],
                                "enabled_tool_ids": ["get_current_time"],
                                "capability_brief": "Enabled tools: get_current_time.",
                            },
                        ],
                        "skill_pool": [{"skill_id": "research"}],
                        "review_agent": {"enabled": True, "name": "Reviewer", "model": "dev-stub"},
                    },
                },
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def update_run_input_json(self, run_id: str, input_json: dict[str, object]):
            saved_inputs["input_json"] = input_json
            return {"run_id": run_id, "input_json": input_json}

        def create_approval_request(self, *, run_id: str, action_type: str, reason: str | None, payload_json: dict[str, object], requested_by: str | None):
            events.append(("approval", action_type, payload_json))
            return {"approval_id": "ha-1", "action_type": action_type}

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"]))

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
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
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)

    async def _blocked_review(**kwargs):
        output = str(kwargs.get("output") or "")
        if "unsafe" in output:
            return {"approved": False, "review_output": "BLOCK: unsafe output"}
        return {"approved": True, "review_output": "PASS"}

    monkeypatch.setattr(arq_jobs, "review_orchestration_output", _blocked_review)

    class _FinalOnlyLLM:
        def bind_tools(self, tools):
            self.tools = list(tools)
            return self

        async def ainvoke(self, messages):
            system_text = "\n".join(str(message.content) for message in messages if getattr(message, "type", "") == "system")
            if "You are Planner" in system_text:
                return AIMessage(content="Planner handoff summary")
            if any(getattr(message, "type", "") == "tool" for message in messages):
                return AIMessage(content="unsafe final output")
            return AIMessage(
                content="",
                tool_calls=[{"name": "get_current_time", "args": {}, "id": "call_1", "type": "tool_call"}],
            )

    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _FinalOnlyLLM())

    ok = await arq_jobs.run_harness_task({}, "hr-orch-blocked")

    assert ok is False
    assert any(event[0] == "approval" and event[1] == "orchestration_review" for event in events)
    approval_event = next(event for event in events if event[0] == "approval")
    assert approval_event[2]["review_stage"] == "agent_output_final"
    assert approval_event[2]["artifact_source"] == "output_artifact"
    assert approval_event[2]["artifact_snapshot"]["agent_name"] == "Builder"
    assert approval_event[2]["artifact_snapshot"]["final_output"] is True
    assert approval_event[2]["artifact_snapshot"]["consumed_handoffs"][0]["source_agent_name"] == "Planner"
    assert approval_event[2]["artifact_snapshot"]["tool_runs"][0]["tool_id"] == "get_current_time"
    assert "orchestration_resume" in saved_inputs["input_json"]
    assert "rollback_state" in saved_inputs["input_json"]["orchestration_resume"]


@pytest.mark.anyio
async def test_run_harness_task_reports_missing_skill_capability_details(monkeypatch):
    events = []
    verifications = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["agent_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [{"agent_id": "agent_a", "name": "Agent A", "skill_ids": ["research"], "model": "dev-stub"}],
                        "edges": [],
                        "skill_pool": [],
                        "review_agent": {"enabled": False},
                    },
                },
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            verifications.append(verification_result)
            events.append(("completed", verification_result["status"]))

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())

    ok = await arq_jobs.run_harness_task({}, "hr-missing-skill")

    assert ok is False
    ready_event = next(event for event in events if event[0] == "event" and event[1] == "orchestration.agent_ready")
    assert ready_event[2]["missing_skills"] == ["research"]
    assert ready_event[2]["missing_skill_details"][0]["title"] == "Research"
    assert "external evidence" in str(ready_event[2]["missing_skill_details"][0]["prompt_hint"] or "")
    assert "web_search" in ready_event[2]["missing_skill_details"][0]["suggested_tool_ids"]

    blocked_agents = verifications[0]["artifacts"]["blocked_agents"]
    assert blocked_agents[0]["agent_name"] == "Agent A"
    assert blocked_agents[0]["missing_skill_details"][0]["skill_id"] == "research"
    assert verifications[0]["artifacts"]["error_code"] == "missing_skill_approval"


@pytest.mark.anyio
async def test_run_harness_task_pauses_agent_orchestration_on_segment_review_block(monkeypatch):
    events = []
    saved_inputs = {}

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["agent_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [{"agent_id": "agent_a", "name": "Agent A", "skill_ids": [], "model": "dev-stub"}],
                        "edges": [],
                        "skill_pool": [],
                        "review_agent": {
                            "enabled": True,
                            "name": "Reviewer",
                            "model": "dev-stub",
                            "pipeline_review_enabled": True,
                            "pipeline_chunk_chars": 10,
                            "pipeline_chunk_overlap_chars": 2,
                        },
                    },
                },
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def update_run_input_json(self, run_id: str, input_json: dict[str, object]):
            saved_inputs["input_json"] = input_json
            return {"run_id": run_id, "input_json": input_json}

        def create_approval_request(self, *, run_id: str, action_type: str, reason: str | None, payload_json: dict[str, object], requested_by: str | None):
            events.append(("approval", action_type, payload_json))
            return {"approval_id": "ha-1", "action_type": action_type}

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"]))

    class _FakeLLM:
        async def ainvoke(self, messages):
            return type("Resp", (), {"content": "1234567890unsafe-fragment-more-output"})()

    async def _review(**kwargs):
        output = str(kwargs.get("output") or "")
        if "unsafe" in output:
            return {"approved": False, "review_output": "BLOCK: unsafe segment"}
        return {"approved": True, "review_output": "PASS"}

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

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)
    monkeypatch.setattr(arq_jobs, "review_orchestration_output", _review)
    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _FakeLLM())

    ok = await arq_jobs.run_harness_task({}, "hr-orch-segment-blocked")

    assert ok is False
    assert any(event[1] == "orchestration.review_segment_scan_completed" for event in events if event[0] == "event")
    approval_event = next(event for event in events if event[0] == "approval")
    assert approval_event[2]["review_stage"] == "agent_output_segment"
    assert approval_event[2]["segment_count"] >= 2
    assert approval_event[2]["segment_preview"]
    assert approval_event[2]["artifact_source"] == "output_artifact"
    assert approval_event[2]["artifact_snapshot"]["agent_name"] == "Agent A"
    assert approval_event[2]["artifact_snapshot"]["output_preview"]
    assert "orchestration_resume" in saved_inputs["input_json"]


@pytest.mark.anyio
async def test_run_harness_task_pauses_agent_orchestration_on_stream_review_block(monkeypatch):
    events = []
    saved_inputs = {}

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["agent_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [{"agent_id": "agent_a", "name": "Agent A", "skill_ids": [], "model": "dev-stub"}],
                        "edges": [],
                        "skill_pool": [],
                        "review_agent": {
                            "enabled": True,
                            "name": "Reviewer",
                            "model": "dev-stub",
                            "stream_review_min_chars": 5,
                            "stream_review_window_chars": 20,
                        },
                    },
                },
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def update_run_input_json(self, run_id: str, input_json: dict[str, object]):
            saved_inputs["input_json"] = input_json
            return {"run_id": run_id, "input_json": input_json}

        def create_approval_request(self, *, run_id: str, action_type: str, reason: str | None, payload_json: dict[str, object], requested_by: str | None):
            events.append(("approval", action_type, payload_json))
            return {"approval_id": "ha-1", "action_type": action_type}

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"]))

    class _StreamingLLM:
        async def astream(self, messages):
            yield "safe "
            yield "unsafe "
            yield "tail"

        async def ainvoke(self, messages):
            raise AssertionError("streaming path should have blocked before ainvoke fallback")

    async def _review(**kwargs):
        output = str(kwargs.get("output") or "")
        if "unsafe" in output:
            return {"approved": False, "review_output": "BLOCK: unsafe streamed output"}
        return {"approved": True, "review_output": "PASS"}

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

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)
    monkeypatch.setattr(arq_jobs, "review_orchestration_output", _review)
    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _StreamingLLM())

    ok = await arq_jobs.run_harness_task({}, "hr-orch-stream-blocked")

    assert ok is False
    assert any(event[1] == "orchestration.review_stream_blocked" for event in events if event[0] == "event")
    approval_event = next(event for event in events if event[0] == "approval")
    assert approval_event[2]["review_stage"] == "agent_output_stream"
    assert approval_event[2]["partial_output"]
    assert approval_event[2]["segment_preview"]
    assert approval_event[2]["artifact_source"] == "partial_stream"
    assert approval_event[2]["artifact_snapshot"]["agent_name"] == "Agent A"
    assert approval_event[2]["artifact_snapshot"]["output_preview"]
    assert "rollback_state" in saved_inputs["input_json"]["orchestration_resume"]


@pytest.mark.anyio
async def test_run_harness_task_stream_review_blocks_before_full_chunk_tail(monkeypatch):
    events = []
    saved_inputs = {}

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["agent_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [{"agent_id": "agent_a", "name": "Agent A", "skill_ids": [], "model": "dev-stub"}],
                        "edges": [],
                        "skill_pool": [],
                        "review_agent": {
                            "enabled": True,
                            "name": "Reviewer",
                            "model": "dev-stub",
                            "stream_review_trigger_chars": 6,
                            "stream_review_window_chars": 24,
                        },
                    },
                },
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def update_run_input_json(self, run_id: str, input_json: dict[str, object]):
            saved_inputs["input_json"] = input_json
            return {"run_id": run_id, "input_json": input_json}

        def create_approval_request(self, *, run_id: str, action_type: str, reason: str | None, payload_json: dict[str, object], requested_by: str | None):
            events.append(("approval", action_type, payload_json))
            return {"approval_id": "ha-1", "action_type": action_type}

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"]))

    class _StreamingLLM:
        async def astream(self, messages):
            yield "safe unsafe tail"

        async def ainvoke(self, messages):
            raise AssertionError("streaming path should block before ainvoke")

    async def _review(**kwargs):
        output = str(kwargs.get("output") or "")
        if "unsafe" in output:
            return {"approved": False, "review_output": "BLOCK: token-ish window"}
        return {"approved": True, "review_output": "PASS"}

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

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)
    monkeypatch.setattr(arq_jobs, "review_orchestration_output", _review)
    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _StreamingLLM())

    ok = await arq_jobs.run_harness_task({}, "hr-orch-stream-token-ish")

    assert ok is False
    approval_event = next(event for event in events if event[0] == "approval")
    assert approval_event[2]["review_stage"] == "agent_output_stream"
    assert approval_event[2]["check_count"] >= 1
    assert approval_event[2]["last_reviewed_char"] == approval_event[2]["segment_end_char"]
    assert "tail" not in saved_inputs["input_json"]["orchestration_resume"]["continuation"]["partial_output"]


@pytest.mark.anyio
async def test_run_harness_task_resumes_stream_block_with_continuation_prefix(monkeypatch):
    events = []
    saved_updates = []
    verifications = []

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["agent_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [{"agent_id": "agent_a", "name": "Agent A", "skill_ids": [], "model": "dev-stub"}],
                        "edges": [],
                        "skill_pool": [],
                        "review_agent": {
                            "enabled": True,
                            "name": "Reviewer",
                            "model": "dev-stub",
                            "stream_review_min_chars": 100,
                            "stream_review_window_chars": 100,
                        },
                    },
                    "orchestration_resume": {
                        "next_step_index": 0,
                        "review_decision": "approved",
                        "continue_mode": "accept_partial_stream_output",
                        "state": {
                            "task": "Coordinate work",
                            "agent_outputs": {},
                            "output_artifacts": {},
                            "current_agent": "",
                            "loop_index": 0,
                            "errors": [],
                        },
                        "rollback_state": {
                            "task": "Coordinate work",
                            "agent_outputs": {},
                            "output_artifacts": {},
                            "current_agent": "",
                            "loop_index": 0,
                            "errors": [],
                        },
                        "continuation": {
                            "agent_id": "agent_a",
                            "agent_name": "Agent A",
                            "partial_output": "safe partial ",
                            "review_output": "BLOCK: inspect before continuing",
                            "review_stage": "agent_output_stream",
                            "step_index": 0,
                            "loop_number": 1,
                        },
                    },
                },
                "metadata_json": {"review_recovery_mode": "continue_with_partial_stream_output"},
                "user_id": "u1",
                "status": "approved",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def update_run_input_json(self, run_id: str, input_json: dict[str, object]):
            saved_updates.append(input_json)
            return {"run_id": run_id, "input_json": input_json}

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            verifications.append(verification_result)
            events.append(("completed", verification_result["status"]))

    class _StreamingLLM:
        async def astream(self, messages):
            yield "continued"
            yield " answer"

        async def ainvoke(self, messages):
            raise AssertionError("streaming continuation should use astream")

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

    async def _review(**kwargs):
        return {"approved": True, "review_output": "PASS"}

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)
    monkeypatch.setattr(arq_jobs, "review_orchestration_output", _review)
    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _StreamingLLM())

    ok = await arq_jobs.run_harness_task({}, "hr-orch-stream-resume")

    assert ok is True
    assert any(event[1] == "orchestration.stream_continuation_resumed" for event in events if event[0] == "event")
    assert any(event[1] == "orchestration.stream_continuation_completed" for event in events if event[0] == "event")
    assert verifications[0]["artifacts"]["agent_outputs"]["agent_a"] == "safe partial continued answer"
    assert any("orchestration_resume" not in update for update in saved_updates)


@pytest.mark.anyio
async def test_run_harness_task_pauses_on_blocked_cluster_research_evidence(monkeypatch):
    events = []
    saved_inputs = {}

    class _RunService:
        def get_run(self, run_id: str):
            return {
                "run_id": run_id,
                "task_type": "agent_orchestration",
                "input_json": {
                    "selected_agent_ids": ["cluster_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [
                            {
                                "agent_id": "cluster_a",
                                "name": "Brainstorm Cluster",
                                "node_kind": "cluster",
                                "cluster_strategy": "brainstorm",
                                "brainstorm_rounds": 1,
                                "cluster_auto_research": True,
                                "cluster_members": [
                                    {"member_id": "m1", "name": "Lead", "role": "chair", "model": "dev-stub"},
                                ],
                            }
                        ],
                        "edges": [],
                        "skill_pool": [],
                        "review_agent": {"enabled": True, "name": "Reviewer", "model": "dev-stub"},
                    },
                },
                "user_id": "u1",
                "status": "queued",
            }

        def mark_running(self, run_id: str):
            events.append(("running", run_id))

        def set_current_step(self, run_id: str, step: str):
            events.append(("step", step))

        def record_event(self, run_id: str, *, event_type: str, actor=None, details=None):
            events.append(("event", event_type, details))

        def update_run_input_json(self, run_id: str, input_json: dict[str, object]):
            saved_inputs["input_json"] = input_json
            return {"run_id": run_id, "input_json": input_json}

        def create_approval_request(self, *, run_id: str, action_type: str, reason: str | None, payload_json: dict[str, object], requested_by: str | None):
            events.append(("approval", action_type, payload_json))
            return {"approval_id": "ha-1", "action_type": action_type}

        def complete_with_verification(self, run_id: str, verification_result: dict[str, object]):
            events.append(("completed", verification_result["status"]))

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
            joined = "\n".join(str(message.content) for message in messages)
            if "Respond using this exact section structure" in joined:
                return type(
                    "Resp",
                    (),
                    {
                        "content": (
                            "Conservative Strategy:\nImprove carefully.\n\n"
                            "Balanced Strategy:\nPilot the change.\n\n"
                            "Aggressive Strategy:\nRoll it out now.\n\n"
                            "Winning Strategy:\nBalanced is safest.\n\n"
                            "Next Step:\nPilot with internal users."
                        )
                    },
                )()
            return type("Resp", (), {"content": "Thoughts\nVOTE: BALANCED"})()

    async def _fake_search(**kwargs):
        return "1. [Paper A](https://example.test/paper-a)\n   Latest progress."

    async def _fake_paper_search(**kwargs):
        result = type(
            "SearchResult",
            (),
            {
                "title": "arXiv Paper",
                "url": "https://arxiv.org/abs/1234.5678",
                "snippet": "Paper abstract highlight.",
                "provider": "arxiv",
            },
        )()
        return type("SearchResponse", (), {"results": [result], "cached": False})()

    async def _fake_review_cluster_research_evidence(**kwargs):
        return {"approved": False, "review_output": "BLOCK: unverified external evidence"}

    monkeypatch.setattr(arq_jobs, "build_run_service", lambda: _RunService())
    monkeypatch.setattr(arq_jobs, "_build_provider_registry", lambda user_id=None: registry)
    monkeypatch.setattr(arq_jobs, "enhanced_web_search", _fake_search)
    monkeypatch.setattr(arq_jobs, "enhanced_search_response", _fake_paper_search)
    monkeypatch.setattr(arq_jobs, "fetch_browser_previews", lambda urls, **kwargs: [])
    monkeypatch.setattr(arq_jobs, "ensure_schema_if_possible", lambda: False)
    monkeypatch.setattr(arq_jobs, "_review_cluster_research_evidence", _fake_review_cluster_research_evidence)
    monkeypatch.setattr("app.runtime.graph.orchestration_graph.get_llm_for_provider", lambda **kwargs: _FakeLLM())

    ok = await arq_jobs.run_harness_task({}, "hr-orch-research-blocked")

    assert ok is False
    assert any(event[1] == "orchestration.cluster_research_completed" for event in events if event[0] == "event")
    assert any(event[1] == "orchestration.cluster_research_review_completed" for event in events if event[0] == "event")
    approval_event = next(event for event in events if event[0] == "approval")
    assert approval_event[2]["review_stage"] == "cluster_research"
    assert approval_event[2]["research_queries"]
    assert approval_event[2]["artifact_source"] == "research_artifact"
    assert approval_event[2]["artifact_snapshot"]["cluster_name"] == "Brainstorm Cluster"
    assert approval_event[2]["artifact_snapshot"]["research"]["blocked"] is True
    assert approval_event[2]["artifact_snapshot"]["research"]["review_output"] == "BLOCK: unverified external evidence"
    rollback_artifact = saved_inputs["input_json"]["orchestration_resume"]["rollback_state"]["output_artifacts"]["cluster_a"]
    assert rollback_artifact["research"]["blocked"] is True
    assert rollback_artifact["research"]["review_output"] == "BLOCK: unverified external evidence"


@pytest.mark.anyio
async def test_resume_harness_task_delegates_to_run(monkeypatch):
    called = {"run_id": None}

    async def _run(ctx, run_id: str):
        called["run_id"] = run_id
        return True

    monkeypatch.setattr(arq_jobs, "run_harness_task", _run)

    ok = await arq_jobs.resume_harness_task({}, "hr-2")

    assert ok is True
    assert called["run_id"] == "hr-2"
