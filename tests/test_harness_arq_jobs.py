import pytest

from app.infrastructure.queue import arq_jobs
from app.runtime.llm.provider_registry import ModelProviderRegistry, RegisteredProvider


def test_chunk_output_for_review_preserves_overlap():
    chunks = arq_jobs._chunk_output_for_review("abcdefghij", chunk_chars=4, overlap_chars=1)

    assert [chunk["content"] for chunk in chunks] == ["abcd", "defg", "ghij"]
    assert chunks[1]["start_char"] == 3
    assert chunks[2]["start_char"] == 6


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

    ok = await arq_jobs.run_harness_task({}, "hr-orch")

    assert ok is True
    assert events[0] == ("running", "hr-orch")
    assert ("step", "prepare_orchestration") in events
    assert any(event[1] == "orchestration.review_agent_attached" for event in events if event[0] == "event")
    assert any(event[1] == "orchestration.review_completed" for event in events if event[0] == "event")
    completed_event = next(event for event in events if event[0] == "event" and event[1] == "orchestration.completed")
    assert set(completed_event[2]["agent_outputs"]) == {"agent_a"}
    assert completed_event[2]["output_artifacts"] == {}
    assert events[-1] == ("completed", "pass", 2)


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
                    "selected_agent_ids": ["agent_a"],
                    "loop_count": 1,
                    "task": "Coordinate work",
                    "graph": {
                        "agents": [{"agent_id": "agent_a", "name": "Agent A", "skill_ids": ["research"], "model": "dev-stub"}],
                        "edges": [],
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
        return {"approved": False, "review_output": "BLOCK: unsafe output"}

    monkeypatch.setattr(arq_jobs, "review_orchestration_output", _blocked_review)

    ok = await arq_jobs.run_harness_task({}, "hr-orch-blocked")

    assert ok is False
    assert any(event[0] == "approval" and event[1] == "orchestration_review" for event in events)
    assert "orchestration_resume" in saved_inputs["input_json"]
    assert "rollback_state" in saved_inputs["input_json"]["orchestration_resume"]


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
