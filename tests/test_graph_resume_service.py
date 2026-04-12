import pytest

from app.infrastructure.config.settings import settings
from app.platform.contracts.runtime_protocol import RuntimeResumePoint
from app.runtime.graph.graph import _after_generate_key, _check_approval, run_app
from app.runtime.graph.resume_service import GraphResumeService, build_runtime_resume_point


class _Snapshot:
    def __init__(self, values: dict[str, object], config: dict[str, object]):
        self.values = values
        self.config = config


def test_run_app_includes_approval_nodes_even_when_flag_defaults_off():
    graph = run_app().get_graph()

    assert "human_interrupt" in graph.nodes
    assert "check_approval" in graph.nodes


def test_after_generate_routes_to_interrupt_before_self_correction(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.feature_flags, "enable_self_correction", True)

    decision = _after_generate_key({"context": {"require_human_approval": True}})

    assert decision == "interrupt"


def test_after_generate_routes_to_grader_without_approval(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.feature_flags, "enable_self_correction", True)

    decision = _after_generate_key({"context": {}})

    assert decision == "grade"


def test_after_generate_routes_to_end_when_self_correction_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings.feature_flags, "enable_self_correction", False)

    decision = _after_generate_key({"context": {}})

    assert decision == "end"


def test_check_approval_routes_approved_when_interrupt_cleared():
    decision = _check_approval(
        {
            "interrupted": False,
            "next_step": "generate",
            "action_required": None,
        }
    )

    assert decision == "approved"


@pytest.mark.anyio
async def test_graph_resume_service_updates_state_and_invokes_graph():
    calls: dict[str, object] = {}

    class _GraphApp:
        async def aget_state(self, config: dict[str, object]):
            checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
            if checkpoint_id == "cp-1":
                return _Snapshot(
                    {
                        "context": {
                            "require_human_approval": True,
                            "interrupt_action_type": "deploy",
                            "interrupt_description": "approve deploy",
                            "interrupt_payload": {"next_step": "generate"},
                            "session_id": "s1",
                        },
                        "action_required": {
                            "action_type": "deploy",
                            "description": "approve deploy",
                            "payload": {"next_step": "generate"},
                            "requires_approval": True,
                            "approved": True,
                        },
                        "interrupted": True,
                    },
                    config,
                )
            return _Snapshot(
                {
                    "interrupted": False,
                    "messages": [{"role": "assistant", "content": "done"}],
                },
                config,
            )

        async def aupdate_state(self, config: dict[str, object], values: dict[str, object], as_node: str | None = None, task_id: str | None = None):
            calls["update"] = (config, values, as_node, task_id)
            return {
                "configurable": {
                    "thread_id": "s1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "cp-2",
                }
            }

        async def ainvoke(self, input_value, config: dict[str, object]):
            calls["invoke"] = (input_value, config)
            return {"interrupted": False}

    service = GraphResumeService(graph_app=_GraphApp())
    result = await service.resume_approved_session(
        session_id="s1",
        checkpoint={
            "checkpoint": {
                "id": "cp-1",
                "checkpoint_id": "cp-1",
                "checkpoint_ns": "",
                "action_required": {"approved": True},
                "interrupted": False,
            }
        },
    )

    assert result["ok"] is True
    assert result["interrupted"] is False
    assert calls["update"][2] == "check_approval"
    assert calls["update"][1]["interrupted"] is False
    assert calls["update"][1]["action_required"] is None
    assert "require_human_approval" not in calls["update"][1]["context"]
    assert calls["invoke"][0] is None
    assert calls["invoke"][1]["configurable"]["checkpoint_id"] == "cp-2"


@pytest.mark.anyio
async def test_graph_resume_service_rejects_unapproved_checkpoint():
    class _GraphApp:
        async def aget_state(self, config: dict[str, object]):
            raise AssertionError("state lookup should not happen for unapproved checkpoints")

    service = GraphResumeService(graph_app=_GraphApp())
    result = await service.resume_approved_session(
        session_id="s1",
        checkpoint={
            "checkpoint": {
                "checkpoint_id": "cp-1",
                "checkpoint_ns": "",
                "action_required": {"approved": False},
                "interrupted": True,
            }
        },
    )

    assert result["ok"] is False
    assert result["error_code"] == "approval_not_granted"


def test_build_runtime_resume_point_extracts_checkpoint_fields():
    point = build_runtime_resume_point(
        {
            "checkpoint": {
                "action_required": {"approved": True},
                "orchestration_resume": {
                    "next_step_index": 3,
                    "rollback_state": {"agent_outputs": {"agent_a": "safe"}},
                    "continuation": {"agent_id": "agent_b"},
                },
            }
        }
    )

    assert isinstance(point, RuntimeResumePoint)
    assert point.next_step_index == 3
    assert point.rollback_state["agent_outputs"]["agent_a"] == "safe"
    assert point.continuation["agent_id"] == "agent_b"
