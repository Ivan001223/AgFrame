import pytest

from app.runtime.graph.graph import _check_approval
from app.runtime.graph.resume_service import GraphResumeService


class _Snapshot:
    def __init__(self, values: dict[str, object], config: dict[str, object]):
        self.values = values
        self.config = config


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
