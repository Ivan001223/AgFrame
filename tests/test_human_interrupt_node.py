import pytest

from app.runtime.graph.nodes.human_interrupt import check_approval_node, human_interrupt_node


@pytest.mark.anyio
async def test_human_interrupt_node_uses_context_payload():
    result = await human_interrupt_node(
        {
            "context": {
                "interrupt_action_type": "deploy",
                "interrupt_description": "approve deploy",
                "interrupt_payload": {"next_step": "generate"},
            }
        }
    )

    assert result["interrupted"] is True
    assert result["action_required"]["action_type"] == "deploy"
    assert result["action_required"]["payload"] == {"next_step": "generate"}


@pytest.mark.anyio
async def test_check_approval_node_clears_human_approval_context():
    result = await check_approval_node(
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
        }
    )

    assert result["interrupted"] is False
    assert result["next_step"] == "generate"
    assert result["action_required"] is None
    assert result["context"]["session_id"] == "s1"
    assert "require_human_approval" not in result["context"]
    assert "interrupt_action_type" not in result["context"]
    assert "interrupt_payload" not in result["context"]
