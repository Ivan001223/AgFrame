from typing import Any

from app.runtime.graph.registry import register_node
from app.runtime.graph.state import ActionRequired, AgentState


@register_node("human_interrupt")
async def human_interrupt_node(state: AgentState) -> dict[str, Any]:
    action_type = state.get("context", {}).get("interrupt_action_type", "unknown")
    description = state.get("context", {}).get("interrupt_description", "需要用户批准的操作")

    action_required: ActionRequired = {
        "action_type": action_type,
        "description": description,
        "payload": state.get("context", {}).get("interrupt_payload", {}),
        "requires_approval": True,
        "approved": False,
        "approved_by": None,
        "approved_at": None,
    }

    return {
        "action_required": action_required,
        "interrupted": True,
    }


@register_node("check_approval")
async def check_approval_node(state: AgentState) -> dict[str, Any]:
    action_required = state.get("action_required")
    if action_required and action_required.get("approved"):
        return {"interrupted": False, "next_step": action_required.get("payload", {}).get("next_step", "generate")}
    return {"interrupted": True, "next_step": "wait_approval"}
