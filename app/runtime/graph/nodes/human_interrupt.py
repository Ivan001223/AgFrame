from typing import Any

from app.runtime.contracts.workflow_context import build_workflow_context_payload
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import ActionRequired, AgentState


@register_node("human_interrupt")
async def human_interrupt_node(state: AgentState) -> dict[str, Any]:
    ctx = build_workflow_context_payload(current=state.get("context"))
    action_type = ctx.get("interrupt_action_type", "unknown")
    description = ctx.get("interrupt_description", "需要用户批准的操作")

    action_required: ActionRequired = {
        "action_type": action_type,
        "description": description,
        "payload": ctx.get("interrupt_payload", {}),
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
        ctx = build_workflow_context_payload(
            current=state.get("context"),
            clear_human_approval=True,
        )
        return {
            "interrupted": False,
            "next_step": action_required.get("payload", {}).get("next_step", "generate"),
            "action_required": None,
            "context": ctx,
        }
    return {"interrupted": True, "next_step": "wait_approval"}
