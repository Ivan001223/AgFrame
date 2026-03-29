from typing import Any

from langchain_core.runnables import RunnableConfig

from app.runtime.contracts.workflow_context import build_workflow_context_payload
from app.runtime.graph.registry import register_node
from app.runtime.graph.state import AgentState


def _get_configurable_str(config: RunnableConfig | None, key: str) -> str | None:
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    value = configurable.get(key)
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


@register_node("bootstrap_request")
async def bootstrap_request_node(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    configured_user_id = _get_configurable_str(config, "user_id")
    configured_session_id = _get_configurable_str(config, "thread_id")

    existing_context = state.get("context")
    existing_context_user_id = None
    existing_context_session_id = None
    if isinstance(existing_context, dict):
        context_user_id = existing_context.get("user_id")
        context_session_id = existing_context.get("session_id")
        if isinstance(context_user_id, str) and context_user_id.strip():
            existing_context_user_id = context_user_id.strip()
        if isinstance(context_session_id, str) and context_session_id.strip():
            existing_context_session_id = context_session_id.strip()

    state_user_id = str(state.get("user_id") or "").strip() or None
    state_session_id = str(state.get("session_id") or "").strip() or None

    # Trust server-injected configurable identity first, then fall back to persisted state.
    resolved_user_id = configured_user_id or state_user_id or existing_context_user_id
    resolved_session_id = configured_session_id or state_session_id or existing_context_session_id

    context = build_workflow_context_payload(
        current=existing_context,
        user_id=resolved_user_id,
        session_id=resolved_session_id,
    )

    result: dict[str, Any] = {"context": context}
    if resolved_user_id:
        result["user_id"] = resolved_user_id
    if resolved_session_id:
        result["session_id"] = resolved_session_id
    return result
