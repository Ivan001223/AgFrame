from __future__ import annotations

from typing import Any

from app.runtime.graph.nodes.human_interrupt import check_approval_node
from app.server.chat_runtime import get_chat_graph_app


def _normalize_message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item.get("text") or ""))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text") or ""))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict) and "text" in value:
        return str(value.get("text") or "")
    if hasattr(value, "text"):
        return str(getattr(value, "text") or "")
    return str(value or "")


def _normalize_message_role(message: Any) -> str:
    role = getattr(message, "type", None) or getattr(message, "role", None)
    if isinstance(message, dict):
        role = message.get("type") or message.get("role") or role
    role_text = str(role or "").lower()
    if role_text in {"human", "user"}:
        return "user"
    if role_text in {"ai", "assistant"}:
        return "assistant"
    return "system"


def serialize_graph_messages(messages: Any) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    if not isinstance(messages, list):
        return serialized
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(message, dict):
            content = message.get("content")
        serialized.append(
            {
                "role": _normalize_message_role(message),
                "content": _normalize_message_content(content),
            }
        )
    return serialized


def extract_last_assistant_reply(messages: list[dict[str, str]]) -> str | None:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = str(message.get("content") or "").strip()
            if content:
                return content
    return None


class GraphResumeService:
    def __init__(self, graph_app: Any | None = None):
        self.graph_app = graph_app or get_chat_graph_app()

    async def resume_approved_session(
        self,
        *,
        session_id: str,
        checkpoint: dict[str, Any],
    ) -> dict[str, object]:
        checkpoint_data = dict(checkpoint.get("checkpoint") or {})
        action_required = dict(checkpoint_data.get("action_required") or {})
        if not action_required:
            return {
                "ok": False,
                "session_id": session_id,
                "interrupted": checkpoint_data.get("interrupted"),
                "error_code": "action_required_missing",
                "error_message": "no action_required payload found in checkpoint",
            }
        if not bool(action_required.get("approved")):
            return {
                "ok": False,
                "session_id": session_id,
                "interrupted": checkpoint_data.get("interrupted"),
                "error_code": "approval_not_granted",
                "error_message": "checkpoint action has not been approved",
            }

        configurable: dict[str, object] = {
            "thread_id": session_id,
            "checkpoint_ns": str(checkpoint_data.get("checkpoint_ns") or ""),
        }
        checkpoint_id = checkpoint_data.get("checkpoint_id") or checkpoint_data.get("id")
        if checkpoint_id:
            configurable["checkpoint_id"] = str(checkpoint_id)
        config = {"configurable": configurable}

        state_snapshot = await self.graph_app.aget_state(config)
        state_values = dict(state_snapshot.values or {})
        resume_values = await check_approval_node(state_values)
        updated_config = await self.graph_app.aupdate_state(
            state_snapshot.config,
            resume_values,
            as_node="check_approval",
        )
        resume_result = await self.graph_app.ainvoke(None, updated_config)
        latest_state = await self.graph_app.aget_state(
            {
                "configurable": {
                    "thread_id": session_id,
                    "checkpoint_ns": str(configurable.get("checkpoint_ns") or ""),
                }
            }
        )
        latest_values = dict(latest_state.values or {})
        serialized_messages = serialize_graph_messages(latest_values.get("messages"))
        interrupted = latest_values.get("interrupted")
        ok = interrupted is False
        return {
            "ok": ok,
            "session_id": session_id,
            "interrupted": bool(interrupted) if interrupted is not None else None,
            "error_code": None if ok else "resume_still_interrupted",
            "error_message": None if ok else "graph resumed but remained interrupted",
            "result": resume_result if isinstance(resume_result, dict) else None,
            "messages": serialized_messages,
            "reply": extract_last_assistant_reply(serialized_messages),
            "context": latest_values.get("context") if isinstance(latest_values.get("context"), dict) else None,
        }
