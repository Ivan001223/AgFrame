from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field

from app.infrastructure.database.models import User
from app.runtime.graph.resume_service import (
    extract_last_assistant_reply,
    serialize_graph_messages,
)
from app.server.api.auth import get_current_active_user
from app.server.chat_runtime import apply_request_runtime_config, get_chat_graph_app
from app.server.session_history import persist_session_messages

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatInvokeResponse(BaseModel):
    session_id: str
    interrupted: bool | None = None
    reply: str | None = None
    messages: list[dict[str, str]] = Field(default_factory=list)
    context: dict[str, Any] | None = None


def _normalize_context(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _normalize_client_messages(value: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(value, list):
        return normalized
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        if role and content:
            normalized.append({"role": role, "content": content})
    return normalized


@router.post("/workbench-invoke", response_model=ChatInvokeResponse)
async def workbench_invoke(
    payload: dict[str, Any],
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    graph_input = dict(payload.get("input") or {})
    graph_input["context"] = _normalize_context(graph_input.get("context"))

    config = apply_request_runtime_config(
        dict(payload.get("config") or {}),
        request,
        user_id=current_user.username,
    )
    configurable = config.get("configurable") or {}
    session_id = str(
        graph_input["context"].get("session_id")
        or configurable.get("thread_id")
        or ""
    ).strip()
    if session_id:
        graph_input["context"]["session_id"] = session_id
        configurable["thread_id"] = session_id

    graph_app = get_chat_graph_app()
    invoke_result = await graph_app.ainvoke(graph_input, config)

    latest_values: dict[str, Any] = {}
    latest_state = await graph_app.aget_state(
        {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": str(configurable.get("checkpoint_ns") or ""),
            }
        }
    )
    if getattr(latest_state, "values", None):
        latest_values = dict(latest_state.values or {})
    elif isinstance(invoke_result, dict):
        latest_values = dict(invoke_result)
    messages = serialize_graph_messages(latest_values.get("messages"))
    if not messages:
        messages = _normalize_client_messages(graph_input.get("messages"))

    persist_session_messages(
        user_id=current_user.username,
        session_id=session_id,
        messages=messages,
        background_tasks=background_tasks,
    )

    context = latest_values.get("context")
    return {
        "session_id": session_id,
        "interrupted": latest_values.get("interrupted"),
        "reply": extract_last_assistant_reply(messages),
        "messages": messages,
        "context": context if isinstance(context, dict) else None,
    }
