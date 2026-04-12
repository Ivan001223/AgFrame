from __future__ import annotations

import uuid
from typing import Any

from app.infrastructure.observability import get_langfuse_callback
from app.runtime.graph.chat_graph_app import get_chat_graph_app as get_chat_graph_app


def apply_request_runtime_config(
    config: dict[str, Any] | None,
    request: Any,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Normalize per-request runtime config for graph execution.

    This keeps LangServe and custom chat endpoints aligned on thread/user identity
    and observability callbacks.
    """
    normalized = dict(config or {})
    normalized.setdefault("configurable", {})
    configurable = normalized["configurable"]
    if not isinstance(configurable, dict):
        configurable = {}
        normalized["configurable"] = configurable

    if "thread_id" not in configurable:
        configurable["thread_id"] = str(uuid.uuid4())

    resolved_user_id = user_id
    if not resolved_user_id:
        request_user = getattr(getattr(request, "state", None), "user", None)
        resolved_user_id = getattr(request_user, "username", None)
    if resolved_user_id:
        configurable["user_id"] = resolved_user_id

    handler = get_langfuse_callback()
    if handler:
        existing_callbacks = normalized.get("callbacks", [])
        if isinstance(existing_callbacks, list):
            normalized["callbacks"] = existing_callbacks + [handler]
        else:
            normalized["callbacks"] = [handler]

    return normalized
