from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks

from app.memory.session_history import persist_session_messages as _persist_session_messages
from app.memory.session_history import update_memory_after_save


def persist_session_messages(
    *,
    user_id: str,
    session_id: str,
    messages: list[dict[str, Any]],
    background_tasks: BackgroundTasks | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    saved = _persist_session_messages(
        user_id=user_id,
        session_id=session_id,
        messages=messages,
        title=title,
    )
    if background_tasks is not None:
        background_tasks.add_task(update_memory_after_save, user_id, session_id, messages)
    return saved
