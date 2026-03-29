from __future__ import annotations

from typing import Any

from fastapi import BackgroundTasks

from app.infrastructure.database.history_manager import history_manager
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.stores import MySQLConversationStore


def update_memory_after_save(user_id: str, session_id: str, messages: list[dict[str, Any]]) -> None:
    from app.memory.long_term.memory_update_service import memory_update_service

    memory_update_service.update_after_save(user_id, session_id, messages)


def persist_session_messages(
    *,
    user_id: str,
    session_id: str,
    messages: list[dict[str, Any]],
    background_tasks: BackgroundTasks | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    if ensure_schema_if_possible():
        saved = MySQLConversationStore().save_session(user_id, session_id, messages, title)
        if background_tasks is not None:
            background_tasks.add_task(update_memory_after_save, user_id, session_id, messages)
        return saved
    return history_manager.save_session(user_id, session_id, messages, title)
