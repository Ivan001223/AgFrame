import json
import os
import time
from typing import Any, cast

from app.infrastructure.database.conversation_utils import (
    derive_session_title,
    should_bump_updated_at,
)

HISTORY_FILE = os.path.join("data", "chat_history.json")


class HistoryManager:
    """JSON-backed fallback conversation history store."""

    def __init__(self) -> None:
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "w", encoding="utf-8") as handle:
                json.dump({}, handle)

    def _load_data(self) -> dict[str, Any]:
        try:
            with open(HISTORY_FILE, encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
        if isinstance(loaded, dict):
            return cast(dict[str, Any], loaded)
        return {}

    def _save_data(self, data: dict[str, Any]) -> None:
        with open(HISTORY_FILE, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def get_history(self, user_id: str) -> list[dict[str, Any]]:
        data = self._load_data()
        user_sessions = data.get(user_id, {})
        if not isinstance(user_sessions, dict):
            return []
        sessions_list = [session for session in user_sessions.values() if isinstance(session, dict)]
        sessions_list.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
        return [cast(dict[str, Any], session) for session in sessions_list]

    def get_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        data = self._load_data()
        user_sessions = data.get(user_id, {})
        if not isinstance(user_sessions, dict):
            return None
        session = user_sessions.get(session_id)
        return cast(dict[str, Any] | None, session if isinstance(session, dict) else None)

    def search_history(self, user_id: str, query: str) -> list[dict[str, Any]]:
        normalized_query = str(query or "").strip().lower()
        sessions = self.get_history(user_id)
        if not normalized_query:
            return sessions

        results: list[dict[str, Any]] = []
        for session in sessions:
            title = str(session.get("title") or "").lower()
            messages = session.get("messages") or []
            if normalized_query in title or any(
                normalized_query in str(message.get("content") or "").lower()
                for message in messages
                if isinstance(message, dict)
            ):
                results.append(session)
        return results

    def save_session(
        self,
        user_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
        title: str | None = None,
    ) -> dict[str, Any]:
        data = self._load_data()
        user_sessions = data.setdefault(user_id, {})
        if not isinstance(user_sessions, dict):
            user_sessions = {}
            data[user_id] = user_sessions

        now = int(time.time())
        existing = user_sessions.get(session_id)
        if isinstance(existing, dict):
            old_messages = existing.get("messages", [])
            existing["messages"] = messages
            existing["title"] = derive_session_title(messages, title)
            if should_bump_updated_at(old_messages, messages):
                existing["updated_at"] = now
            saved = existing
        else:
            saved = {
                "id": session_id,
                "title": derive_session_title(messages, title),
                "created_at": now,
                "updated_at": now,
                "messages": messages,
            }
            user_sessions[session_id] = saved

        self._save_data(data)
        return cast(dict[str, Any], saved)

    def delete_session(self, user_id: str, session_id: str) -> bool:
        data = self._load_data()
        user_sessions = data.get(user_id, {})
        if isinstance(user_sessions, dict) and session_id in user_sessions:
            del user_sessions[session_id]
            self._save_data(data)
            return True
        return False

    def rename_session(self, user_id: str, session_id: str, title: str) -> dict[str, Any] | None:
        data = self._load_data()
        user_sessions = data.get(user_id, {})
        if not isinstance(user_sessions, dict):
            return None
        session = user_sessions.get(session_id)
        if not isinstance(session, dict):
            return None
        session["title"] = str(title).strip() or str(session.get("title") or "") or "New conversation"
        self._save_data(data)
        return cast(dict[str, Any], session)


history_manager = HistoryManager()
