from __future__ import annotations

import time
from typing import Any, Dict, List

from app.core.database.schema import ensure_schema_if_possible
from app.core.database.stores import MySQLConversationStore
from app.core.services.chat_memory_engine import ChatSummaryIndex, split_messages_for_memory, summarize_chat_messages
from app.core.services.profile_engine import UserProfileEngine, incremental_update_profile, extract_base_profile


class MemoryUpdateService:
    def update_after_save(self, user_id: str, session_id: str, messages: List[Dict[str, Any]]) -> None:
        if not ensure_schema_if_possible():
            return

        store = MySQLConversationStore()
        try:
            meta = store.get_session_meta(user_id, session_id) or {}
            last_summarized = int(meta.get("last_summarized_msg_id") or 0)
            last_profiled = int(meta.get("last_profiled_msg_id") or 0)
        except Exception:
            last_summarized = 0
            last_profiled = 0

        recent_turns = 5
        older, _ = split_messages_for_memory(messages, recent_turns=recent_turns)
        older_end = len(older)

        if older_end > last_summarized and older_end - last_summarized >= 6:
            segment = older[last_summarized:older_end]
            summary_index = ChatSummaryIndex()
            summary_text = summarize_chat_messages(segment)
            summary_index.add_summary(
                user_id=user_id,
                session_id=session_id,
                summary_text=summary_text,
                start_msg_id=last_summarized,
                end_msg_id=older_end - 1,
            )
            store.update_session_markers(user_id, session_id, last_summarized_msg_id=older_end)

        if len(messages) - last_profiled >= 20:
            profile_engine = UserProfileEngine()
            profile = profile_engine.get_profile(user_id)
            chat_log = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in messages[last_profiled:]])
            if profile and any(profile.values()):
                new_profile = incremental_update_profile(profile, chat_log=chat_log)
                version = int(time.time())
                profile_engine.upsert_profile(user_id, new_profile, version=version)
            else:
                full_log = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in messages])
                base_profile = extract_base_profile(full_log)
                version = int(time.time())
                profile_engine.upsert_profile(user_id, base_profile, version=version)
            store.update_session_markers(user_id, session_id, last_profiled_msg_id=len(messages))


memory_update_service = MemoryUpdateService()

