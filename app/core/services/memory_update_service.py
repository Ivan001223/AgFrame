from __future__ import annotations

import time
from typing import Any, Dict, List

from app.core.database.schema import ensure_schema_if_possible
from app.core.database.stores import MySQLConversationStore
from app.core.services.chat_memory_engine import ChatSummaryIndex, split_messages_for_memory, summarize_chat_messages
from app.core.services.profile_engine import UserProfileEngine, incremental_update_profile, extract_base_profile


class MemoryUpdateService:
    """
    记忆更新服务。
    负责在对话结束后，异步更新长期记忆（包括对话摘要和用户画像）。
    """
    def update_after_save(self, user_id: str, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        在消息保存后触发的更新逻辑。
        
        策略：
        1. 检查距离上次摘要更新的消息数，如果超过阈值（6条），则生成新摘要。
        2. 检查距离上次画像更新的消息数，如果超过阈值（20条），则增量更新用户画像。
        
        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            messages: 当前会话的所有消息列表
        """
        if not ensure_schema_if_possible():
            return

        store = MySQLConversationStore()
        try:
            # 获取会话元数据，了解上次更新的位置
            meta = store.get_session_meta(user_id, session_id) or {}
            last_summarized = int(meta.get("last_summarized_msg_id") or 0)
            last_profiled = int(meta.get("last_profiled_msg_id") or 0)
        except Exception:
            last_summarized = 0
            last_profiled = 0

        # 切分消息：保留最近 5 轮（10条），剩下的视为“旧消息”
        recent_turns = 5
        older, _ = split_messages_for_memory(messages, recent_turns=recent_turns)
        older_end = len(older)

        # 1. 检查是否需要更新摘要
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
            # 更新摘要进度标记
            store.update_session_markers(user_id, session_id, last_summarized_msg_id=older_end)

        # 2. 检查是否需要更新画像
        if len(messages) - last_profiled >= 20:
            profile_engine = UserProfileEngine()
            profile = profile_engine.get_profile(user_id)
            # 提取自上次画像更新以来的新对话
            chat_log = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in messages[last_profiled:]])
            
            if profile and any(profile.values()):
                # 增量更新
                new_profile = incremental_update_profile(profile, chat_log=chat_log)
                version = int(time.time())
                profile_engine.upsert_profile(user_id, new_profile, version=version)
            else:
                # 首次全量提取
                full_log = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in messages])
                base_profile = extract_base_profile(full_log)
                version = int(time.time())
                profile_engine.upsert_profile(user_id, base_profile, version=version)
            # 更新画像进度标记
            store.update_session_markers(user_id, session_id, last_profiled_msg_id=len(messages))


memory_update_service = MemoryUpdateService()

