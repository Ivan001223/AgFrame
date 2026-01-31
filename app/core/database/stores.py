from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select, update, func

from app.core.database.models import ChatHistory, ChatSession, DocContent, Document, UserProfile
from app.core.database.orm import get_session
from app.core.database.conversation_utils import derive_session_title, should_bump_updated_at


class MySQLConversationStore:
    """MySQL 对话存储实现"""
    
    def save_session(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, Any]],
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """保存或更新会话及消息记录"""
        now = int(time.time())
        title = derive_session_title(messages, title)

        with get_session() as session:
            existing = session.execute(
                select(ChatSession).where(ChatSession.session_id == session_id, ChatSession.user_id == user_id)
            ).scalar_one_or_none()
            created_at = int(existing.created_at) if existing else now
            old_len = 0
            if existing:
                old_len = int(
                    session.execute(
                        select(func.count())
                        .select_from(ChatHistory)
                        .where(ChatHistory.session_id == session_id, ChatHistory.user_id == user_id)
                    ).scalar_one()
                )
            # 只有当有新消息时才更新 updated_at
            bump_updated_at = (not existing) or should_bump_updated_at(range(old_len), messages)

            if existing:
                existing.title = title
                if bump_updated_at:
                    existing.updated_at = now
            else:
                session.add(
                    ChatSession(
                        session_id=session_id,
                        user_id=user_id,
                        title=title,
                        created_at=created_at,
                        updated_at=now,
                        last_summarized_msg_id=None,
                        last_profiled_msg_id=None,
                    )
                )

            # 全量替换消息策略（简单但可能低效，生产环境可优化为增量更新）
            session.execute(
                delete(ChatHistory).where(ChatHistory.session_id == session_id, ChatHistory.user_id == user_id)
            )

            if messages:
                rows: List[ChatHistory] = []
                for m in messages:
                    role = str(m.get("role", ""))
                    content = str(m.get("content", ""))
                    created_at_msg = int(m.get("created_at") or now)
                    token_count = m.get("token_count")
                    rows.append(
                        ChatHistory(
                            session_id=session_id,
                            user_id=user_id,
                            role=role,
                            content=content,
                            created_at=created_at_msg,
                            token_count=int(token_count) if token_count is not None else None,
                        )
                    )
                session.add_all(rows)

        return {
            "id": session_id,
            "title": title,
            "created_at": created_at,
            "updated_at": now if bump_updated_at else int(existing.updated_at),
            "messages": messages,
        }

    def list_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """列出用户的所有会话，按更新时间倒序"""
        with get_session() as session:
            sessions = session.execute(
                select(ChatSession).where(ChatSession.user_id == user_id).order_by(ChatSession.updated_at.desc())
            ).scalars().all()
            out: List[Dict[str, Any]] = []
            for s in sessions:
                # 获取会话的所有消息
                msgs = session.execute(
                    select(ChatHistory)
                    .where(ChatHistory.user_id == user_id, ChatHistory.session_id == s.session_id)
                    .order_by(ChatHistory.msg_id.asc())
                ).scalars().all()
                out.append(
                    {
                        "id": s.session_id,
                        "title": s.title,
                        "created_at": int(s.created_at),
                        "updated_at": int(s.updated_at),
                        "messages": [
                            {
                                "role": m.role,
                                "content": m.content,
                                "created_at": int(m.created_at),
                                "token_count": m.token_count,
                            }
                            for m in msgs
                        ],
                    }
                )
            return out

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定会话"""
        with get_session() as session:
            session.execute(
                delete(ChatSession).where(ChatSession.user_id == user_id, ChatSession.session_id == session_id)
            )
        return True

    def get_recent_messages(
        self, user_id: str, session_id: str, limit_messages: int
    ) -> List[Dict[str, Any]]:
        """获取指定会话的最近 N 条消息"""
        if limit_messages <= 0:
            return []
        with get_session() as session:
            msgs = session.execute(
                select(ChatHistory)
                .where(ChatHistory.user_id == user_id, ChatHistory.session_id == session_id)
                .order_by(ChatHistory.msg_id.desc())
                .limit(limit_messages)
            ).scalars().all()
            msgs = list(reversed(msgs))
            return [
                {"role": m.role, "content": m.content, "created_at": int(m.created_at), "token_count": m.token_count}
                for m in msgs
            ]

    def get_session_meta(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话元数据（不含消息内容）"""
        with get_session() as session:
            s = session.execute(
                select(ChatSession).where(ChatSession.user_id == user_id, ChatSession.session_id == session_id)
            ).scalar_one_or_none()
            if not s:
                return None
            return {
                "id": s.session_id,
                "title": s.title,
                "created_at": int(s.created_at),
                "updated_at": int(s.updated_at),
                "last_summarized_msg_id": s.last_summarized_msg_id,
                "last_profiled_msg_id": s.last_profiled_msg_id,
            }

    def update_session_markers(
        self,
        user_id: str,
        session_id: str,
        last_summarized_msg_id: Optional[int] = None,
        last_profiled_msg_id: Optional[int] = None,
    ) -> None:
        """更新会话的处理进度标记"""
        values: Dict[str, Any] = {}
        if last_summarized_msg_id is not None:
            values["last_summarized_msg_id"] = int(last_summarized_msg_id)
        if last_profiled_msg_id is not None:
            values["last_profiled_msg_id"] = int(last_profiled_msg_id)
        if not values:
            return
        with get_session() as session:
            session.execute(
                update(ChatSession)
                .where(ChatSession.user_id == user_id, ChatSession.session_id == session_id)
                .values(**values)
            )

    def get_messages_after(
        self, user_id: str, session_id: str, after_msg_id: int, limit_messages: int
    ) -> List[Dict[str, Any]]:
        """获取指定 msg_id 之后的消息（用于增量处理）"""
        if limit_messages <= 0:
            return []
        with get_session() as session:
            msgs = session.execute(
                select(ChatHistory)
                .where(
                    ChatHistory.user_id == user_id,
                    ChatHistory.session_id == session_id,
                    ChatHistory.msg_id > int(after_msg_id),
                )
                .order_by(ChatHistory.msg_id.asc())
                .limit(limit_messages)
            ).scalars().all()
            return [
                {
                    "msg_id": int(m.msg_id),
                    "role": m.role,
                    "content": m.content,
                    "created_at": int(m.created_at),
                    "token_count": m.token_count,
                }
                for m in msgs
            ]


class MySQLProfileStore:
    """MySQL 用户画像存储实现"""
    
    def get_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户画像"""
        with get_session() as session:
            row = session.get(UserProfile, user_id)
            if not row:
                return None
            return {"profile": row.profile_json, "version": int(row.version), "updated_at": int(row.updated_at)}

    def upsert_profile(self, user_id: str, profile: Dict[str, Any], version: int) -> None:
        """更新或插入用户画像"""
        now = int(time.time())
        with get_session() as session:
            row = session.get(UserProfile, user_id)
            if row:
                row.profile_json = profile
                row.version = int(version)
                row.updated_at = now
            else:
                session.add(UserProfile(user_id=user_id, profile_json=profile, version=int(version), updated_at=now))


class MySQLDocStore:
    """MySQL 文档存储实现 (Parent Retrieval)"""
    
    def upsert_document(
        self, source_path: str, created_at: Optional[int] = None, user_id: Optional[str] = None, checksum: Optional[str] = None
    ) -> int:
        """记录上传的文档元数据"""
        created_at_val = int(created_at or time.time())
        with get_session() as session:
            existing = session.execute(select(Document).where(Document.source_path == source_path)).scalar_one_or_none()
            if existing:
                existing.user_id = user_id
                existing.checksum = checksum
                session.flush()
                return int(existing.doc_id)
            doc = Document(user_id=user_id, source_path=source_path, checksum=checksum, created_at=created_at_val)
            session.add(doc)
            session.flush()
            return int(doc.doc_id)

    def insert_parent_chunks(self, doc_id: int, chunks: List[Dict[str, Any]]) -> List[int]:
        """插入父文档切片"""
        if not chunks:
            return []
        now = int(time.time())
        with get_session() as session:
            rows: List[DocContent] = []
            for c in chunks:
                content = str(c.get("content", ""))
                page_num = c.get("page_num")
                rows.append(DocContent(doc_id=int(doc_id), content=content, page_num=page_num, created_at=now))
            session.add_all(rows)
            session.flush()
            return [int(r.parent_chunk_id) for r in rows]

    def fetch_parent_chunks(self, parent_chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """根据 ID 列表批量获取父文档切片"""
        if not parent_chunk_ids:
            return []
        with get_session() as session:
            rows = session.execute(
                select(DocContent).where(DocContent.parent_chunk_id.in_([int(x) for x in parent_chunk_ids]))
            ).scalars().all()
            row_by_id = {int(r.parent_chunk_id): r for r in rows}
            out: List[Dict[str, Any]] = []
            for i in parent_chunk_ids:
                r = row_by_id.get(int(i))
                if not r:
                    continue
                out.append(
                    {
                        "parent_chunk_id": int(r.parent_chunk_id),
                        "doc_id": int(r.doc_id),
                        "content": r.content,
                        "page_num": r.page_num,
                    }
                )
            return out
