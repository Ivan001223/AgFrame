from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from sqlalchemy import bindparam, delete, select, update, func, cast, Float
from pgvector.sqlalchemy import Vector

from app.infrastructure.database.models import (
    ChatHistory,
    ChatSession,
    DocContent,
    DocEmbedding,
    Document,
    UserProfile,
    UserMemoryEmbedding,
    UserMemoryItem,
)
from app.infrastructure.database.orm import get_session
from app.infrastructure.database.conversation_utils import derive_session_title, should_bump_updated_at


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


class PgDocEmbeddingStore:
    def delete_by_doc_id(self, doc_id: int) -> int:
        with get_session() as session:
            res = session.execute(delete(DocEmbedding).where(DocEmbedding.doc_id == int(doc_id)))
            return int(res.rowcount or 0)

    def add_embeddings(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        now = int(time.time())
        to_add: List[DocEmbedding] = []
        for r in rows:
            to_add.append(
                DocEmbedding(
                    doc_id=r.get("doc_id"),
                    parent_chunk_id=r.get("parent_chunk_id"),
                    child_index=r.get("child_index"),
                    source_path=r.get("source_path"),
                    content=str(r.get("content") or ""),
                    embedding=list(r.get("embedding") or []),
                    metadata_json=r.get("metadata_json"),
                    created_at=int(r.get("created_at") or now),
                )
            )
        with get_session() as session:
            session.add_all(to_add)
        return len(to_add)

    def dense_search(
        self, query_vec: List[float], *, k: int, filter: Optional[Dict[str, Any]] = None
    ) -> List[DocEmbedding]:
        if not query_vec or k <= 0:
            return []
        q = bindparam("query_vec", value=list(query_vec), type_=Vector)
        distance = cast(DocEmbedding.embedding.op("<=>")(q), Float)
        stmt = select(DocEmbedding).order_by(distance).limit(int(k))

        if filter:
            allowed_keys = {"user_id", "doc_id", "source", "type"}
            for key, value in filter.items():
                if key not in allowed_keys:
                    continue
                stmt = stmt.where(func.json_extract(DocEmbedding.metadata_json, f"$.{key}") == value)

        with get_session() as session:
            return list(session.execute(stmt).scalars().all())

    def sparse_search(
        self, query: str, *, k: int, filter: Optional[Dict[str, Any]] = None
    ) -> List[DocEmbedding]:
        q = str(query or "").strip()
        if not q or k <= 0:
            return []
        tsq = func.plainto_tsquery("simple", bindparam("q", value=q))
        tsv = func.to_tsvector("simple", DocEmbedding.content)
        stmt = (
            select(DocEmbedding)
            .where(tsv.op("@@")(tsq))
            .order_by(func.ts_rank_cd(tsv, tsq).desc())
            .limit(int(k))
        )
        
        if filter:
            allowed_keys = {"user_id", "doc_id", "source", "type"}
            for key, value in filter.items():
                if key not in allowed_keys:
                    continue
                stmt = stmt.where(func.json_extract(DocEmbedding.metadata_json, f"$.{key}") == value)

        with get_session() as session:
            return list(session.execute(stmt).scalars().all())


class PgUserMemoryStore:
    def upsert_items(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        now = int(time.time())
        count = 0
        with get_session() as session:
            grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
            for r in rows:
                user_id = str(r.get("user_id") or "")
                kind = str(r.get("kind") or "")
                if not user_id or not kind:
                    continue
                grouped.setdefault((user_id, kind), []).append(r)

            existing_by_key: Dict[tuple[str, str, str], UserMemoryItem] = {}
            for (user_id, kind), g in grouped.items():
                hashes = [str(x.get("item_hash") or "") for x in g if str(x.get("item_hash") or "")]
                if not hashes:
                    continue
                found = session.execute(
                    select(UserMemoryItem).where(
                        UserMemoryItem.user_id == user_id,
                        UserMemoryItem.kind == kind,
                        UserMemoryItem.item_hash.in_(hashes),
                    )
                ).scalars().all()
                for it in found:
                    if it.item_hash:
                        existing_by_key[(str(it.user_id), str(it.kind), str(it.item_hash))] = it

            for r in rows:
                user_id = str(r.get("user_id") or "")
                kind = str(r.get("kind") or "")
                item_hash = str(r.get("item_hash") or "")
                if not user_id or not kind or not item_hash:
                    continue
                key = (user_id, kind, item_hash)
                subkind = r.get("subkind")
                session_id = r.get("session_id")
                text = str(r.get("text") or "")
                embedding = list(r.get("embedding") or [])
                metadata_json = r.get("metadata_json")
                confidence_score = r.get("confidence_score")
                last_verified_at = r.get("last_verified_at")

                it = existing_by_key.get(key)
                if it is None:
                    it = UserMemoryItem(
                        user_id=user_id,
                        kind=kind,
                        subkind=str(subkind) if subkind is not None else None,
                        session_id=str(session_id) if session_id is not None else None,
                        text=text,
                        item_hash=item_hash,
                        confidence_score=float(confidence_score) if confidence_score is not None else None,
                        last_verified_at=int(last_verified_at) if last_verified_at is not None else None,
                        created_at=now,
                        updated_at=now,
                        metadata_json=metadata_json,
                    )
                    session.add(it)
                    session.flush()
                    session.add(UserMemoryEmbedding(item_id=int(it.item_id), embedding=embedding))
                else:
                    it.subkind = str(subkind) if subkind is not None else it.subkind
                    it.session_id = str(session_id) if session_id is not None else it.session_id
                    it.text = text or it.text
                    it.confidence_score = float(confidence_score) if confidence_score is not None else it.confidence_score
                    it.last_verified_at = int(last_verified_at) if last_verified_at is not None else it.last_verified_at
                    it.updated_at = now
                    it.metadata_json = metadata_json if metadata_json is not None else it.metadata_json
                    emb = session.get(UserMemoryEmbedding, int(it.item_id))
                    if emb is None:
                        session.add(UserMemoryEmbedding(item_id=int(it.item_id), embedding=embedding))
                    else:
                        emb.embedding = embedding
                count += 1
        return count

    def delete_by_user(self, user_id: str, *, kind: Optional[str] = None, subkind: Optional[str] = None) -> int:
        uid = str(user_id or "").strip()
        if not uid:
            return 0
        with get_session() as session:
            stmt = delete(UserMemoryItem).where(UserMemoryItem.user_id == uid)
            if kind:
                stmt = stmt.where(UserMemoryItem.kind == str(kind))
            if subkind:
                stmt = stmt.where(UserMemoryItem.subkind == str(subkind))
            res = session.execute(stmt)
            return int(res.rowcount or 0)

    def dense_search(
        self,
        query_vec: List[float],
        *,
        user_id: str,
        kind: str,
        k: int,
        subkind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        uid = str(user_id or "").strip()
        knd = str(kind or "").strip()
        if not uid or not knd or not query_vec or k <= 0:
            return []
        q = bindparam("query_vec", value=list(query_vec), type_=Vector)
        distance = cast(UserMemoryEmbedding.embedding.op("<=>")(q), Float)
        stmt = (
            select(UserMemoryItem, distance.label("distance"))
            .join(UserMemoryEmbedding, UserMemoryEmbedding.item_id == UserMemoryItem.item_id)
            .where(UserMemoryItem.user_id == uid, UserMemoryItem.kind == knd)
        )
        if subkind:
            stmt = stmt.where(UserMemoryItem.subkind == str(subkind))
        stmt = stmt.order_by(distance).limit(int(k))
        with get_session() as session:
            rows = session.execute(stmt).all()
            out: List[Dict[str, Any]] = []
            for it, dist in rows:
                out.append(
                    {
                        "item_id": int(it.item_id),
                        "user_id": str(it.user_id),
                        "kind": str(it.kind),
                        "subkind": it.subkind,
                        "session_id": it.session_id,
                        "text": it.text,
                        "confidence_score": it.confidence_score,
                        "last_verified_at": it.last_verified_at,
                        "created_at": int(it.created_at),
                        "updated_at": int(it.updated_at),
                        "metadata_json": it.metadata_json,
                        "distance": float(dist) if dist is not None else None,
                    }
                )
            return out


class PgChatSummaryStore:
    """聊天摘要向量存储 (pgvector)"""

    def add_summary(
        self,
        user_id: str,
        session_id: str,
        summary_text: str,
        embedding: List[float],
        start_msg_id: Optional[int] = None,
        end_msg_id: Optional[int] = None,
        created_at: Optional[int] = None,
    ) -> int:
        now = int(created_at or time.time())
        with get_session() as session:
            item = UserMemoryItem(
                user_id=user_id,
                kind="chat_summary",
                session_id=session_id,
                text=summary_text,
                item_hash=None,
                created_at=now,
                updated_at=now,
                metadata_json={
                    "start_msg_id": start_msg_id,
                    "end_msg_id": end_msg_id,
                },
            )
            session.add(item)
            session.flush()
            session.add(UserMemoryEmbedding(item_id=int(item.item_id), embedding=embedding))
            return int(item.item_id)

    def search(
        self,
        user_id: str,
        query_vec: List[float],
        k: int = 3,
        filter_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        uid = str(user_id or "").strip()
        if not uid or not query_vec or k <= 0:
            return []
        q = bindparam("query_vec", value=list(query_vec), type_=Vector)
        distance = cast(UserMemoryEmbedding.embedding.op("<=>")(q), Float)
        stmt = (
            select(UserMemoryItem, distance.label("distance"))
            .join(UserMemoryEmbedding, UserMemoryEmbedding.item_id == UserMemoryItem.item_id)
            .where(UserMemoryItem.user_id == uid, UserMemoryItem.kind == "chat_summary")
        )
        if filter_session_id:
            stmt = stmt.where(UserMemoryItem.session_id == filter_session_id)
        stmt = stmt.order_by(distance).limit(int(k))
        with get_session() as session:
            rows = session.execute(stmt).all()
            out: List[Dict[str, Any]] = []
            for it, dist in rows:
                out.append(
                    {
                        "item_id": int(it.item_id),
                        "user_id": str(it.user_id),
                        "session_id": it.session_id,
                        "text": it.text,
                        "start_msg_id": it.metadata_json.get("start_msg_id") if it.metadata_json else None,
                        "end_msg_id": it.metadata_json.get("end_msg_id") if it.metadata_json else None,
                        "created_at": int(it.created_at),
                        "distance": float(dist) if dist is not None else None,
                    }
                )
            return out

    def delete_by_session(self, user_id: str, session_id: str) -> int:
        uid = str(user_id or "").strip()
        sid = str(session_id or "").strip()
        if not uid or not sid:
            return 0
        with get_session() as session:
            res = session.execute(
                delete(UserMemoryItem).where(
                    UserMemoryItem.user_id == uid,
                    UserMemoryItem.session_id == sid,
                    UserMemoryItem.kind == "chat_summary",
                )
            )
            return int(res.rowcount or 0)

    def delete_by_user(self, user_id: str) -> int:
        uid = str(user_id or "").strip()
        if not uid:
            return 0
        with get_session() as session:
            res = session.execute(
                delete(UserMemoryItem).where(
                    UserMemoryItem.user_id == uid,
                    UserMemoryItem.kind == "chat_summary",
                )
            )
            return int(res.rowcount or 0)
