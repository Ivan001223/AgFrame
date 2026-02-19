from __future__ import annotations

import hashlib
import time
from typing import Any

from langchain_core.documents import Document

from app.infrastructure.database.stores import PgUserMemoryStore
from app.runtime.llm.embeddings import ModelEmbeddings
from app.runtime.llm.reranker import ModelReranker


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class UserMemoryEngine:
    def __init__(self):
        self.store = PgUserMemoryStore()
        self.embeddings = ModelEmbeddings()
        self.reranker = ModelReranker()

    def add_chat_summary(
        self,
        *,
        user_id: str,
        session_id: str,
        summary_text: str,
        start_msg_id: int | None = None,
        end_msg_id: int | None = None,
        created_at: int | None = None,
    ) -> None:
        uid = str(user_id or "").strip()
        sid = str(session_id or "").strip()
        text = str(summary_text or "").strip()
        if not uid or not sid or not text:
            return
        now = int(time.time())
        created_at_val = int(created_at or now)
        item_hash = _sha256_hex(f"chat_summary|{uid}|{sid}|{start_msg_id}|{end_msg_id}|{text}")
        embedding = self.embeddings.embed_documents([text])[0]
        self.store.upsert_items(
            [
                {
                    "user_id": uid,
                    "kind": "episodic",
                    "subkind": "chat_summary",
                    "session_id": sid,
                    "text": text,
                    "item_hash": item_hash,
                    "confidence_score": None,
                    "last_verified_at": created_at_val,
                    "metadata_json": {
                        "type": "chat_summary",
                        "session_id": sid,
                        "start_msg_id": start_msg_id,
                        "end_msg_id": end_msg_id,
                        "created_at": created_at_val,
                    },
                    "embedding": embedding,
                }
            ]
        )

    def retrieve_chat_summaries(
        self,
        *,
        user_id: str,
        query: str,
        k: int = 3,
        fetch_k: int = 20,
    ) -> list[Document]:
        uid = str(user_id or "").strip()
        q = str(query or "").strip()
        if not uid or not q:
            return []
        query_vec = self.embeddings.embed_query(q)
        candidates = self.store.dense_search(
            query_vec,
            user_id=uid,
            kind="episodic",
            subkind="chat_summary",
            k=max(int(fetch_k), int(k)),
        )
        if not candidates:
            return []
        texts = [str(c.get("text") or "") for c in candidates]
        reranked = self.reranker.rerank(q, texts, top_k=min(int(k), len(texts)))
        out: list[Document] = []
        for _, score, idx in reranked:
            c = candidates[idx]
            meta = dict(c.get("metadata_json") or {})
            meta["rerank_score"] = score
            out.append(Document(page_content=str(c.get("text") or ""), metadata=meta))
        return out

    def replace_profile_semantic_memory(self, *, user_id: str, profile: dict[str, Any]) -> int:
        uid = str(user_id or "").strip()
        if not uid:
            return 0
        items = self._profile_items(profile)
        if not items:
            self.store.delete_by_user(uid, kind="semantic", subkind="profile_preference")
            self.store.delete_by_user(uid, kind="semantic", subkind="profile_fact")
            return 0
        texts = [it["text"] for it in items]
        embeddings = self.embeddings.embed_documents(texts)
        now = int(time.time())
        rows: list[dict[str, Any]] = []
        for it, emb in zip(items, embeddings):
            rows.append(
                {
                    "user_id": uid,
                    "kind": "semantic",
                    "subkind": it.get("subkind"),
                    "session_id": None,
                    "text": it.get("text"),
                    "item_hash": it.get("item_hash"),
                    "confidence_score": it.get("confidence_score"),
                    "last_verified_at": it.get("last_verified_at") or now,
                    "metadata_json": it.get("metadata_json"),
                    "embedding": emb,
                }
            )
        self.store.delete_by_user(uid, kind="semantic", subkind="profile_preference")
        self.store.delete_by_user(uid, kind="semantic", subkind="profile_fact")
        return self.store.upsert_items(rows)

    def retrieve_profile_items(
        self,
        *,
        user_id: str,
        query: str,
        k: int = 6,
        fetch_k: int = 30,
    ) -> list[dict[str, Any]]:
        uid = str(user_id or "").strip()
        q = str(query or "").strip()
        if not uid or not q:
            return []
        query_vec = self.embeddings.embed_query(q)
        candidates = self.store.dense_search(
            query_vec,
            user_id=uid,
            kind="semantic",
            k=max(int(fetch_k), int(k)),
        )
        if not candidates:
            return []
        texts = [str(c.get("text") or "") for c in candidates]
        reranked = self.reranker.rerank(q, texts, top_k=min(int(k), len(texts)))
        out: list[dict[str, Any]] = []
        for _, score, idx in reranked:
            c = dict(candidates[idx])
            meta = dict(c.get("metadata_json") or {})
            meta["rerank_score"] = score
            c["metadata_json"] = meta
            c["rerank_score"] = score
            out.append(c)
        return out

    def _profile_items(self, profile: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(profile, dict):
            return []
        out: list[dict[str, Any]] = []
        prefs = profile.get("preferences") or {}
        if isinstance(prefs, dict):
            for key in ["language", "communication_style", "interaction_protocol", "tone_instruction"]:
                val = prefs.get(key)
                if val is None:
                    continue
                text = f"偏好：{key}={str(val).strip()}"
                item_hash = _sha256_hex(f"profile_preference|{key}|{text}")
                out.append(
                    {
                        "subkind": "profile_preference",
                        "text": text,
                        "item_hash": item_hash,
                        "confidence_score": 0.9,
                        "last_verified_at": None,
                        "metadata_json": {"type": "profile_preference", "key": key},
                    }
                )
        facts = profile.get("facts") or []
        if isinstance(facts, list):
            for f in facts:
                if isinstance(f, str):
                    txt = f.strip()
                    conf = 0.6
                    last = None
                elif isinstance(f, dict):
                    txt = str(f.get("text") or "").strip()
                    conf = f.get("confidence_score")
                    last = f.get("last_verified_at")
                else:
                    continue
                if not txt:
                    continue
                text = f"事实/偏好：{txt}"
                item_hash = _sha256_hex(f"profile_fact|{text}")
                out.append(
                    {
                        "subkind": "profile_fact",
                        "text": text,
                        "item_hash": item_hash,
                        "confidence_score": float(conf) if conf is not None else 0.6,
                        "last_verified_at": int(last) if last is not None else None,
                        "metadata_json": {"type": "profile_fact"},
                    }
                )
        return out
