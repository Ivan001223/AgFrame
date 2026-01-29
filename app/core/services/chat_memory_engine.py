from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.llm.embeddings import ModelEmbeddings
from app.core.llm.llm_factory import get_llm
from app.core.llm.reranker import ModelReranker
from app.core.utils.faiss_store import load_faiss, save_faiss


CHAT_SUMMARY_STORE_BASE = os.path.join("data", "vector_store_chat_summary")


def _format_chat_for_summary(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if not role or content is None:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def summarize_chat_messages(messages: List[Dict[str, Any]]) -> str:
    llm = get_llm(temperature=0, streaming=False)
    chat_log = _format_chat_for_summary(messages)
    prompt = (
        "你是长期记忆摘要器。只提炼对未来有价值的信息，忽略寒暄与即时情绪。\n"
        "输出要求：中文，尽量信息密度高，最多 8 条要点，每条一句话。\n"
        "优先级：事实与偏好 > 约束条件 > 当前目标 > 已解决/未解决的问题。\n"
        "不要复述无意义内容（如“你好/谢谢”）。\n\n"
        "<chat_log>\n"
        f"{chat_log}\n"
        "</chat_log>"
    )
    return str(llm.invoke(prompt).content).strip()


class ChatSummaryIndex:
    def __init__(self, base_dir: str = CHAT_SUMMARY_STORE_BASE):
        self.base_dir = base_dir
        self.embeddings = ModelEmbeddings()
        self.reranker = ModelReranker()
        self._stores: Dict[str, Optional[FAISS]] = {}

    def _user_dir(self, user_id: str) -> str:
        return os.path.join(self.base_dir, user_id)

    def _load(self, user_id: str) -> Optional[FAISS]:
        if user_id in self._stores:
            return self._stores[user_id]
        user_dir = self._user_dir(user_id)
        store = load_faiss(user_dir, self.embeddings, allow_dangerous_deserialization=True)
        self._stores[user_id] = store
        return store

    def _persist(self, user_id: str) -> None:
        user_dir = self._user_dir(user_id)
        save_faiss(user_dir, self._stores.get(user_id))

    def add_summary(
        self,
        user_id: str,
        session_id: str,
        summary_text: str,
        start_msg_id: Optional[int] = None,
        end_msg_id: Optional[int] = None,
        created_at: Optional[int] = None,
    ) -> None:
        created_at_val = int(created_at or time.time())
        doc = Document(
            page_content=summary_text,
            metadata={
                "type": "chat_summary",
                "user_id": user_id,
                "session_id": session_id,
                "start_msg_id": start_msg_id,
                "end_msg_id": end_msg_id,
                "created_at": created_at_val,
            },
        )
        store = self._load(user_id)
        if store is None:
            self._stores[user_id] = FAISS.from_documents([doc], self.embeddings)
        else:
            store.add_documents([doc])
        self._persist(user_id)

    def retrieve(
        self, user_id: str, query: str, k: int = 3, fetch_k: int = 20
    ) -> List[Document]:
        store = self._load(user_id)
        if store is None:
            return []
        candidates = store.similarity_search(query, k=fetch_k)
        if not candidates:
            return []
        candidate_texts = [d.page_content for d in candidates]
        reranked = self.reranker.rerank(query, candidate_texts, top_k=min(k, len(candidates)))
        out: List[Document] = []
        for _, score, idx in reranked:
            d = candidates[idx]
            d.metadata["rerank_score"] = score
            out.append(d)
        return out


def select_recent_turn_messages(messages: List[Dict[str, Any]], recent_turns: int) -> List[Dict[str, Any]]:
    if recent_turns <= 0:
        return []
    limit = recent_turns * 2
    return messages[-limit:] if len(messages) > limit else messages


def split_messages_for_memory(messages: List[Dict[str, Any]], recent_turns: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    recent = select_recent_turn_messages(messages, recent_turns=recent_turns)
    older = messages[: max(0, len(messages) - len(recent))]
    return older, recent
