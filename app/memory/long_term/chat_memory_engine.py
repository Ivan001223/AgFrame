from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from app.runtime.llm.embeddings import ModelEmbeddings
from app.runtime.llm.llm_factory import get_llm
from app.runtime.llm.reranker import ModelReranker
from app.infrastructure.database.stores import PgChatSummaryStore


def _format_chat_for_summary(messages: List[Dict[str, Any]]) -> str:
    """格式化对话消息为文本，用于生成摘要"""
    lines: List[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if not role or content is None:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def summarize_chat_messages(messages: List[Dict[str, Any]]) -> str:
    """
    使用 LLM 生成对话片段的摘要。

    Args:
        messages: 对话消息列表

    Returns:
        str: 生成的中文摘要
    """
    llm = get_llm(temperature=0, streaming=False)
    chat_log = _format_chat_for_summary(messages)
    prompt = (
        "你是长期记忆摘要器。只提炼对未来有价值的信息，忽略寒暄与即时情绪。"
        "输出要求：中文，尽量信息密度高，最多 8 条要点，每条一句话。"
        "优先级：事实与偏好 > 约束条件 > 当前目标 > 已解决/未解决的问题。"
        "不要复述无意义内容（如你好/谢谢）。"
        "<chat_log>\n"
        f"{chat_log}\n"
        "</chat_log>"
    )
    return str(llm.invoke(prompt).content).strip()


class ChatSummaryIndex:
    """
    聊天摘要索引服务。
    负责管理用户对话历史的摘要，并支持语义检索。
    使用 pgvector 存储向量，支持分布式部署。
    """
    def __init__(self):
        self.embeddings = ModelEmbeddings()
        self.reranker = ModelReranker()
        self._store = PgChatSummaryStore()

    def add_summary(
        self,
        user_id: str,
        session_id: str,
        summary_text: str,
        start_msg_id: Optional[int] = None,
        end_msg_id: Optional[int] = None,
        created_at: Optional[int] = None,
    ) -> int:
        """
        添加新的对话摘要到索引中。

        Args:
            user_id: 用户 ID
            session_id: 会话 ID
            summary_text: 摘要内容
            start_msg_id: 摘要对应的起始消息 ID
            end_msg_id: 摘要对应的结束消息 ID
            created_at: 创建时间戳

        Returns:
            int: 摘要项 ID
        """
        embedding = self.embeddings.embed_query(summary_text)
        return self._store.add_summary(
            user_id=user_id,
            session_id=session_id,
            summary_text=summary_text,
            embedding=embedding,
            start_msg_id=start_msg_id,
            end_msg_id=end_msg_id,
            created_at=created_at,
        )

    def retrieve(
        self, user_id: str, query: str, k: int = 3, fetch_k: int = 20
    ) -> List[Document]:
        """
        检索相关的历史对话摘要。
        包含召回和重排两个步骤。

        Args:
            user_id: 用户 ID
            query: 查询语句
            k: 返回结果数量
            fetch_k: 召回数量

        Returns:
            List[Document]: 相关的摘要文档列表
        """
        query_vec = self.embeddings.embed_query(query)
        rows = self._store.search(user_id, query_vec, k=fetch_k)
        if not rows:
            return []
        candidates = [
            Document(
                page_content=r["text"],
                metadata={
                    "type": "chat_summary",
                    "user_id": r["user_id"],
                    "session_id": r["session_id"],
                    "start_msg_id": r.get("start_msg_id"),
                    "end_msg_id": r.get("end_msg_id"),
                    "created_at": r.get("created_at"),
                },
            )
            for r in rows
        ]
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
    """选择最近的 N 轮对话消息（2 * recent_turns 条）"""
    if recent_turns <= 0:
        return []
    limit = recent_turns * 2
    return messages[-limit:] if len(messages) > limit else messages


def split_messages_for_memory(messages: List[Dict[str, Any]], recent_turns: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    将消息列表切分为"旧消息"（用于生成摘要）和"近期消息"（保留在上下文中）。

    Args:
        messages: 完整消息列表
        recent_turns: 保留的最近轮数

    Returns:
        Tuple[List, List]: (older_messages, recent_messages)
    """
    recent = select_recent_turn_messages(messages, recent_turns=recent_turns)
    older = messages[: max(0, len(messages) - len(recent))]
    return older, recent
