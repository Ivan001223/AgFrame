from __future__ import annotations

from app.memory.long_term.chat_memory_engine import ChatSummaryIndex
from app.memory.long_term.user_memory_engine import UserMemoryEngine


def test_chat_summary_index_uses_local_ranking(monkeypatch):
    class _Embeddings:
        def embed_query(self, query: str):
            return [0.1, 0.2]

    class _Store:
        def search(self, user_id: str, query_vec, k: int):
            assert user_id == "u1"
            assert k == 5
            return [
                {
                    "text": "billing retry uses exponential backoff",
                    "user_id": "u1",
                    "session_id": "s1",
                },
                {
                    "text": "ui theme button color tweaks",
                    "user_id": "u1",
                    "session_id": "s2",
                },
            ]

    monkeypatch.setattr("app.memory.long_term.chat_memory_engine.get_embeddings", lambda: _Embeddings())
    monkeypatch.setattr("app.memory.long_term.chat_memory_engine.PgChatSummaryStore", lambda: _Store())

    engine = ChatSummaryIndex()
    docs = engine.retrieve("u1", "how does billing retry backoff work", k=1, fetch_k=5)

    assert len(docs) == 1
    assert "retry" in docs[0].page_content
    assert docs[0].metadata["rank_score"] > 0


def test_user_memory_engine_profile_items_use_local_ranking(monkeypatch):
    class _Embeddings:
        def embed_query(self, query: str):
            return [0.1, 0.2]

    class _Store:
        def dense_search(self, query_vec, *, user_id: str, kind: str, k: int, subkind=None):
            assert user_id == "u1"
            assert kind == "semantic"
            assert k == 5
            return [
                {
                    "text": "偏好：language=python",
                    "metadata_json": {"type": "profile_preference"},
                },
                {
                    "text": "事实/偏好：喜欢简洁界面",
                    "metadata_json": {"type": "profile_fact"},
                },
            ]

    monkeypatch.setattr("app.memory.long_term.user_memory_engine.get_embeddings", lambda: _Embeddings())
    monkeypatch.setattr("app.memory.long_term.user_memory_engine.PgUserMemoryStore", lambda: _Store())

    engine = UserMemoryEngine()
    items = engine.retrieve_profile_items(user_id="u1", query="python preference", k=1, fetch_k=5)

    assert len(items) == 1
    assert items[0]["text"] == "偏好：language=python"
    assert items[0]["rank_score"] > 0
