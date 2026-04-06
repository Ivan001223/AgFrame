from __future__ import annotations

from langchain_core.documents import Document

from app.skills.rag.rag_engine import RAGEngine


class _HybridRetriever:
    def __init__(self, docs: list[Document]):
        self._docs = docs
        self.calls: list[tuple[str, object, object]] = []

    def retrieve_candidates(self, query: str, config=None, filter=None) -> list[Document]:
        self.calls.append((query, config, filter))
        return list(self._docs)


def test_retrieve_context_restores_parents_from_rrf_candidates(monkeypatch):
    monkeypatch.setattr("app.skills.rag.rag_engine.ensure_schema_if_possible", lambda: True)
    monkeypatch.setattr("app.skills.rag.rag_engine.get_embeddings", lambda: object())

    class _Store:
        pass

    engine = RAGEngine()
    engine._vectorstore = _Store()
    engine._hybrid_retriever = _HybridRetriever(
        [
            Document(
                page_content="child chunk",
                metadata={
                    "doc_id": 7,
                    "parent_chunk_id": 42,
                    "retrieval_rrf_score": 1.5,
                },
            )
        ]
    )

    class _DocStore:
        def fetch_parent_chunks(self, parent_ids: list[int]):
            assert parent_ids == [42]
            return [
                {
                    "parent_chunk_id": 42,
                    "doc_id": 7,
                    "page_num": 1,
                    "content": "full parent chunk content",
                    "source_path": "/tmp/source.md",
                    "knowledge_base_id": "kb-1",
                    "knowledge_base_name": "Project KB",
                }
            ]

    monkeypatch.setattr("app.skills.rag.rag_engine.MySQLDocStore", lambda: _DocStore())

    docs = engine.retrieve_context("retry flow", k=1, fetch_k=5, user_id="u1")

    assert len(docs) == 1
    assert docs[0].page_content == "full parent chunk content"
    assert docs[0].metadata["parent_chunk_id"] == 42
    assert docs[0].metadata["retrieval_rrf_score"] == 1.5
    assert docs[0].metadata["source"] == "/tmp/source.md"
    assert docs[0].metadata["knowledge_base_id"] == "kb-1"
    assert docs[0].metadata["knowledge_base_name"] == "Project KB"
    _, _, filter_dict = engine._hybrid_retriever.calls[0]
    assert filter_dict == {"user_id": "u1"}
