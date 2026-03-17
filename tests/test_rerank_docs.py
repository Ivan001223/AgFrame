from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.skills.rag import rerank_docs as rerank_docs_module


class _Budget:
    max_docs = 2


class _Retrieval:
    final_k = 2


class _Rag:
    retrieval = _Retrieval()


class _Settings:
    rag = _Rag()
    prompt = type("_Prompt", (), {"budget": _Budget()})()


class _Engine:
    def rerank_candidates(self, query: str, candidates: list[Document], *, k: int):
        assert query == "How does retry work?"
        assert k == 2
        assert len(candidates) == 1
        assert candidates[0].page_content == "billing retry flow\nretry uses exponential backoff"
        assert candidates[0].metadata["parent_chunk_id"] == 42
        meta = dict(candidates[0].metadata)
        meta["rerank_score"] = 1.5
        return [Document(page_content=candidates[0].page_content, metadata=meta)]

    def restore_parents(self, docs: list[Document], *, k: int):
        assert k == 2
        assert docs[0].metadata["parent_chunk_id"] == 42
        assert docs[0].metadata["rerank_score"] == 1.5
        return [
            Document(
                page_content="full parent chunk content",
                metadata={"parent_chunk_id": 42, "doc_id": 7, "rerank_score": 1.5},
            )
        ]


@pytest.mark.anyio
async def test_rerank_docs_node_restores_parents_from_pruned_candidates(monkeypatch):
    monkeypatch.setattr(rerank_docs_module, "settings", _Settings())
    monkeypatch.setattr(rerank_docs_module, "get_rag_engine", lambda: _Engine())

    state = {
        "messages": [HumanMessage(content="How does retry work?")],
        "context": {
            "session_id": "s1",
            "retrieved_docs_candidates": [
                Document(
                    page_content="billing retry flow\nretry uses exponential backoff",
                    metadata={"doc_id": 7, "parent_chunk_id": 42},
                )
            ],
        },
    }

    result = await rerank_docs_module.rerank_docs_node(state)

    docs = result["retrieved_docs"]
    assert len(docs) == 1
    assert docs[0].page_content == "full parent chunk content"
    assert docs[0].metadata["parent_chunk_id"] == 42
