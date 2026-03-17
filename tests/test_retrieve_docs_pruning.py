from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.skills.rag import retrieve_docs as retrieve_docs_module


class _RagRetrieval:
    candidate_k = 5


class _ContextPruning:
    enabled = True
    method = "heuristic"
    auto_reranker_min_lines = 40
    auto_reranker_min_chars = 2500
    min_keywords = 2
    min_keep_lines = 2
    max_keep_ratio = 0.6
    neighbor_window = 0
    reranker_window_radius = 1
    max_lines_per_item = 3
    score_threshold = 0.2


class _Prompt:
    context_pruning = _ContextPruning()


class _Settings:
    rag = type("_Rag", (), {"retrieval": _RagRetrieval()})()
    prompt = _Prompt()


class _Engine:
    def retrieve_candidates(self, query: str, fetch_k: int, user_id: str | None = None):
        assert query == "How does retry work?"
        assert fetch_k == 5
        assert user_id == "u1"
        return [
            Document(
                page_content="\n".join(
                    [
                        "module boot",
                        "billing retry flow",
                        "retry uses exponential backoff",
                        "cleanup handler",
                    ]
                ),
                metadata={"doc_id": "doc-1"},
            )
        ]


@pytest.mark.anyio
async def test_retrieve_docs_node_prunes_candidates_with_focus_hint(monkeypatch):
    monkeypatch.setattr(retrieve_docs_module, "settings", _Settings())
    monkeypatch.setattr(retrieve_docs_module, "get_rag_engine", lambda: _Engine())

    state = {
        "messages": [HumanMessage(content="How does retry work?")],
        "user_id": "u1",
        "context": {
            "session_id": "s1",
            "context_focus_hint": "billing retry",
        },
        "route": {"reasoning": "Focus on retry and backoff implementation"},
    }

    result = await retrieve_docs_module.retrieve_docs_node(state)

    candidates = result["retrieved_docs_candidates"]
    assert len(candidates) == 1
    assert "billing retry flow" in candidates[0].page_content
    assert "retry uses exponential backoff" in candidates[0].page_content
    assert "module boot" not in candidates[0].page_content

    debug = result["retrieval_debug"]["candidate_pruning"]
    assert debug["focus_hint"] == "billing retry"
    assert debug["items_pruned"] == 1
    assert debug["char_savings"]["saved"] > 0
    assert debug["scoring_source"] == "heuristic"
    assert result["trace"]["candidate_pruning"]["scoring_source"] == "heuristic"
    assert result["trace"]["candidate_pruning"]["char_savings"]["saved"] > 0
