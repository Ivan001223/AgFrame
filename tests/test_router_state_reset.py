from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from app.skills.common import router as router_module


@pytest.mark.anyio
async def test_router_clears_stale_retrieval_artifacts_at_turn_start(monkeypatch: pytest.MonkeyPatch):
    class _Decision:
        needs_docs = False
        needs_history = False
        reasoning = "no retrieval needed"

    monkeypatch.setattr(router_module, "route_memory", lambda state: _Decision())

    result = await router_module.router_node(
        {
            "messages": [HumanMessage(content="hello")],
            "context": {
                "session_id": "s1",
                "retrieved_docs": [Document(page_content="stale doc")],
                "retrieved_memories": [Document(page_content="stale memory")],
                "retrieved_profile_items": [{"text": "stale profile"}],
                "citations": [{"kind": "doc", "label": "Doc 1"}],
            },
            "retrieved_docs": [Document(page_content="stale doc")],
            "retrieved_docs_candidates": [Document(page_content="stale candidate")],
            "retrieved_docs_candidates_raw": [Document(page_content="stale raw candidate")],
            "retrieved_memories": [Document(page_content="stale memory")],
            "citations": [{"kind": "doc", "label": "Doc 1"}],
        }
    )

    assert result["route"]["needs_docs"] is False
    assert result["retrieved_docs"] == []
    assert result["retrieved_docs_candidates"] == []
    assert result["retrieved_docs_candidates_raw"] == []
    assert result["retrieved_memories"] == []
    assert result["retrieved_profile_items"] == []
    assert result["citations"] == []
    assert "retrieved_docs" not in result["context"]
    assert "retrieved_memories" not in result["context"]
    assert "retrieved_profile_items" not in result["context"]
    assert "citations" not in result["context"]
