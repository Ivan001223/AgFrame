from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.skills.common import grader as grader_module


@pytest.mark.anyio
async def test_grader_accept_clears_stale_search_state(monkeypatch: pytest.MonkeyPatch):
    async def _fake_invoke_structured(*args, **kwargs):
        return grader_module.GraderResult(
            verdict="accept",
            reasoning="looks good",
            issues=[],
            rewrite_instructions=None,
            search_query=None,
        )

    monkeypatch.setattr(grader_module, "invoke_structured", _fake_invoke_structured)

    result = await grader_module.grader_node(
        {
            "messages": [
                HumanMessage(content="What changed?"),
                AIMessage(content="Here is the answer."),
            ],
            "context": {
                "search_query": "stale query",
                "web_search": {"query": "stale query", "result": "stale result"},
                "self_correction": "stale rewrite",
            },
            "trace": {"trace_id": "t1", "self_correction_attempts": 1},
        }
    )

    context = result["context"]
    assert context["grade"]["verdict"] == "accept"
    assert "search_query" not in context
    assert "web_search" not in context
    assert "self_correction" not in context


@pytest.mark.anyio
async def test_grader_rewrite_clears_search_state_but_keeps_new_rewrite(monkeypatch: pytest.MonkeyPatch):
    async def _fake_invoke_structured(*args, **kwargs):
        return grader_module.GraderResult(
            verdict="rewrite",
            reasoning="missing support",
            issues=["missing_info"],
            rewrite_instructions="rewrite with cited support",
            search_query=None,
        )

    monkeypatch.setattr(grader_module, "invoke_structured", _fake_invoke_structured)

    result = await grader_module.grader_node(
        {
            "messages": [
                HumanMessage(content="What changed?"),
                AIMessage(content="Here is the answer."),
            ],
            "context": {
                "search_query": "stale query",
                "web_search": {"query": "stale query", "result": "stale result"},
                "self_correction": "stale rewrite",
            },
            "trace": {"trace_id": "t1", "self_correction_attempts": 0},
        }
    )

    context = result["context"]
    assert context["grade"]["verdict"] == "rewrite"
    assert "search_query" not in context
    assert "web_search" not in context
    assert context["self_correction"] == "rewrite with cited support"
