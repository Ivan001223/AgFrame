from __future__ import annotations

from langchain_core.documents import Document

from app.runtime.prompts import prompt_builder
from app.runtime.prompts import context_pruner
from app.runtime.prompts.context_pruner import ContextPruningConfig, prune_document_content
from app.runtime.prompts.prompt_builder import PromptBudget, build_citations, build_system_prompt


def test_truncate_max_chars_zero():
    assert prompt_builder._truncate("abc", 0) == ""


def test_truncate_noop_when_short():
    assert prompt_builder._truncate("abc", 3) == "abc"


def test_truncate_adds_ellipsis_and_length_respects_max():
    out = prompt_builder._truncate("abcdef", 4)
    assert out.endswith("…")
    assert len(out) == 4


def test_take_with_budget_stops_when_exhausted():
    items = ["aa", "bb", "cc"]
    out = prompt_builder._take_with_budget(items, max_total_chars=4)
    assert out == ["aa", "bb"]


def test_take_with_budget_truncates_last_item():
    items = ["aaaa", "bbbb"]
    out = prompt_builder._take_with_budget(items, max_total_chars=5)
    assert out[0] == "aaaa"
    assert len(out[1]) == 1


def test_build_citations_parses_doc_and_memory_meta():
    docs = [
        Document(page_content="d1", metadata={"doc_id": "id1", "page_num": "2", "source": "s1"}),
        Document(page_content="d2", metadata={"source": "s2", "page_num": "not_int"}),
    ]
    memories = [Document(page_content="m1", metadata={"session_id": "sess_1", "source": "mem_src"})]

    c = build_citations(docs=docs, memories=memories)
    assert c[0]["kind"] == "doc"
    assert c[0]["doc_id"] == "id1"
    assert c[0]["page"] == 2
    assert c[1]["doc_id"] == "s2"
    assert c[1]["page"] is None
    assert c[2]["kind"] == "memory"
    assert c[2]["session_id"] == "sess_1"


def test_get_meta_int_returns_none_when_missing_key():
    assert prompt_builder._get_meta_int({}, "page_num") is None


def test_build_system_prompt_respects_budget_limits():
    docs = [
        Document(page_content=("x" * 5000), metadata={"doc_id": "d1", "parent_chunk_id": "p1", "page_num": 1}),
        Document(page_content=("y" * 5000), metadata={"doc_id": "d2", "parent_chunk_id": "p2", "page_num": 2}),
    ]
    memories = [
        Document(
            page_content=("m" * 5000),
            metadata={"session_id": "s1", "start_msg_id": 1, "end_msg_id": 2},
        )
    ]

    budget = PromptBudget(
        max_recent_history_lines=2,
        max_docs=1,
        max_memories=1,
        max_doc_chars_total=50,
        max_memory_chars_total=50,
        max_profile_chars_total=10,
        max_item_chars=20,
    )

    prompt, citations, pruning = build_system_prompt(
        profile="profile" * 10,
        recent_history_lines=["l1", "l2", "l3"],
        docs=docs,
        memories=memories,
        query="find x",
        focus_hint="focus on x",
        web_search={"query": "q" * 400, "result": "r" * 4000},
        self_correction="c" * 4000,
        budget=budget,
    )

    assert "<recent_history>\n" in prompt
    assert "l2" in prompt and "l3" in prompt
    assert "l1" not in prompt
    assert "<retrieved_docs>\n" in prompt
    assert "<context_focus_hint>\nfocus on x\n</context_focus_hint>" in prompt
    assert "Doc 1" in prompt
    assert "Doc 2" not in prompt
    assert len(citations) == 2
    assert citations[0]["kind"] == "doc"
    assert citations[1]["kind"] == "memory"
    assert pruning["focus_hint"] == "focus on x"
    assert pruning["docs"]["char_savings"]["saved"] >= 0
    assert pruning["memories"]["char_savings"]["saved"] >= 0
    assert pruning["method"] == "heuristic"
    assert pruning["scoring_source"] == "heuristic"


def test_prune_document_content_keeps_goal_relevant_lines():
    content = "\n".join(
        [
            "alpha setup",
            "billing retry pipeline",
            "retry uses exponential backoff",
            "metrics and tracing",
            "final cleanup",
        ]
    )

    pruned, stats = prune_document_content(
        content,
        query="How does retry work?",
        focus_hint="billing retry",
        config=ContextPruningConfig(
            enabled=True,
            min_keep_lines=2,
            max_keep_ratio=0.6,
            neighbor_window=0,
            max_lines_per_item=3,
            score_threshold=0.2,
        ),
    )

    assert "billing retry pipeline" in pruned
    assert "retry uses exponential backoff" in pruned
    assert stats["char_count_after"] < stats["char_count_before"]


def test_prune_document_content_supports_reranker_mode(monkeypatch):
    content = "\n".join(
        [
            "setup",
            "retry loop",
            "backoff strategy",
            "cleanup",
        ]
    )

    pruned, stats = prune_document_content(
        content,
        query="How does retry work?",
        focus_hint="retry backoff",
        config=ContextPruningConfig(
            enabled=True,
            method="reranker",
            min_keep_lines=1,
            max_keep_ratio=0.5,
            neighbor_window=0,
            max_lines_per_item=2,
            score_threshold=0.2,
        ),
    )

    assert "backoff strategy" in pruned
    assert stats["method"] == "reranker"
    assert stats["scoring_source"] == "lightweight_ranker"


def test_prune_document_content_auto_mode_falls_back_to_heuristic_for_short_text(monkeypatch):
    pruned, stats = prune_document_content(
        "alpha\nretry handler\ncleanup",
        query="retry",
        focus_hint="retry",
        config=ContextPruningConfig(
            enabled=True,
            method="auto",
            auto_reranker_min_lines=10,
            auto_reranker_min_chars=100,
            min_keep_lines=1,
            max_keep_ratio=0.5,
            neighbor_window=0,
            max_lines_per_item=2,
            score_threshold=0.2,
        ),
    )

    assert "retry handler" in pruned
    assert stats["method"] == "heuristic"


def test_prune_document_content_auto_mode_uses_reranker_for_long_text(monkeypatch):
    content = "\n".join(["setup"] * 8 + ["critical backoff line"] + ["cleanup"] * 8)
    pruned, stats = prune_document_content(
        content,
        query="retry",
        focus_hint="retry backoff",
        config=ContextPruningConfig(
            enabled=True,
            method="auto",
            auto_reranker_min_lines=10,
            auto_reranker_min_chars=50,
            min_keep_lines=1,
            max_keep_ratio=0.2,
            neighbor_window=0,
            max_lines_per_item=2,
            score_threshold=0.2,
        ),
    )

    assert "critical backoff line" in pruned
    assert stats["method"] == "reranker"


def test_prune_document_content_heuristic_covers_focus_keywords_on_distractors():
    content = "\n".join(
        [
            "provider health check",
            "retry button label in ui",
            "provider fallback copy text",
            "provider retry scheduler",
            "backoff state machine with jitter",
            "manual retry notification",
            "metrics snapshot",
        ]
    )

    pruned, stats = prune_document_content(
        content,
        query="What code applies exponential retry after provider failures?",
        focus_hint="provider retry backoff",
        config=ContextPruningConfig(
            enabled=True,
            method="heuristic",
            min_keep_lines=2,
            max_keep_ratio=0.45,
            neighbor_window=0,
            max_lines_per_item=4,
            score_threshold=0.18,
        ),
    )

    assert "provider retry scheduler" in pruned
    assert "backoff state machine with jitter" in pruned
    assert stats["line_count_after"] <= 4


def test_prune_document_content_heuristic_uses_semantic_aliases():
    content = "\n".join(
        [
            "oauth callback received",
            "mint fresh session secret for the browser",
            "extend identity lease in storage",
            "ui success banner",
        ]
    )

    pruned, stats = prune_document_content(
        content,
        query="Which code renews login credentials after callback?",
        focus_hint="refresh token rotate after oauth callback",
        config=ContextPruningConfig(
            enabled=True,
            method="heuristic",
            min_keep_lines=2,
            max_keep_ratio=0.6,
            neighbor_window=0,
            max_lines_per_item=3,
            score_threshold=0.18,
        ),
    )

    assert "mint fresh session secret for the browser" in pruned
    assert "extend identity lease in storage" in pruned
    assert stats["method"] == "heuristic"


def test_prune_document_content_reranker_window_uses_neighbor_context(monkeypatch):
    content = "\n".join(
        [
            "oauth callback received",
            "issue browser grant from callback state",
            "ui success banner",
        ]
    )

    pruned, stats = prune_document_content(
        content,
        query="Which code renews login credentials after callback?",
        focus_hint="refresh token rotate after oauth callback",
        config=ContextPruningConfig(
            enabled=True,
            method="reranker",
            reranker_window_radius=1,
            min_keep_lines=1,
            max_keep_ratio=0.34,
            neighbor_window=0,
            max_lines_per_item=1,
            score_threshold=0.18,
        ),
    )

    assert "issue browser grant from callback state" in pruned
    assert stats["method"] == "reranker"
    assert stats["scoring_source"] == "lightweight_ranker"


def test_prune_document_content_reranker_uses_lightweight_ranker():
    content = "\n".join(
        [f"filler line {i}" for i in range(45)]
        + [
            "provider callback telemetry",
            "session token badge",
            "discard browser grant from callback state",
            "purge persisted identity lease",
            "success toast copy",
        ]
    )

    pruned, stats = prune_document_content(
        content,
        query="Which code revokes stale credentials after the provider callback finishes?",
        focus_hint="oauth callback revoke refresh token after login",
        config=ContextPruningConfig(
            enabled=True,
            method="reranker",
            reranker_window_radius=1,
            min_keep_lines=2,
            max_keep_ratio=0.2,
            neighbor_window=0,
            max_lines_per_item=6,
            score_threshold=0.18,
        ),
    )

    assert "discard browser grant from callback state" in pruned
    assert "purge persisted identity lease" in pruned
    assert stats["method"] == "reranker"
    assert stats["scoring_source"] == "lightweight_ranker"


def test_prune_document_content_neighbor_window_preserves_protected_lines():
    content = "\n".join(
        [
            "alpha setup",
            "billing retry coordinator",
            "backoff strategy implementation",
            "cleanup",
        ]
    )

    pruned, stats = prune_document_content(
        content,
        query="How does billing retry backoff work?",
        focus_hint="billing retry backoff",
        config=ContextPruningConfig(
            enabled=True,
            method="heuristic",
            min_keep_lines=2,
            max_keep_ratio=0.5,
            neighbor_window=1,
            max_lines_per_item=2,
            score_threshold=0.18,
        ),
    )

    assert "billing retry coordinator" in pruned
    assert "backoff strategy implementation" in pruned
    assert stats["line_count_after"] <= 2
