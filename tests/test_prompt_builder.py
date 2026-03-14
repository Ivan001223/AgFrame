from __future__ import annotations

from langchain_core.documents import Document

from app.runtime.prompts import prompt_builder
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

    prompt, citations = build_system_prompt(
        profile="profile" * 10,
        recent_history_lines=["l1", "l2", "l3"],
        docs=docs,
        memories=memories,
        web_search={"query": "q" * 400, "result": "r" * 4000},
        self_correction="c" * 4000,
        budget=budget,
    )

    assert "<recent_history>\n" in prompt
    assert "l2" in prompt and "l3" in prompt
    assert "l1" not in prompt
    assert "<retrieved_docs>\n" in prompt
    assert "Doc 1" in prompt
    assert "Doc 2" not in prompt
    assert len(citations) == 2
    assert citations[0]["kind"] == "doc"
    assert citations[1]["kind"] == "memory"
