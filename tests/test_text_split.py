from __future__ import annotations

from app.infrastructure.utils.text_split import split_text_by_chars


def test_split_text_chunk_size_zero_returns_original():
    assert split_text_by_chars("abc", chunk_size=0, overlap=0) == ["abc"]


def test_split_text_overlap_is_capped():
    parts = split_text_by_chars("0123456789", chunk_size=5, overlap=10)
    assert parts[0] == "01234"
    assert parts[-1] == "89"


def test_split_text_filters_blank_chunks():
    parts = split_text_by_chars("a   b", chunk_size=2, overlap=0)
    assert all(p.strip() for p in parts)
