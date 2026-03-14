import pytest
from app.skills.rag.bm25.tokenizer import Tokenizer
from app.skills.rag.bm25.inverted_index import InvertedIndex


def test_inverted_index_build():
    tokenizer = Tokenizer()
    index = InvertedIndex(tokenizer)

    docs = [
        "hello world",
        "hello python",
        "world of python",
    ]

    index.build(docs)

    assert index.doc_count == 3
    assert "hello" in index.term_dict
    assert "world" in index.term_dict
    assert "python" in index.term_dict


def test_inverted_index_postings():
    tokenizer = Tokenizer()
    index = InvertedIndex(tokenizer)

    docs = [
        "hello world",
        "hello python",
        "world of python",
    ]

    index.build(docs)

    hello_postings = index.get_postings("hello")
    assert len(hello_postings) == 2

    world_postings = index.get_postings("world")
    assert len(world_postings) == 2


def test_inverted_index_idf():
    tokenizer = Tokenizer()
    index = InvertedIndex(tokenizer)

    docs = [
        "hello world",
        "hello python",
        "world of python",
    ]

    index.build(docs)

    assert "hello" in index.idf
    assert "python" in index.idf
    assert index.idf["hello"] > 0
