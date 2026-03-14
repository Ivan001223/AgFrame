import pytest
from app.skills.rag.bm25.tokenizer import Tokenizer
from app.skills.rag.bm25.inverted_index import InvertedIndex
from app.skills.rag.bm25.bm25_scorer import BM25Scorer


def test_bm25_score():
    tokenizer = Tokenizer()
    index = InvertedIndex(tokenizer)
    index.build(["hello world python", "world is big", "python is great"])

    scorer = BM25Scorer(index, k1=1.5, b=0.75)
    score = scorer.score(doc_id=0, query="hello world")

    assert score > 0
    assert isinstance(score, float)


def test_bm25_score_with_different_params():
    tokenizer = Tokenizer()
    index = InvertedIndex(tokenizer)
    index.build(["hello world python python python", "world is big", "python is great"])

    scorer_k1 = BM25Scorer(index, k1=0.5, b=0.75)
    scorer_k2 = BM25Scorer(index, k1=2.0, b=0.75)

    score_k1 = scorer_k1.score(doc_id=0, query="python world")
    score_k2 = scorer_k2.score(doc_id=0, query="python world")

    assert isinstance(score_k1, float)
    assert isinstance(score_k2, float)


def test_bm25_score_empty_query():
    tokenizer = Tokenizer()
    index = InvertedIndex(tokenizer)
    index.build(["hello world python"])

    scorer = BM25Scorer(index)
    score = scorer.score(doc_id=0, query="")

    assert score == 0.0


def test_bm25_score_no_matching_terms():
    tokenizer = Tokenizer()
    index = InvertedIndex(tokenizer)
    index.build(["hello world python"])

    scorer = BM25Scorer(index)
    score = scorer.score(doc_id=0, query="xyz123")

    assert score == 0.0
