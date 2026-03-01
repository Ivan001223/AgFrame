import pytest
from langchain_core.documents import Document

from app.skills.rag.bm25.retriever import BM25Retriever


def test_bm25_retriever_search():
    docs = [
        Document(page_content="hello world python", metadata={"id": "1"}),
        Document(page_content="python is great", metadata={"id": "2"}),
        Document(page_content="machine learning ai", metadata={"id": "3"}),
    ]

    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.add_documents(docs)

    results = retriever.search("python programming", top_k=2)

    assert len(results) == 2
    assert all(isinstance(doc, Document) for doc, score in results)
    assert all(isinstance(score, float) for doc, score in results)


def test_bm25_retriever_empty_index():
    retriever = BM25Retriever()
    results = retriever.search("test", top_k=10)

    assert results == []


def test_bm25_retriever_batch_search():
    docs = [
        Document(page_content="hello world python", metadata={"id": "1"}),
        Document(page_content="python is great", metadata={"id": "2"}),
        Document(page_content="machine learning ai", metadata={"id": "3"}),
    ]

    retriever = BM25Retriever()
    retriever.add_documents(docs)

    queries = ["python", "machine learning"]
    results = retriever.batch_search(queries, top_k=2)

    assert len(results) == 2
    assert all(len(r) <= 2 for r in results)
