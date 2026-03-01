import time

import pytest
from langchain_core.documents import Document

from app.skills.rag.bm25.retriever import BM25Retriever


def generate_test_docs(n: int = 1000) -> list[Document]:
    topics = [
        "python programming",
        "machine learning",
        "data science",
        "web development",
        "AI",
    ]
    docs = []
    for i in range(n):
        topic = topics[i % len(topics)]
        docs.append(
            Document(
                page_content=f"{topic} tutorial {i} - learn {topic} with examples",
                metadata={"id": str(i)},
            )
        )
    return docs


def test_bm25_build_performance():
    docs = generate_test_docs(1000)

    start = time.perf_counter()
    retriever = BM25Retriever()
    retriever.add_documents(docs)
    build_time = time.perf_counter() - start

    print(f"Build time for 1000 docs: {build_time*1000:.2f}ms")
    assert build_time < 5.0


def test_bm25_search_performance():
    docs = generate_test_docs(1000)
    retriever = BM25Retriever()
    retriever.add_documents(docs)

    queries = ["python", "machine learning", "web development"]

    start = time.perf_counter()
    for q in queries:
        retriever.search(q, top_k=10)
    search_time = time.perf_counter() - start

    print(f"Search time for 3 queries: {search_time*1000:.2f}ms")
    assert search_time < 1.0


def test_bm25_recall():
    docs = [
        Document(
            page_content="python is a programming language", metadata={"id": "1"}
        ),
        Document(page_content="java is also programming", metadata={"id": "2"}),
        Document(page_content="python machine learning", metadata={"id": "3"}),
    ]

    retriever = BM25Retriever()
    retriever.add_documents(docs)

    results = retriever.search("python programming", top_k=2)
    doc_ids = [doc.metadata.get("id") for doc, _ in results]

    assert "1" in doc_ids
    assert "3" in doc_ids


def test_bm25_cached_search_performance():
    docs = generate_test_docs(1000)
    retriever = BM25Retriever()
    retriever.add_documents(docs)

    queries = ["python", "machine learning", "web development"]

    for q in queries:
        retriever.search(q, top_k=10)

    start = time.perf_counter()
    for q in queries:
        retriever.search(q, top_k=10)
    cached_search_time = time.perf_counter() - start

    print(f"Cached search time for 3 queries: {cached_search_time*1000:.2f}ms")
    assert cached_search_time < 0.1
