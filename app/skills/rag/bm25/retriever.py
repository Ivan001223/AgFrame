import threading
from typing import Dict, List, Tuple

from langchain_core.documents import Document

from app.skills.rag.bm25.tokenizer import Tokenizer
from app.skills.rag.bm25.index_builder import IndexBuilder
from app.skills.rag.bm25.bm25_scorer import BM25Scorer


class BM25Retriever:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        persist_path: str = None,
        cache_size: int = 100,
    ):
        self.tokenizer = Tokenizer()
        self.builder = IndexBuilder(self.tokenizer, persist_path)
        self.k1 = k1
        self.b = b
        self._documents: List[Document] = []
        self._indexed = False
        self._cache: Dict[str, List[Tuple[Document, float]]] = {}
        self._cache_lock = threading.Lock()
        self._cache_size = cache_size

    def add_documents(self, documents: List[Document]) -> None:
        self._documents = documents
        texts = [doc.page_content for doc in documents]
        self.builder.build(texts)
        self.scorer = BM25Scorer(self.builder.index, k1=self.k1, b=self.b)
        self._indexed = True
        self.clear_cache()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        cache_key = f"{query}:{top_k}"

        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        if not self._indexed:
            return []

        query_terms = self.builder.index.tokenizer.tokenize(query)
        if not query_terms:
            return []

        candidate_docs = set()
        for term in query_terms:
            for doc_id, tf in self.builder.index.get_postings(term):
                candidate_docs.add(doc_id)

        scores = []
        for doc_id in candidate_docs:
            score = self.scorer.score(doc_id, query)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scores[:top_k]:
            if 0 <= doc_id < len(self._documents):
                doc = self._documents[doc_id]
                doc.metadata["bm25_score"] = score
                results.append((doc, score))

        with self._cache_lock:
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[cache_key] = results

        return results

    def batch_search(
        self, queries: List[str], top_k: int = 10
    ) -> List[List[Tuple[Document, float]]]:
        return [self.search(q, top_k) for q in queries]

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
