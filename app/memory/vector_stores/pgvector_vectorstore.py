from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from app.infrastructure.database.stores import PgDocEmbeddingStore
from app.runtime.llm.embeddings import ModelEmbeddings


class PgVectorVectorStore:
    def __init__(self, *, embeddings: ModelEmbeddings):
        self._embeddings = embeddings
        self._store = PgDocEmbeddingStore()

    def similarity_search(self, query: str, k: int = 20) -> List[Document]:
        query_vec = self._embeddings.embed_query(str(query or ""))
        rows = self._store.dense_search(query_vec, k=int(k))
        out: List[Document] = []
        for r in rows:
            meta = dict(r.metadata_json or {})
            if r.doc_id is not None:
                meta.setdefault("doc_id", int(r.doc_id))
            if r.parent_chunk_id is not None:
                meta.setdefault("parent_chunk_id", int(r.parent_chunk_id))
            if r.child_index is not None:
                meta.setdefault("child_index", int(r.child_index))
            if r.source_path:
                meta.setdefault("source", r.source_path)
            out.append(Document(page_content=r.content, metadata=meta))
        return out

    def sparse_search(self, query: str, k: int = 20) -> List[Document]:
        rows = self._store.sparse_search(str(query or ""), k=int(k))
        out: List[Document] = []
        for r in rows:
            meta = dict(r.metadata_json or {})
            if r.doc_id is not None:
                meta.setdefault("doc_id", int(r.doc_id))
            if r.parent_chunk_id is not None:
                meta.setdefault("parent_chunk_id", int(r.parent_chunk_id))
            if r.child_index is not None:
                meta.setdefault("child_index", int(r.child_index))
            if r.source_path:
                meta.setdefault("source", r.source_path)
            out.append(Document(page_content=r.content, metadata=meta))
        return out

