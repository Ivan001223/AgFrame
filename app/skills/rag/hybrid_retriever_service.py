from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from app.infrastructure.utils.logging import get_logger

_log = get_logger("services.hybrid_retriever")


def _stable_doc_key(doc: Document) -> str:
    meta = dict(getattr(doc, "metadata", {}) or {})
    parts = [
        str(meta.get("type") or ""),
        str(meta.get("doc_id") or ""),
        str(meta.get("parent_chunk_id") or ""),
        str(meta.get("child_index") or ""),
        str(meta.get("source") or ""),
    ]
    content = str(getattr(doc, "page_content", "") or "")
    digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()
    return "|".join(parts) + "|" + digest


def _iter_vectorstore_docs(vectorstore: Any) -> List[Document]:
    docstore = getattr(vectorstore, "docstore", None)
    if docstore is None:
        return []
    data = getattr(docstore, "_dict", None)
    if isinstance(data, dict):
        docs = [d for d in data.values() if isinstance(d, Document)]
        return docs
    if isinstance(docstore, dict):
        docs = [d for d in docstore.values() if isinstance(d, Document)]
        return docs
    return []


def _invoke_retriever(retriever: Any, query: str) -> List[Document]:
    if hasattr(retriever, "invoke"):
        out = retriever.invoke(query)
        if isinstance(out, list):
            return out
        return list(out or [])
    if hasattr(retriever, "get_relevant_documents"):
        out = retriever.get_relevant_documents(query)
        if isinstance(out, list):
            return out
        return list(out or [])
    if hasattr(retriever, "_get_relevant_documents"):
        try:
            out = retriever._get_relevant_documents(query, run_manager=None)
        except TypeError:
            out = retriever._get_relevant_documents(query)
        if isinstance(out, list):
            return out
        return list(out or [])
    return []


def _rrf_fuse(
    ranked_lists: Sequence[Tuple[str, Sequence[Document], float]],
    *,
    rrf_k: int,
    top_n: int,
) -> List[Document]:
    scores: Dict[str, float] = {}
    best_doc: Dict[str, Document] = {}
    ranks: Dict[str, Dict[str, int]] = {}

    for name, docs, weight in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            key = _stable_doc_key(doc)
            best_doc.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + float(weight) * (1.0 / (rrf_k + rank))
            ranks.setdefault(key, {})[name] = rank

    out: List[Document] = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[
        :top_n
    ]:
        d = best_doc[key]
        meta = dict(getattr(d, "metadata", {}) or {})
        meta["retrieval_rrf_score"] = float(score)
        for name, _, _ in ranked_lists:
            rank = ranks.get(key, {}).get(name)
            if rank is not None:
                meta[f"retrieval_{name}_rank"] = int(rank)
        d.metadata = meta
        out.append(d)
    return out


@dataclass(frozen=True)
class HybridRetrievalConfig:
    mode: str = "hybrid"
    dense_k: int = 20
    sparse_k: int = 20
    candidate_k: int = 20
    rrf_k: int = 60
    weights: Tuple[float, float] = (0.5, 0.5)


class HybridRetrieverService:
    def __init__(self, *, vectorstore: Any):
        self._vectorstore = vectorstore
        self._bm25 = None
        self._bm25_doc_count = -1

    def _ensure_bm25(self) -> bool:
        docs = _iter_vectorstore_docs(self._vectorstore)
        if not docs:
            self._bm25 = None
            self._bm25_doc_count = 0
            return False
        if self._bm25 is not None and self._bm25_doc_count == len(docs):
            return True

        try:
            from langchain_community.retrievers.bm25 import BM25Retriever
        except Exception as e:
            _log.warning("BM25Retriever import failed: %s", e)
            self._bm25 = None
            self._bm25_doc_count = len(docs)
            return False

        t0 = time.perf_counter()
        bm25 = BM25Retriever.from_documents(docs)
        self._bm25 = bm25
        self._bm25_doc_count = len(docs)
        _log.info(
            "bm25 rebuilt docs=%d cost_ms=%d",
            len(docs),
            int((time.perf_counter() - t0) * 1000),
        )
        return True

    def retrieve_candidates(
        self,
        query: str,
        *,
        config: Optional[HybridRetrievalConfig] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        cfg = config or HybridRetrievalConfig()
        mode = (cfg.mode or "hybrid").lower()
        dense_k = max(1, int(cfg.dense_k))
        sparse_k = max(1, int(cfg.sparse_k))
        candidate_k = max(1, int(cfg.candidate_k))
        rrf_k = max(1, int(cfg.rrf_k))
        w_sparse, w_dense = cfg.weights if cfg.weights else (0.5, 0.5)

        # Pass filter to similarity_search (supported by PgVectorVectorStore)
        dense_docs = list(
            self._vectorstore.similarity_search(query, k=dense_k, filter=filter)
        )

        if mode == "dense":
            for i, d in enumerate(dense_docs, start=1):
                meta = dict(getattr(d, "metadata", {}) or {})
                meta["retrieval_dense_rank"] = i
                d.metadata = meta
            return dense_docs[:candidate_k]

        # BM25 In-Memory doesn't support filter easily unless we filter results post-retrieval
        # OR we rebuild BM25 with filtered docs (expensive).
        # OR if vectorstore supports sparse_search with filter.

        # If filter provided and we fall back to BM25 (in-memory),
        # we should ideally filter the sparse_docs.

        if hasattr(self._vectorstore, "sparse_search"):
            # Assume sparse_search supports filter if it exists on PgVector store (custom implementation?)
            # Standard PGVector doesn't have sparse_search.
            # If it's custom, let's hope it supports filter.
            # Checking PgVectorVectorStore later.
            # For now passing filter as kwargs if supported.
            try:
                sparse_docs = list(
                    self._vectorstore.sparse_search(query, k=sparse_k, filter=filter)
                )[:sparse_k]
            except TypeError:
                sparse_docs = list(self._vectorstore.sparse_search(query, k=sparse_k))[
                    :sparse_k
                ]

            return _rrf_fuse(
                [
                    ("sparse", sparse_docs, float(w_sparse)),
                    ("dense", dense_docs, float(w_dense)),
                ],
                rrf_k=rrf_k,
                top_n=candidate_k,
            )

        has_bm25 = self._ensure_bm25()
        if not has_bm25:
            for i, d in enumerate(dense_docs, start=1):
                meta = dict(getattr(d, "metadata", {}) or {})
                meta["retrieval_dense_rank"] = i
                d.metadata = meta
            return dense_docs[:candidate_k]

        try:
            self._bm25.k = sparse_k
        except Exception:
            pass

        # BM25 is in-memory over ALL docs. We need to filter results.
        # This is suboptimal for multi-tenancy if BM25 index is shared.
        # Ideally, we shouldn't use shared in-memory BM25 for multi-tenant.
        # But for now, let's filter the results.
        all_sparse_docs = _invoke_retriever(self._bm25, query)

        sparse_docs = []
        if filter:
            for d in all_sparse_docs:
                meta = d.metadata or {}
                # Simple exact match for now.
                match = True
                for k, v in filter.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if match:
                    sparse_docs.append(d)
        else:
            sparse_docs = all_sparse_docs

        sparse_docs = sparse_docs[:sparse_k]

        return _rrf_fuse(
            [
                ("sparse", sparse_docs, float(w_sparse)),
                ("dense", dense_docs, float(w_dense)),
            ],
            rrf_k=rrf_k,
            top_n=candidate_k,
        )
