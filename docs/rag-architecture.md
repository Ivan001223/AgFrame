# RAG Architecture Design

<div align="center">
  <a href="rag-architecture-cn.md">中文文档</a>
</div>

## Overview

AgFrame's default RAG (Retrieval-Augmented Generation) pipeline remains intentionally lightweight and transparent. In the current system, this retrieval path is used by a backend-owned chat runtime and related document workflows rather than a frontend-owned demo pipeline.

Default retrieval flow:

```text
Dense Search + BM25 (Sparse Search)
  -> RRF Fusion (Reciprocal Rank Fusion)
  -> Candidate Pruning (Lightweight Candidate Pruning)
  -> Parent Restore (Parent Document Restoration)
  -> Prompt Assembly
```

The goal is to maximize transparency, recall quality, and maintainability without introducing a heavyweight model-reranking stage.

## Design Principles

- Only use two recall channels on the main path: Dense vector retrieval and BM25 sparse retrieval.
- Use RRF (Reciprocal Rank Fusion) for ranking fusion, replacing model-based reranking.
- Maintain fine-grained retrieval of child Chunks, but restore the complete Parent context before LLM generation.
- Use Lightweight pruning strategies to reduce noise in the Prompt.

## Main Code Paths

- Ingestion and chunking: `app/skills/rag/rag_engine.py`
- Hybrid retrieval and RRF: `app/skills/rag/hybrid_retriever_service.py`
- Vector and sparse retrieval adapters: `app/memory/vector_stores/pgvector_vectorstore.py`
- Context Pruning: `app/runtime/prompts/context_pruner.py`
- Local lightweight ranker: `app/infrastructure/utils/lightweight_ranker.py`
- Chat runtime integration: `app/server/api/chat.py`
- LangGraph orchestration entry: `app/runtime/graph/graph.py`

## Runtime Context

The current runtime uses the RAG pipeline in a broader execution environment:

- the workbench UI calls `POST /chat/workbench-invoke`
- the backend applies runtime config and invokes the LangGraph app
- interrupt and resume can pause or continue the same retrieval-backed conversation flow
- harness and studio features are adjacent control-plane capabilities, but they do not replace the core lightweight RAG retrieval path

## Current Defaults

### Chunking

Current document ingestion default parameters:

- Parent chunk: `6000` characters, `400` overlap
- Child chunk: `1400` characters, `120` overlap

Reasons:
- Child chunks can improve the accuracy of recall.
- Parent chunks can provide a more complete context background for LLM generation.

### Retrieval

Current default values in `settings.rag.retrieval`:

- `dense_k=20`
- `sparse_k=20`
- `candidate_k=20`
- `final_k=3`
- `rrf_k=60`

### Pruning

Current default values in `settings.prompt.context_pruning`:

- `method="heuristic"`
- `auto_reranker_min_lines=40`
- `auto_reranker_min_chars=2500`
- `neighbor_window=1`
- `reranker_window_radius=1`
- `max_lines_per_item=24`
- `score_threshold=0.18`

Note:
- It still accepts the configuration item named `reranker` to maintain compatibility.
- However, its underlying implementation has now been changed to scoring based on a lightweight local algorithm, rather than large model inference.

## Tuning Order

It is recommended to perform effectiveness tuning in the following order:

1. `dense_k` and `sparse_k`
2. `candidate_k`
3. Child chunk size and overlap
4. `final_k`
5. Pruning thresholds

Reasons:
- Omissions in the recall phase cannot be compensated for in subsequent pruning or reranking.
- Most Hit-rate issues primarily stem from the chunking strategy and candidate pool depth.

## Practical Starting Points

**General Knowledge Base:**
- Child chunk: `1000-1600` characters
- Overlap: `80-180` characters
- `dense_k=20-40`
- `sparse_k=20-40`
- `candidate_k=20-30`
- `final_k=3-6`

**Code or API Documentation:**
- Use slightly smaller child chunks.
- Keep overlap around 10%.
- If strongly relying on professional terminology, prioritize increasing `sparse_k`.

**Long Policy or OCR-heavy Documents:**
- Keep parent chunks relatively large.
- Avoid excessively large child chunks to prevent diluting the precise terminology signals of BM25.
- Before increasing retrieval depth, prioritize cleaning up garbled OCR outputs.

## Legacy-Compatible Items

The following configurations are retained only for backward compatibility and are no longer part of the recommended default main path:

- `reranker.*`
- `local_models.rerank_model`
- The naming convention of `context_pruning.method="reranker"`

## Recommended Next Steps

- Build an evaluation Benchmark based on real business data.
- Gradually rename the concept of `reranker` to `lightweight_ranker` in UI and documentation.
- Introduce Type-aware chunking and pruning strategies for code, logs, and tables.

---

## Migration Guide from Legacy RAG

If you are migrating from an older, heavier RAG architecture to AgFrame's currently recommended lightweight default pipeline, please refer to the following guide.

Typical situation:
- You previously used: Dense + Sparse + Model reranking
- Your configuration file still contains items related to `reranker.*`
- Your previous Graph relied on an independent `rerank_docs` node

### Configuration and Behavior Changes

**Old pattern:**
```text
Query -> Hybrid Retrieve -> Model Rerank -> Parent Restore
```

**Current pattern:**
```text
Query -> Dense + BM25 -> RRF -> Candidate Pruning (Lightweight) -> Parent Restore
```

**Retained and tunable items:**
- `embeddings.*`
- `rag.retrieval.*`
- `prompt.context_pruning.*`

**Normally should be empty (unless you need legacy compatibility):**
- `reranker.model_name`
- `local_models.rerank_model`

If your old configuration still contains the `rerank_docs` node, please remove it.

### Important Compatibility Notes

The configuration item `context_pruning.method="reranker"` is still accepted by the system. **However, it no longer means "use an LLM-based reranker".**
It has now been redirected and mapped to a lightweight local scoring algorithm. This is done so that old configuration files can continue to work without forcing heavy model inference dependencies at runtime.

### Migration Checklist

1. Set `reranker.model_name=""`
2. Set `local_models.rerank_model=""`
3. Remove `rerank_docs` from the list of enabled nodes (if it exists)
4. Keep `prompt.context_pruning.method="auto"`, unless you have a specific reason not to
5. Before modifying pruning thresholds, verify the recall quality of the underlying retrieval

### Common Mistakes

- Tuning pruning parameters before tuning recall depth
- Mistakenly believing that configuration items containing the word `reranker` will still call an LLM for reranking
- Removing compatibility fields too early, causing legacy deployment environments to crash
