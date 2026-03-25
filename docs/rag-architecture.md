# RAG Architecture

## Overview

AgFrame's default RAG path is intentionally simple:

```text
Dense Search + BM25
  -> RRF
  -> Candidate Pruning
  -> Parent Restore
  -> Prompt Assembly
```

The goal is to maximize transparency, recall, and maintainability without introducing a heavyweight document reranking stage.

## Design Principles

- Use only two recall channels in the main path: dense and BM25
- Fuse ranks with RRF instead of model-based reranking
- Keep chunk retrieval fine-grained, but restore larger parent context before generation
- Reduce prompt noise with lightweight pruning

## Main Code Paths

- ingestion and chunking: `app/skills/rag/rag_engine.py`
- hybrid retrieval and RRF: `app/skills/rag/hybrid_retriever_service.py`
- vector and sparse search adapters: `app/memory/vector_stores/pgvector_vectorstore.py`
- pruning: `app/runtime/prompts/context_pruner.py`
- local lightweight ranking: `app/infrastructure/utils/lightweight_ranker.py`

## Current Defaults

### Chunking

Current ingestion defaults:

- parent chunk: `6000` chars, `400` overlap
- child chunk: `1400` chars, `120` overlap

Why:

- child chunks improve recall precision
- parent chunks provide fuller context to the model

### Retrieval

Current defaults in `settings.rag.retrieval`:

- `dense_k=20`
- `sparse_k=20`
- `candidate_k=20`
- `final_k=3`
- `rrf_k=60`

### Pruning

Current defaults in `settings.prompt.context_pruning`:

- `method="auto"`
- `auto_reranker_min_lines=40`
- `auto_reranker_min_chars=2500`
- `neighbor_window=1`
- `reranker_window_radius=1`
- `max_lines_per_item=24`
- `score_threshold=0.18`

Note:

- the config name `reranker` is still accepted
- the implementation behind it is now lightweight local scoring

## Tuning Order

Tune in this order:

1. `dense_k` and `sparse_k`
2. `candidate_k`
3. child chunk size and overlap
4. `final_k`
5. pruning thresholds

Reason:

- poor recall cannot be fixed later by pruning
- most hit-rate issues come from chunking and candidate depth first

## Practical Starting Points

General knowledge base:

- child chunk: `1000-1600` chars
- overlap: `80-180` chars
- `dense_k=20-40`
- `sparse_k=20-40`
- `candidate_k=20-30`
- `final_k=3-6`

Code or API docs:

- use slightly smaller child chunks
- keep overlap around 10%
- if terminology is strong, increase `sparse_k` first

Long policy or OCR-heavy docs:

- keep parent chunks relatively large
- avoid oversized child chunks that dilute BM25 signals
- clean OCR output before increasing retrieval depth

## Legacy-Compatible Items

These remain in config for compatibility, but are not part of the recommended default path:

- `reranker.*`
- `local_models.rerank_model`
- `context_pruning.method="reranker"` naming

## Recommended Next Steps

- build a benchmark on real business data
- gradually rename UI and docs from `reranker` to `lightweight_ranker`
- add type-aware chunking and pruning for code, logs, and tables
