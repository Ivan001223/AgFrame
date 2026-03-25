# RAG Migration Guide

## Who This Is For

Use this guide if you are moving from an older, heavier RAG setup to AgFrame's current lightweight default.

Typical cases:

- you previously used dense + sparse + model reranking
- you still have `reranker.*` config in place
- you previously relied on a separate `rerank_docs` step

## Target State

Recommended end state:

- document retrieval: `Dense + BM25 + RRF`
- pruning: lightweight local ranking / heuristic pruning
- parent restore: enabled
- model reranker: disabled by default

## Configuration Changes

Keep and tune:

- `embeddings.*`
- `rag.retrieval.*`
- `prompt.context_pruning.*`

Usually set empty unless you need legacy compatibility:

- `reranker.model_name`
- `local_models.rerank_model`

Recommended node list:

```json
{
  "nodes": {
    "enabled": [
      "router",
      "retrieve_docs",
      "retrieve_memories",
      "assemble",
      "generate"
    ]
  }
}
```

If your old config still references `rerank_docs`, remove it.

## Behavior Changes

Old pattern:

```text
Query
  -> Hybrid Retrieve
  -> Model Rerank
  -> Parent Restore
```

Current pattern:

```text
Query
  -> Dense + BM25
  -> RRF
  -> Candidate Pruning
  -> Parent Restore
```

What changed:

- document model reranking is gone from the default path
- ranking quality now depends mainly on recall depth, chunking, and RRF
- pruning is lighter and cheaper

## Important Compatibility Note

`context_pruning.method="reranker"` is still accepted.

But it no longer means "use a model reranker".

It now maps to lightweight local scoring so that old configs keep working without forcing heavyweight runtime dependencies.

## Migration Checklist

1. Set `reranker.model_name=""`
2. Set `local_models.rerank_model=""`
3. Remove `rerank_docs` from enabled nodes if it exists
4. Keep `prompt.context_pruning.method="auto"` unless you have a reason not to
5. Validate retrieval quality before changing pruning thresholds

## Validation Checklist

After migration, verify:

- BM25 still catches exact terminology and identifiers
- dense recall still catches paraphrases and semantic variants
- parent restore still returns enough context for generation
- latency improves or stays stable
- prompt size decreases or stays controlled

## Common Mistakes

- tuning pruning before recall depth
- assuming `reranker` config names still imply model-based reranking
- removing compatibility fields too early and breaking old deployments

## Minimal Recommended Config

```json
{
  "embeddings": {
    "model_name": "Qwen/Qwen3-Embedding-0.6B"
  },
  "rag": {
    "retrieval": {
      "mode": "hybrid",
      "dense_k": 20,
      "sparse_k": 20,
      "candidate_k": 20,
      "final_k": 3,
      "rrf_k": 60
    }
  },
  "prompt": {
    "context_pruning": {
      "enabled": true,
      "method": "auto"
    }
  },
  "reranker": {
    "model_name": ""
  },
  "local_models": {
    "rerank_model": ""
  }
}
```
