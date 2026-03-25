# Deployment Guide

## Requirements

- Python 3.11.15
- `uv`
- Docker and Docker Compose
- reachable PostgreSQL and Redis ports

Using the Python version declared by the repository is recommended.

## Install

```bash
uv python install 3.11.15
uv sync --no-dev
cp configs/config.example.json configs/config.json
```

Optional groups:

- `uv sync --no-dev --group local-inference`
  for local embeddings, OCR, or legacy reranker compatibility
- `uv sync --no-dev --group document-ai`
  for higher-accuracy PDF / Office parsing

The default document RAG, memory retrieval, and context pruning paths do not require a model reranker.

## Required Configuration

At minimum, set:

- `auth.secret_key`
- `database.url` or equivalent database credentials
- `database.password`
- `llm.api_key` if using hosted models

For the recommended lightweight path, keep:

- `reranker.model_name=""`
- `local_models.rerank_model=""`

## Start Dependencies

```bash
docker-compose up -d
docker-compose ps
```

## Start Service

```bash
uv run python -m app.server.main
```

Default address:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

## Stop Service

```bash
docker-compose down
```
