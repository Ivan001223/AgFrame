# Deployment Guide

<div align="center">
  <a href="deployment-cn.md">中文文档</a>
</div>

## Requirements

- Python 3.11.15
- `uv` (Package manager)
- Docker and Docker Compose
- Accessible PostgreSQL and Redis ports

It is highly recommended to strictly use the Python version declared in this repository.

## Install

```bash
uv python install 3.11.15
uv sync --no-dev
cp configs/config.example.json configs/config.json
```

Optional dependency groups:

- `uv sync --no-dev --group document-ai`
  Used to support higher precision PDF / Office document parsing.

The default installation already includes the dependencies required for local Embeddings, local OCR, Torch, Transformers, and legacy Reranker compatibility.

The default document RAG, memory retrieval, and context pruning paths **do not require** loading a local large model Reranker.

## Required Configuration

At least the following must be set in `config.json`:

- `auth.secret_key` (please generate a long random string)
- `database.url` or equivalent database credentials
- `database.password`
- `llm.api_key` (if using a cloud model provider)

For the recommended lightweight pipeline, please keep the following configurations empty:

- `reranker.model_name=""`
- `local_models.rerank_model=""`

## Start The Full Stack with Docker Compose

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

By default, this will start:

- `postgres`
- `redis` (uses Redis Stack, supporting queues, rate limiting, and LangGraph checkpoints)
- `backend`
- `worker`
- `frontend`

Default addresses:

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

Notes:

- If you want to use the chat capability directly, please fill in `LLM_API_KEY` in `.env`
- If you want to enable document ingestion / RAG, it is still recommended to supplement the embeddings configuration or remote embeddings service address in `configs/config.json`

If additional observability components are needed:

```bash
docker compose --profile observability up --build -d
```

This will additionally start:

- `clickhouse`
- `minio`
- `langfuse-server`
- `langfuse-worker`

The default Langfuse address is `http://localhost:3001`.

## Start Service

If you don't use Docker Compose, you can also continue using the original manual method. Since `v0.2.1` introduced Harness and the Agent task execution pipeline, you need to ensure both the backend and Worker are running healthily:

**Start the backend API service:**

```bash
uv run python -m app.server.main
```

Default addresses:
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

**Start the asynchronous Worker (Required):**

```bash
uv run arq app.infrastructure.queue.worker_settings.WorkerSettings
```

## Stop Service

```bash
docker compose down
```