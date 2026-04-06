# Deployment Guide

<div align="center">
  <a href="deployment-cn.md">中文文档</a>
</div>

## Scope

- **Backend version**: `0.3.1`
- **Frontend version**: `0.3.1`
- **Python constraint**: `>=3.11,<3.12`
- **Primary local stack**: Postgres + Redis + FastAPI + ARQ Worker + Next.js

## Requirements

- Python 3.11
- `uv`
- Node.js 22 LTS recommended for frontend validation
- Docker and Docker Compose
- Reachable PostgreSQL and Redis ports

## Install

Backend:

```bash
uv sync
```

Frontend:

```bash
cd frontend
npm install
```

Optional backend groups:

- `uv sync --group document-ai`
- `uv sync --group evals`

## Required Configuration

Prepare local files:

```bash
cp .env.example .env
cp configs/config.example.json configs/config.json
```

At minimum, set secure values for:

- `AUTH_SECRET_KEY`
- `DATABASE_URL` or `DB_HOST` / `DB_PORT` / `DB_USER` / `DB_PASSWORD` / `DB_NAME`
- `DB_PASSWORD`
- `LLM_API_KEY` when using a cloud model

Security-sensitive behavior:

- startup validation rejects insecure `auth.secret_key`
- startup validation rejects insecure `database.password`
- `server.cors_allow_credentials=true` cannot be paired with `server.cors_origins=["*"]`

Recommended compatibility settings for the lightweight retrieval path:

- `reranker.model_name=""`
- `local_models.rerank_model=""`

## Docker Compose

Start the full stack:

```bash
docker compose up --build -d
docker compose ps
```

Default services:

- `postgres`
- `redis`
- `backend`
- `worker`
- `frontend`

Default local addresses:

- Frontend: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

Environment behavior to know:

- `.env.example` now defaults to `LLM_MODEL=dev-stub` and `MODEL_PATH_EMBEDDING=dev-stub`, so local startup works without a cloud key
- switch `.env` to a cloud model and set `LLM_API_KEY` when you want real remote generation
- chat and harness approvals are now surfaced per run when execution actually pauses; there is no global Docker approval toggle

Optional observability profile:

```bash
docker compose --profile observability up --build -d
```

This additionally starts:

- `clickhouse`
- `minio`
- `langfuse-server`
- `langfuse-worker`

Langfuse is exposed at `http://127.0.0.1:3001`.

## Manual Startup

Use manual startup when running services outside Docker Compose.

If you copied `.env.example`, override the Docker-only hostnames before starting backend or worker on the host:

```bash
export AUTH_SECRET_KEY='replace-with-at-least-32-random-chars'
export DATABASE_URL='postgresql+psycopg://agframe:agframe_secret@127.0.0.1:5432/agframe'
export REDIS_URL='redis://:redissecret@127.0.0.1:6379/0'
```

For a local smoke path without cloud credentials, also override:

```bash
export LLM_MODEL='dev-stub'
export MODEL_PATH_EMBEDDING='dev-stub'
```

Backend:

```bash
./scripts/start-backend.sh
```

Worker:

```bash
./scripts/start-worker.sh
```

Frontend:

```bash
cd frontend
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

Manual startup requires both backend and worker to be healthy for harness runs, document ingestion, and resume flows.

## Shutdown

```bash
docker compose down
```

## Post-Deployment Checks

- `GET /health`
- `GET /health/ready`
- `GET /health/live`
- open `http://127.0.0.1:3000/login`
- confirm worker logs show no startup import or Redis connectivity failures
