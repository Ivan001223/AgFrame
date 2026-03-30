<div align="center">
  <h1>🚀 AgFrame</h1>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-cyan?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-0.3+-FF6B6B?style=flat-square&logoColor=white" alt="LangGraph">
  <img src="https://img.shields.io/badge/Version-0.3.1-blue?style=flat-square" alt="Version">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
</div>

<div align="center">
  <a href="./README-CN.md">中文文档</a>
</div>

<div align="center">
  <b>Backend-owned chat runtime, harness control plane, and Agent Studio workbench</b>
</div>

---

![AgFrame Banner](data/assets/banner.png)

## What AgFrame Is

AgFrame is a full-stack agent platform built around three coordinated layers:

- **Chat runtime**: FastAPI + LangGraph powers the main conversation path and persists turns through a backend-owned workbench invoke flow.
- **Harness control plane**: run creation, approval, verification, retry, event evidence, runtime-state history, and model-provider management live under `/harness`.
- **Workbench frontend**: a Next.js application provides chat, knowledge, conversations, memory, tasks, settings, and a dedicated Harness Agent Studio.

The current repository state is centered on a controllable agent runtime rather than a demo-only RAG chatbot.

## Architecture Snapshot

```mermaid
flowchart TD
    FE[Next.js Workbench<br/>Chat / Knowledge / Memory / Tasks / Agent Studio]
    API[FastAPI Backend<br/>REST APIs / LangServe / Auth / Rate Limit]
    CHAT["Chat Runtime<br/>/chat/workbench-invoke"]
    GRAPH[LangGraph Runtime]
    INTERRUPT["Interrupt Flow<br/>Approval / Resume"]
    HARNESS["Harness Control Plane<br/>Runs / Verification / Policies / Studio"]
    QUEUE[Redis + ARQ Worker]
    REDIS[Redis Checkpoints]
    PG[Postgres Persistence]

    FE --> API
    API --> CHAT
    CHAT --> GRAPH
    GRAPH --> REDIS
    API --> INTERRUPT
    INTERRUPT --> GRAPH
    API --> HARNESS
    HARNESS --> QUEUE
    HARNESS --> PG
    CHAT --> PG
```

### Runtime shape

- **Backend API**: `app/server/main.py` mounts FastAPI routers, LangServe `/chat` routes, static file serving, auth, and rate limiting.
- **Chat flow**: `POST /chat/workbench-invoke` is the main workbench entry. The backend applies runtime config, invokes the graph, reads the latest state, and persists messages.
- **Harness flow**: `app/server/api/harness.py` exposes runs, studio projects, skill requests, approvals, verification, runtime-state history, and model providers.
- **Queue execution**: ARQ workers execute document ingestion, harness runs, and harness resume jobs.
- **Frontend**: `frontend/` uses Next.js App Router, React Query, shared HTTP utilities, and domain-specific hooks.

### Retrieval defaults

```text
Query
  -> Dense Search + BM25 Search
  -> RRF Fusion
  -> Candidate Pruning
  -> Parent Restore
  -> Prompt Assembly
```

Recommended defaults:

- Keep `Dense + BM25 + RRF` on the main retrieval path
- Use lightweight pruning before prompt assembly
- Treat `reranker.*` as compatibility-only configuration, not the default path

## Repository Map

```text
app/
  server/           FastAPI entry, REST APIs, chat runtime integration
  runtime/          LangGraph graph, state, prompts, resume service
  harness/          Harness contracts, persistence, runtime services
  infrastructure/   config, database, queue, checkpoint, utilities
  memory/           long-term memory engines and pgvector integration
frontend/           Next.js workbench and Agent Studio
docs/               formal product and engineering documentation
scripts/            smoke tests, security checks, report generation
tests/              API, harness, runtime, and smoke regression tests
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
cd frontend && npm install
```

Optional dependency groups:

- `uv sync --group document-ai`
- `uv sync --group evals`

### 2. Prepare local configuration

```bash
cp .env.example .env
cp configs/config.example.json configs/config.json
```

Minimum secure changes before startup:

- `AUTH_SECRET_KEY`
- `DATABASE_URL` or `DB_*`
- `DB_PASSWORD`
- `LLM_API_KEY` when using a cloud model

Notes:

- `docker-compose.yml` falls back to `LLM_MODEL=dev-stub` only when `.env` does not override it.
- `.env.example` currently sets `LLM_MODEL=gpt-4o-mini`, so local live smoke without a cloud key requires adjusting `.env`.

### 3. Start the full stack with Docker Compose

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

Optional observability profile:

```bash
docker compose --profile observability up --build -d
```

### 4. Start services manually

Backend:

```bash
./.venv/bin/python -m app.server.main
```

Worker:

```bash
./.venv/bin/arq app.infrastructure.queue.worker_settings.WorkerSettings
```

Frontend:

```bash
cd frontend
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

## Main Capabilities

- **Chat workbench**: backend-owned invoke flow with session persistence and interrupt-aware resume support
- **Harness runs**: create runs, inspect events, review verification evidence, resolve approvals, and retry
- **Harness Agent Studio**: manage projects, edit graph JSON, request skills, resolve skill approvals, and launch orchestration runs
- **Knowledge ingestion**: upload files, queue parsing/indexing, reindex documents, and inspect task status
- **Memory and history**: maintain long-term profile memory, atomic memory items, and conversation history
- **Model provider management**: create, update, list, and delete harness model providers

## Health and Runtime Signals

Key health endpoints:

- `GET /health`
- `GET /health/ready`
- `GET /health/live`

The ready check reports retrieval and runtime component readiness, including the lightweight retrieval path and dependencies such as DB and Redis.

## Documentation Map

- [Deployment Guide](./docs/deployment.md)
- [API Reference](./docs/api.md)
- [Testing Guide](./docs/testing.md)
- [Security Notes](./docs/security.md)
- [Frontend Architecture](./docs/frontend-architecture.md)
- [RAG Architecture](./docs/rag-architecture.md)
- [Documentation Governance](./docs/documentation-governance.md)

## Version Scope

- Backend version: `0.3.1`
- Frontend version: `0.3.1`
- Python constraint: `>=3.11,<3.12`
- This README is aligned with the current repository state on 2026-03-30
