# Testing Guide

## Quick Unit Test Run

```bash
uv python install 3.11.15
uv run pytest -v --tb=short
```

Notes:

- evaluation dependencies are optional
- install `uv sync --group evals` for evaluation-focused runs
- install `uv sync --group local-inference --group document-ai` for local embeddings, OCR, or advanced document parsing
- the default document RAG, memory retrieval, and pruning paths do not require a local reranker
- Node 22 LTS is recommended for frontend verification

## Targeted Regression

```bash
uv run pytest tests/test_settings_security.py tests/test_api_misc.py -v --tb=short
```

## Workbench Smoke

```bash
uv run ./scripts/smoke_workbench.sh
```

Or, inside an activated virtual environment:

```bash
./scripts/smoke_workbench.sh
```

This covers the main upload, task, document, conversation, and memory APIs.

## Live Smoke Against a Running Service

```bash
uv run ./scripts/live_workbench_smoke.sh --base-url http://127.0.0.1:8000
```

Prerequisites:

- service is running
- PostgreSQL and Redis are reachable
- current config allows upload and indexing

## Infrastructure Integration Tests

```bash
export DATABASE_URL='postgresql+psycopg://<user>:<password>@<host>:<port>/<db>'
export REDIS_URL='redis://:<password>@<host>:6379/0'
uv run pytest tests/test_integration_postgres_store.py tests/test_integration_queue_redis.py -v --tb=short
```

## Full Test Gate

```bash
uv run ./scripts/run_test_suite.sh
```

To include live smoke:

```bash
export LIVE_SMOKE_BASE_URL='http://127.0.0.1:8000'
uv run ./scripts/run_test_suite.sh
```

Frontend validation:

```bash
cd frontend
export NEXT_PUBLIC_API_URL='http://127.0.0.1:8000'
npm install
npm run lint -- --max-warnings=0
npx next build
```

Generated reports:

- `./reports/test_suite/<timestamp>/pytest.json`
- `./reports/test_suite/<timestamp>/coverage.xml`
- `./reports/test_suite/<timestamp>/perf.json`
- `./reports/test_suite/<timestamp>/context_pruning_eval.json`
- `./reports/test_suite/<timestamp>/security.json`
- `./reports/test_suite/<timestamp>/live_smoke.log`
- `./reports/test_suite/<timestamp>/report.md`
- `./reports/test_suite/<timestamp>/defects.md`
