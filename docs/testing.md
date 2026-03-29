# Testing Guide

<div align="center">
  <a href="testing-cn.md">中文文档</a>
</div>

## Scope

- use `./.venv/bin/python` for Python test commands when available
- use Node.js 22 LTS for frontend validation
- treat the harness regression suite as the primary backend gate for runtime changes

## Quick Test Run

```bash
./.venv/bin/python -m pytest -v --tb=short
```

Useful install notes:

- evaluation tests require `uv sync --group evals`
- advanced document parsing requires `uv sync --group document-ai`
- the default lightweight retrieval path does not require a model reranker

## Targeted Harness Regression

```bash
./.venv/bin/python -m pytest tests/test_api_chat.py tests/test_api_harness.py tests/test_api_harness_approval.py tests/test_api_interrupt.py tests/test_graph_resume_service.py tests/test_harness_arq_jobs.py tests/test_harness_run_service.py tests/test_harness_approval_service.py tests/test_workbench_smoke.py -q
```

This is the maintained regression set for:

- workbench invoke persistence
- interrupt approval and resume
- harness run lifecycle
- harness approval and verification behavior
- worker-driven harness execution

## Workbench Smoke Test

```bash
./scripts/smoke_workbench.sh
```

This covers the local workbench flow, including upload, queueing, parsing, `/chat/workbench-invoke`, and interrupt approve or resume behavior.

## Live Smoke Against Running Services

```bash
./scripts/live_workbench_smoke.sh --base-url http://127.0.0.1:8000
```

Prerequisites:

- backend and worker are running
- PostgreSQL and Redis are reachable
- current config allows upload, parsing, and harness scheduling
- if `.env` still sets `LLM_MODEL=gpt-4o-mini`, provide a valid `LLM_API_KEY` or change `.env` to a stub model before startup

Optional switches:

- `--skip-upload`
- `--skip-chat`
- `--skip-interrupt`
- `--exercise-reject`
- `--task-timeout <seconds>`

When debugging approval or resume behavior, also inspect:

- `/harness/runs/{run_id}/events`
- `/interrupt/{session_id}/events`

## Infrastructure Integration Tests

```bash
export DATABASE_URL='postgresql+psycopg://<user>:<password>@<host>:<port>/<db>'
export REDIS_URL='redis://:<password>@<host>:6379/0'
./.venv/bin/python -m pytest tests/test_integration_postgres_store.py tests/test_integration_queue_redis.py -v --tb=short
```

## Full Test Gate

```bash
./scripts/run_test_suite.sh
```

To include the live smoke stage:

```bash
export LIVE_SMOKE_BASE_URL='http://127.0.0.1:8000'
./scripts/run_test_suite.sh
```

Supported environment variables:

- `LIVE_SMOKE_SKIP_UPLOAD=1`
- `LIVE_SMOKE_SKIP_CHAT=1`
- `LIVE_SMOKE_SKIP_INTERRUPT=1`
- `LIVE_SMOKE_EXERCISE_REJECT=1`
- `LIVE_SMOKE_TASK_TIMEOUT=<seconds>`

## Frontend Validation

```bash
cd frontend
export NEXT_PUBLIC_API_URL='http://127.0.0.1:8000'
npm run lint -- --max-warnings=0
npm run build
```

## Generated Reports

- `./reports/test_suite/<timestamp>/pytest.json`
- `./reports/test_suite/<timestamp>/harness_regression.log`
- `./reports/test_suite/<timestamp>/coverage.xml`
- `./reports/test_suite/<timestamp>/perf.json`
- `./reports/test_suite/<timestamp>/context_pruning_eval.json`
- `./reports/test_suite/<timestamp>/security.json`
- `./reports/test_suite/<timestamp>/live_smoke.log`
- `./reports/test_suite/<timestamp>/frontend_lint.log`
- `./reports/test_suite/<timestamp>/frontend_build.log`
- `./reports/test_suite/<timestamp>/report.md`
- `./reports/test_suite/<timestamp>/defects.md`
