# Testing Guide

<div align="center">
  <a href="testing-cn.md">中文文档</a>
</div>

## Quick Unit Test Run

```bash
./.venv/bin/python -m pytest -v --tb=short
```

Notes:
- If the local `uv` version does not match the constraints in `pyproject.toml`, prioritize using `.venv/bin/python` within the repository.
- Evaluation-related dependencies are optional.
- To run tests focused on evaluation, please first install `uv sync --group evals`.
- The default installation already includes local Embedding, OCR, Torch, and Transformers dependencies.
- If advanced document parsing capabilities are needed, please additionally install `uv sync --group document-ai`.
- The default document RAG, memory retrieval, and context pruning paths **do not depend** on a local large model Reranker.
- It is recommended to use Node 22 LTS for frontend validation.

## Targeted Regression

```bash
./.venv/bin/python -m pytest tests/test_settings_security.py tests/test_api_tasks.py tests/test_api_harness.py -v --tb=short
```

## Workbench Smoke Test

```bash
uv run ./scripts/smoke_workbench.sh
```

Or run directly in an activated virtual environment:

```bash
./scripts/smoke_workbench.sh
```

This covers the main workbench pipelines: file upload, task queues, document parsing, normal `/chat/workbench-invoke` persistence, and interrupt approve / reject / resume.
For the Harness approval flow, additionally run `tests/test_api_chat.py`, `tests/test_api_interrupt.py`, `tests/test_graph_resume_service.py`, `tests/test_api_harness.py`, `tests/test_api_harness_approval.py`, and `tests/test_harness_arq_jobs.py`.
If you want to verify the approval audit trail, you can additionally check `/harness/runs/{run_id}/events` and `/interrupt/{session_id}/events`, which will respectively expose the run lifecycle and session interrupt event streams.
If you want to verify whether the Harness control plane is fully connected, you can also check if `/harness/policies`, `POST /harness/runs`, and `POST /harness/runs/{run_id}/retry` are consistent with the frontend `/harness` page.

## Live Smoke Test Against a Running Service

```bash
uv run ./scripts/live_workbench_smoke.sh --base-url http://127.0.0.1:8000
```

Prerequisites:
- Backend service and Worker are running.
- PostgreSQL and Redis are accessible.
- Current configuration allows document upload, parsing, and Agent scheduling.
- The repository's default `docker compose` local orchestration will set `LLM_MODEL=dev-stub`, `MODEL_PATH_EMBEDDING=dev-stub`, and enable `ENABLE_HUMAN_APPROVAL=true`, facilitating the live smoke run without external API Keys.
- If you want to switch back to real cloud models, explicitly override `LLM_MODEL` and `LLM_API_KEY` before starting.
- By default, it will first verify the normal `/chat/workbench-invoke -> history` persistence, and then verify the `/chat -> interrupt -> approve -> resume -> history` pipeline.
- If you only want to skip the normal chat pipeline, add `--skip-chat`; if you only want to skip the approval resume pipeline, add `--skip-interrupt`.
- If you also want to additionally verify the reject branch, add `--exercise-reject`.
- If you need to troubleshoot harness / interrupt lifecycle issues, it is recommended to synchronously check the event interfaces in the live environment to confirm whether `approved`, `resume_requested`, `resumed`, or `resume_blocked` are persisted in order.

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

If the live smoke test needs to be included:

```bash
export LIVE_SMOKE_BASE_URL='http://127.0.0.1:8000'
./scripts/run_test_suite.sh
```

Optional gate switches:

```bash
export LIVE_SMOKE_BASE_URL='http://127.0.0.1:8000'
export LIVE_SMOKE_EXERCISE_REJECT=1
export LIVE_SMOKE_TASK_TIMEOUT=180
./scripts/run_test_suite.sh
```

Supported environment variables:
- `LIVE_SMOKE_SKIP_UPLOAD=1`
- `LIVE_SMOKE_SKIP_CHAT=1`
- `LIVE_SMOKE_SKIP_INTERRUPT=1`
- `LIVE_SMOKE_EXERCISE_REJECT=1`
- `LIVE_SMOKE_TASK_TIMEOUT=<seconds>`

Frontend validation:

```bash
cd frontend
export NEXT_PUBLIC_API_URL='http://127.0.0.1:8000'
npm install
npm run lint -- --max-warnings=0
npx next build
```

Generated report paths:

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