#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$PROJECT_ROOT/reports/test_suite/$TIMESTAMP"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN=("$PROJECT_ROOT/.venv/bin/python")
elif command -v uv >/dev/null 2>&1 && uv run python -c "import sys" >/dev/null 2>&1; then
  PYTHON_BIN=("uv" "run" "python")
else
  PYTHON_BIN=("python3")
fi

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

"${PYTHON_BIN[@]}" -m pytest \
  tests/test_main_import.py \
  tests/test_api_chat.py \
  tests/test_api_interrupt.py \
  tests/test_graph_resume_service.py \
  tests/test_harness_arq_jobs.py \
  tests/test_api_harness.py \
  tests/test_api_harness_approval.py \
  tests/test_harness_run_service.py \
  tests/test_harness_approval_service.py \
  tests/test_workbench_smoke.py \
  -q > "$OUT_DIR/harness_regression.log"

"${PYTHON_BIN[@]}" -m pytest \
  -s \
  --json-report \
  --json-report-file="$OUT_DIR/pytest.json" \
  --cov=app.runtime.prompts \
  --cov=app.runtime.llm.model_manager \
  --cov=app.infrastructure.utils \
  --cov=app.infrastructure.config.env \
  --cov=app.server.api \
  --cov-config=.coveragerc \
  --cov-report=term-missing \
  --cov-report=xml:"$OUT_DIR/coverage.xml" \
  --cov-fail-under=80

"${PYTHON_BIN[@]}" "$PROJECT_ROOT/scripts/perf_bench.py" --out "$OUT_DIR/perf.json"
"${PYTHON_BIN[@]}" "$PROJECT_ROOT/scripts/eval_context_pruning.py" --out "$OUT_DIR/context_pruning_eval.json"
"${PYTHON_BIN[@]}" "$PROJECT_ROOT/scripts/security_scan.py" --out "$OUT_DIR/security.json"
"${PYTHON_BIN[@]}" "$PROJECT_ROOT/scripts/generate_test_report.py" \
  --pytest-json "$OUT_DIR/pytest.json" \
  --coverage-xml "$OUT_DIR/coverage.xml" \
  --perf-json "$OUT_DIR/perf.json" \
  --context-pruning-eval-json "$OUT_DIR/context_pruning_eval.json" \
  --security-json "$OUT_DIR/security.json" \
  --out "$OUT_DIR/report.md" \
  --defects "$OUT_DIR/defects.md"

if [[ -n "${LIVE_SMOKE_BASE_URL:-}" ]]; then
  LIVE_SMOKE_CMD=(bash "$PROJECT_ROOT/scripts/live_workbench_smoke.sh" --base-url "$LIVE_SMOKE_BASE_URL")
  if [[ "${LIVE_SMOKE_SKIP_UPLOAD:-0}" == "1" ]]; then
    LIVE_SMOKE_CMD+=(--skip-upload)
  fi
  if [[ "${LIVE_SMOKE_SKIP_CHAT:-0}" == "1" ]]; then
    LIVE_SMOKE_CMD+=(--skip-chat)
  fi
  if [[ "${LIVE_SMOKE_SKIP_INTERRUPT:-0}" == "1" ]]; then
    LIVE_SMOKE_CMD+=(--skip-interrupt)
  fi
  if [[ "${LIVE_SMOKE_EXERCISE_REJECT:-0}" == "1" ]]; then
    LIVE_SMOKE_CMD+=(--exercise-reject)
  fi
  if [[ -n "${LIVE_SMOKE_TASK_TIMEOUT:-}" ]]; then
    LIVE_SMOKE_CMD+=(--task-timeout "$LIVE_SMOKE_TASK_TIMEOUT")
  fi
  "${LIVE_SMOKE_CMD[@]}" > "$OUT_DIR/live_smoke.log"
fi

if [[ -f "$PROJECT_ROOT/frontend/package.json" ]]; then
  if [[ -d "$PROJECT_ROOT/frontend/node_modules" ]]; then
    (
      cd "$PROJECT_ROOT/frontend"
      npm run lint -- --max-warnings=0 > "$OUT_DIR/frontend_lint.log"
      npm run build > "$OUT_DIR/frontend_build.log"
    )
  else
    echo "frontend checks skipped: node_modules not installed" > "$OUT_DIR/frontend_checks.log"
  fi
fi

echo "$OUT_DIR"
