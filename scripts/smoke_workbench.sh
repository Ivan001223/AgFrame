#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN=("$PROJECT_ROOT/.venv/bin/python")
elif command -v uv >/dev/null 2>&1 && uv run python -c "import sys" >/dev/null 2>&1; then
  PYTHON_BIN=("uv" "run" "python")
else
  PYTHON_BIN=("python3")
fi

"${PYTHON_BIN[@]}" -m pytest tests/test_workbench_smoke.py -v --tb=short
