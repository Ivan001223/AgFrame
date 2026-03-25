#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "AgFrame Evaluation Runner"
echo "========================================"

if [[ -n "$(git diff --name-only -- 'tests/' ':!tests/fixtures/' 2>/dev/null)" ]]; then
  echo "Detected test changes, running tests..."
  cd "$PROJECT_ROOT"
  uv run python -m pytest tests/ -v --tb=short --color=yes
  echo "All tests passed"
else
  echo "No test file changes detected, skipping evaluation run"
fi

echo "========================================"
echo "Evaluation Complete"
echo "========================================"
