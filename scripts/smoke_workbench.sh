#!/bin/zsh
set -e

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
python -m pytest tests/test_workbench_smoke.py -v --tb=short
