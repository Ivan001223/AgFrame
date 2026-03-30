#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN=("$PROJECT_ROOT/.venv/bin/python")
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=("python3")
else
  echo "Python 3 is not available. Create the project virtualenv first." >&2
  exit 1
fi

exec "${PYTHON_BIN[@]}" -m app.server.main
