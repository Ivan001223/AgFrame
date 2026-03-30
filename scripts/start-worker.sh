#!/usr/bin/env bash
# AgFrame ARQ Worker 启动脚本
# 用法: ./scripts/start-worker.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/.venv/bin/arq" ]]; then
  ARQ_BIN=("$PROJECT_ROOT/.venv/bin/arq")
elif command -v arq >/dev/null 2>&1; then
  ARQ_BIN=("arq")
else
  echo "ARQ is not available. Create the project virtualenv first." >&2
  exit 1
fi

echo "启动 ARQ Worker..."
exec "${ARQ_BIN[@]}" app.infrastructure.queue.worker_settings.WorkerSettings
