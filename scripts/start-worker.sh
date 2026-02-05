#!/bin/bash
# AgFrame ARQ Worker 启动脚本
# 用法: ./scripts/start-worker.sh

set -e

cd "$(dirname "$0")/.."

echo "启动 ARQ Worker..."
arq app.infrastructure.queue.worker_settings
