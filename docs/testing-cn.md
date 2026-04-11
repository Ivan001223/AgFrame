# 测试指南

<div align="center">
  <a href="testing.md">English</a>
</div>

## 范围

- Python 测试优先使用 `./.venv/bin/python`
- 前端校验建议使用 Node.js 22 LTS
- Harness 回归集是运行时相关改动的主要后端门禁

## 快速测试

```bash
./.venv/bin/python -m pytest -v --tb=short
```

安装说明：

- 评测相关测试需要先执行 `uv sync --group evals`
- 高级文档解析能力需要先执行 `uv sync --group document-ai`
- 默认轻量检索链路不依赖模型重排器

## Harness 定向回归

```bash
./.venv/bin/python -m pytest tests/test_api_chat.py tests/test_api_harness.py tests/test_api_harness_approval.py tests/test_api_interrupt.py tests/test_graph_resume_service.py tests/test_harness_arq_jobs.py tests/test_harness_run_service.py tests/test_harness_approval_service.py tests/test_workbench_smoke.py -q
```

这组回归覆盖：

- workbench invoke 持久化
- interrupt 审批与恢复
- harness run 生命周期
- harness approval 与 verification 行为
- worker 驱动的 harness 执行

## 工作台冒烟测试

```bash
./scripts/smoke_workbench.sh
```

该脚本覆盖本地工作台主流程，包括上传、排队、解析、`/chat/workbench-invoke` 以及 interrupt approve 或 resume。

## 面向运行中服务的 Live Smoke

```bash
./scripts/live_workbench_smoke.sh --base-url http://127.0.0.1:8000
```

前置条件：

- backend 与 worker 正在运行
- PostgreSQL 与 Redis 可访问
- 当前配置允许上传、解析与 harness 调度
- 如果你把 `.env` 改回云模型，则需要在启动前提供有效的 `LLM_API_KEY`

可选参数：

- `--skip-upload`
- `--skip-chat`
- `--skip-interrupt`
- `--exercise-reject`
- `--task-timeout <seconds>`

排查审批或恢复问题时，建议同步查看：

- `/harness/runs/{run_id}/events`
- `/interrupt/{session_id}/events`

## 基础设施集成测试

```bash
export DATABASE_URL='postgresql+psycopg://<user>:<password>@<host>:<port>/<db>'
export REDIS_URL='redis://:<password>@<host>:6379/0'
./.venv/bin/python -m pytest tests/test_integration_postgres_store.py tests/test_integration_queue_redis.py -v --tb=short
```

## 完整测试门禁

```bash
./scripts/run_test_suite.sh
```

如需包含 live smoke：

```bash
export LIVE_SMOKE_BASE_URL='http://127.0.0.1:8000'
./scripts/run_test_suite.sh
```

支持的环境变量：

- `LIVE_SMOKE_SKIP_UPLOAD=1`
- `LIVE_SMOKE_SKIP_CHAT=1`
- `LIVE_SMOKE_SKIP_INTERRUPT=1`
- `LIVE_SMOKE_EXERCISE_REJECT=1`
- `LIVE_SMOKE_TASK_TIMEOUT=<seconds>`

## 前端校验

```bash
cd frontend
export NEXT_PUBLIC_API_URL='http://127.0.0.1:8000'
npm run lint -- --max-warnings=0
npm run typecheck
npm run build
```

## 产物路径

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
