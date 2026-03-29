# 测试指南 (Testing Guide)

## 快速单元测试 (Quick Unit Test Run)

```bash
./.venv/bin/python -m pytest -v --tb=short
```

注意：
- 如果本地 `uv` 版本与 `pyproject.toml` 中的约束不匹配，优先使用仓库内的 `.venv/bin/python`。
- 评测相关的依赖为可选项。
- 运行专注评测的测试，请先安装 `uv sync --group evals`。
- 默认安装已经包含本地 Embedding、OCR、Torch 与 Transformers 依赖。
- 如果需要高级文档解析能力，请额外安装 `uv sync --group document-ai`。
- 默认的文档 RAG、记忆检索和裁剪路径**不依赖**本地大模型 Reranker。
- 建议使用 Node 22 LTS 进行前端校验。

## 核心回归测试 (Targeted Regression)

```bash
./.venv/bin/python -m pytest tests/test_settings_security.py tests/test_api_tasks.py tests/test_api_harness.py -v --tb=short
```

## 工作台冒烟测试 (Workbench Smoke)

```bash
uv run ./scripts/smoke_workbench.sh
```

或者在已激活的虚拟环境中直接运行：

```bash
./scripts/smoke_workbench.sh
```

这覆盖了主要的文件上传、任务队列、文档解析、普通 `/chat/workbench-invoke` 持久化，以及 interrupt 的 approve / reject / resume 工作台链路。
Harness 审批流请额外运行 `tests/test_api_chat.py`、`tests/test_api_interrupt.py`、`tests/test_graph_resume_service.py`、`tests/test_api_harness.py`、`tests/test_api_harness_approval.py` 与 `tests/test_harness_arq_jobs.py`。
如果要核对审批审计轨迹，可额外查看 `/harness/runs/{run_id}/events` 与 `/interrupt/{session_id}/events`，它们会分别暴露 run lifecycle 和 session interrupt 的事件流。
如果要核对 Harness 控制面是否完整接通，还可以检查 `/harness/policies`、`POST /harness/runs` 与 `POST /harness/runs/{run_id}/retry` 是否和前端 `/harness` 页面一致。

## 针对运行中服务的活体冒烟测试 (Live Smoke Against a Running Service)

```bash
uv run ./scripts/live_workbench_smoke.sh --base-url http://127.0.0.1:8000
```

前置条件：
- 后端服务与 Worker 正在运行。
- PostgreSQL 和 Redis 处于可访问状态。
- 当前配置允许执行文档上传、解析和 Agent 调度。
- 仓库默认 `docker compose` 本地编排会将 `LLM_MODEL=dev-stub`、`MODEL_PATH_EMBEDDING=dev-stub` 且开启 `ENABLE_HUMAN_APPROVAL=true`，便于在没有外部 API Key 时跑通 live smoke。
- 若要切回真实云端模型，请在启动前显式覆盖 `LLM_MODEL` 和 `LLM_API_KEY`。
- 默认会先验证普通 `/chat/workbench-invoke -> history` 持久化，再验证 `/chat -> interrupt -> approve -> resume -> history` 链路。
- 若只想跳过普通聊天链路，可加 `--skip-chat`；若只想跳过审批恢复链路，可加 `--skip-interrupt`。
- 若还想额外验证 reject 分支，可加 `--exercise-reject`。
- 若需要排查 harness / interrupt 生命周期问题，建议在活体环境里同步检查事件接口，确认 `approved`、`resume_requested`、`resumed` 或 `resume_blocked` 是否按顺序落库。

## 基础设施集成测试 (Infrastructure Integration Tests)

```bash
export DATABASE_URL='postgresql+psycopg://<user>:<password>@<host>:<port>/<db>'
export REDIS_URL='redis://:<password>@<host>:6379/0'
uv run pytest tests/test_integration_postgres_store.py tests/test_integration_queue_redis.py -v --tb=short
```

## 完整测试门禁 (Full Test Gate)

```bash
uv run ./scripts/run_test_suite.sh
```

如果需要包含活体冒烟测试：

```bash
export LIVE_SMOKE_BASE_URL='http://127.0.0.1:8000'
./scripts/run_test_suite.sh
```

可选门禁开关：

```bash
export LIVE_SMOKE_BASE_URL='http://127.0.0.1:8000'
export LIVE_SMOKE_EXERCISE_REJECT=1
export LIVE_SMOKE_TASK_TIMEOUT=180
./scripts/run_test_suite.sh
```

支持的环境变量：
- `LIVE_SMOKE_SKIP_UPLOAD=1`
- `LIVE_SMOKE_SKIP_CHAT=1`
- `LIVE_SMOKE_SKIP_INTERRUPT=1`
- `LIVE_SMOKE_EXERCISE_REJECT=1`
- `LIVE_SMOKE_TASK_TIMEOUT=<seconds>`

前端校验：

```bash
cd frontend
export NEXT_PUBLIC_API_URL='http://127.0.0.1:8000'
npm install
npm run lint -- --max-warnings=0
npx next build
```

生成的报告路径：

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
