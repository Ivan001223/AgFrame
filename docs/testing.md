# 测试指南 (Testing Guide)

## 快速单元测试 (Quick Unit Test Run)

```bash
uv python install 3.11.15
uv run pytest -v --tb=short
```

注意：
- 评测相关的依赖为可选项。
- 运行专注评测的测试，请先安装 `uv sync --group evals`。
- 如果需要测试本地 Embedding、OCR 或是高级文档解析能力，请安装 `uv sync --group local-inference --group document-ai`。
- 默认的文档 RAG、记忆检索和裁剪路径**不依赖**本地大模型 Reranker。
- 建议使用 Node 22 LTS 进行前端校验。

## 核心回归测试 (Targeted Regression)

```bash
uv run pytest tests/test_settings_security.py tests/test_api_misc.py -v --tb=short
```

## 工作台冒烟测试 (Workbench Smoke)

```bash
uv run ./scripts/smoke_workbench.sh
```

或者在已激活的虚拟环境中直接运行：

```bash
./scripts/smoke_workbench.sh
```

这覆盖了主要的文件上传、任务队列、文档解析、对话流、长期记忆以及新增的 Harness 审批流 APIs。

## 针对运行中服务的活体冒烟测试 (Live Smoke Against a Running Service)

```bash
uv run ./scripts/live_workbench_smoke.sh --base-url http://127.0.0.1:8000
```

前置条件：
- 后端服务与 Worker 正在运行。
- PostgreSQL 和 Redis 处于可访问状态。
- 当前配置允许执行文档上传、解析和 Agent 调度。

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
uv run ./scripts/run_test_suite.sh
```

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
- `./reports/test_suite/<timestamp>/coverage.xml`
- `./reports/test_suite/<timestamp>/perf.json`
- `./reports/test_suite/<timestamp>/context_pruning_eval.json`
- `./reports/test_suite/<timestamp>/security.json`
- `./reports/test_suite/<timestamp>/live_smoke.log`
- `./reports/test_suite/<timestamp>/report.md`
- `./reports/test_suite/<timestamp>/defects.md`
