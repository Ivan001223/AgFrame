# 测试最小文档

## 1. 快速单测

```bash
uv run pytest -v --tb=short
```

说明：

- `ragas` 等评测依赖安装在开发环境中，不作为服务运行时必需依赖
- 前端门禁建议使用 Node 22 LTS

## 2. 指定模块回归

```bash
uv run pytest tests/test_settings_security.py tests/test_api_misc.py -v --tb=short
```

## 3. 工作台主链路 Smoke

```bash
uv run ./scripts/smoke_workbench.sh
```

或在已激活的 `.venv` 中直接执行：

```bash
./scripts/smoke_workbench.sh
```

覆盖上传、任务、文档、会话、记忆的主接口流。

## 4. 运行中服务 Live Smoke

```bash
uv run ./scripts/live_workbench_smoke.sh --base-url http://127.0.0.1:8000
```

或：

```bash
./scripts/live_workbench_smoke.sh --base-url http://127.0.0.1:8000
```

前提：

- 服务已启动
- 数据库和 Redis 可用
- 当前配置允许上传和索引

## 5. 基础设施集成测试

```bash
export DATABASE_URL='postgresql+psycopg://<user>:<password>@<host>:<port>/<db>'
export REDIS_URL='redis://:<password>@<host>:6379/0'
uv run pytest tests/test_integration_postgres_store.py tests/test_integration_queue_redis.py -v --tb=short
```

说明：

- `tests/test_integration_postgres_store.py` 需要可访问的 PostgreSQL
- `tests/test_integration_queue_redis.py` 需要可访问的 Redis
- 环境不可达时，相关测试会显式 `skip`

## 6. 一键测试门禁

```bash
uv run ./scripts/run_test_suite.sh
```

如需把运行中服务的工作台 live smoke 一并纳入门禁：

```bash
export LIVE_SMOKE_BASE_URL='http://127.0.0.1:8000'
uv run ./scripts/run_test_suite.sh
```

前端单独验证：

```bash
cd frontend
export NEXT_PUBLIC_API_URL='http://127.0.0.1:8000'
npm install
npm run lint -- --max-warnings=0
npx next build
```

输出目录：

- `./reports/test_suite/<timestamp>/pytest.json`
- `./reports/test_suite/<timestamp>/coverage.xml`
- `./reports/test_suite/<timestamp>/security.json`
- `./reports/test_suite/<timestamp>/live_smoke.log`（仅当设置 `LIVE_SMOKE_BASE_URL` 时生成）
- `./reports/test_suite/<timestamp>/report.md`
- `./reports/test_suite/<timestamp>/defects.md`
