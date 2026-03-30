# 部署指南

## 范围

- **后端版本**：`0.3.1`
- **前端版本**：`0.3.1`
- **Python 约束**：`>=3.11,<3.12`
- **本地默认栈**：Postgres + Redis + FastAPI + ARQ Worker + Next.js

## 环境要求

- Python 3.11
- `uv`
- 前端验证建议使用 Node.js 22 LTS
- Docker 和 Docker Compose
- 可访问的 PostgreSQL 与 Redis 端口

## 安装

Backend：

```bash
uv sync
```

Frontend：

```bash
cd frontend
npm install
```

可选依赖组：

- `uv sync --group document-ai`
- `uv sync --group evals`

## 必要配置

先准备本地配置文件：

```bash
cp .env.example .env
cp configs/config.example.json configs/config.json
```

至少需要为以下项设置安全值：

- `AUTH_SECRET_KEY`
- `DATABASE_URL` 或 `DB_HOST` / `DB_PORT` / `DB_USER` / `DB_PASSWORD` / `DB_NAME`
- `DB_PASSWORD`
- 使用云端模型时配置 `LLM_API_KEY`

需要注意的安全行为：

- 启动校验会拒绝不安全的 `auth.secret_key`
- 启动校验会拒绝不安全的 `database.password`
- `server.cors_allow_credentials=true` 不能与 `server.cors_origins=["*"]` 同时出现

轻量检索链路推荐保持以下兼容配置为空：

- `reranker.model_name=""`
- `local_models.rerank_model=""`

## Docker Compose 启动

```bash
docker compose up --build -d
docker compose ps
```

默认服务：

- `postgres`
- `redis`
- `backend`
- `worker`
- `frontend`

默认访问地址：

- Frontend: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

环境变量行为说明：

- `.env.example` 现在默认写入 `LLM_MODEL=dev-stub` 与 `MODEL_PATH_EMBEDDING=dev-stub`，因此本地启动默认不依赖云端 Key
- 如果需要真实云端生成，再把 `.env` 改成云模型并补齐 `LLM_API_KEY`
- `ENABLE_HUMAN_APPROVAL` 在 Docker Compose 中默认开启

如需额外观测组件：

```bash
docker compose --profile observability up --build -d
```

这会额外启动：

- `clickhouse`
- `minio`
- `langfuse-server`
- `langfuse-worker`

Langfuse 默认暴露在 `http://127.0.0.1:3001`。

## 手动启动

不使用 Docker Compose 时，可按以下方式手动启动。

如果你是从 `.env.example` 复制得到 `.env`，在宿主机直接启动 backend 或 worker 之前，请先覆盖其中仅适用于 Docker 网络的主机名：

```bash
export AUTH_SECRET_KEY='replace-with-at-least-32-random-chars'
export DATABASE_URL='postgresql+psycopg://agframe:agframe_secret@127.0.0.1:5432/agframe'
export REDIS_URL='redis://:redissecret@127.0.0.1:6379/0'
```

如果希望在本地走无云端依赖的 smoke 路径，还需要额外覆盖：

```bash
export LLM_MODEL='dev-stub'
export MODEL_PATH_EMBEDDING='dev-stub'
```

Backend：

```bash
./scripts/start-backend.sh
```

Worker：

```bash
./scripts/start-worker.sh
```

Frontend：

```bash
cd frontend
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

手动启动时，backend 与 worker 必须同时健康运行，否则 harness run、文档入库和 resume 链路都无法完整工作。

## 停止服务

```bash
docker compose down
```

## 部署后检查

- `GET /health`
- `GET /health/ready`
- `GET /health/live`
- 打开 `http://127.0.0.1:3000/login`
- 确认 worker 日志中没有启动导入失败或 Redis 连接失败
