# 部署指南

<div align="center">
  <a href="deployment.md">English</a>
</div>

## 范围

- **后端版本**：`0.3.1`
- **前端版本**：`0.3.1`
- **Python 约束**：`>=3.12,<3.13`
- **本地默认栈**：Postgres + Redis + FastAPI + 3 类 ARQ Worker + Next.js

## 环境要求

- Python 3.12
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
- 当前端从 `http://127.0.0.1:3000` 访问后端时，请保持 `CORS_ALLOW_CREDENTIALS=true`，否则浏览器不会携带认证 Cookie

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
- `worker-ingest`
- `worker-resume`
- `frontend`

默认访问地址：

- Frontend: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`

环境变量行为说明：

- `.env.example` 现在默认写入 `LLM_MODEL=dev-stub` 与 `MODEL_PATH_EMBEDDING=dev-stub`，因此本地启动默认不依赖云端 Key
- 如果需要真实云端生成，再把 `.env` 改成云模型并补齐 `LLM_API_KEY`
- chat 与 harness 的审批现在按运行时实际暂停来显式展示，不再提供全局 Docker 审批开关

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
export CORS_ORIGINS='["http://127.0.0.1:3000","http://localhost:3000"]'
export CORS_ALLOW_CREDENTIALS='true'
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

Workers：

```bash
./scripts/start-worker.sh
```

该脚本会并行拉起：

- `IngestWorkerSettings`，负责文档入库任务
- `RuntimeWorkerSettings`，负责 harness run 执行
- `ResumeWorkerSettings`，负责 interrupt 与 harness resume 任务

Frontend：

```bash
cd frontend
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

手动启动时，backend 与三类 worker 都必须健康运行，否则 harness run、文档入库和 resume 链路都无法完整工作。

认证说明：

- 前端现在依赖 `POST /auth/token` 下发的 HttpOnly Cookie，而不是把 JWT 持久化到 `localStorage`
- 页面加载后的当前用户恢复依赖 `GET /auth/users/me`
- 当前端和后端分端口或分源运行时，必须打开 `CORS_ALLOW_CREDENTIALS=true`，否则浏览器会在鉴权前直接拦截请求

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
