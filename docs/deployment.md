# 部署指南 (Deployment Guide)

## 环境要求 (Requirements)

- Python 3.11.15
- `uv` (包管理器)
- Docker 和 Docker Compose
- 可访问的 PostgreSQL 与 Redis 端口

建议严格使用本仓库声明的 Python 版本。

## 安装步骤 (Install)

```bash
uv python install 3.11.15
uv sync --no-dev
cp configs/config.example.json configs/config.json
```

可选依赖组：

- `uv sync --no-dev --group local-inference`
  用于支持本地 Embeddings 模型、本地 OCR 或遗留的 Reranker 模型兼容。
- `uv sync --no-dev --group document-ai`
  用于支持更高精度的 PDF / Office 文档解析。

默认的文档 RAG、记忆检索与上下文裁剪路径**不需要**加载本地大模型 Reranker。

## 必填配置 (Required Configuration)

至少需要在 `config.json` 中设置以下内容：

- `auth.secret_key` (请生成长随机字符串)
- `database.url` 或同等的数据库凭证
- `database.password`
- `llm.api_key` (若使用云端模型提供商)

对于推荐的轻量级链路，请保持以下配置为空：

- `reranker.model_name=""`
- `local_models.rerank_model=""`

## 使用 Docker Compose 启动整套项目 (Start The Full Stack)

```bash
cp .env.example .env
docker compose up --build -d
docker compose ps
```

默认会启动：

- `postgres`
- `redis`（使用 Redis Stack，支持队列、限流与 LangGraph checkpoint）
- `backend`
- `worker`
- `frontend`

默认地址：

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

说明：

- 若要直接使用聊天能力，请在 `.env` 中填写 `LLM_API_KEY`
- 若要启用文档摄取 / RAG，仍建议在 `configs/config.json` 中补充 embeddings 配置或远程 embeddings 服务地址

如果需要额外的观测组件：

```bash
docker compose --profile observability up --build -d
```

这会额外启动：

- `clickhouse`
- `minio`
- `langfuse-server`
- `langfuse-worker`

Langfuse 默认地址为 `http://localhost:3001`。

## 启动主服务 (Start Service)

如果你不使用 Docker Compose，也可以沿用原来的手动方式。由于 `v0.2.1` 引入了 Harness 和 Agent 任务执行链路，需确保后端和 Worker 同时健康运行：

**启动后端 API 服务：**

```bash
uv run python -m app.server.main
```

默认地址：
- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`

**启动异步 Worker（必须）：**

```bash
uv run arq app.infrastructure.queue.worker_settings.WorkerSettings
```

## 停止服务 (Stop Service)

```bash
docker compose down
```
