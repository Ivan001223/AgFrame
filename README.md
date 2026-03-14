# 🚀 AgFrame (Agent Framework)

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-cyan?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-FF6B6B?style=flat-square&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)

**⚡ 生产级 Agent/RAG 后端框架 | 基于 FastAPI + LangGraph 构建**  
专注于复杂工作流编排、混合检索、分层记忆与可观测性

---

```text
  █████   ██████  ███████ ██████   █████  ███    ███ ███████
 ██   ██ ██       ██      ██   ██ ██   ██ ████  ████ ██
 ███████ ██   ███ █████   ██████  ███████ ██ ████ ██ █████
 ██   ██ ██    ██ ██      ██   ██ ██   ██ ██  ██  ██ ██
 ██   ██  ██████  ██      ██   ██ ██   ██ ██      ██ ███████
```

---

## ✨ 核心特性速览

| 🏗️ 工作流编排 | 🧠 混合 RAG | 💾 分层记忆 | 🔮 LLM 工厂 | 📊 可观测性 | 🛠️ 基础设施 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| LangGraph | 双路检索 | 短期+长期 | 多模型支持 | Langfuse | Redis |
| Checkpoint | RRF 融合 | pgvector | 嵌入模型 | DeepEval | PostgreSQL |
| Human-in-Loop | 重排序 | 用户画像 | 结构化输出 | 任务诊断 | Docker |

---

## 📐 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│                  (Auth / REST / LangServe)                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Runtime                        │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│   │ Orchestr │  │  State   │  │  Nodes   │  │ Routers  │  │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Skills / Services                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
│  │  RAG   │ │ Memory │ │ Profile│ │ Research│ │ Tools  │    │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐    │
│  │ pgvector │ │  Redis   │ │ PostgreSQL│ │ Langfuse     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 📂 目录结构

```
╭────────────────────────────────────────────────────────────────────────────╮
│                              AgFrame /                                     │
╰────────────────────────────────────────────────────────────────────────────╯
│
├── app/
│   ├── server/              # 🚀 FastAPI 入口
│   │   ├── api/             #   路由层
│   │   └── main.py          #   应用启动
│   ├── runtime/             # ⚙️ 运行时核心
│   │   ├── graph/           #   LangGraph 工作流
│   │   │   ├── graph.py     #   图定义
│   │   │   ├── state.py     #   State Schema
│   │   │   ├── orchestrator.py # 编排器
│   │   │   ├── registry.py  #   节点注册表
│   │   │   └── nodes/       #   节点实现
│   │   ├── llm/             #   LLM 工厂
│   │   └── prompts/         #   Prompt 模板
│   ├── skills/              # 🛠️ 原子能力层
│   │   ├── rag/             #   混合检索
│   │   ├── memory/          #   记忆检索技能
│   │   ├── profile/         #   用户画像
│   │   ├── research/        #   网络搜索
│   │   ├── ocr/             #   图片 OCR
│   │   ├── common/          #   公共技能
│   │   └── tools/           #   代码执行
│   ├── infrastructure/      # 🏗️ 基础设施
│   │   ├── config/          #   配置管理
│   │   ├── database/        #   SQLAlchemy ORM
│   │   ├── checkpoint/      #   Redis Checkpoint
│   │   ├── queue/           #   ARQ 异步任务
│   │   ├── sandbox/         #   代码沙箱
│   │   ├── observability/   #   可观测性
│   │   └── utils/           #   工具函数
│   ├── agents/              # 🤖 Agent 节点工厂
│   ├── memory/              # 🧠 记忆模块
│   │   ├── long_term/       #   长期记忆引擎
│   │   └── vector_stores/   #   向量存储 (pgvector)
│   └── examples/            # 🔬 调试脚本与示例
│
├── configs/                 # ⚙️ 配置文件
├── docker/                 # 🐳 Docker 初始化脚本
├── docs/                   # 📖 部署/安全/测试文档
├── frontend/               # 💻 Next.js 工作台前端
├── scripts/                # 🧰 工具脚本
├── data/                   # 📁 运行时数据
├── tests/                  # 🧪 单元测试
│
├── docker-compose.yml      # 🐳 基础设施编排
├── pyproject.toml         # 🐍 Python 项目配置
└── uv.lock                 # 🔒 依赖锁文件
```

## ⚡ 核心特性

### 🏗️ 工作流编排 (LangGraph)

| 特性 | 说明 |
|:---|:---|
| **Stateful Graph** | 支持循环、分支、条件判断 |
| **Checkpoint** | Redis 持久化断点恢复 |
| **Human-in-the-Loop** | 支持人工中断与反馈 |
| **节点注册表** | 动态节点管理 |

### 🧠 混合 RAG

| 特性 | 说明 |
|:---|:---|
| **双路检索** | BM25 (关键词) + Embedding (语义) |
| **重排序** | 支持 Cross-Encoder 重排序 |
| **多格式解析** | PDF/DOCX/Excel/图片 OCR |
| **RRF 融合** | RRF 排序融合算法 |

### 💾 分层记忆系统

| 特性 | 说明 |
|:---|:---|
| **短期记忆** | 对话上下文窗口管理 |
| **长期记忆** | 用户画像 + 历史向量存储 |
| **pgvector** | 向量检索持久化 |
| **对话管理** | 历史记录持久化 |

### 🔮 LLM 工厂

| 特性 | 说明 |
|:---|:---|
| **多模型支持** | OpenAI API / 本地 Qwen (Ollama/VLLM) |
| **嵌入模型** | Sentence-Transformers / ModelScope |
| **重排序** | Cross-Encoder |
| **结构化输出** | 原生 Pydantic + JSON 模式 |

### 📊 可观测性

| 特性 | 说明 |
|:---|:---|
| **Langfuse** | 全链路追踪、Prompt 管理 |
| **DeepEval** | RAG 评测 (Context Recall/Precision) |
| **任务运营** | 任务摘要、失败事件流、结构化诊断 |
| **事件处置** | 支持事件已处理/归档与筛选 |

### 🛠️ 基础设施

| 特性 | 说明 |
|:---|:---|
| **Redis** | 缓存 / Checkpoint / 任务队列 (ARQ) |
| **PostgreSQL** | 持久化存储 |
| **Docker** | 一键启动全部依赖 |
| **Sandbox** | 代码安全执行环境 |

## 🆕 最新进展

本轮迭代已补齐一组更接近生产可用的后端能力：

| 模块 | 新增功能 |
|:---|:---|
| 📚 **知识库管理** | 文档列表、搜索、详情、预览、删除、重建索引 |
| 💬 **会话中心** | 会话查询、详情、重命名、删除 |
| 🧠 **记忆控制台** | 画像查看/更新、长期记忆查看/新增/删除 |
| 📋 **任务运营** | 任务诊断、超时疑似标记、失败重试、事件流、事件归档/已处理 |
| ❤️ **健康检查** | 数据库、Redis、向量库、模型组件配置检查 |
| ✅ **真实依赖验证** | PostgreSQL、Redis、远端 vLLM embedding/reranker 已完成集成验证 |
| 🎯 **服务级验收** | 已提供 smoke 脚本，覆盖注册、登录、上传、索引、文档、会话、记忆主链路 |
| 💻 **前端工作台** | `/login`、`/chat`、`/knowledge`、`/conversations`、`/memory`、`/tasks`、`/settings`、`/admin/settings` 已实现并对齐后端 |
| 🛡️ **前端门禁** | Node 22 环境下 `npm run lint` 与 `npx next build` 已通过 |

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
uv sync
```

说明：

- 默认会创建 `.venv/` 并安装开发依赖
- 如需显式指定 Python 3.11，可先执行 `uv python install 3.11`
- 常用命令建议统一使用 `uv run ...`

### 2️⃣ 配置

```bash
cp configs/config.example.json configs/config.json
# 编辑 configs/config.json 配置各项参数
```

**配置结构示例**：

```json
{
  "llm": {
    "api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o"
  },
  "model_manager": {
    "provider": "modelscope",
    "cache_dir": "",
    "revision": "",
    "trust_remote_code": true,
    "modelscope_fallback_to_hf": true
  },
  "local_models": {
    "ocr_model": "",
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "rerank_model": "Qwen/Qwen3-Reranker-0.6B"
  },
  "embeddings": {
    "provider": "modelscope",
    "backend": "sentence_transformers",
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "base_url": "",
    "api_key": "",
    "timeout_seconds": 30,
    "env_var": "MODEL_PATH_EMBEDDING",
    "device": "auto"
  },
  "reranker": {
    "provider": "modelscope",
    "backend": "sentence_transformers",
    "model_name": "Qwen/Qwen3-Reranker-0.6B",
    "base_url": "",
    "api_key": "",
    "timeout_seconds": 30,
    "env_var": "MODEL_PATH_RERANKER",
    "device": "auto"
  },
  "search": {
    "provider": "duckduckgo",
    "tavily_api_key": ""
  },
  "database": {
    "type": "postgres",
    "url": "postgresql+psycopg://<DB_USER>:<DB_PASSWORD>@localhost:5432/<DB_NAME>",
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "<DB_PASSWORD>",
    "db_name": "agent_app"
  },
  "queue": {
    "redis_url": "redis://:redissecret@localhost:6379/0"
  },
  "storage_s3": {
    "s3_endpoint": "",
    "s3_access_key": "",
    "s3_secret_key": "",
    "s3_bucket": "agframe",
    "s3_secure": false
  },
  "auth": {
    "secret_key": "<AUTH_SECRET_KEY_AT_LEAST_32_CHARS>",
    "algorithm": "HS256",
    "access_token_expire_minutes": 30
  },
  "general": {
    "app_name": "My Agent App"
  },
  "rag": {
    "retrieval": {
      "mode": "hybrid",
      "dense_k": 20,
      "sparse_k": 20,
      "candidate_k": 20,
      "final_k": 3,
      "rrf_k": 60,
      "weights": [0.5, 0.5]
    }
  },
  "prompt": {
    "budget": {
      "max_recent_history_lines": 10,
      "max_docs": 3,
      "max_memories": 3
    }
  },
  "nodes": {
    "enabled": [
      "router",
      "retrieve_docs",
      "rerank_docs",
      "retrieve_memories",
      "assemble",
      "generate"
    ]
  },
  "feature_flags": {
    "enable_docs_rag": true,
    "enable_chat_memory": true,
    "enable_self_correction": true,
    "enable_human_approval": false,
    "pgvector_dimension": 1024
  },
  "sandbox": {
    "enabled": false,
    "image": "python:3.11-slim",
    "timeout": 30
  },
  "self_correction": {
    "max_attempts": 2
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "cors_origins": [],
    "cors_allow_credentials": false
  },
  "storage_local": {
    "documents_dir": "data/documents",
    "uploads_dir": "data/uploads",
    "data_dir": "data"
  }
}
```

完整样例请直接参考 `configs/config.example.json`。

**环境变量覆盖**：

敏感配置可通过环境变量覆盖：

```bash
export LLM_API_KEY="sk-xxx"
export DATABASE_URL="postgresql+psycopg://<DB_USER>:<DB_PASSWORD>@localhost:5432/<DB_NAME>"
export REDIS_URL="redis://:redissecret@localhost:6379/0"
```

说明：

- 远端 Hugging Face / ModelScope 模型下载建议显式设置 `revision`
- 前端建议使用 Node 22 LTS
- 常见敏感项优先用环境变量覆盖，完整映射见 `configs/config.example.json` 中 `env_overrides`

### 3️⃣ 启动依赖

```bash
# 复制环境变量模板 (如果存在，否则手动创建 .env)
# cp .env.example .env

# 启动所有基础设施
docker-compose up -d

# 验证服务状态
docker-compose ps
```

### 3️⃣.1 启动前端

```bash
cd frontend
export NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
npm install
npm run lint -- --max-warnings=0
npx next build
npm run dev
```

**启动的服务：**

| 🔌 服务 | 🚪 端口 | 📝 用途 |
|:---|:---:|:---|
| PostgreSQL + pgvector | 5432 | 主数据库 + 向量存储 |
| Redis | 6379 | 缓存、Checkpoint、任务队列 |
| RabbitMQ | 5672/15672 | 预留消息队列（当前 ARQ 不依赖） |
| ClickHouse | 8123 | Langfuse 指标存储 |
| MinIO | 9000/9001 | S3 对象存储 |
| Langfuse | 3000 | 可观测性追踪 |

### 4️⃣ 启动后端与 Worker

```bash
uv run python -m app.server.main
uv run arq app.infrastructure.queue.worker_settings
```

后端服务运行在 `http://localhost:8000`

- **Swagger Docs**: http://localhost:8000/docs

说明：

- 如果你启用了 MinIO/S3 存储，需要自行在 MinIO 中创建 `agframe` bucket
- 仓库当前未内置 `mc` 初始化脚本，建议通过 MinIO Console 或你自己的 `mc` 客户端完成

## ⏹️ 停止服务

```bash
# 停止所有基础设施
docker-compose down

# 停止并删除数据卷（慎用！）
docker-compose down -v
```

## 📖 文档索引

- [部署最小文档](./docs/deployment.md)
- [安全最小文档](./docs/security.md)
- [测试最小文档](./docs/testing.md)
- [前端架构文档](./docs/frontend-architecture.md)
- [0.1.1 发布计划](./docs/release-0.1.1-plan.md)
- [Roadmap](./docs/roadmap.md)

## 🔧 开发指南

### 🆕 新增 Skill 流程

1. **定义能力** → `app/skills/<领域>/<能力>.py`
2. **注册节点** → `app/runtime/graph/nodes/<节点>.py`
3. **编排流程** → `app/runtime/graph/graph.py` 更新 Graph
4. **验证** → `python -m app.examples.graph_demo`

### ⚙️ 配置管理

```python
from app.infrastructure.config.settings import settings

# 访问配置（类型安全）
llm_model = settings.llm.model
db_host = settings.database.host

# 或获取字典格式
config = settings.model_dump()
```

### 🔌 核心 API

| 🏷️ 方法 | 📡 端点 | 🎯 功能 |
|:---:|:---|:---|
| `POST` | `/auth/token` | JWT 登录 |
| `POST` | `/auth/register` | 用户注册 |
| `GET` | `/auth/users/me` | 获取当前用户 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/health/ready` | 就绪检查 |
| `GET` | `/health/live` | 存活检查 |
| `POST` | `/chat/invoke` | LangGraph 对话触发 |
| `POST` | `/upload` | PDF 上传并入队索引 |
| `POST` | `/upload/image` | 图片上传（OCR 占位） |
| `GET` | `/documents` | 文档列表/搜索 |
| `GET` | `/documents/{doc_id}` | 文档详情与预览 |
| `DELETE` | `/documents/{doc_id}` | 删除文档 |
| `POST` | `/documents/{doc_id}/reindex` | 文档重建索引 |
| `GET` | `/tasks/summary` | 任务聚合摘要 |
| `GET` | `/tasks/incidents` | 任务失败事件流 |
| `PATCH` | `/tasks/incidents/{incident_id}` | 标记事件已处理/归档 |
| `GET` | `/tasks/{task_id}` | 查询异步任务状态 |
| `POST` | `/tasks/{task_id}/retry` | 重试失败任务 |
| `GET` | `/history/{user_id}` | 查询历史会话 |
| `GET` | `/history/{user_id}/{session_id}` | 查询单个会话详情 |
| `POST` | `/history/{user_id}/save` | 保存历史会话 |
| `PATCH` | `/history/{user_id}/{session_id}` | 重命名历史会话 |
| `DELETE` | `/history/{user_id}/{session_id}` | 删除历史会话 |
| `GET` | `/interrupt/{session_id}` | 查询中断状态 |
| `POST` | `/interrupt/{session_id}/approve` | 审批中断动作 |
| `GET` | `/interrupt/{session_id}/resume` | 获取恢复参数 |
| `GET` | `/memory/profile` | 获取当前用户画像 |
| `PUT` | `/memory/profile` | 更新当前用户画像 |
| `GET` | `/memory/items` | 获取长期记忆列表 |
| `POST` | `/memory/items` | 新增长期记忆 |
| `DELETE` | `/memory/items/{item_id}` | 删除长期记忆 |
| `GET` | `/settings` | 获取系统配置（Admin） |
| `POST` | `/settings` | 更新系统配置（Admin） |
| `GET` | `/settings/user` | 获取个人配置 |
| `POST` | `/settings/user` | 更新个人配置 |
| `GET` | `/profile/{user_id}` | 获取用户画像 |
| `POST` | `/vectorstore/docs/clear` | 清空向量库（Admin） |

### 🧪 测试与验收

```bash
# 运行本地回归
uv run ./scripts/run_test_suite.sh

# 运行双桩 smoke
uv run ./scripts/smoke_workbench.sh

# 对运行中的服务执行 live smoke
uv run ./scripts/live_workbench_smoke.sh --base-url http://127.0.0.1:8000
```

如果需要把 live smoke 纳入测试门禁，可设置：

```bash
export LIVE_SMOKE_BASE_URL="http://127.0.0.1:8000"
uv run ./scripts/run_test_suite.sh
```

### 📈 运行评测

```bash
# 运行所有评估与测试
uv run ./scripts/run_evals.sh
```
