# AgFrame (Agent Framework)

AgFrame 是一个生产级 Agent/RAG 后端框架，基于 **FastAPI** + **LangGraph** 构建。专注于复杂工作流编排、混合检索、分层记忆与可观测性。

## 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│                  (Auth / REST / LangServe)                 │
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
│                      Skills / Services                     │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐   │
│  │  RAG   │ │ Memory │ │ Profile│ │ Research│ │ Tools  │   │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │ pgvector │ │  Redis  │ │ PostgreSQL│ │ Langfuse     │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
AgFrame/
├── app/
│   ├── server/              # FastAPI 入口
│   │   ├── api/             # 路由层
│   │   └── main.py           # 应用启动
│   ├── runtime/             # 运行时核心
│   │   ├── graph/           # LangGraph 工作流
│   │   │   ├── graph.py      # 图定义
│   │   │   ├── state.py      # State Schema
│   │   │   ├── orchestrator.py # 编排器
│   │   │   ├── registry.py   # 节点注册表
│   │   │   ├── json_router.py # JSON 路由
│   │   │   ├── memory_router.py
│   │   │   └── nodes/        # 节点实现
│   │   ├── llm/             # LLM 工厂
│   │   │   ├── llm_factory.py
│   │   │   ├── embeddings.py
│   │   │   ├── reranker.py
│   │   │   └── local_qwen.py
│   │   └── prompts/         # Prompt 模板
│   ├── skills/              # 原子能力层
│   │   ├── rag/             # 混合检索
│   │   ├── memory/          # 分层记忆
│   │   ├── profile/         # 用户画像
│   │   ├── research/        # 网络搜索
│   │   ├── ocr/             # 图片 OCR
│   │   ├── common/          # 公共技能
│   │   └── tools/           # 代码执行
│   ├── infrastructure/      # 基础设施
│   │   ├── config/          # 配置管理
│   │   ├── database/        # SQLAlchemy ORM
│   │   ├── vector_stores/   # pgvector 集成
│   │   ├── checkpoint/      # Redis Checkpoint
│   │   ├── queue/           # ARQ 异步任务
│   │   ├── sandbox/         # 代码沙箱
│   │   ├── observability/   # 可观测性
│   │   └── utils/           # 工具函数
│   ├── agents/              # Agent 节点工厂
│   ├── memory/              # 记忆模块
│   │   ├── long_term/       # 长期记忆
│   │   └── vector_stores/   # 向量存储
│   └── examples/           # 调试脚本
├── configs/                 # 配置文件
│   ├── config.example.json
│   └── config.json          # 本地配置（需创建）
├── docker/                  # Docker 初始化脚本
├── scripts/                 # 工具脚本
├── data/                    # 运行时数据
├── tests/                   # 单元测试
├── docker-compose.yml       # 基础设施编排
└── requirements.txt         # Python 依赖
```

## 核心特性

### 1. 工作流编排 (LangGraph)
- **Stateful Graph**: 支持循环、分支、条件判断
- **Checkpoint**: Redis 持久化断点恢复
- **Human-in-the-Loop**: 支持人工中断与反馈
- **节点注册表**: 动态节点管理

### 2. 混合 RAG
- **双路检索**: BM25 (关键词) + Embedding (语义)
- **重排序**: 支持 Cross-Encoder 重排序
- **多格式解析**: PDF/DOCX/Excel/图片 OCR
- **RRF 融合**: RRF 排序融合算法

### 3. 分层记忆系统
- **短期记忆**: 对话上下文窗口管理
- **长期记忆**: 用户画像 + 历史向量存储
- **pgvector**: 向量检索持久化
- **对话管理**: 历史记录持久化

### 4. LLM 工厂
- **多模型支持**: OpenAI API / 本地 Qwen (Ollama/VLLM)
- **嵌入模型**: Sentence-Transformers / ModelScope
- **重排序**: Cross-Encoder
- **结构化输出**: 原生 Pydantic + JSON 模式

### 5. 可观测性
- **Langfuse**: 全链路追踪、Prompt 管理
- **DeepEval**: RAG 评测 (Context Recall/Precision)

### 6. 基础设施
- **Redis**: 缓存 / Checkpoint / 任务队列 (ARQ)
- **PostgreSQL**: 持久化存储
- **Docker**: 一键启动全部依赖

## 快速开始

### 1. 环境准备

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置

```bash
cp configs/config.example.json configs/config.json
# 编辑 configs/config.json 配置各项参数
```

**配置结构**：

```json
{
  "llm": {
    "api_key": "sk-xxx",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o"
  },
  "model_manager": {
    "provider": "modelscope",
    "cache_dir": ""
  },
  "local_models": {
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "rerank_model": "Qwen/Qwen3-Reranker-0.6B"
  },
  "embeddings": {
    "provider": "modelscope",
    "model_name": "Qwen/Qwen3-Embedding-0.6B"
  },
  "reranker": {
    "provider": "modelscope",
    "model_name": "Qwen/Qwen3-Reranker-0.6B"
  },
  "database": {
    "type": "postgres",
    "url": "postgresql+psycopg://user:pass@localhost:5432/db"
  },
  "queue": {
    "redis_url": "redis://localhost:6379/0"
  },
  "rag": {
    "retrieval": {
      "mode": "hybrid",
      "dense_k": 20,
      "sparse_k": 20,
      "final_k": 3
    }
  },
  "auth": {
    "secret_key": "your-secret-key",
    "algorithm": "HS256"
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000
  }
}
```

**环境变量覆盖**：

敏感配置可通过环境变量覆盖：

```bash
export LLM_API_KEY="sk-xxx"
export DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/db"
export REDIS_URL="redis://localhost:6379/0"
```

### 3. 启动依赖

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 配置数据库连接等（可选，使用默认值可直接启动）
# vim .env

# 启动所有基础设施
docker-compose up -d

# 验证服务状态
docker-compose ps
```

**启动的服务：**

| 服务 | 端口 | 用途 |
|------|------|------|
| PostgreSQL + pgvector | 5432 | 主数据库 + 向量存储 |
| Redis | 6379 | 缓存、Checkpoint、任务队列 |
| RabbitMQ | 5672/15672 | ARQ 异步任务队列 |
| ClickHouse | 8123 | Langfuse 指标存储 |
| MinIO | 9000/9001 | S3 对象存储 |
| Langfuse | 3000 | 可观测性追踪 |

**验证命令：**

```bash
# PostgreSQL
psql -h localhost -p 5432 -U agframe -d agframe -c "SELECT 1"

# Redis
redis-cli -h localhost -p 6379 -a redissecret ping

# RabbitMQ Management
curl -u agframe:rabbitmq_secret http://localhost:15672/api/overview

# ClickHouse
curl http://localhost:8123/ping

# MinIO
mc alias set local http://localhost:9000 minioadmin minioadmin_secret
mc ls local

# Langfuse
curl http://localhost:3000/api/public
```

### 4. 初始化 MinIO Bucket

```bash
# 创建 bucket
docker-compose exec minio mc mb local/agframe

# 设置公开访问策略（可选，用于存储公开资源）
docker-compose exec minio mc anonymous set public local/agframe
```

### 5. 启动服务

```bash
python -m app.server.main
```

服务运行在 `http://localhost:8000`

- **Swagger Docs**: http://localhost:8000/docs

## 停止服务

```bash
# 停止所有基础设施
docker-compose down

# 停止并删除数据卷（慎用！）
docker-compose down -v
```

## 开发指南

### 新增 Skill 流程

1. **定义能力** → `app/skills/<领域>/<能力>.py`
2. **注册节点** → `app/runtime/graph/nodes/<节点>.py`
3. **编排流程** → `app/runtime/graph/graph.py` 更新 Graph
4. **验证** → `python -m app.examples.graph_demo`

### 配置管理

```python
from app.infrastructure.config.config_manager import config_manager

# 获取配置
config = config_manager.get_config()
llm_config = config.get("llm")

# 更新配置
config_manager.update_config({"llm": {"model": "gpt-4-turbo"}})
```

### 核心 API

| 端点 | 功能 |
|------|------|
| `POST /auth/token` | JWT 登录 |
| `POST /chat/invoke` | 对话触发 |
| `POST /upload` | 文件上传 (RAG 入库) |
| `GET /history/{user_id}` | 对话历史 |
| `GET /settings` | 获取配置 |
| `POST /tasks/background` | 异步任务提交 |

### 运行评测

```bash
# RAG 评测
python -m tests.test_deepeval_rag

# 或使用评测脚本
./scripts/run_evals.sh
```
