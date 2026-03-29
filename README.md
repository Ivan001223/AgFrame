# 🚀 AgFrame (Agent Framework)

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-cyan?style=flat-square&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-FF6B6B?style=flat-square&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)

**⚡ 生产级 Agent/RAG 后端框架 | 基于 FastAPI + LangGraph 构建**  
专注于复杂工作流编排、轻量级混合检索、分层记忆与可观测性

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

| 🏗️ 工作流编排 | 🧠 轻量级 RAG | 💾 分层记忆 | 🔮 LLM 工厂 | 📊 可观测性 | 🛠️ 基础设施 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| LangGraph | 双路检索 (Dense+BM25) | 短期对话窗口 | 多模型支持 | Langfuse | Redis Worker |
| Harness 运行态 | RRF 排序融合 | 长期画像更新 | 嵌入模型 | 任务诊断队列 | PostgreSQL |
| Human-in-Loop | 上下文轻量裁剪 | pgvector 检索 | 结构化输出 | Checkpoint 追踪 | Docker 编排 |

---

## 📐 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│                  (Auth / REST / LangServe)                  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph / Harness                      │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│   │ Orchestr │  │  State   │  │  Nodes   │  │ Interrupt│  │
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
│  │ pgvector │ │  Redis   │ │ ARQ Queue│ │ Observability│    │
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
│   │   └── prompts/         #   Prompt 模板与裁剪策略
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
├── docs/                   # 📖 部署/架构/安全/测试文档
├── frontend/               # 💻 Next.js 工作台前端
├── scripts/                # 🧰 工具与冒烟测试脚本
├── data/                   # 📁 运行时数据
├── tests/                  # 🧪 单元测试与评测
│
├── docker-compose.yml      # 🐳 基础设施编排
├── pyproject.toml         # 🐍 Python 项目配置
└── uv.lock                 # 🔒 依赖锁文件
```

## 默认检索链路

```text
Query
  -> Dense Search + BM25 Search
  -> RRF Fusion
  -> Candidate Pruning (轻量级候选裁剪)
  -> Parent Restore (父文档还原)
  -> Prompt Assembly (组装 Prompt)
```

当前推荐最佳实践：
- 在主文档检索路径中仅保留 `Dense + BM25 + RRF`
- 在组装 Prompt 之前使用轻量级裁剪策略
- 将基于大模型的重排器 (Reranker) 视为遗留兼容组件，而非默认必选项

## 🚀 快速开始

### 1. 环境安装

```bash
uv python install 3.11
uv sync
```

可选依赖组：
- `uv sync --group document-ai`: 高精度 PDF / Office 文档解析能力
- `uv sync --group evals`: 离线评估与 Benchmark 工具

默认安装现在已经包含本地 Embedding / OCR / Transformers / Torch 运行时依赖，因此 `uv sync` 后即可直接使用本地推理链路。

### 2. 配置说明

```bash
cp configs/config.example.json configs/config.json
```

至少需要更新以下配置项：
- `auth.secret_key`
- `database.url` 或数据库凭证
- `llm.api_key` (若使用云端模型)

轻量级推荐配置参考：

```json
{
  "embeddings": {
    "model_name": "Qwen/Qwen3-Embedding-0.6B"
  },
  "rag": {
    "retrieval": {
      "mode": "hybrid",
      "dense_k": 20,
      "sparse_k": 20,
      "candidate_k": 20,
      "final_k": 3,
      "rrf_k": 60
    }
  },
  "prompt": {
    "context_pruning": {
      "enabled": true,
      "method": "auto"
    }
  },
  "reranker": {
    "model_name": ""
  }
}
```

### 3. 使用 Docker Compose 启动整套项目

```bash
cp .env.example .env
docker compose up --build -d
```

默认会启动：
- `postgres`
- `redis`（基于 Redis Stack，提供队列、限流与 LangGraph checkpoint 所需的 RediSearch 能力）
- `backend`
- `worker`
- `frontend`

默认访问地址：
- Frontend: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8000`
- Swagger 文档: `http://127.0.0.1:8000/docs`

说明：
- 若要直接使用聊天能力，请在 `.env` 中填写 `LLM_API_KEY`
- 若要启用文档摄取 / RAG，仍建议在 `configs/config.json` 中补充 embeddings 配置或远程 embeddings 服务地址

可选观测组件：

```bash
docker compose --profile observability up --build -d
```

这会额外启动 `clickhouse`、`minio`、`langfuse-server`、`langfuse-worker`。
Langfuse 默认暴露在 `http://127.0.0.1:3001`。

### 4. 手动启动 Backend API

```bash
uv run python -m app.server.main
```

### 5. 手动启动异步 Worker

```bash
uv run arq app.infrastructure.queue.worker_settings
```

## 🩺 健康检查与运行时信号

`readiness` 端点 (Endpoint) 会暴露当前轻量级检索状态与 Harness 引擎状态：
- `components.retrieval == "hybrid_rrf"`
- `components.context_pruning == "lightweight_ranker"`

`reranker` 组件为向后兼容保留，但默认检索路径不再强依赖。

## 📖 文档指引

- [部署指南 (Deployment Guide)](./docs/deployment.md)
- [RAG 架构设计 (RAG Architecture)](./docs/rag-architecture.md)
- [RAG 迁移指南 (RAG Migration Guide)](./docs/rag-migration.md)
- [测试指南 (Testing Guide)](./docs/testing.md)
- [前端架构 (Frontend Architecture)](./docs/frontend-architecture.md)
- [安全规范 (Security Notes)](./docs/security.md)
- [路线图 (Roadmap)](./docs/roadmap.md)

## 📌 当前状态

当前代码库反映了最新的轻量级 RAG 与 Agent 编排设计：
- 文档检索已彻底移除对模型重排器 (Model Reranker) 的硬依赖
- 记忆检索采用轻量级本地打分排序
- 上下文裁剪采用轻量级排序 / 启发式评分
- **(v0.2.1)** 引入 Harness 执行引擎，支持 LangGraph 任务中断 (Interrupt) 与 Checkpoint 审批恢复
- 基础配置项和健康度报告均与该轻量化、高可控链路保持一致
