# AgFrame (Agent Backend Scaffold)

AgFrame 是一个纯净的 Agent/RAG 后端开发脚手架，基于 **FastAPI** + **LangGraph** + **LangServe** 构建。  
项目移除了具体的业务逻辑，保留了通用的架构、基础设施与核心原子能力，旨在为构建复杂的 Agentic Application 提供坚实、可扩展的基础。

## 核心特性

- **现代架构**:  
  遵循 `FastAPI (Web层) -> LangGraph (编排层) -> Services/Skills (能力层)` 分层设计，职责清晰。
- **工作流编排 (LangGraph)**:  
  原生支持 Stateful Graph，易于构建循环、分支、多 Agent 协作等复杂逻辑。
- **API 服务化 (LangServe)**:  
  集成 LangServe，自动将 Graph 暴露为标准 REST API (`/chat`)，支持流式输出与反馈。
- **企业级鉴权**:  
  内置 JWT 身份认证 (User/Admin 角色体系) 与 `FastAPI-Limiter` 速率限制。
- **混合存储架构**:
  - **Vector**: FAISS (本地持久化) / 易于扩展至 pgvector 等。
  - **Relational**: SQLAlchemy ORM (默认支持 MySQL/PostgreSQL)。
  - **Cache/Queue**: 深度集成 Redis，用于缓存、限流及异步任务队列。
- **原子能力 (Skills)**:
  - **RAG**: PDF/各类文档异步解析、切片、向量化入库。
  - **OCR**: 图片文字识别模块接口预留。
  - **Memory**: 分层记忆系统（短期对话上下文 + 长期历史持久化）。
  - **Profile**: 用户画像构建与更新机制。
- **可观测性**:  
  深度集成 **Langfuse**，支持全链路追踪 (Tracing)、Prompt 管理与评估。

## 目录结构

```text
AgFrame/
├── app/
│   ├── server/             # FastAPI 服务入口 (Main, Routers, Middleware)
│   │   ├── api/            # 路由定义 (Auth, Upload, History, Tasks...)
│   │   └── main.py         # App 启动入口
│   ├── runtime/            # 核心运行时
│   │   ├── graph/          # LangGraph 工作流定义与编排
│   │   ├── llm/            # LLM 工厂与配置
│   │   └── prompts/        # Prompt 模板管理
│   ├── skills/             # 原子能力模块 (RAG, Memory, Profile, OCR...)
│   ├── infrastructure/     # 基础设施 (Database, Redis, Logging, Utils)
│   ├── agents/             # Agent 节点构建工厂
│   └── examples/           # 调试脚本与使用示例
├── configs/                # 配置文件模板
├── data/                   # 运行时数据目录 (忽略提交)
├── docker-compose.yml      # 基础设施编排
└── requirements.txt        # Python 依赖
```

## 快速开始

### 1. 环境准备

推荐使用 Python 3.10+。

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

复制示例配置文件，或直接通过环境变量配置（推荐生产环境使用环境变量）：

```bash
cp configs/config.example.json config.json
```

**关键配置项 (Environment Variables)**:

```bash
# LLM Settings
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"
export LLM_BASE_URL="https://api.openai.com/v1"

# Auth Settings
export SECRET_KEY="your-super-secret-key"  # 用于 JWT 签名
export ALGORITHM="HS256"

# Infrastructure
export REDIS_URL="redis://localhost:6379/0"
export DATABASE_URL="mysql+mysqlconnector://user:pass@localhost/dbname" # 或 sqlite

# Observability
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
```

### 3. 启动基础设施

使用 Docker Compose 启动 Redis、MySQL 等依赖服务：

```bash
docker-compose up -d
```

### 4. 启动后端服务

```bash
python -m app.server.main
```

服务启动后默认监听 `http://localhost:8000`。

- **Swagger API 文档**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **LangServe Playground**: [http://localhost:8000/chat/playground](http://localhost:8000/chat/playground)

## 开发指南

### 常用 API Endpoint

- `POST /auth/token`: 用户登录获取 Token。
- `POST /chat/invoke`: 核心对话接口 (LangServe)。
- `POST /upload`: 上传文件 (RAG 入库)。
- `GET /history/{user_id}`: 获取用户对话历史。
- `POST /vectorstore/docs/clear`: 清空向量库 (Admin only)。
