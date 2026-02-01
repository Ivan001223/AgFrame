# AgFrame（Agent 后端脚手架）

一个用于快速搭建 Agent/RAG 后端的 Python 脚手架：FastAPI + LangServe + LangGraph，并内置 RAG、聊天记忆与用户画像等可复用组件。

## 功能概览
- 对话编排：基于 LangGraph 的工作流（路由/检索/提示词组装/生成）
- 服务化 API：FastAPI + LangServe（默认暴露 `/chat`）
- RAG（PDF）：上传 PDF 后异步入库；向量库基于 FAISS 持久化到 `data/`
- 聊天历史：优先 MySQL（可选），否则降级到本地文件存储
- 用户画像：MySQL 落库与增量更新（保存历史后触发）
- 配置中心：`config.json` + 环境变量覆盖（敏感信息建议只用环境变量）

## 快速开始
### 1) 创建虚拟环境并安装依赖
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) 配置
方式 A（推荐）：仅用环境变量（避免把 Key 写进文件）
```bash
export OPENAI_API_KEY="YOUR_KEY"
export LLM_MODEL="gpt-4o"
export LLM_BASE_URL="https://api.openai.com/v1"
```

方式 B：使用配置文件（会读取根目录 `config.json`；该文件已在 `.gitignore` 中忽略）
```bash
cp configs/config.example.json config.json
```

MySQL（可选）：如果你希望启用 SQL 存储/画像/三桶分层，可配置数据库环境变量：
```bash
export DB_HOST="localhost"
export DB_PORT="3306"
export DB_USER="root"
export DB_PASSWORD="password"
export DB_NAME="agent_app"
```

### 3) 启动服务
```bash
python -m app.server.main
```

启动后默认监听 `http://localhost:8000`。

## 常用 API
- 对话（LangServe）：`/chat`
- 上传 PDF 入库（RAG）：`POST /upload`（multipart，字段名 `files`）
- 上传图片（OCR 预留）：`POST /upload/image`
- 查看/更新配置：`GET /settings`、`POST /settings`
- 历史：`GET /history/{user_id}`、`POST /history/{user_id}/save`、`DELETE /history/{user_id}/{session_id}`
- 清空文档向量库：`POST /vectorstore/docs/clear`

示例：上传 PDF
```bash
curl -F "files=@./example.pdf" http://localhost:8000/upload
```

示例：保存历史（会在启用 MySQL 时触发画像增量更新）
```bash
curl -X POST "http://localhost:8000/history/u1/save" \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","messages":[{"role":"user","content":"你好"}],"title":"demo"}'
```

## 代码导航
- [app/server/main.py](app/server/main.py)：FastAPI 入口 + LangServe 路由 + 文件上传/历史/配置 API
- [app/runtime/graph/graph.py](app/runtime/graph/graph.py)：LangGraph 工作流编排入口
- `app/skills/`：原子能力 (RAG/OCR/Memory/Profile)
- `app/infrastructure/`：基础设施 (DB, Redis, Utils)
- `app/agents/`：Agent 定义与编排
- `app/memory/`：记忆系统 (Long-term, Vector Stores)

## 运行时目录
`data/` 为运行时目录（上传文件、向量索引、历史缓存等），会自动创建并写入内容，通常不建议提交到仓库。

## 开发与调试
- Swagger：启动后访问 `http://localhost:8000/docs`
- CORS：默认全放开（见 [server.py](file:///Users/ivan/Documents/Code/Agent_infra/server.py)）
- 本地调试 LangGraph：可直接运行示例脚本（`app/examples/`）
