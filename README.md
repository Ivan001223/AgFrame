# AgFrame（Agent 后端脚手架）

一个用于快速搭建 Agent/RAG 后端的 Python 脚手架：FastAPI + LangServe + LangGraph，并内置 RAG、聊天记忆与用户画像等可复用组件。

> 说明：本仓库默认忽略 `docs/` 与 `tests/` 目录（不会提交到 GitHub）。

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
python server.py
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
- [server.py](file:///Users/ivan/Documents/Code/Agent_infra/server.py)：FastAPI 入口 + LangServe 路由 + 文件上传/历史/配置 API
- [graph.py](file:///Users/ivan/Documents/Code/Agent_infra/app/core/workflow/graph.py)：LangGraph 工作流编排入口
- `app/core/services/`：RAG/OCR/记忆/画像等服务层
- `app/core/utils/`：消息清洗、JSON 解析、FAISS 持久化等通用工具
- `app/agents/`：自定义 Agent 与节点工厂（可按业务扩展）

## 运行时目录
`data/` 为运行时目录（上传文件、向量索引、历史缓存等），会自动创建并写入内容，通常不建议提交到仓库。

## 开发与调试
- Swagger：启动后访问 `http://localhost:8000/docs`
- CORS：默认全放开（见 [server.py](file:///Users/ivan/Documents/Code/Agent_infra/server.py)）
- 本地调试 LangGraph：可直接运行示例脚本（`app/examples/`）

## 常见问题
- 远端已存在提交导致 push 被拒绝：先 `git pull --rebase` 再 push
- 不想提交本地配置：`config.json` 已被 `.gitignore` 忽略；敏感信息建议只用环境变量
- 不想提交运行数据：向量库/上传文件默认落在 `data/` 下，建议保持忽略
