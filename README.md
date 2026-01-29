# Agent 后端脚手架

这是一个用于构建 Agent 应用的纯后端脚手架。

## 项目结构
- `server.py`: 主入口（FastAPI + LangServe）。
- `app/core/workflow/graph.py`: LangGraph 定义（路由 + 检索 + 组装提示词 + 生成）。
- `app/core/services/`: RAG、OCR、聊天记忆、画像等服务层。
- `app/core/utils/`: 通用工具（消息清洗、JSON 解析、FAISS 持久化等）。
- `app/agents/`: 放置你的 Agent 实现与节点工厂。
- `docs/PROJECT_STRUCTURE.md`: 更完整的目录结构与职责说明。

## 安装与运行
1. 创建虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 配置：
   - 复制 `configs/config.example.json` 为 `config.json` 并填写必要字段（或通过环境变量覆盖）。
   - `data/` 为运行时目录，会自动创建并写入历史与索引文件，不建议提交到仓库。
4. 启动服务：
   ```bash
   python server.py
   ```
