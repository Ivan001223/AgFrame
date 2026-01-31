---
trigger: always_on
---

1. 核心工作流：微步迭代循环 (Micro-Step Loop)
所有功能开发必须遵循 D-V-C-N 循环，严禁“大爆炸”式提交：

拆解 (Decompose)

将大任务拆解为最小可验证单元（例如：只写一个 Tool 函数，而不是整个 Agent）。

粒度标准：单个单元的代码修改量应控制在 50 行以内。

开发 (Develop)

接口层 (server.py)：仅挂载路由，无业务逻辑。

编排层 (workflow/)：先定义 State，再写 Node 逻辑。

能力层 (services/)：算法封装为无状态函数。

验证 (Verify)

代码编写完成后，必须立即运行本地测试（Unit Test 或 app/examples/ 下的调试脚本）。

标准：确保无 Syntax Error，且逻辑符合预期。禁止在未运行验证的情况下进入下一步。

存档 (Commit)

验证通过后，立即执行本地 Git 提交。

格式：git commit -m "feat/fix: 具体变更内容(中文)"。

注意：仅提交到本地仓库，提交的具体内容使用中文，测试代码严禁提交。

下一步 (Next)

重复上述步骤进入下一个功能单元。

2. 架构红线
分层原则：FastAPI (外壳) -> LangGraph (大脑) -> Services (手脚)。

IO 规范：所有文件读写强制限制在 data/ 目录；所有 IO/LLM 操作必须 async。

配置安全：敏感 Key 必须走 os.getenv，禁止硬编码。

3. 错误处理
防崩设计：Node 内部必须捕获异常并返回“错误状态”，严禁导致服务 Crash。

日志：关键节点需输出 logger.info 包含 session_id。