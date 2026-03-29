# AgFrame API 文档

本文档提供了 AgFrame 项目中可用的 RESTful API 端点概述。后端基于 FastAPI 构建，由多个模块化的路由组件组成。

## 基础信息
- **默认端口**: 8000 (本地)
- **基础路径**: `/`
- **认证方式**: JWT Bearer Token (通过请求头 `Authorization: Bearer <token>` 传递)。

---

## 1. 认证鉴权 (`/auth`)
处理用户认证、令牌生成以及注册。

- `POST /auth/token`: 登录获取 JWT Access Token。
- `POST /auth/register`: 注册新用户账号。
- `GET /auth/users/me`: 获取当前认证用户的基本信息。

## 2. 调度执行 (`/harness`)
用于管理 Agent 运行、审批工作流及其策略的控制平面 API。

- `GET /harness/runs`: 列出所有的 Harness 运行记录。
- `POST /harness/runs`: 创建一个新的 Harness 运行。
- `GET /harness/runs/{run_id}`: 获取指定 Harness 运行的详细信息。
- `POST /harness/runs/{run_id}/retry`: 重试失败的执行。
- `GET /harness/runs/{run_id}/events`: 获取与某次运行相关的事件流。
- `GET /harness/runs/{run_id}/approval`: 获取某次运行的审批工作流状态。
- `POST /harness/runs/{run_id}/approval`: 提交审批决策 (通过/拒绝)。
- `GET /harness/runs/{run_id}/verification`: 查看执行验证及证据信息。
- `GET /harness/policies`: 列出所有运行时执行策略。

## 3. 循环中人为介入 / 中断 (`/interrupt`)
处理处于中断状态的执行，以便用户进行审查或授权。

- `GET /interrupt/{session_id}`: 获取会话的中断状态。
- `GET /interrupt/{session_id}/events`: 列出队列中等待审批的事件。
- `POST /interrupt/{session_id}/approve`: 针对特定的状态变更提供审批决策。
- `GET /interrupt/{session_id}/resume`: 获取恢复执行的请求参数结构。
- `POST /interrupt/{session_id}/resume`: 从中断状态恢复代码及流程继续执行。

## 4. 聊天与交互 (`/chat`)
实时交互和推理接口。

- `POST /chat/workbench-invoke`: 供工作台界面调用的主推理端点。

## 5. 会话历史 (`/history`)
管理聊天会话及历史事件的接口。

- `GET /history/{user_id}`: 获取指定用户的聊天会话列表。
- `GET /history/{user_id}/{session_id}`: 抓取某一特定会话的历史消息。
- `POST /history/{user_id}/save`: 手动将当前会话状态持久化。
- `PATCH /history/{user_id}/{session_id}`: 更新会话元数据（例如重命名对话）。
- `DELETE /history/{user_id}/{session_id}`: 彻底删除指定的会话历史。

## 6. 文档与上传管理 (`/documents` & `/upload`)
管理知识库文件和媒体资源。

- `GET /documents`: 列出用户已存入存储体系的文档。
- `GET /documents/{doc_id}`: 获取指定文档详情及元数据。
- `DELETE /documents/{doc_id}`: 从存储与向量库中移除指定文档。
- `POST /documents/{doc_id}/reindex`: 重新启动对文档的解析和向量库索引映射。
- `POST /upload`: 上传标准文档或数据文件。
- `POST /upload/image`: 上传图片文件。

## 7. 向量库管理 (`/vectorstore`)
- `POST /vectorstore/docs/clear`: 清空向量库中的历史文档节点数据（通常需 Admin 权限）。

## 8. 后台任务 (`/tasks`)
管理脱离主干流程的后台任务及其执行异常监控。

- `GET /tasks/summary`: 获取活动和完成状态下的全盘任务汇总看板数据。
- `GET /tasks/{task_id}`: 监控和查看单一 Worker 任务状态。
- `POST /tasks/{task_id}/retry`: 重新触发未完成或失败的任务。
- `GET /tasks/incidents`: 列出由于任务执行导致的突发事件/错误。
- `PATCH /tasks/incidents/{incident_id}`: 标记接收、确认或者缓解指定的报警事件。

## 9. 记忆节点管理 (`/memory`)
管理用户全局或者上下文相关的持续记忆知识。

- `GET /memory/profile`: 查看目前的合成用户偏好与身份记忆文件。
- `PUT /memory/profile`: 手工修正及更新偏好记忆档案内容。
- `GET /memory/items`: 呈现离散式的记忆知识片段。
- `POST /memory/items`: 手动登记一条新的记忆条目。
- `DELETE /memory/items/{item_id}`: 删除指定的记忆条目。

## 10. 用户配置与偏好 (`/profile` & `/settings`)
管理平台及个人层级的配置设定项。

- `GET /profile/{user_id}`: 获取用户自己的个人档案数据详情。
- `GET /settings`: 读取全站范围内的全局应用配置 (需要 Admin 权限)。
- `POST /settings`: 对当前全局应用配置下发更新 (需要 Admin 权限)。
- `GET /settings/user`: 读取用户私人层级的参数设定。
- `POST /settings/user`: 持久化及修改当前调用用户的私人配置参数。

## 11. 系统健康监控 (`/health`)
用于监控报警以及容灾节点的生命周期校验（开放访问）。

- `GET /health`: 基础存活性探测回执节点。
- `GET /health/ready`: 测试与下游数据设施集群（如数据库、Redis）的联通就绪状况。
- `GET /health/live`: 用于 Kubernetes 等编排工具的轻量级轮询探活接口。
