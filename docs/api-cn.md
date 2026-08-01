# AgFrame API 文档

<div align="center">
  <a href="api.md">English</a>
</div>

## 范围

- **Base URL**：`http://127.0.0.1:8000`
- **认证方式**：`POST /auth/token` 会返回 JWT，同时写入 HttpOnly 认证 Cookie；受保护接口优先支持 Cookie，也继续兼容 `Authorization: Bearer <token>`
- **后端版本**：`0.3.3`
- **主运行时入口**：`POST /chat/workbench-invoke`

## 访问模型

- **开放接口**：`/health/*`、`/auth/*`
- **登录后可用**：`/chat/*`、`/interrupt/*`、`/history/*`、`/documents/*`、`/upload/*`、`/tasks/*`、`/memory/*`、`/profile/*`、`/knowledge-bases/*`、`/files/*`、`/uploads/*`、`/harness/*`、`GET|POST /settings/user`
- **仅管理员**：`GET|POST /settings`、`POST /vectorstore/docs/clear`

## 认证鉴权

- `POST /auth/token` — 使用用户名和密码换取访问令牌，并设置认证 Cookie
- `POST /auth/logout` — 清除认证 Cookie
- `POST /auth/register` — 注册新用户
- `GET /auth/users/me` — 获取当前登录用户信息

浏览器客户端说明：

- 当前前端默认使用后端下发的 HttpOnly Cookie 鉴权，而不是把 JWT 保存在 `localStorage`
- 如果前端和后端不在同一个 origin 或端口，请先把对应来源加入 `CORS_ORIGINS`
- 浏览器使用 Cookie 鉴权时还需要 `CORS_ALLOW_CREDENTIALS=true`

## 聊天接口

### Workbench 主链路

- `POST /chat/workbench-invoke` — 工作台主调用入口；负责注入运行时配置、执行图、持久化消息，并返回 `reply`、`messages`、`context` 与 interrupt 状态

### LangServe 运行时接口

同一个 graph 还会通过 LangServe 暴露在 `/chat` 前缀下。当前已明确校验的标准入口为：

- `POST /chat/invoke` — LangServe 标准 invoke 接口

此外，LangServe 还会在同一 `/chat` 前缀下自动挂载 schema、stream、playground 与 feedback 等标准子路由，且应用启动时已启用 feedback 能力。

## Interrupt 中断与恢复

面向会话级的人审与恢复链路：

- `GET /interrupt/{session_id}` — 查询会话是否处于中断状态，以及是否需要人工动作
- `GET /interrupt/{session_id}/events` — 查看该会话的 interrupt 事件流
- `POST /interrupt/{session_id}/approve` — 对待审批动作进行通过或拒绝
- `GET /interrupt/{session_id}/resume` — 获取恢复执行所需的 payload 结构
- `POST /interrupt/{session_id}/resume` — 恢复图执行并持久化恢复后的消息

## Harness 控制平面

### Runs

- `POST /harness/runs` — 创建 harness run
- `GET /harness/runs` — 列出当前用户可见的 run
- `GET /harness/runs/{run_id}` — 获取 run 详情
- `POST /harness/runs/{run_id}/retry` — 基于已有 run 创建重试 run
- `GET /harness/runs/{run_id}/events` — 查看 run 生命周期事件
- `GET /harness/runs/{run_id}/runtime-state/history` — 读取运行时状态历史
- `GET /harness/runs/{run_id}/approval` — 读取待审批状态
- `POST /harness/runs/{run_id}/approval` — 提交 run 审批决策
- `GET /harness/runs/{run_id}/verification` — 读取最近一次验证证据
- `GET /harness/policies` — 列出可用的 harness policies

### Studio Projects

- `GET /harness/studio/projects` — 列出当前用户的 studio 项目
- `GET /harness/studio/projects/current` — 获取当前 studio 项目
- `POST /harness/studio/projects` — 创建 studio 项目
- `GET /harness/studio/projects/{project_id}` — 读取项目详情
- `PUT /harness/studio/projects/{project_id}` — 更新项目元数据或 `graph_json`
- `POST /harness/studio/projects/{project_id}/skill-requests` — 为某个 agent 发起技能申请
- `POST /harness/studio/projects/{project_id}/skill-requests/{request_id}` — 对技能申请做通过或拒绝
- `POST /harness/studio/projects/{project_id}/run` — 基于 studio 项目发起编排运行

### Model Providers

- `GET /harness/model-providers` — 列出当前用户可见的模型提供方配置
- `POST /harness/model-providers` — 创建模型提供方
- `PUT /harness/model-providers/{provider_id}` — 更新模型提供方
- `DELETE /harness/model-providers/{provider_id}` — 删除模型提供方

> Harness run 在内部通过平台运行时命令层（`RuntimeApplicationService`）分发执行。参见
> [平台治理控制平面](architecture/platform-governance-control-plane.md)
> 了解运行时命令流转与生命周期治理详情。

## 会话历史

- `GET /history/{user_id}` — 列出会话，可通过 `q` 搜索
- `GET /history/{user_id}/{session_id}` — 读取单个会话详情
- `POST /history/{user_id}/save` — 手动持久化当前会话
- `PATCH /history/{user_id}/{session_id}` — 更新会话元数据，例如重命名标题
- `DELETE /history/{user_id}/{session_id}` — 删除指定会话

## 文档与上传

### Documents

- `GET /documents` — 列出已上传文档
- `GET /documents/{doc_id}` — 读取文档详情与预览片段
- `GET /documents/{doc_id}/download` — 下载文档原始文件
- `DELETE /documents/{doc_id}` — 删除文档记录与源文件
- `PUT /documents/{doc_id}/knowledge-base` — 绑定或清空文档所属知识库
- `POST /documents/{doc_id}/reindex` — 重新入队文档索引任务

### Knowledge Bases

- `GET /knowledge-bases` — 列出当前可见知识库及文档数量
- `POST /knowledge-bases` — 创建当前用户拥有的知识库
- `PUT /knowledge-bases/{knowledge_base_id}` — 更新知识库名称或描述
- `DELETE /knowledge-bases/{knowledge_base_id}` — 删除当前用户拥有的知识库

### Upload

- `POST /upload` — 上传文档文件并触发入库
- `POST /upload/image` — 上传图片并返回文件访问地址

### File Access

- `GET /uploads/{owner}/{relative_path}` — 按 owner 或 admin 权限读取上传资产
- `GET /files/{owner}/{relative_path}` — 按 owner 或 admin 权限读取文档资产

## 后台任务

- `GET /tasks/summary` — 汇总后台任务状态、incident 与超时信号
- `GET /tasks/incidents` — 列出 incident，可按 handled / archived 过滤
- `PATCH /tasks/incidents/{incident_id}` — 更新 incident 的处理状态
- `GET /tasks/{task_id}` — 查看单个任务的详情与诊断信息
- `POST /tasks/{task_id}/retry` — 重试失败任务

## 记忆与画像

### Memory

- `GET /memory/profile` — 读取合成后的用户画像
- `PUT /memory/profile` — 更新画像并同步语义记忆
- `GET /memory/items` — 列出原子记忆条目
- `POST /memory/items` — 创建记忆条目
- `DELETE /memory/items/{item_id}` — 删除记忆条目

### Profile

- `GET /profile/{user_id}` — 读取指定用户画像信息

## 设置

- `GET /settings` — 读取全局应用设置
- `POST /settings` — 更新全局应用设置
- `GET /settings/user` — 读取用户级偏好
- `POST /settings/user` — 更新用户级偏好

## 向量库

- `POST /vectorstore/docs/clear` — 清空文档向量库

## 健康检查

- `GET /health` — 基础健康状态
- `GET /health/ready` — 依赖项与运行时就绪检查
- `GET /health/live` — 轻量级存活检查

## 说明

- `/chat/workbench-invoke` 是工作台前端使用的 UI 主链路
- `/interrupt/*` 是会话级恢复链路，与 LangGraph checkpoint 恢复直接相关
- `/harness/*` 是 run 级控制平面，也是 Agent Studio 的后端接口集合
- 上传、文档、任务与 harness 接口共同组成“上传 -> 入队 -> 观察 -> 重试 -> 审计”的操作闭环
