# AgFrame 前端架构

<div align="center">
  <a href="frontend-architecture.md">English</a>
</div>

## 范围

- **框架**：Next.js `16.1.6`
- **React**：`19.2.3`
- **数据层**：TanStack React Query `5.x`
- **样式体系**：Tailwind CSS `4`
- **表单**：React Hook Form + Zod

## 概览

前端是构建在 FastAPI 后端之上的 Next.js App Router 工作台。它的职责不是承载 Agent 业务逻辑本身，而是为后端自管执行流提供操作与交互界面，核心包括：

- chat workbench
- knowledge ingestion
- conversations
- memory management
- task observation
- settings
- Harness Agent Studio

主要业务逻辑由后端 API 承担，前端重点负责路由组织、鉴权访问、轮询刷新和运维级交互状态。

## 分层结构

### App 层

- `src/app/layout.tsx` 定义根布局并注入共享 query provider
- `src/app/(workspace)/layout.tsx` 为已登录工作区路由统一套用 `AuthGuard` 与 `AppShell`
- `src/app/(public)/login/page.tsx` 提供公开登录入口

### Domain 层

`src/domains/*/hooks.ts` 负责定义带类型的 API 访问，当前覆盖：

- auth
- chat
- conversations
- documents
- harness
- memory
- settings
- tasks

这是当前前端最重要的抽象边界。页面应优先消费 domain hooks，而不是直接手写 `fetch`。

### Shared Infrastructure 层

- `src/lib/http/client.ts` 统一处理 base URL、Bearer Token 注入、超时和 `ApiError` 归一化
- `src/lib/auth/session.ts` 管理客户端 token 持久化
- `src/components/layout/*` 提供通用登录保护与工作台壳层

## 当前技术选型

当前代码实际使用：

- Next.js App Router
- React 19 client components
- TanStack React Query 管理服务端状态
- React Hook Form 与 Zod 处理表单
- Lucide React 提供图标
- Tailwind CSS 4 处理样式

当前前端 **没有** 使用：

- Zustand
- Radix UI
- TanStack Table
- Recharts

## 路由结构

当前工作区路由：

- `/chat`
- `/harness`
- `/knowledge`
- `/conversations`
- `/conversations/[conversationId]`
- `/memory`
- `/tasks`
- `/settings`
- `/admin/settings`

公开路由：

- `/login`

`AppShell` 当前默认展示 chat、knowledge、conversations、memory、tasks、settings，以及管理员可见的 admin settings。Harness 是独立工作区页面，页面本身承担较大的 Agent Studio 功能面。

## 数据流

### 鉴权

- 登录使用 `POST /auth/token`
- 当前用户初始化使用 `GET /auth/users/me`
- Token 失效或缺失时会回跳 `/login`

### Chat

前端聊天主链路使用 `POST /chat/workbench-invoke`，而不是纯前端直连 LangServe。后端统一负责：

- 注入运行时配置
- 调用 graph 执行
- 读取最新 state
- 持久化消息
- 返回 interrupt 状态

### Harness

Harness 前端已经不是简单的 run 看板，而是 Agent Studio 工作台，当前覆盖：

- run 列表与详情
- approval 与 retry 操作
- policy 可见性
- studio project 列表与 current project 加载
- graph 编辑与保存
- skill request / skill approval 工作流
- studio run 启动
- model provider 管理

### Knowledge 与 Tasks

Knowledge 页面负责上传、文档列表、预览与 reindex。Tasks 页面负责异步任务状态、incident 和 retry 流程的可视化。

## 状态管理

- **服务端状态**：由 React Query 负责获取、缓存、失效与刷新
- **会话状态**：token 与当前用户引导由 auth 工具与 hooks 负责
- **本地 UI 状态**：依赖页面级 `useState`、refs 与组件内部状态
- **表单状态**：由 React Hook Form 在局部页面或弹层中闭环管理

## HTTP 与错误处理

共享 API Client 负责：

- 读取 `NEXT_PUBLIC_API_URL`
- 自动附带 Bearer Token
- 统一设置请求超时
- 将失败响应转换成统一的 `ApiError`

这样可以保证页面与 domain hooks 使用统一的 HTTP 合约。

## 架构备注

- 前端采用 API-first 模式，工作流逻辑归后端所有
- 聊天体验围绕 workbench invoke 与 interrupt 恢复构建
- Harness 体验围绕 Agent Studio 和控制平面操作构建
- 未来新增 UI 功能，优先延续 typed domain hooks 的接入方式
