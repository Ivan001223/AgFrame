## AgFrame 前端

<div align="center">
  <a href="README.md">English</a>
</div>

本 Next.js 应用是 AgFrame 后端配套的认证工作台。

## 技术栈

- Next.js `16.2.3`
- React `19.2.3`
- TanStack React Query
- React Hook Form + Zod
- Tailwind CSS `4`

## 路由

- `/register` — 首个管理员引导与可选开放注册页
- `/login` — 登录页
- `/chat` — 对话工作台
- `/harness` — Harness Agent Studio 与控制平面
- `/knowledge` — 知识库管理
- `/knowledge/[docId]` — 文档详情、预览、下载、重建索引与知识库绑定
- `/conversations` — 会话列表
- `/conversations/[conversationId]` — 会话详情
- `/memory` — 记忆控制台
- `/tasks` — 任务与 incident 观测
- `/settings` — 个人设置
- `/admin/settings` — 管理员全局设置

## 环境变量

```bash
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

启动后访问 `http://127.0.0.1:3000`。

如果后端运行在不同的 origin 或端口，请确保后端配置了：

- `CORS_ORIGINS=["http://127.0.0.1:3000","http://localhost:3000"]`
- `CORS_ALLOW_CREDENTIALS=true`

## 鉴权模型

- 登录使用 `POST /auth/token`
- 登出使用 `POST /auth/logout`
- 当前用户初始化使用 `GET /auth/users/me`
- 浏览器鉴权依赖后端设置的 HttpOnly Cookie
- 前端本地仅保留用户名，用于缓存分区和偏好设置
- 鉴权失效时会清理本地会话提示并跳回 `/login`
- 管理员界面依赖 `role === "admin"`

## API 接入说明

### Chat

- 主 UI 链路使用 `POST /chat/workbench-invoke`
- 后端统一负责 graph 执行、最新 state 读取、消息持久化与 interrupt 状态返回

### Harness

Harness 已经不是简单的 run 看板，当前页面整合了：

- run 列表与详情
- approval 与 retry
- policy 可见性
- studio project 加载与编辑
- skill request / skill approval 工作流
- studio run 创建
- model provider 管理

前端实际消费的核心 harness 接口包括：

- `GET /harness/runs`
- `GET /harness/runs/{run_id}`
- `GET /harness/policies`
- `GET /harness/studio/projects`
- `GET /harness/studio/projects/current`
- `GET /harness/model-providers`
- `POST /harness/runs`
- `POST /harness/runs/{run_id}/approval`
- `POST /harness/runs/{run_id}/retry`
- `POST /harness/studio/projects`
- `PUT /harness/studio/projects/{project_id}`
- `POST /harness/studio/projects/{project_id}/skill-requests`
- `POST /harness/studio/projects/{project_id}/skill-requests/{request_id}`
- `POST /harness/studio/projects/{project_id}/run`

### 其他模块

- 用户个人设置对应 `GET|POST /settings/user`
- 管理员设置对应 `GET|POST /settings`
- 文档上传使用 `POST /upload`，multipart 字段名为 `files`
- 知识库管理使用 `GET|POST|PUT|DELETE /knowledge-bases`
- 文档操作还会调用 `GET /documents/{doc_id}/download` 与 `PUT /documents/{doc_id}/knowledge-base`

## 校验命令

```bash
npm run lint -- --max-warnings=0
npm run typecheck
npm run build
```

如果本地安装损坏，先删除 `node_modules` 并重新安装，再判断是否为应用代码问题。
