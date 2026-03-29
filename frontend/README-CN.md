## AgFrame 前端 (Frontend)

本 Next.js 应用是为 AgFrame FastAPI 后端配套的运维与交互工作台 UI。

目前已实现的路由：
- `/login` (登录)
- `/chat` (对话工作台)
- `/harness` (Harness 控制面：run、approval、verification、timeline、retry)
- `/knowledge` (知识库管理)
- `/conversations` (会话中心)
- `/conversations/[conversationId]` (会话详情)
- `/memory` (记忆控制台)
- `/tasks` (任务与事件观测)
- `/settings` (个人设置)
- `/admin/settings` (系统安全与配置)

## 环境变量 (Environment)

在启动前端之前，请先设置后端服务的 Base URL：

```bash
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

启动后访问 [http://localhost:3000](http://localhost:3000)。

## 鉴权模型 (Auth Model)

- 登录使用接口 `POST /auth/token` 获取凭证。
- 当前用户信息初始化使用 `GET /auth/users/me`。
- 工作台路由需要存储合法的 Token，若 Token 过期将自动重定向回 `/login`。
- 管理员专属导航栏以及 `/admin/settings` 页面依赖当前用户的 `role === "admin"` 字段。

## 开发须知 (Notes)

- 对话界面使用 `POST /chat/workbench-invoke`，由后端统一完成 graph 执行与轮次持久化。
- Harness 控制面读取 `GET /harness/runs`、`GET /harness/runs/{run_id}`、`GET /harness/policies`，并支持 `POST /harness/runs`、`POST /harness/runs/{run_id}/approval`、`POST /harness/runs/{run_id}/retry`。
- 用户个人设置读写对应 `GET|POST /settings/user`。
- 管理员全局设置读写对应 `GET|POST /settings`。
- 文档上传请求指向 `POST /upload`，其 multipart 字段名为 `files`。
- 如果在 `npm install` 之后运行 `next build` 或 `npm run lint` 发生报错，请尝试删除并重新安装 `node_modules`。过去曾因本地包缓存损坏导致构建失败，并非当前的 TypeScript 接口定义有问题。