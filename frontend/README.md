## AgFrame Frontend

<div align="center">
  <a href="README-CN.md">中文文档</a>
</div>

This Next.js application is the operations and interactive workbench UI accompanying the AgFrame FastAPI backend.

Currently implemented routes:
- `/login` (Login)
- `/chat` (Chat Workbench)
- `/harness` (Harness Control Plane: run, approval, verification, timeline, retry)
- `/knowledge` (Knowledge Base Management)
- `/conversations` (Conversation Center)
- `/conversations/[conversationId]` (Conversation Details)
- `/memory` (Memory Console)
- `/tasks` (Task and Event Observation)
- `/settings` (Personal Settings)
- `/admin/settings` (System Security and Configuration)

## Environment

Before starting the frontend, please set the Base URL for the backend service:

```bash
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

Access [http://localhost:3000](http://localhost:3000) after starting.

## Auth Model

- Login uses the `POST /auth/token` endpoint to get credentials.
- Current user information initialization uses `GET /auth/users/me`.
- Workbench routes need to store valid Tokens, if the Token expires it will automatically redirect back to `/login`.
- Admin exclusive navigation bar and `/admin/settings` page depend on the `role === "admin"` field of the current user.

## Notes

- Chat interface uses `POST /chat/workbench-invoke`, the backend uniformly completes graph execution and turn persistence.
- Harness control plane reads `GET /harness/runs`, `GET /harness/runs/{run_id}`, `GET /harness/policies`, and supports `POST /harness/runs`, `POST /harness/runs/{run_id}/approval`, `POST /harness/runs/{run_id}/retry`.
- User personal settings read/write corresponds to `GET|POST /settings/user`.
- Admin global settings read/write corresponds to `GET|POST /settings`.
- Document upload request points to `POST /upload`, its multipart field name is `files`.
- If you encounter errors running `next build` or `npm run lint` after `npm install`, please try deleting and reinstalling `node_modules`. In the past, build failures were caused by corrupted local package caches, not current TypeScript interface definition issues.