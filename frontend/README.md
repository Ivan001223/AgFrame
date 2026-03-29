## AgFrame Frontend

<div align="center">
  <a href="README-CN.md">中文文档</a>
</div>

This Next.js application is the authenticated workbench for the AgFrame backend.

## Stack

- Next.js `16.1.6`
- React `19.2.3`
- TanStack React Query
- React Hook Form + Zod
- Tailwind CSS `4`

## Routes

- `/login` — login page
- `/chat` — chat workbench
- `/harness` — Harness Agent Studio and control-plane surface
- `/knowledge` — knowledge base management
- `/conversations` — conversation list
- `/conversations/[conversationId]` — conversation detail
- `/memory` — memory console
- `/tasks` — task and incident observation
- `/settings` — personal settings
- `/admin/settings` — admin-only global settings

## Environment

```bash
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

Open `http://127.0.0.1:3000` after startup.

## Auth Model

- login uses `POST /auth/token`
- current-user bootstrap uses `GET /auth/users/me`
- authenticated workspace routes require a stored token
- expired or invalid auth redirects back to `/login`
- admin-only UI depends on `role === "admin"`

## API Integration Notes

### Chat

- the main UI path is `POST /chat/workbench-invoke`
- the backend owns graph execution, latest-state loading, persistence, and interrupt reporting

### Harness

Harness is no longer just a run dashboard. The page integrates:

- run listing and run detail
- approvals and retries
- policy visibility
- studio project loading and editing
- skill request and skill approval workflows
- studio run creation
- model provider management

Primary harness endpoints consumed by the frontend include:

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

### Other modules

- personal settings map to `GET|POST /settings/user`
- admin settings map to `GET|POST /settings`
- document uploads use `POST /upload` with multipart field `files`

## Verification

```bash
npm run lint -- --max-warnings=0
npm run build
```

If a local install becomes corrupted, remove `node_modules` and reinstall before treating the issue as an application bug.
