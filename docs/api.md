# AgFrame API Reference

<div align="center">
  <a href="api-cn.md">中文文档</a>
</div>

## Scope

- **Base URL**: `http://127.0.0.1:8000`
- **Auth model**: `POST /auth/token` returns a JWT and also sets an HttpOnly auth cookie; protected routes accept the cookie and still accept `Authorization: Bearer <token>`
- **Backend version**: `0.3.1`
- **Primary runtime entry**: `POST /chat/workbench-invoke`

## Access Model

- **Public**: `/health/*`, `/auth/*`
- **Authenticated user**: `/chat/*`, `/interrupt/*`, `/history/*`, `/documents/*`, `/upload/*`, `/tasks/*`, `/memory/*`, `/profile/*`, `/knowledge-bases/*`, `/files/*`, `/uploads/*`, `/harness/*`, `GET|POST /settings/user`
- **Admin only**: `GET|POST /settings`, `POST /vectorstore/docs/clear`

## Authentication

- `POST /auth/token` — exchange username/password for an access token and set the auth cookie
- `POST /auth/logout` — clear the auth cookie
- `POST /auth/register` — create a new user account
- `GET /auth/users/me` — retrieve the current authenticated user

Browser clients:

- the frontend primarily authenticates with the HttpOnly cookie rather than storing the JWT in `localStorage`
- if frontend and backend run on different origins or ports, the backend must allow that origin in `CORS_ORIGINS`
- browser-based cookie auth also requires `CORS_ALLOW_CREDENTIALS=true`

## Chat

### Workbench flow

- `POST /chat/workbench-invoke` — main workbench invocation endpoint; applies runtime config, invokes the graph, persists messages, and returns `reply`, `messages`, `context`, and interrupt state

### LangServe runtime routes

The same graph is also exposed under `/chat` through LangServe. The explicitly verified runtime route is:

- `POST /chat/invoke` — LangServe invoke endpoint for the graph runtime

Additional LangServe schema, stream, playground, and feedback routes are mounted automatically under the same `/chat` prefix. Feedback support is enabled in the application bootstrap.

## Interrupt

Session-level human approval and resume flow for interrupted chat execution:

- `GET /interrupt/{session_id}` — read whether the session is interrupted and whether action is required
- `GET /interrupt/{session_id}/events` — list interrupt event records for the session
- `POST /interrupt/{session_id}/approve` — approve or reject the pending action
- `GET /interrupt/{session_id}/resume` — retrieve the resume payload structure
- `POST /interrupt/{session_id}/resume` — resume graph execution and persist the resumed messages

## Harness

### Runs

- `POST /harness/runs` — create a harness run
- `GET /harness/runs` — list visible runs for the current user
- `GET /harness/runs/{run_id}` — get run detail
- `POST /harness/runs/{run_id}/retry` — create a retry run
- `GET /harness/runs/{run_id}/events` — list run lifecycle events
- `GET /harness/runs/{run_id}/runtime-state/history` — read persisted runtime-state history
- `GET /harness/runs/{run_id}/approval` — read pending approval state
- `POST /harness/runs/{run_id}/approval` — resolve a run approval
- `GET /harness/runs/{run_id}/verification` — read the latest verification evidence
- `GET /harness/policies` — list available harness policies

### Studio projects

- `GET /harness/studio/projects` — list studio projects owned by the current user
- `GET /harness/studio/projects/current` — read the current studio project
- `POST /harness/studio/projects` — create a new studio project
- `GET /harness/studio/projects/{project_id}` — read studio project detail
- `PUT /harness/studio/projects/{project_id}` — update project metadata or `graph_json`
- `POST /harness/studio/projects/{project_id}/skill-requests` — request additional skills for an agent
- `POST /harness/studio/projects/{project_id}/skill-requests/{request_id}` — approve or reject a skill request
- `POST /harness/studio/projects/{project_id}/run` — launch an orchestration run from a studio project

### Model providers

- `GET /harness/model-providers` — list configured model providers visible to the current user
- `POST /harness/model-providers` — create a provider entry
- `PUT /harness/model-providers/{provider_id}` — update an existing provider
- `DELETE /harness/model-providers/{provider_id}` — delete a provider

## History

- `GET /history/{user_id}` — list conversation sessions, optionally filtered with `q`
- `GET /history/{user_id}/{session_id}` — read one session detail
- `POST /history/{user_id}/save` — persist a session manually
- `PATCH /history/{user_id}/{session_id}` — rename session metadata
- `DELETE /history/{user_id}/{session_id}` — delete a session

## Documents and Uploads

### Documents

- `GET /documents` — list uploaded documents
- `GET /documents/{doc_id}` — read document detail and preview fragments
- `GET /documents/{doc_id}/download` — download the original stored file for a document
- `DELETE /documents/{doc_id}` — remove the document record and stored file
- `PUT /documents/{doc_id}/knowledge-base` — assign or clear the knowledge base binding for a document
- `POST /documents/{doc_id}/reindex` — enqueue reindexing for a document

### Knowledge bases

- `GET /knowledge-bases` — list visible knowledge bases with document counts
- `POST /knowledge-bases` — create a knowledge base owned by the current user
- `PUT /knowledge-bases/{knowledge_base_id}` — update name or description
- `DELETE /knowledge-bases/{knowledge_base_id}` — delete an owned knowledge base

### Uploads

- `POST /upload` — upload document files and enqueue ingestion
- `POST /upload/image` — upload an image and return a file URL

### File access

- `GET /uploads/{owner}/{relative_path}` — read an uploaded asset with owner or admin authorization
- `GET /files/{owner}/{relative_path}` — read a stored document asset with owner or admin authorization

## Tasks

- `GET /tasks/summary` — aggregate background task status, incidents, and timeout signals
- `GET /tasks/incidents` — list incidents, optionally filtered by handled or archived state
- `PATCH /tasks/incidents/{incident_id}` — update incident handling state
- `GET /tasks/{task_id}` — read task detail and diagnostics
- `POST /tasks/{task_id}/retry` — retry a failed background task

## Memory and Profile

### Memory

- `GET /memory/profile` — read the synthesized user profile
- `PUT /memory/profile` — update the profile and sync semantic memory
- `GET /memory/items` — list atomic memory items
- `POST /memory/items` — create a memory item
- `DELETE /memory/items/{item_id}` — delete a memory item

### Profile

- `GET /profile/{user_id}` — read profile information for a user

## Settings

- `GET /settings` — read global application settings
- `POST /settings` — update global application settings
- `GET /settings/user` — read user-level preferences
- `POST /settings/user` — update user-level preferences

## Vector Store

- `POST /vectorstore/docs/clear` — clear the document vector store

## Health

- `GET /health` — basic health status
- `GET /health/ready` — dependency and runtime readiness check
- `GET /health/live` — lightweight liveness check

## Notes

- `/chat/workbench-invoke` is the UI-facing path used by the workbench frontend.
- `/interrupt/*` is session-scoped and tied to LangGraph checkpoint recovery.
- `/harness/*` is run-scoped and used by the harness control plane and Agent Studio.
- Upload, document, task, and harness APIs are designed to form an operational loop: upload -> enqueue -> observe -> retry -> audit.
