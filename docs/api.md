# AgFrame API Documentation

This document provides an overview of the RESTful API endpoints available in the AgFrame project. The backend is built with FastAPI and is composed of several modular routing components.

## Base Information
- **Default Port**: 8000 (Local)
- **Base Path**: `/`
- **Authentication**: JWT Bearer Token (passed via `Authorization: Bearer <token>` header).

---

## 1. Authentication (`/auth`)
Handles user authentication, token generation, and registration.

- `POST /auth/token`: Login to obtain a JWT access token.
- `POST /auth/register`: Register a new user account.
- `GET /auth/users/me`: Get the profile of the currently authenticated user.

## 2. Platform Harness (`/harness`)
Control-plane APIs for agent execution runs, approval workflows, and policies.

- `GET /harness/runs`: List agent harness runs.
- `POST /harness/runs`: Create a new harness run.
- `GET /harness/runs/{run_id}`: Retrieve details of a specific harness run.
- `POST /harness/runs/{run_id}/retry`: Retry a failed execution run.
- `GET /harness/runs/{run_id}/events`: Retrieve events associated with a run.
- `GET /harness/runs/{run_id}/approval`: Get the approval workflow status for a run.
- `POST /harness/runs/{run_id}/approval`: Submit an approval decision (Approve/Reject).
- `GET /harness/runs/{run_id}/verification`: Read execution verification evidence.
- `GET /harness/policies`: List runtime execution policies.

## 3. Human-In-The-Loop / Interrupts (`/interrupt`)
Endpoints handling execution interruptions requiring user review or approval.

- `GET /interrupt/{session_id}`: Get interrupt status for a session.
- `GET /interrupt/{session_id}/events`: List queued events pending approval.
- `POST /interrupt/{session_id}/approve`: Provide approval (or rejection) for a specific state.
- `GET /interrupt/{session_id}/resume`: Get the resume payload requirements.
- `POST /interrupt/{session_id}/resume`: Resume execution from an interrupted state.

## 4. Chat & Interactions (`/chat`)
Real-time interaction and inference interfaces.

- `POST /chat/workbench-invoke`: Main invocation endpoint for the chat workbench interface.

## 5. Session History (`/history`)
APIs to manage conversations and event histories.

- `GET /history/{user_id}`: Get a list of chat sessions for a user.
- `GET /history/{user_id}/{session_id}`: Retrieve message history for a specific session.
- `POST /history/{user_id}/save`: Persist current session state manually.
- `PATCH /history/{user_id}/{session_id}`: Update session metadata (e.g., renaming the session).
- `DELETE /history/{user_id}/{session_id}`: Delete a session entirely.

## 6. Document Management (`/documents` & `/upload`)
Manage knowledge base files and media.

- `GET /documents`: List stored documents.
- `GET /documents/{doc_id}`: Get metadata of a specific document.
- `DELETE /documents/{doc_id}`: Remove a document from storage.
- `POST /documents/{doc_id}/reindex`: Re-trigger parsing and vector store indexing.
- `POST /upload`: Upload a standard document or file.
- `POST /upload/image`: Upload an image file.

## 7. Vector Store Administrative (`/vectorstore`)
- `POST /vectorstore/docs/clear`: Clear documents in the vector store (Requires Admin rights).

## 8. Agent Tasks (`/tasks`)
Manage detached background tasks and review issues.

- `GET /tasks/summary`: Get a summary dashboard of active and completed tasks.
- `GET /tasks/{task_id}`: View status and details of a background worker task.
- `POST /tasks/{task_id}/retry`: Retry an uncompleted or failed task.
- `GET /tasks/incidents`: List task execution incidents.
- `PATCH /tasks/incidents/{incident_id}`: Acknowledge or mitigate an incident.

## 9. Memory Storage (`/memory`)
Endpoints to handle user-wide or context-wide persistent memories.

- `GET /memory/profile`: View the current synthesized memory profile.
- `PUT /memory/profile`: Manually update profile memories.
- `GET /memory/items`: List atomic memory items.
- `POST /memory/items`: Store a new memory item.
- `DELETE /memory/items/{item_id}`: Remove a memory item.

## 10. User Profile & Settings (`/profile` & `/settings`)
Manage platform configurations.

- `GET /profile/{user_id}`: Retrieve user profile metadata.
- `GET /settings`: Read global application settings (Admin).
- `POST /settings`: Update global application settings (Admin).
- `GET /settings/user`: Read user-specific preferences.
- `POST /settings/user`: Save user-specific preferences.

## 11. System Health (`/health`)
Observability and liveness checks (Publicly accessible).

- `GET /health`: Basic health endpoint.
- `GET /health/ready`: Checks if dependencies (DB, Redis, etc.) are up.
- `GET /health/live`: Lightweight ping for orchestration tools (e.g. K8s).
