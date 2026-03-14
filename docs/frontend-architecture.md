# AgFrame Frontend Architecture

## 1. Scope

This document defines the frontend architecture for a new AgFrame workbench UI that connects to the existing FastAPI and LangServe backend.

Implementation status in the current repository:
- `/login`, `/chat`, `/knowledge`, `/conversations`, `/memory`, `/tasks`, `/settings`, `/admin/settings` are implemented
- current-user bootstrap and workspace route guards are implemented
- the frontend is aligned to the current FastAPI routes, not a hypothetical future API

Current backend status:
- REST APIs for auth, documents, tasks, history, memory, profile, settings, and health already exist.
- Chat entrypoint is `POST /chat/invoke`.
- Upload, reindex, and ingest are asynchronous and observable through `/tasks/*`.

Frontend goal:
- Build a production-oriented operations workbench instead of a demo chat page.
- Make the existing backend capabilities usable through a coherent UI.
- Keep the frontend thin: business rules stay in backend APIs, frontend focuses on orchestration, rendering, and interaction state.

## 2. Product Modules

The frontend should be split by user-facing domains, not by backend files.

Core modules:
- Auth Center
- Chat Workbench
- Knowledge Base
- Task Operations
- Conversation Center
- Memory Console
- User Settings
- Admin Settings

Recommended first release:
- Login and session bootstrap
- Chat workbench
- Document upload and task tracking
- Document list and detail
- Conversation list and detail
- Memory profile and memory items
- Task incidents and diagnostics
- User settings and admin settings

## 3. Recommended Stack

Use a single frontend application with server-side rendering support.

Recommended stack:
- Framework: Next.js 15 with App Router
- Language: TypeScript
- UI: React 19
- Data fetching: TanStack Query
- Local client state: Zustand
- Forms: React Hook Form + Zod
- Styling: Tailwind CSS
- Component primitives: Radix UI
- Tables: TanStack Table
- Charts: Recharts
- HTTP client: `fetch` wrapper, not axios

Why this stack:
- Next.js reduces routing and build setup overhead.
- TanStack Query fits this backend well because most state is server-owned.
- Zustand is enough for UI state like active session, filters, drawer state, and upload queue.
- React Hook Form keeps admin/settings forms maintainable.

Do not use:
- Redux for this project size
- GraphQL layer on top of current REST APIs
- Large client-side domain logic that duplicates backend rules

## 4. Application Architecture

Use a layered frontend:

1. App shell layer
- route layout
- auth bootstrap
- navigation
- global error boundaries

2. Feature layer
- pages and containers for each product module
- user flows
- feature-specific hooks

3. Domain layer
- typed API clients
- query keys
- mutation hooks
- DTO to view-model mapping

4. Shared layer
- UI components
- table/filter primitives
- upload widgets
- task status badges
- date/number/diagnostic formatters

Design rule:
- Pages call feature hooks.
- Feature hooks call domain clients.
- Domain clients are the only place that knows raw endpoint paths.

## 5. Proposed Directory Structure

```text
frontend/
  src/
    app/
      (public)/
        login/page.tsx
      (workspace)/
        layout.tsx
        chat/page.tsx
        knowledge/page.tsx
        knowledge/[docId]/page.tsx
        conversations/page.tsx
        conversations/[sessionId]/page.tsx
        memory/page.tsx
        tasks/page.tsx
        settings/page.tsx
        admin/settings/page.tsx
      api/
    features/
      auth/
      chat/
      documents/
      history/
      memory/
      tasks/
      settings/
    domains/
      auth/
      chat/
      documents/
      history/
      memory/
      tasks/
      settings/
      shared/
    components/
      layout/
      feedback/
      forms/
      tables/
      status/
    stores/
      app-shell.ts
      chat-ui.ts
      upload-ui.ts
    lib/
      http/
      env/
      auth/
      query/
      utils/
    styles/
    types/
```

Rule of thumb:
- `features/` owns screens and interaction logic.
- `domains/` owns API contracts and reusable hooks.
- `stores/` only stores transient UI state, not server truth.

## 6. Routing Design

Recommended routes:

Public:
- `/login`

Workspace:
- `/chat`
- `/knowledge`
- `/knowledge/[docId]`
- `/conversations`
- `/conversations/[sessionId]`
- `/memory`
- `/tasks`
- `/settings`
- `/admin/settings`

Default landing:
- If authenticated, redirect `/` to `/chat`
- If not authenticated, redirect `/` to `/login`

Current implementation note:
- `/` performs a client-side redirect based on the stored session token

## 7. Backend Mapping

### 7.1 Auth

Backend:
- `POST /auth/token`
- `POST /auth/register`
- `GET /auth/users/me`

Frontend responsibility:
- login form
- token persistence
- bootstrap current user
- route guard

Token strategy:
- Store access token in HTTP-only cookie if a BFF layer is introduced later.
- For the first version, store JWT in memory plus `localStorage` fallback.
- Add `Authorization: Bearer <token>` in the shared fetch wrapper.

Current implementation:
- JWT and username are stored in `localStorage`
- `GET /auth/users/me` is queried on workspace entry to validate the session

### 7.2 Chat Workbench

Backend:
- `POST /chat/invoke`
- `GET /interrupt/{session_id}`
- `POST /interrupt/{session_id}/approve`
- `GET /interrupt/{session_id}/resume`
- `GET /history/{user_id}`
- `POST /history/{user_id}/save`

Frontend layout:
- left: session list
- center: conversation timeline
- right: optional diagnostics drawer

Chat sub-modules:
- message composer
- message timeline
- interrupt approval bar
- citation/document reference area
- session metadata panel

Implementation note:
- Hide raw LangServe payload shape behind `domains/chat/client.ts`.
- The rest of the app should work with a normalized message model.

Current implementation:
- chat payload normalization lives in the frontend domain layer
- successful chat turns are persisted through `POST /history/{user_id}/save`

### 7.3 Knowledge Base

Backend:
- `POST /upload`
- `GET /documents`
- `GET /documents/{doc_id}`
- `DELETE /documents/{doc_id}`
- `POST /documents/{doc_id}/reindex`
- `GET /tasks/{task_id}`

Frontend layout:
- upload area
- document table
- document detail drawer/page
- task status side panel

Key behavior:
- upload returns queued or duplicate
- queued uploads must register into a task polling manager
- reindex should create a new task record in UI immediately

### 7.4 Task Operations

Backend:
- `GET /tasks/summary`
- `GET /tasks/incidents`
- `PATCH /tasks/incidents/{incident_id}`
- `GET /tasks/{task_id}`
- `POST /tasks/{task_id}/retry`

Frontend layout:
- top summary cards
- incident table
- task detail drawer
- retry and incident action toolbar

Important current semantics:
- incidents support `handled` and `archived`
- incidents can be filtered by `handled` and `archived`
- `summary.recent_incidents` excludes archived items by default

### 7.5 Conversation Center

Backend:
- `GET /history/{user_id}`
- `GET /history/{user_id}/{session_id}`
- `PATCH /history/{user_id}/{session_id}`
- `DELETE /history/{user_id}/{session_id}`

Frontend layout:
- searchable conversation list
- detail preview
- rename and delete actions

### 7.6 Memory Console

Backend:
- `GET /memory/profile`
- `PUT /memory/profile`
- `GET /memory/items`
- `POST /memory/items`
- `DELETE /memory/items/{item_id}`

Frontend layout:
- profile editor
- memory item table
- add/delete controls

### 7.7 Settings

Backend:
- `GET /settings/user`
- `POST /settings/user`
- `GET /settings`
- `POST /settings`

Frontend split:
- personal settings page
- admin settings page

## 8. State Management Rules

Use three state classes.

### 8.1 Server state

Managed by TanStack Query:
- current user
- session lists
- document lists
- document detail
- task summary
- incidents
- memory profile
- memory items
- user settings
- admin settings

### 8.2 UI state

Managed by Zustand:
- active navigation item
- selected document row
- selected task id
- open drawers and dialogs
- active chat session id
- optimistic upload queue items
- table filter persistence

### 8.3 Form state

Managed locally via React Hook Form:
- login form
- profile form
- add memory form
- settings forms

Do not put form state into Zustand.

## 9. Data Fetching and Cache Policy

Use typed query hooks per domain.

Examples:
- `useCurrentUserQuery()`
- `useDocumentsQuery(filters)`
- `useDocumentDetailQuery(docId)`
- `useTaskSummaryQuery()`
- `useTaskIncidentsQuery(filters)`
- `useMemoryProfileQuery()`

Suggested polling:
- task detail: every 2 seconds while `queued` or `running`
- task summary: every 10 seconds on tasks page
- task incidents: every 10 to 15 seconds on tasks page
- documents/history/memory: manual invalidation after mutation

Invalidation rules:
- upload success -> invalidate `documents`, `taskSummary`, `taskIncidents`
- reindex start -> invalidate `taskSummary`, task detail for that task id
- incident patch -> invalidate `taskIncidents`, `taskSummary`
- memory update -> invalidate `memoryProfile`, `memoryItems`
- history rename/delete -> invalidate history list and current session detail

## 10. HTTP Client Design

Create one shared HTTP wrapper:

Responsibilities:
- inject token
- set `X-Request-ID`
- parse JSON
- normalize backend error shape
- map `401` to logout flow
- expose typed `ApiError`

Suggested files:
- `src/lib/http/client.ts`
- `src/lib/http/errors.ts`

Recommended error shape:

```ts
type ApiError = {
  status: number
  code: string
  message: string
  requestId?: string
  detail?: unknown
}
```

## 11. Domain Models

Frontend should not directly bind tables and forms to backend raw payloads. Introduce normalized view models.

Examples:

```ts
type TaskIncident = {
  incidentId: string
  taskId: string
  userId: string
  errorCode: string
  errorMessage: string
  stage: string
  handled: boolean
  archived: boolean
  handledAt?: number | null
  archivedAt?: number | null
  updatedAt?: number | null
  timestamp?: number | null
}

type TaskDiagnosticsView = {
  status: string
  stage: string
  title: string
  userMessage: string
  suggestedAction: string
  retryable: boolean
  timeoutExceeded: boolean
}
```

Benefits:
- isolates backend field naming
- reduces component conditionals
- makes refactoring cheaper

## 12. Page-Level Design

### 12.1 Chat Page

Sections:
- session sidebar
- timeline
- composer
- interrupt banner
- optional references panel

Actions:
- send message
- resume interrupted session
- save current session metadata
- open related document or memory references

### 12.2 Knowledge Page

Sections:
- upload dropzone
- active ingest tasks strip
- document table
- right-side preview panel

Actions:
- upload
- search
- view preview
- delete
- reindex

### 12.3 Tasks Page

Sections:
- summary cards
- incidents table
- task detail drawer

Actions:
- filter by handled/archived/error code
- mark handled
- archive
- retry failed task
- inspect diagnostics

### 12.4 Memory Page

Sections:
- profile form
- memory list
- add memory dialog

Actions:
- edit profile
- add item
- delete item

## 13. Shared Components

Build reusable components early:
- `AppShell`
- `SidebarNav`
- `PageHeader`
- `StatusBadge`
- `TaskStatusBadge`
- `DiagnosticsCard`
- `IncidentTable`
- `FilterBar`
- `UploadDropzone`
- `ConfirmDialog`
- `EmptyState`
- `ErrorState`
- `LoadingBlock`

Avoid building page-specific versions of status cards, filters, and tables.

## 14. Chat Integration Strategy

The `/chat/invoke` contract should be wrapped behind a frontend adapter.

Recommended adapter responsibilities:
- send normalized conversation input
- map LangServe response to message list updates
- extract interrupt state if present
- emit frontend events for:
  - conversation appended
  - approval required
  - references updated

Do not let page components construct LangServe request payloads directly.

## 15. Upload and Task Event Flow

Recommended frontend flow:

1. User uploads file
2. Upload mutation returns result list
3. For each queued result:
   - create temporary UI row
   - start polling `/tasks/{task_id}`
4. On task success:
   - invalidate document list
   - update task summary
5. On task failure:
   - show diagnostics
   - allow retry
   - allow marking incident handled after triage

This flow should be centralized in `features/documents` and `features/tasks`, not duplicated across pages.

## 16. Permissions and Guards

Frontend permission model:
- regular user
- admin

Guards:
- all workspace routes require authenticated user
- admin settings and vectorstore operations require admin role
- incident and task actions should be hidden when backend would reject them

Do not rely only on UI guards. Backend remains source of truth.

## 17. Error Handling

Implement three layers:

1. Request-level
- toast for mutation failure
- field error display for forms

2. Module-level
- table empty/error state
- retry button

3. App-level
- route error boundary
- auth expiration redirect

For tasks and incidents, prefer backend diagnostics text over generic frontend messages.

## 18. Observability

Frontend should emit:
- page view
- upload started/completed/failed
- reindex started/completed/failed
- incident handled/archived
- retry clicked
- chat invoke started/completed/failed

If no analytics system exists yet, start with a typed event logger abstraction so analytics can be added later without rewriting feature code.

## 19. Testing Strategy

Required test layers:

1. Unit tests
- DTO mappers
- filter helpers
- diagnostics formatting

2. Component tests
- incident table actions
- upload flow states
- settings forms

3. Integration tests
- auth bootstrap
- upload -> task polling -> documents refresh
- incident patch -> summary refresh

Recommended tools:
- Vitest
- Testing Library
- MSW
- Playwright for smoke flows

## 20. Delivery Plan

### Phase 1

Goal: usable internal workbench

Build:
- auth
- app shell
- documents page
- tasks page
- conversation page
- memory page

### Phase 2

Goal: complete daily-use console

Build:
- chat workbench
- interrupt approval UX
- admin settings
- richer diagnostics views

### Phase 3

Goal: operational maturity

Build:
- analytics hooks
- frontend feature flags
- dark-launch support for new modules
- e2e regression suite

## 21. Suggested Implementation Order

1. Scaffold Next.js app and app shell
2. Implement auth bootstrap and fetch client
3. Implement documents and task operations first
4. Implement conversation center
5. Implement memory console
6. Implement chat workbench
7. Implement admin settings
8. Add Playwright smoke coverage

Reason:
- documents + tasks already form the strongest backend workflow
- they validate auth, polling, diagnostics, incidents, and CRUD patterns in one pass

## 22. Non-Goals for V1

Do not include in the first frontend version:
- multi-tenant theme system
- offline mode
- websocket infrastructure
- drag-and-drop layout builder
- custom plugin runtime in browser

These add complexity without improving initial product usability.

## 23. Final Recommendation

Use one Next.js workspace frontend, organized by product modules and backed by TanStack Query plus a thin typed API layer.

The first frontend milestone should not start with chat. It should start with:
- Knowledge Base
- Task Operations
- Conversation Center

This order is better aligned with the current backend maturity and will expose integration gaps earlier than a chat-first UI.
