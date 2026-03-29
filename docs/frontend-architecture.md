# AgFrame Frontend Architecture

<div align="center">
  <a href="frontend-architecture-cn.md">中文文档</a>
</div>

## Scope

- **Framework**: Next.js `16.1.6`
- **React**: `19.2.3`
- **Data layer**: TanStack React Query `5.x`
- **Styling**: Tailwind CSS `4`
- **Forms**: React Hook Form + Zod

## Overview

The frontend is a Next.js App Router workbench that sits on top of the FastAPI backend. Its primary role is not to own agent logic, but to provide operational and interactive views over backend-owned execution flows:

- chat workbench
- knowledge ingestion
- conversations
- memory management
- task observation
- settings
- Harness Agent Studio

Most business behavior lives in backend APIs. The frontend focuses on route composition, authenticated API access, polling, and operational UI state.

## Layering

### App layer

- `src/app/layout.tsx` defines the root layout and wraps the app with the shared query provider
- `src/app/(workspace)/layout.tsx` applies `AuthGuard` and `AppShell` to authenticated workspace routes
- `src/app/(public)/login/page.tsx` handles the public login entry

### Domain layer

Domain hooks under `src/domains/*/hooks.ts` define typed API access for:

- auth
- chat
- conversations
- documents
- harness
- memory
- settings
- tasks

This is the main frontend abstraction boundary. Pages should consume domain hooks instead of calling `fetch` directly.

### Shared infrastructure layer

- `src/lib/http/client.ts` centralizes base URL resolution, bearer token injection, timeout handling, and `ApiError` normalization
- `src/lib/auth/session.ts` manages client-side token persistence
- `src/components/layout/*` contains the shared authenticated shell and guard

## Current Technology Choices

The current codebase uses:

- Next.js App Router
- React 19 client components
- TanStack React Query for server state
- React Hook Form and Zod for forms
- Lucide React for icons
- Tailwind CSS 4 for styling

The current frontend does **not** depend on:

- Zustand
- Radix UI
- TanStack Table
- Recharts

## Route Structure

Current workspace routes:

- `/chat`
- `/harness`
- `/knowledge`
- `/conversations`
- `/conversations/[conversationId]`
- `/memory`
- `/tasks`
- `/settings`
- `/admin/settings`

Public route:

- `/login`

`AppShell` currently renders primary navigation for chat, knowledge, conversations, memory, tasks, settings, and conditionally admin settings for admin users. Harness is a dedicated workspace route with its own large surface area.

## Data Flow

### Authentication

- login uses `POST /auth/token`
- current-user bootstrap uses `GET /auth/users/me`
- invalid or expired auth state redirects the user back to `/login`

### Chat

The main frontend chat path uses `POST /chat/workbench-invoke`, not a direct LangServe-only client flow. The backend owns:

- runtime config injection
- graph invocation
- latest state retrieval
- message persistence
- interrupt state reporting

### Harness

The harness frontend is now an Agent Studio surface rather than a simple run dashboard. It includes:

- run list and run detail
- approval and retry actions
- policy visibility
- studio project list and current project loading
- graph editing and persistence
- skill request and skill approval workflows
- studio run launching
- model provider management

### Knowledge and tasks

Knowledge pages coordinate upload, document listing, preview, and reindexing. Task pages surface queue status, incidents, and retry flows for asynchronous backend work.

## State Management

- **Server state**: React Query handles fetching, caching, invalidation, and refresh
- **Session state**: auth token and current-user bootstrap live in the auth utilities and auth hooks
- **Local UI state**: page-local `useState`, refs, and component-level state manage transient interaction state
- **Form state**: React Hook Form keeps submission state local to the relevant page or modal

## HTTP and Error Handling

The shared API client is responsible for:

- reading `NEXT_PUBLIC_API_URL`
- attaching bearer tokens
- applying request timeouts
- converting failed responses into normalized `ApiError` instances

This keeps pages and domain hooks aligned on one HTTP contract.

## Architectural Notes

- the frontend is API-first and backend-owned for workflow logic
- the chat experience is centered on workbench invoke plus interrupt recovery
- the harness experience is centered on Agent Studio and control-plane operations
- typed domain hooks are the preferred integration seam for future UI changes
