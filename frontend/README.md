## AgFrame Frontend

This Next.js app is the workbench UI for the AgFrame FastAPI backend.

Implemented routes:
- `/login`
- `/chat`
- `/knowledge`
- `/conversations`
- `/conversations/[conversationId]`
- `/memory`
- `/tasks`
- `/settings`
- `/admin/settings`

## Environment

Set the backend base URL before starting the frontend:

```bash
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Auth Model

- Login uses `POST /auth/token`
- Current user bootstrap uses `GET /auth/users/me`
- Workspace routes require a stored token and redirect back to `/login` on expiry
- Admin navigation and `/admin/settings` depend on `role === "admin"`

## Notes

- Chat uses `POST /chat/invoke` and persists turns through `/history/{user}/save`
- User settings read/write `GET|POST /settings/user`
- Admin settings read/write `GET|POST /settings`
- Document upload targets `POST /upload` with multipart field name `files`
- If `next build` or `npm run lint` fail after install, refresh `node_modules`; earlier local failures were caused by corrupted package artifacts, not by the current TypeScript contracts
