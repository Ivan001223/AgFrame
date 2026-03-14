# AgFrame 0.1.1 Release Plan

## Positioning

`0.1.1` should be treated as the first stable workbench-oriented patch release for the current `0.1.x` line.

This version is not primarily about new experiments. It is about packaging the work already completed on:

- workbench frontend
- knowledge management
- conversation management
- memory operations
- task operations
- health checks
- smoke-based release validation

If you want the version number to reflect the actual scope of change, `0.2.0` is also defensible. If the team wants to preserve the current branch and milestone naming, keep `0.1.1` and treat it as an internal milestone release.

## Release Scope

### In scope

- Frontend workbench routes are available and wired to backend APIs.
- Backend management APIs for documents, conversations, memory, tasks, settings, and health checks are included.
- Queue, persistence, and model-loading improvements from the current branch are included.
- Smoke scripts and minimum documentation are included as release assets.

### Out of scope

- Retrieval operations dashboard with quality analytics.
- Approval-resume service wrapper for human-in-the-loop.
- Rich user preference controls for model, style, and retrieval strategy.
- Admin governance features such as quota, audit, and tenant management.

## Must-Fix Before Release

### 1. Version alignment

The repository currently exposes conflicting versions and needs a single release number:

- backend package: `1.0.1`
- FastAPI app: `1.0`
- health endpoint: `1.0.1`
- frontend package: `0.1.0`

Required action:

- change all user-facing version strings to `0.1.1`
- update lockfiles or generated metadata if versioned
- verify `/health` returns `0.1.1`

## Release Checklist

### Product

- Confirm the release name and external version number.
- Confirm whether this is internal-only or user-facing.
- Freeze `0.1.1` scope and move all remaining P1 items out of the release.

### Engineering

- Align version strings across backend, frontend, and API metadata.
- Add release notes to `CHANGELOG.md`.
- Recheck README so the "latest progress" section reads like release content, not draft iteration notes.
- Confirm no temporary debug code or draft endpoints are being shipped.

### Validation

- Run backend test suite.
- Run targeted regression for settings, misc APIs, and workbench-related coverage.
- Run `./scripts/smoke_workbench.sh`.
- If an environment is available, run `./scripts/live_workbench_smoke.sh --base-url http://127.0.0.1:8000`.
- Run frontend `npm run lint -- --max-warnings=0`.
- Run frontend `npx next build`.
- Run security scan and confirm no high-risk findings.

### Release artifact

- Publish a short changelog focused on user-visible capabilities.
- Record known limitations for P1 items that were intentionally deferred.
- Tag the release only after the above checks pass.

## Suggested User-Facing Changelog

### 0.1.1 highlights

- Added a complete first-pass workbench frontend covering login, chat, knowledge, conversations, memory, tasks, settings, and admin settings.
- Added backend management APIs for documents, conversations, memories, and task operations.
- Added richer health and readiness checks for database, Redis, vector store, and model dependencies.
- Added smoke-based validation scripts for the main workbench flow.
- Improved queue persistence, model component loading, and release security checks.

## Exit Criteria

`0.1.1` is ready when all conditions below are true:

- every visible version string reports `0.1.1`
- backend tests pass
- frontend lint and build pass
- smoke scripts pass or any skipped item is explicitly documented
- release notes are committed
- deferred P1 work is not implied to be part of the release
