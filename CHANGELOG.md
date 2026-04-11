# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- Standardized the technical documentation set so README, API, deployment, testing, security, frontend architecture, and frontend subsystem docs reflect the current chat runtime, harness control plane, and Agent Studio behavior.
- Added bilingual documentation governance rules covering ownership, update triggers, review expectations, and traceability.
- Synced version, Python runtime, worker topology, knowledge-base APIs, document download and assignment flows, and frontend verification commands across the maintained documentation set.

## [0.3.1] - 2026-04-08

### Added

- Knowledge base CRUD APIs and frontend management flows.
- Document download and knowledge-base assignment endpoints.
- Dedicated ingest and resume worker services alongside the runtime worker.
- Bilingual documentation audit artifacts covering change tracking and quality checks.

### Changed

- Raised the supported Python runtime to `>=3.12,<3.13`.
- Default local startup now documents `dev-stub` chat and embedding configuration for no-cloud development.
- Frontend validation gate now explicitly includes `npm run typecheck`.
- Deployment and README guides now document the three-worker topology used by Docker Compose and manual startup.

### Security

- Startup validation now remains part of the documented deployment baseline for JWT, database password, and CORS credential checks.

## [0.1.1] - 2026-03-14

### Added

- Workbench frontend pages for login, chat, knowledge, conversations, memory, tasks, settings, and admin settings.
- Knowledge management APIs for document listing, search, detail, preview, delete, and reindex flows.
- Conversation center APIs for querying, inspecting, renaming, and deleting conversations.
- Memory console APIs for viewing, updating, creating, and deleting profile and long-term memory data.
- Task operations APIs for diagnostics, suspected timeout marking, retry, event stream review, and archive/resolve actions.
- Health and readiness probes covering database, Redis, vector store, and model component checks.
- Smoke and live-smoke scripts for end-to-end workbench validation.
- Deployment, testing, security, and frontend architecture documentation.

### Changed

- Improved queue persistence and worker integration around Redis and ARQ.
- Improved model component loading and retrieval compatibility for embedding and reranker backends.
- Tightened security scanning and workbench smoke tooling.
- Expanded API and integration test coverage for workbench-facing flows.

### Release Notes

- This release packages the first usable workbench-oriented slice of AgFrame.
