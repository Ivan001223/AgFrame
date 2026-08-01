# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.3.3] - 2026-08-01

### Added

- `RuntimeApplicationService` now dispatches real runtime commands (`start`, `resume`, `step`, `cancel`) instead of returning a placeholder `accepted` result, integrating with `HarnessRunService`, governance authorization, and the worker adapter execution plan.
- ARQ worker execution entry point (`execute_harness_run`) now routes through `RuntimeApplicationService` and `build_runtime_command_for_run` so the platform runtime command is the single authoritative entry.
- Platform event constructors for `runtime.started`, `runtime.resumed`, `runtime.interrupted`, `runtime.failed` in `app/platform/runtime/events.py`.
- Governance lifecycle transitions for `failed -> created` (retry) and `resumed -> waiting_approval` with audit context (`actor`, `triggered_by`, `correlation_id`).
- `openapi.json` regenerated from the FastAPI application covering all 72 public endpoints including 20 harness and interrupt routes.
- New test coverage: `tests/test_governance_lifecycle.py` (boundary and illegal transitions), updated `tests/test_platform_runtime_service.py` (real execution paths), `tests/test_platform_runtime_events.py` (all event types), `tests/test_platform_runtime_worker_adapter.py` (plan-to-phase alignment).
- `defusedxml` added as an explicit runtime dependency for safe XML parsing in enhanced search.

### Changed

- Unified version to `0.3.3` across `pyproject.toml`, `frontend/package.json`, `app/server/main.py`, README, and all documentation.
- `GovernanceService.authorize_transition` and `GovernanceLifecycleManager.transition` now accept audit context parameters (`actor`, `triggered_by`, `correlation_id`) for governance traceability.
- All `_transition_run_status` call sites in `HarnessRunService` now pass `actor` and `reason` for lifecycle audit.
- `HarnessEventService` now supports writing events in `EventEnvelopeV1` canonical format via `record_runtime_event`.
- Audit details in `app/platform/governance/audit.py` now include `correlation_id` field for cross-service traceability.

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
