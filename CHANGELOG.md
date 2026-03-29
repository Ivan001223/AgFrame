# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- Standardized the technical documentation set so README, API, deployment, testing, security, frontend architecture, and frontend subsystem docs reflect the current chat runtime, harness control plane, and Agent Studio behavior.
- Added bilingual documentation governance rules covering ownership, update triggers, review expectations, and traceability.

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
- Before tagging the release, align all in-repo version strings to `0.1.1`.
