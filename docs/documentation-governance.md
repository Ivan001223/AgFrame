# Documentation Governance

<div align="center">
  <a href="documentation-governance-cn.md">中文文档</a>
</div>

## Goal

Keep repository documentation aligned with the current codebase, runtime behavior, and operational workflows in both English and Chinese.

## Ownership Model

### Repository entry docs

- `README.md`
- `README-CN.md`

Own:

- project positioning
- architecture summary
- quick start
- document navigation

Do not own:

- full API details
- detailed deployment runbooks
- detailed frontend module behavior

### Formal docs under `docs/`

- `api*.md` own endpoint coverage and interface grouping
- `deployment*.md` own environment, startup, Docker, and operational guidance
- `testing*.md` own regression, smoke, and gate instructions
- `security*.md` own security baseline and release checks
- `frontend-architecture*.md` own frontend structure and data flow
- `rag-architecture*.md` own retrieval design and migration context
- `documentation-package-index.md`, `documentation-change-record.md`, and `documentation-quality-report.md` own documentation-audit inventory, change traceability, and quality sign-off artifacts

### Subsystem entry docs

- `frontend/README*.md` own frontend-specific routes, auth model, and integration notes

### Version and release tracking

- `CHANGELOG.md` owns user-visible release notes
- this governance document owns documentation process rules

## Source of Truth Rules

- code and tests are the primary source of truth
- configuration docs must be verified against `.env.example`, `docker-compose.yml`, and runtime settings models
- API docs must be verified against FastAPI routers and application bootstrap
- frontend docs must be verified against `frontend/package.json`, route files, shared layout, and domain hooks

## Required Update Triggers

Update documentation whenever any of the following changes:

- API endpoints, request shapes, auth requirements, or response behavior
- environment variables, Docker defaults, startup commands, or dependency constraints
- frontend routes, navigation structure, or domain-level integration patterns
- harness run lifecycle, approvals, verification, studio, or provider management behavior
- retrieval defaults, memory behavior, or interrupt and resume flow semantics

## Bilingual Sync Rules

- English and Chinese counterparts must be updated in the same change set
- section structure should remain symmetrical unless a language-specific note is necessary
- endpoint lists, commands, version numbers, and file paths must remain equivalent
- internal audit artifacts may remain Chinese-only when they are used as release-review deliverables and no maintained English counterpart exists

## Versioning and Traceability

- use `CHANGELOG.md` for externally visible release changes
- include version scope or validation date in major technical documents when the context may drift
- keep documentation links pointed to canonical owner documents instead of duplicating large sections
- when behavior changes, update both the owner document and any overview document that references it

## Review Checklist

Before merging a documentation change, verify:

- the described behavior matches code and tests
- English and Chinese files are synchronized
- commands are executable from the repository root
- version numbers and dependency names are current
- file paths and endpoint paths are correct
- overview docs do not contradict canonical docs

## Team Review Roles

- backend reviewer checks API, config, queue, persistence, and runtime behavior
- frontend reviewer checks routes, auth, UI integration, and domain hooks
- platform reviewer checks deployment, Docker, smoke, and release gates

## Recommended Change Workflow

1. identify the canonical document owner
2. verify behavior in code or tests
3. update English and Chinese documents together
4. update `CHANGELOG.md` if the change is externally visible
5. run the relevant validation commands
6. request cross-role review when the change spans backend, frontend, and operations

## Non-Goals

- this document does not replace product planning
- this document does not define implementation architecture beyond documentation ownership
- this document does not require every internal scratch note to be bilingual
