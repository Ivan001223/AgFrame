# Platform Domain Stage 0 Inventory

This inventory captures the Stage 0 baseline before deeper domain extraction.
The current authoritative write path for harness run lifecycle changes starts in
`app/harness/runtime/run_service.py` and is exercised by runtime orchestration
through `app/infrastructure/queue/arq_jobs.py`.

## Current authoritative write path

- `app/harness/runtime/run_service.py`
  - Owns harness run lifecycle transitions, approval state changes, and
    verification recording for the existing control plane.
- `app/infrastructure/queue/arq_jobs.py`
  - Owns runtime execution and review interruption handling, and writes the
    orchestration resume payload that feeds retries and resumes.

## Stage 0 extraction targets

- Preserve a single authoritative write path for lifecycle state changes while
  extracting canonical platform contracts.
- Preserve a single authoritative write path for orchestration resume payloads
  while moving review and interruption semantics into the runtime protocol.
