# Platform Governance Control Plane

The governance control plane is responsible for lifecycle authorization and for
maintaining a single authoritative write path for run status changes.

## Single authoritative write path

The intended single authoritative write path for governance decisions is
`app/platform/governance/service.py`.

Current control-plane integrations route harness lifecycle decisions through:

- `app/platform/governance/service.py`
- `app/platform/governance/lifecycle.py`
- `app/harness/runtime/run_service.py`

This keeps runtime execution separate from governance authorization while the
monolith is still being decomposed.

## Platform runtime command dispatch

As of v0.3.3, the platform runtime (`app/platform/runtime/`) serves as the
single entry point for harness execution. The `RuntimeApplicationService`
dispatches `RuntimeCommandV1` instances (`start`, `resume`, `step`, `cancel`)
and validates run state, governance authorization, and execution planning
before delegating to the underlying harness execution path.

### Command flow

1. **Governance phase** — `GovernanceService.authorize_transition` validates the
   lifecycle transition and records audit context (`actor`, `triggered_by`,
   `correlation_id`).
2. **Runtime phase** — `RuntimeApplicationService.accept()` dispatches the
   command and returns a `RuntimeResultV1` with a typed `result_type`
   (`execution_ready`, `resume_ready`, `step_acknowledged`, `cancelled`).
3. **Completion phase** — `HarnessRunService.complete_with_verification` records
   the verification result and transitions the run to a terminal status.

### Lifecycle transitions

The governance lifecycle manager (`GovernanceLifecycleManager`) enforces the
authoritative transition graph including retry (`failed -> created`) and
resume-blocking (`resumed -> waiting_approval`) paths. All transitions carry
full audit context for traceability.

### Event pipeline

Platform events are emitted as `EventEnvelopeV1` envelopes with canonical
event types: `runtime.started`, `runtime.resumed`, `runtime.interrupted`,
`runtime.failed`, `runtime.step_completed`, and `runtime.completed`. The
`HarnessEventService.record_runtime_event` method persists these envelopes.

### Worker integration

The ARQ worker (`app/infrastructure/queue/arq_jobs.py`) routes all task types
(document_ingest, agent_orchestration, session_resume_approval) through
`_accept_runtime_command_for_run`, which builds the runtime command, accepts
it via `RuntimeApplicationService`, and records the result before executing
the task-specific logic.
