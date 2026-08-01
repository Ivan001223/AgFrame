from __future__ import annotations

from app.platform.contracts.event import EventEnvelopeV1


def build_runtime_step_completed_event(
    *,
    run_id: str,
    step_index: int,
    agent_id: str,
    agent_name: str | None,
    loop_number: int,
) -> EventEnvelopeV1:
    return EventEnvelopeV1(
        event_id=f"runtime-step-{run_id}-{step_index}",
        event_type="runtime.step_completed",
        aggregate_id=run_id,
        payload={
            "step_index": step_index,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "loop_number": loop_number,
        },
    )


def build_runtime_started_event(
    *,
    run_id: str,
    task_type: str,
    actor: str | None = None,
) -> EventEnvelopeV1:
    return EventEnvelopeV1(
        event_id=f"runtime-started-{run_id}",
        event_type="runtime.started",
        aggregate_id=run_id,
        actor=actor,
        payload={
            "task_type": task_type,
        },
    )


def build_runtime_resumed_event(
    *,
    run_id: str,
    task_type: str,
    next_step_index: int = 0,
    recovery_mode: str | None = None,
    actor: str | None = None,
) -> EventEnvelopeV1:
    return EventEnvelopeV1(
        event_id=f"runtime-resumed-{run_id}",
        event_type="runtime.resumed",
        aggregate_id=run_id,
        actor=actor,
        payload={
            "task_type": task_type,
            "next_step_index": next_step_index,
            "recovery_mode": recovery_mode,
        },
    )


def build_runtime_interrupted_event(
    *,
    run_id: str,
    interrupt_type: str,
    reason: str | None = None,
    resumable: bool = True,
) -> EventEnvelopeV1:
    return EventEnvelopeV1(
        event_id=f"runtime-interrupted-{run_id}",
        event_type="runtime.interrupted",
        aggregate_id=run_id,
        payload={
            "interrupt_type": interrupt_type,
            "reason": reason,
            "resumable": resumable,
        },
    )


def build_runtime_failed_event(
    *,
    run_id: str,
    error_type: str,
    error_message: str | None = None,
    task_type: str | None = None,
) -> EventEnvelopeV1:
    return EventEnvelopeV1(
        event_id=f"runtime-failed-{run_id}",
        event_type="runtime.failed",
        aggregate_id=run_id,
        payload={
            "error_type": error_type,
            "error_message": error_message,
            "task_type": task_type,
        },
    )


def build_runtime_completed_event(
    *,
    run_id: str,
    agent_outputs: dict[str, object],
    output_artifacts: dict[str, object],
    errors: list[str],
    review_agent_enabled: bool,
    recovery_mode: str | None,
) -> EventEnvelopeV1:
    return EventEnvelopeV1(
        event_id=f"runtime-completed-{run_id}",
        event_type="runtime.completed",
        aggregate_id=run_id,
        payload={
            "agent_outputs": agent_outputs,
            "output_artifacts": output_artifacts,
            "errors": errors,
            "review_agent_enabled": review_agent_enabled,
            "recovery_mode": recovery_mode,
        },
    )
