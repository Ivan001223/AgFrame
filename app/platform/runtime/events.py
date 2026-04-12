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
