from app.platform.runtime.events import (
    build_runtime_completed_event,
    build_runtime_step_completed_event,
)


def test_runtime_step_completion_can_be_emitted_as_canonical_event():
    event = build_runtime_step_completed_event(
        run_id="hr-1",
        step_index=0,
        agent_id="agent_a",
        agent_name="Agent A",
        loop_number=1,
    )

    assert event.version == "event.v1"
    assert event.event_type == "runtime.step_completed"
    assert event.payload["step_index"] == 0


def test_runtime_completed_event_preserves_result_payload():
    event = build_runtime_completed_event(
        run_id="hr-1",
        agent_outputs={"agent_a": "done"},
        output_artifacts={"agent_a": {"node_kind": "agent"}},
        errors=[],
        review_agent_enabled=True,
        recovery_mode="standard",
    )

    assert event.event_type == "runtime.completed"
    assert event.payload["agent_outputs"]["agent_a"] == "done"
