from app.platform.runtime.events import (
    build_runtime_completed_event,
    build_runtime_failed_event,
    build_runtime_interrupted_event,
    build_runtime_resumed_event,
    build_runtime_started_event,
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


def test_runtime_started_event():
    event = build_runtime_started_event(
        run_id="hr-1",
        task_type="agent_orchestration",
        actor="user_1",
    )

    assert event.version == "event.v1"
    assert event.event_type == "runtime.started"
    assert event.aggregate_id == "hr-1"
    assert event.actor == "user_1"
    assert event.payload["task_type"] == "agent_orchestration"


def test_runtime_resumed_event():
    event = build_runtime_resumed_event(
        run_id="hr-1",
        task_type="agent_orchestration",
        next_step_index=2,
        recovery_mode="continue_with_research",
        actor="system",
    )

    assert event.event_type == "runtime.resumed"
    assert event.payload["next_step_index"] == 2
    assert event.payload["recovery_mode"] == "continue_with_research"


def test_runtime_interrupted_event():
    event = build_runtime_interrupted_event(
        run_id="hr-1",
        interrupt_type="approval_required",
        reason="awaiting human review",
        resumable=True,
    )

    assert event.event_type == "runtime.interrupted"
    assert event.payload["interrupt_type"] == "approval_required"
    assert event.payload["resumable"] is True


def test_runtime_failed_event():
    event = build_runtime_failed_event(
        run_id="hr-1",
        error_type="execution_error",
        error_message="agent failed to produce output",
        task_type="agent_orchestration",
    )

    assert event.event_type == "runtime.failed"
    assert event.payload["error_type"] == "execution_error"
    assert event.payload["task_type"] == "agent_orchestration"


def test_all_runtime_events_share_envelope_version():
    events = [
        build_runtime_started_event(run_id="hr-1", task_type="t"),
        build_runtime_resumed_event(run_id="hr-1", task_type="t"),
        build_runtime_interrupted_event(run_id="hr-1", interrupt_type="t"),
        build_runtime_failed_event(run_id="hr-1", error_type="t"),
        build_runtime_completed_event(
            run_id="hr-1",
            agent_outputs={},
            output_artifacts={},
            errors=[],
            review_agent_enabled=False,
            recovery_mode=None,
        ),
        build_runtime_step_completed_event(
            run_id="hr-1",
            step_index=0,
            agent_id="a",
            agent_name=None,
            loop_number=1,
        ),
    ]

    for event in events:
        assert event.version == "event.v1"
        assert event.aggregate_id == "hr-1"
        assert isinstance(event.payload, dict)
