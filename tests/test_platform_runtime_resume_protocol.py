from app.platform.contracts.runtime_protocol import (
    RuntimeCommandV1,
    RuntimeInterruption,
    RuntimeResumePoint,
    runtime_resume_point_from_payload,
    runtime_resume_point_to_payload,
)


def test_runtime_protocol_supports_resume_command():
    command = RuntimeCommandV1(
        command_id="cmd-resume-1",
        run_id="hr-1",
        command_type="resume",
        payload={"next_step_index": 2},
    )

    assert command.command_type == "resume"
    assert command.payload["next_step_index"] == 2


def test_runtime_resume_point_round_trips_legacy_payload():
    resume_point = RuntimeResumePoint(
        next_step_index=2,
        rollback_state={"agent_outputs": {"agent_a": "safe"}},
        continuation={"agent_id": "agent_b", "partial_output": "prefix"},
    )

    payload = runtime_resume_point_to_payload(resume_point)
    restored = runtime_resume_point_from_payload(payload)

    assert restored.next_step_index == 2
    assert restored.rollback_state["agent_outputs"]["agent_a"] == "safe"
    assert restored.continuation["agent_id"] == "agent_b"


def test_runtime_interruption_defaults_to_resumable():
    interruption = RuntimeInterruption(interrupt_type="human_review", reason="approval required")

    assert interruption.resumable is True
