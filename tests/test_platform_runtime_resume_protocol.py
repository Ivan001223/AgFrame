from app.platform.contracts.runtime_protocol import (
    RuntimeCommandV1,
    RuntimeInterruption,
    RuntimeResumePoint,
    normalize_review_approval_resume_payload,
    normalize_review_rejection_resume_payload,
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


def test_review_rejection_resume_payload_rewinds_stream_review_with_protocol_helper():
    payload, recovery_mode = normalize_review_rejection_resume_payload(
        {
            "next_step_index": 4,
            "rollback_state": {"agent_outputs": {"agent_a": "safe"}},
            "continuation": {"agent_id": "agent_a", "partial_output": "draft"},
            "review_decision": "rejected",
            "continue_mode": "accept_partial_stream_output",
        },
        review_stage="agent_output_stream",
        review_step_index=2,
    )

    assert payload["next_step_index"] == 2
    assert payload["state"] == {"agent_outputs": {"agent_a": "safe"}}
    assert "continuation" not in payload
    assert "review_decision" not in payload
    assert "continue_mode" not in payload
    assert recovery_mode == "continue_from_stream_block"


def test_review_rejection_resume_payload_keeps_position_when_research_is_discarded():
    payload, recovery_mode = normalize_review_rejection_resume_payload(
        {
            "next_step_index": 3,
            "rollback_state": {"research": {"cluster_a": "baseline"}},
            "continue_mode": "discard_research_evidence",
        },
        review_stage="cluster_research",
    )

    assert payload["next_step_index"] == 3
    assert payload["state"] == {"research": {"cluster_a": "baseline"}}
    assert recovery_mode == "continue_without_research"


def test_review_approval_resume_payload_marks_stream_continuation_mode():
    payload, recovery_mode = normalize_review_approval_resume_payload(
        {"next_step_index": 2},
        review_stage="agent_output_stream",
    )

    assert payload["review_decision"] == "approved"
    assert payload["continue_mode"] == "accept_partial_stream_output"
    assert recovery_mode == "continue_with_partial_stream_output"
