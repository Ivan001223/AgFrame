from app.platform.contracts.approval import ApprovalDecisionState, ApprovalRecordV1
from app.platform.contracts.event import EventEnvelopeV1
from app.platform.contracts.run import RunEnvelopeV1, RunLifecycleStatus
from app.platform.contracts.runtime_protocol import RuntimeCommandV1, RuntimeResultV1
from app.platform.contracts.verification import VerificationRecordV1


def test_platform_contracts_expose_v1_envelopes():
    run = RunEnvelopeV1(
        version="run.v1",
        run_id="hr-1",
        task_type="agent_orchestration",
        lifecycle_status=RunLifecycleStatus.CREATED,
        input={"task": "Coordinate work"},
        metadata={"source": "api"},
    )
    event = EventEnvelopeV1(
        version="event.v1",
        event_id="he-1",
        event_type="run.created",
        aggregate_id="hr-1",
        payload={"lifecycle_status": "created"},
    )
    approval = ApprovalRecordV1(
        version="approval.v1",
        approval_id="ha-1",
        target_run_id="hr-1",
        decision_state=ApprovalDecisionState.PENDING,
        requested_decision="approve",
    )
    verification = VerificationRecordV1(
        version="verification.v1",
        verification_id="hv-1",
        profile="document_ingest_basic",
        subject_run_id="hr-1",
        result_status="pass",
    )
    command = RuntimeCommandV1(
        version="runtime_command.v1",
        command_id="cmd-1",
        run_id="hr-1",
        command_type="start",
        payload={"task": "Coordinate work"},
    )
    result = RuntimeResultV1(
        version="runtime_result.v1",
        command_id="cmd-1",
        run_id="hr-1",
        result_type="accepted",
        payload={"accepted": True},
    )

    assert run.version == "run.v1"
    assert event.version == "event.v1"
    assert approval.version == "approval.v1"
    assert verification.version == "verification.v1"
    assert command.version == "runtime_command.v1"
    assert result.version == "runtime_result.v1"
