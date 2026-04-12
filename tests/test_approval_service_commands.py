from app.harness.runtime.approval_service import build_approval_resolution_command, build_approval_resolution_verification_command
from app.platform.governance.commands import ApprovalResolutionCommand, VerificationRecordCommand


def test_build_approval_resolution_command_preserves_resolution_inputs():
    command = build_approval_resolution_command(
        run_id="hr-1",
        approval_id="ha-1",
        approved=True,
        resolved_by="u1",
        comment="ok",
    )

    assert isinstance(command, ApprovalResolutionCommand)
    assert command.run_id == "hr-1"
    assert command.approval_id == "ha-1"
    assert command.approved is True
    assert command.comment == "ok"


def test_build_approval_resolution_verification_command_maps_outcome():
    approved = build_approval_resolution_verification_command(run_id="hr-1", approved=True)
    rejected = build_approval_resolution_verification_command(run_id="hr-1", approved=False)

    assert isinstance(approved, VerificationRecordCommand)
    assert approved.result_status == "pass"
    assert approved.artifacts == {"approved": True}
    assert rejected.result_status == "partial"
    assert rejected.summary == "approval rejected"
