from app.platform.governance.commands import (
    ApprovalResolutionCommand,
    VerificationRecordCommand,
)


def test_governance_commands_are_explicit_and_versioned():
    approval = ApprovalResolutionCommand(
        version="governance_command.v1",
        run_id="hr-1",
        approval_id="ha-1",
        approved=True,
        resolved_by="u1",
    )
    verification = VerificationRecordCommand(
        version="governance_command.v1",
        run_id="hr-1",
        verification_profile="document_ingest_basic",
        result_status="pass",
    )

    assert approval.version == "governance_command.v1"
    assert verification.result_status == "pass"
