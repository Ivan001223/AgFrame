from __future__ import annotations

from app.platform.contracts.approval import ApprovalDecisionState, ApprovalRecordV1
from app.platform.contracts.run import RunEnvelopeV1, RunLifecycleStatus
from app.platform.contracts.verification import VerificationRecordV1


def legacy_run_to_run_envelope(payload: dict[str, object]) -> RunEnvelopeV1:
    return RunEnvelopeV1(
        run_id=str(payload.get("run_id") or ""),
        task_type=str(payload.get("task_type") or ""),
        lifecycle_status=RunLifecycleStatus(str(payload.get("status") or RunLifecycleStatus.CREATED.value)),
        input=dict(payload.get("input_json") or {}),
        metadata=dict(payload.get("metadata_json") or {}) or None,
    )


def run_envelope_to_legacy_payload(payload: RunEnvelopeV1 | dict[str, object]) -> dict[str, object]:
    envelope = payload if isinstance(payload, RunEnvelopeV1) else RunEnvelopeV1.model_validate(payload)
    return {
        "run_id": envelope.run_id,
        "task_type": envelope.task_type,
        "status": envelope.lifecycle_status.value,
        "input_json": dict(envelope.input),
        "metadata_json": dict(envelope.metadata or {}) or None,
    }


def legacy_approval_to_record(payload: dict[str, object]) -> ApprovalRecordV1:
    return ApprovalRecordV1(
        approval_id=str(payload.get("approval_id") or ""),
        target_run_id=str(payload.get("run_id") or ""),
        decision_state=ApprovalDecisionState(str(payload.get("status") or ApprovalDecisionState.PENDING.value)),
        requested_decision=str(payload.get("action_type") or ""),
        approver_identity=str(payload.get("resolved_by") or "") or None,
    )


def approval_record_to_legacy_payload(payload: ApprovalRecordV1 | dict[str, object]) -> dict[str, object]:
    record = payload if isinstance(payload, ApprovalRecordV1) else ApprovalRecordV1.model_validate(payload)
    return {
        "approval_id": record.approval_id,
        "run_id": record.target_run_id,
        "status": record.decision_state.value,
        "action_type": record.requested_decision,
    }


def legacy_verification_to_record(payload: dict[str, object]) -> VerificationRecordV1:
    return VerificationRecordV1(
        verification_id=str(payload.get("verification_id") or ""),
        profile=str(payload.get("profile") or payload.get("verification_profile") or "unknown"),
        subject_run_id=str(payload.get("run_id") or ""),
        result_status=str(payload.get("status") or "fail"),
        evidence=dict(payload.get("artifacts_json") or {}),
        summary=str(payload.get("summary") or "") or None,
    )
