from __future__ import annotations

from app.platform.contracts.runtime_protocol import (
    normalize_review_approval_resume_payload,
    normalize_review_rejection_resume_payload,
)
from app.platform.governance.commands import ApprovalResolutionCommand, VerificationRecordCommand
from app.harness.runtime.checkpoint_adapter import CheckpointAdapter
from app.harness.runtime.run_service import build_run_service
from app.infrastructure.queue.client import enqueue_harness_resume


def build_approval_resolution_command(
    *,
    run_id: str,
    approval_id: str,
    approved: bool,
    resolved_by: str,
    comment: str | None,
) -> ApprovalResolutionCommand:
    return ApprovalResolutionCommand(
        run_id=run_id,
        approval_id=approval_id,
        approved=approved,
        resolved_by=resolved_by,
        comment=comment,
    )


def build_approval_resolution_verification_command(
    *,
    run_id: str,
    approved: bool,
) -> VerificationRecordCommand:
    return VerificationRecordCommand(
        run_id=run_id,
        verification_profile="approval_resolution",
        result_status="pass" if approved else "partial",
        checks_run=["approval_resolution"],
        artifacts={"approved": approved},
        summary="approval accepted" if approved else "approval rejected",
    )


class ApprovalService:
    def __init__(self, checkpoint_adapter: CheckpointAdapter | None = None):
        self.checkpoint_adapter = checkpoint_adapter or CheckpointAdapter()
        self.run_service = build_run_service()

    def get_pending_approval(self, run_id: str):
        approval = self.run_service.get_pending_approval_for_run(run_id)
        if approval is not None:
            return approval
        return self.run_service.get_latest_approval(run_id)

    async def resolve(
        self,
        command: ApprovalResolutionCommand,
    ) -> dict[str, object]:
        run_id = command.run_id
        approved = command.approved
        resolved_by = command.resolved_by
        comment = command.comment
        approval = self.run_service.get_pending_approval_for_run(run_id)
        if approval is None:
            return {
                "run_id": run_id,
                "status": "not_found",
                "resolved_by": resolved_by,
                "comment": comment,
            }

        approval_status = "approved" if approved else "rejected"
        resolved_command = build_approval_resolution_command(
            run_id=run_id,
            approval_id=str(approval.get("approval_id") or command.approval_id),
            approved=approved,
            resolved_by=resolved_by,
            comment=comment,
        )
        updated = self.run_service.resolve_approval_command(resolved_command)

        run = self.run_service.get_run(run_id)
        action_type = str(approval.get("action_type") or "resume")
        session_id = None if run is None else run.get("session_id")
        if session_id:
            checkpoint = await self.checkpoint_adapter.load(str(session_id))
            if checkpoint is not None:
                checkpoint_data = dict(checkpoint.get("checkpoint") or {})
                action_required = dict(checkpoint_data.get("action_required") or {})
                action_required["approved"] = approved
                action_required["approved_by"] = resolved_by
                checkpoint_data["action_required"] = action_required
                checkpoint_data["interrupted"] = not approved
                await self.checkpoint_adapter.save(str(session_id), checkpoint_data)

        if action_type == "orchestration_review":
            input_value = run.get("input_json") if isinstance(run, dict) else None
            metadata_value = run.get("metadata_json") if isinstance(run, dict) else None
            input_json = dict(input_value) if isinstance(input_value, dict) else {}
            metadata_json = dict(metadata_value) if isinstance(metadata_value, dict) else {}
            resume_state_value = input_json.get("orchestration_resume")
            payload_value = approval.get("payload_json")
            resume_state = dict(resume_state_value) if isinstance(resume_state_value, dict) else {}
            payload_json = dict(payload_value) if isinstance(payload_value, dict) else {}
            review_stage = str(payload_json.get("review_stage") or "").strip()
            if approved:
                resume_state, recovery_mode = normalize_review_approval_resume_payload(
                    resume_state,
                    review_stage=review_stage,
                )
                if recovery_mode:
                    metadata_json["review_recovery_mode"] = recovery_mode
                input_json["orchestration_resume"] = resume_state
                self.run_service.update_run_input_json(run_id, input_json)
                if metadata_json:
                    self.run_service.update_run_metadata_json(run_id, metadata_json)
                self.run_service.mark_approved(run_id)
                self.run_service.create_verification_command(
                    build_approval_resolution_verification_command(run_id=run_id, approved=True)
                )
                await enqueue_harness_resume(run_id)
            else:
                resume_state, recovery_mode = normalize_review_rejection_resume_payload(
                    resume_state,
                    review_stage=review_stage,
                    continue_mode=(
                        "discard_research_evidence"
                        if review_stage == "cluster_research"
                        else str(resume_state.get("continue_mode") or "")
                    ),
                    review_step_index=int(payload_json.get("step_index") or 0) or None,
                )
                resume_state["review_decision"] = "rejected"
                if review_stage == "cluster_research":
                    resume_state["continue_mode"] = "discard_research_evidence"
                if recovery_mode:
                    metadata_json["review_recovery_mode"] = recovery_mode
                input_json["orchestration_resume"] = resume_state
                self.run_service.update_run_input_json(run_id, input_json)
                if metadata_json:
                    self.run_service.update_run_metadata_json(run_id, metadata_json)
                self.run_service.mark_rejected(run_id)
                self.run_service.create_verification_command(
                    build_approval_resolution_verification_command(run_id=run_id, approved=False)
                )
        elif approved:
            self.run_service.mark_approved(run_id)
            self.run_service.create_verification_command(
                build_approval_resolution_verification_command(run_id=run_id, approved=True)
            )
            await enqueue_harness_resume(run_id)
        else:
            self.run_service.mark_rejected(run_id)
            self.run_service.create_verification_command(
                build_approval_resolution_verification_command(run_id=run_id, approved=False)
            )

        return {
            "run_id": run_id,
            "status": approval_status,
            "resolved_by": resolved_by,
            "comment": comment,
            "approval": updated,
        }
