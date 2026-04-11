from __future__ import annotations

from app.harness.runtime.checkpoint_adapter import CheckpointAdapter
from app.harness.runtime.run_service import build_run_service
from app.infrastructure.queue.client import enqueue_harness_resume


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
        run_id: str,
        approved: bool,
        resolved_by: str,
        comment: str | None,
    ) -> dict[str, object]:
        approval = self.run_service.get_pending_approval_for_run(run_id)
        if approval is None:
            return {
                "run_id": run_id,
                "status": "not_found",
                "resolved_by": resolved_by,
                "comment": comment,
            }

        approval_status = "approved" if approved else "rejected"
        updated = self.run_service.update_approval(
            str(approval["approval_id"]),
            status=approval_status,
            resolved_by=resolved_by,
            comment=comment,
        )

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
                if review_stage == "cluster_research":
                    resume_state["continue_mode"] = "accept_research_evidence"
                    metadata_json["review_recovery_mode"] = "continue_with_research"
                elif review_stage == "agent_output_stream":
                    resume_state["continue_mode"] = "accept_partial_stream_output"
                    metadata_json["review_recovery_mode"] = "continue_with_partial_stream_output"
                resume_state["review_decision"] = "approved"
                input_json["orchestration_resume"] = resume_state
                self.run_service.update_run_input_json(run_id, input_json)
                if metadata_json:
                    self.run_service.update_run_metadata_json(run_id, metadata_json)
                self.run_service.mark_approved(run_id)
                self.run_service.persist_approval_resolution(run_id, approved=True)
                await enqueue_harness_resume(run_id)
            else:
                rollback_state = dict(resume_state.get("rollback_state") or {})
                if rollback_state:
                    resume_state["state"] = rollback_state
                    if review_stage == "cluster_research":
                        resume_state["next_step_index"] = int(resume_state.get("next_step_index") or 0)
                        resume_state["continue_mode"] = "discard_research_evidence"
                        metadata_json["review_recovery_mode"] = "continue_without_research"
                    else:
                        resume_state["next_step_index"] = int(resume_state.get("next_step_index") or 0)
                resume_state["review_decision"] = "rejected"
                input_json["orchestration_resume"] = resume_state
                self.run_service.update_run_input_json(run_id, input_json)
                if metadata_json:
                    self.run_service.update_run_metadata_json(run_id, metadata_json)
                self.run_service.mark_rejected(run_id)
                self.run_service.persist_approval_resolution(run_id, approved=False)
        elif approved:
            self.run_service.mark_approved(run_id)
            self.run_service.persist_approval_resolution(run_id, approved=True)
            await enqueue_harness_resume(run_id)
        else:
            self.run_service.mark_rejected(run_id)
            self.run_service.persist_approval_resolution(run_id, approved=False)

        return {
            "run_id": run_id,
            "status": approval_status,
            "resolved_by": resolved_by,
            "comment": comment,
            "approval": updated,
        }
