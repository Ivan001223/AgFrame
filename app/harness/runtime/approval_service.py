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

        if approved:
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
