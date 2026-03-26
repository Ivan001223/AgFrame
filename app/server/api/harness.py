from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.harness.contracts.approval import HarnessApprovalDecision
from app.harness.contracts.run import HarnessRunCreate
from app.harness.runtime.approval_service import ApprovalService
from app.harness.runtime.run_service import build_run_service
from app.infrastructure.database.models import User
from app.infrastructure.queue.client import enqueue_harness_run
from app.server.api.auth import get_current_active_user

router = APIRouter(prefix="/harness", tags=["harness"])


def get_run_service():
    return build_run_service()


def get_approval_service():
    return ApprovalService()


@router.post("/runs")
async def create_harness_run(
    payload: HarnessRunCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_run_service()
    run = service.create_run(
        user_id=current_user.username,
        task_type=payload.task_type,
        input_json=payload.input,
        session_id=payload.session_id,
        metadata_json=payload.metadata,
    )
    if not bool(run.get("approval_required")):
        await enqueue_harness_run(str(run["run_id"]))
        queued = service.mark_queued(str(run["run_id"]))
        if queued is not None:
            run = queued
    return run


@router.get("/runs")
async def list_harness_runs(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_run_service()
    return {"runs": service.list_runs(user_id=current_user.username)}


@router.get("/runs/{run_id}")
async def get_harness_run(
    run_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_run_service()
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/runs/{run_id}/approval")
async def get_harness_approval(
    run_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_approval_service()
    return service.get_pending_approval(run_id)


@router.post("/runs/{run_id}/approval")
async def resolve_harness_approval(
    run_id: str,
    payload: HarnessApprovalDecision,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_approval_service()
    return await service.resolve(
        run_id=run_id,
        approved=payload.approved,
        resolved_by=current_user.username,
        comment=payload.comment,
    )
