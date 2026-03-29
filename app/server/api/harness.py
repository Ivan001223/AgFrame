from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.harness.contracts.approval import HarnessApprovalDecision
from app.harness.contracts.run import HarnessRunCreate
from app.harness.runtime.approval_service import ApprovalService
from app.harness.runtime.policy_registry import UnknownHarnessTaskTypeError
from app.harness.runtime.run_service import HarnessRetryNotAllowedError, build_run_service
from app.infrastructure.database.models import User
from app.infrastructure.queue.client import enqueue_harness_run
from app.server.api.auth import get_current_active_user

router = APIRouter(prefix="/harness", tags=["harness"])


def get_run_service():
    return build_run_service()


def get_approval_service():
    return ApprovalService()


def _run_visible_to_user(run: dict[str, object], current_user: User) -> bool:
    run_user_id = str(run.get("user_id") or "")
    return run_user_id == current_user.username or current_user.role == "admin"


def _require_authorized_run(*, service, run_id: str, current_user: User) -> dict[str, object]:
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _run_visible_to_user(run, current_user):
        raise HTTPException(status_code=403, detail="Not authorized to access this harness run")
    return run


@router.post("/runs")
async def create_harness_run(
    payload: HarnessRunCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_run_service()
    try:
        run = service.create_run(
            user_id=current_user.username,
            task_type=payload.task_type.value,
            input_json=payload.input,
            session_id=payload.session_id,
            metadata_json=payload.metadata,
        )
    except UnknownHarnessTaskTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not bool(run.get("approval_required")):
        await enqueue_harness_run(str(run["run_id"]))
        queued = service.mark_queued(str(run["run_id"]))
        if queued is not None:
            run = queued
    detail_loader = getattr(service, "get_run_detail", None)
    if callable(detail_loader):
        detail = detail_loader(str(run.get("run_id") or ""))
        if detail is not None:
            return detail
    return run


@router.get("/policies")
async def list_harness_policies(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_run_service()
    return {"policies": service.list_policies()}


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
    run = _require_authorized_run(service=service, run_id=run_id, current_user=current_user)
    detail_loader = getattr(service, "get_run_detail", None)
    if callable(detail_loader):
        detail = detail_loader(run_id)
        if detail is not None:
            return detail
    return run


@router.post("/runs/{run_id}/retry")
async def retry_harness_run(
    run_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_run_service()
    _require_authorized_run(service=service, run_id=run_id, current_user=current_user)
    try:
        run = service.create_retry_run(run_id, requested_by=current_user.username)
    except HarnessRetryNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not bool(run.get("approval_required")):
        await enqueue_harness_run(str(run["run_id"]))
        queued = service.mark_queued(str(run["run_id"]))
        if queued is not None:
            run = queued
    detail_loader = getattr(service, "get_run_detail", None)
    if callable(detail_loader):
        detail = detail_loader(str(run.get("run_id") or ""))
        if detail is not None:
            return detail
    return run


@router.get("/runs/{run_id}/events")
async def list_harness_run_events(
    run_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_run_service()
    _require_authorized_run(service=service, run_id=run_id, current_user=current_user)
    user_id = None if current_user.role == "admin" else current_user.username
    return {"events": service.list_run_events(run_id=run_id, user_id=user_id)}


@router.get("/runs/{run_id}/approval")
async def get_harness_approval(
    run_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    run_service = get_run_service()
    _require_authorized_run(service=run_service, run_id=run_id, current_user=current_user)
    service = get_approval_service()
    return service.get_pending_approval(run_id)


@router.get("/runs/{run_id}/verification")
async def get_harness_verification(
    run_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    run_service = get_run_service()
    _require_authorized_run(service=run_service, run_id=run_id, current_user=current_user)
    loader = getattr(run_service, "get_latest_verification", None)
    if not callable(loader):
        return None
    return loader(run_id)


@router.post("/runs/{run_id}/approval")
async def resolve_harness_approval(
    run_id: str,
    payload: HarnessApprovalDecision,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    run_service = get_run_service()
    _require_authorized_run(service=run_service, run_id=run_id, current_user=current_user)
    service = get_approval_service()
    return await service.resolve(
        run_id=run_id,
        approved=payload.approved,
        resolved_by=current_user.username,
        comment=payload.comment,
    )
