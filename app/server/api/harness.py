from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException

from app.harness.contracts.approval import HarnessApprovalDecision
from app.harness.contracts.provider import (
    HarnessModelProviderCreate,
    HarnessModelProviderUpdate,
)
from app.harness.contracts.run import HarnessRunCreate
from app.harness.contracts.studio import (
    HarnessStudioProjectCreate,
    HarnessStudioProjectUpdate,
    HarnessStudioRunCreate,
    HarnessStudioSkillDecision,
    HarnessStudioSkillRequestCreate,
)
from app.harness.persistence.stores import HarnessModelProviderStore
from app.harness.runtime.approval_service import ApprovalService
from app.harness.runtime.policy_registry import UnknownHarnessTaskTypeError
from app.harness.runtime.run_service import HarnessRetryNotAllowedError, build_run_service
from app.harness.runtime.studio_service import (
    HarnessStudioAgentNotFoundError,
    HarnessStudioProjectAccessError,
    HarnessStudioProjectNotFoundError,
    build_studio_service,
)
from app.infrastructure.database.models import User
from app.infrastructure.queue.client import enqueue_harness_run
from app.infrastructure.utils.secrets import encrypt_secret
from app.server.api.auth import get_current_active_user

router = APIRouter(prefix="/harness", tags=["harness"])


def get_run_service():
    return build_run_service()


def get_provider_store():
    return HarnessModelProviderStore()


def get_approval_service():
    return ApprovalService()


def get_studio_service():
    return build_studio_service()


def _serialize_provider(provider: dict[str, object]) -> dict[str, object]:
    payload = dict(provider)
    payload.pop("api_key_encrypted", None)
    raw_models = payload.pop("models_json", [])
    payload["models"] = list(raw_models) if isinstance(raw_models, list) else []
    return payload


def _encrypt_provider_api_key(api_key: str) -> str:
    try:
        return encrypt_secret(api_key)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _run_visible_to_user(run: dict[str, object], current_user: User) -> bool:
    run_user_id = str(run.get("user_id") or "")
    return run_user_id == current_user.username or current_user.role == "admin"


def _require_authorized_run(*, service, run_id: str, current_user: User) -> dict[str, object]:
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if not _run_visible_to_user(run, current_user):
        raise HTTPException(status_code=403, detail="Not authorized to access this harness run")
    return cast(dict[str, object], run)


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


@router.get("/studio/projects")
async def list_harness_studio_projects(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_studio_service()
    return {"projects": service.list_projects(user_id=current_user.username)}


@router.get("/studio/projects/current")
async def get_current_harness_studio_project(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_studio_service()
    return service.get_current_project(user_id=current_user.username)


@router.post("/studio/projects")
async def create_harness_studio_project(
    payload: HarnessStudioProjectCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_studio_service()
    return service.create_project(
        user_id=current_user.username,
        name=payload.name,
        description=payload.description,
    )


@router.get("/studio/projects/{project_id}")
async def get_harness_studio_project(
    project_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_studio_service()
    try:
        return service.get_project(project_id=project_id, user_id=current_user.username)
    except HarnessStudioProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HarnessStudioProjectAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.put("/studio/projects/{project_id}")
async def update_harness_studio_project(
    project_id: str,
    payload: HarnessStudioProjectUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_studio_service()
    try:
        return service.update_project(
            project_id=project_id,
            user_id=current_user.username,
            name=payload.name,
            description=payload.description,
            graph_json=payload.graph_json.model_dump() if payload.graph_json is not None else None,
        )
    except HarnessStudioProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HarnessStudioProjectAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/studio/projects/{project_id}/skill-requests")
async def request_harness_studio_skills(
    project_id: str,
    payload: HarnessStudioSkillRequestCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_studio_service()
    try:
        return service.request_skills(
            project_id=project_id,
            user_id=current_user.username,
            agent_id=payload.agent_id,
            requested_skills=payload.requested_skills,
        )
    except HarnessStudioProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HarnessStudioProjectAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except HarnessStudioAgentNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/studio/projects/{project_id}/skill-requests/{request_id}")
async def resolve_harness_studio_skill_request(
    project_id: str,
    request_id: str,
    payload: HarnessStudioSkillDecision,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_studio_service()
    try:
        return service.resolve_skill_request(
            project_id=project_id,
            user_id=current_user.username,
            request_id=request_id,
            approved=payload.approved,
        )
    except HarnessStudioProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HarnessStudioProjectAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/studio/projects/{project_id}/run")
async def create_harness_studio_run(
    project_id: str,
    payload: HarnessStudioRunCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    studio_service = get_studio_service()
    run_service = get_run_service()
    try:
        run = studio_service.create_orchestration_run(
            project_id=project_id,
            user_id=current_user.username,
            run_scope=payload.run_scope,
            agent_ids=payload.agent_ids,
            loop_count=payload.loop_count,
            task=payload.task,
            timeout_seconds=payload.timeout_seconds,
        )
    except HarnessStudioProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HarnessStudioProjectAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    run_id = str(run.get("run_id") or "")
    if run_id and not bool(run.get("approval_required")):
        await enqueue_harness_run(run_id)
        queued = run_service.mark_queued(run_id)
        if queued is not None:
            run = queued
    detail_loader = getattr(run_service, "get_run_detail", None)
    if callable(detail_loader):
        detail = detail_loader(run_id)
        if detail is not None:
            return detail
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


@router.get("/runs/{run_id}/runtime-state/history")
async def list_harness_run_runtime_state_history(
    run_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    service = get_run_service()
    _require_authorized_run(service=service, run_id=run_id, current_user=current_user)
    loader = getattr(service, "list_runtime_state_history", None)
    if not callable(loader):
        return {"history": []}
    return {"history": loader(run_id=run_id)}


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


@router.get("/model-providers")
async def list_harness_model_providers(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    store = get_provider_store()
    providers = store.list_providers(user_id=None if current_user.role == "admin" else current_user.username)
    return {"providers": [_serialize_provider(provider) for provider in providers]}


@router.post("/model-providers")
async def create_harness_model_provider(
    payload: HarnessModelProviderCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    import uuid
    store = get_provider_store()

    provider_id = f"provider_{uuid.uuid4().hex[:8]}"
    provider = store.create_provider(
        provider_id=provider_id,
        user_id=current_user.username,
        name=payload.name,
        base_url=payload.base_url,
        api_key_encrypted=_encrypt_provider_api_key(payload.api_key),
        models_json=payload.models,
        is_default=payload.is_default,
        enabled=payload.enabled,
    )
    return _serialize_provider(provider)


@router.put("/model-providers/{provider_id}")
async def update_harness_model_provider(
    provider_id: str,
    payload: HarnessModelProviderUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    store = get_provider_store()
    existing = store.get_provider(provider_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Provider not found")

    if existing["user_id"] != current_user.username and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")

    changes: dict[str, object] = {}
    if payload.name is not None:
        changes["name"] = payload.name
    if payload.base_url is not None:
        changes["base_url"] = payload.base_url
    if payload.api_key is not None:
        changes["api_key_encrypted"] = _encrypt_provider_api_key(payload.api_key)
    if payload.models is not None:
        changes["models_json"] = payload.models
    if payload.is_default is not None:
        changes["is_default"] = payload.is_default
    if payload.enabled is not None:
        changes["enabled"] = payload.enabled

    updated = store.update_provider(provider_id, **changes)
    if updated is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    return _serialize_provider(updated)


@router.delete("/model-providers/{provider_id}")
async def delete_harness_model_provider(
    provider_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    store = get_provider_store()
    existing = store.get_provider(provider_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Provider not found")

    if existing["user_id"] != current_user.username and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized")

    store.delete_provider(provider_id)
    return {"success": True}
