import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException

from app.infrastructure.database.models import User
from app.infrastructure.queue.client import enqueue_ingest_pdf
from app.infrastructure.queue.redis_client import get_redis
from app.infrastructure.queue.redis_client import (
    get_task,
    init_task,
    list_task_incidents,
    update_task_incident,
)
from app.server.api.auth import get_current_active_user

router = APIRouter()
RUNNING_TIMEOUT_SECONDS = 15 * 60

ERROR_CATALOG: dict[str, dict[str, str]] = {
    "document_load_failed": {
        "title": "文档读取失败",
        "message": "系统无法读取这个文件的内容。",
        "suggested_action": "请确认文件未损坏，并重新上传。",
    },
    "no_text_extracted": {
        "title": "未提取到文本",
        "message": "文档中没有提取到可用于索引的文本。",
        "suggested_action": "请检查文档是否为扫描件，或启用 OCR 后重试。",
    },
    "database_not_ready": {
        "title": "数据库未就绪",
        "message": "索引写入依赖的数据库当前不可用。",
        "suggested_action": "请检查数据库和向量库状态后重试。",
    },
    "no_chunks_generated": {
        "title": "未生成切片",
        "message": "系统没有生成可索引的文档切片。",
        "suggested_action": "请检查文档内容是否为空，或重新上传。",
    },
    "embedding_failed": {
        "title": "向量化失败",
        "message": "文档文本在生成 embedding 时失败。",
        "suggested_action": "请检查 embedding 服务状态后重试。",
    },
    "vectorstore_write_failed": {
        "title": "向量写入失败",
        "message": "生成的向量未能写入向量库。",
        "suggested_action": "请检查 pgvector 或向量存储连接后重试。",
    },
    "ingest_failed": {
        "title": "索引任务失败",
        "message": "文档索引过程中发生未分类错误。",
        "suggested_action": "请查看详细错误并重试。",
    },
    "ingest_returned_false": {
        "title": "索引任务失败",
        "message": "文档索引未成功完成。",
        "suggested_action": "请查看任务详情并重试。",
    },
    "task_timeout_suspected": {
        "title": "任务疑似超时",
        "message": "任务运行时间过长，可能已卡住。",
        "suggested_action": "请检查依赖服务状态，必要时重新入队。",
    },
}


def _to_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None


def _build_diagnostics(task: dict[str, str]) -> dict[str, object]:
    now = int(time.time())
    status = str(task.get("status") or "")
    started_at = _to_int(task.get("started_at"))
    created_at = _to_int(task.get("created_at"))
    age_seconds = None
    if started_at is not None:
        age_seconds = max(0, now - started_at)
    elif created_at is not None:
        age_seconds = max(0, now - created_at)

    timeout_exceeded = bool(
        status == "running"
        and age_seconds is not None
        and age_seconds > RUNNING_TIMEOUT_SECONDS
    )

    diagnostics: dict[str, object] = {
        "status": status or "unknown",
        "stage": task.get("result_stage") or task.get("step") or "",
        "error_code": task.get("error_code") or "",
        "error_message": task.get("error") or "",
        "retryable": str(task.get("retryable") or "false").lower() == "true",
        "age_seconds": age_seconds,
        "timeout_exceeded": timeout_exceeded,
    }
    if timeout_exceeded and not diagnostics["error_code"]:
        diagnostics["error_code"] = "task_timeout_suspected"
    if timeout_exceeded and not diagnostics["error_message"]:
        diagnostics["error_message"] = "任务运行时间过长，疑似卡住或超时"
    error_code = str(diagnostics["error_code"] or "")
    catalog_entry = ERROR_CATALOG.get(error_code, {})
    diagnostics["title"] = catalog_entry.get("title", "任务状态")
    diagnostics["user_message"] = catalog_entry.get(
        "message",
        str(diagnostics["error_message"] or "任务当前没有额外诊断信息。"),
    )
    diagnostics["suggested_action"] = catalog_entry.get(
        "suggested_action",
        "如需继续处理，请查看任务明细或稍后重试。",
    )
    return diagnostics


def _task_visible_to_user(task: dict[str, str], current_user: User) -> bool:
    task_user_id = task.get("user_id")
    if not task_user_id or task_user_id == "unknown":
        return True
    return task_user_id == current_user.username or current_user.role == "admin"


def _incident_visible_to_user(incident: dict[str, object], current_user: User) -> bool:
    user_id = str(incident.get("user_id") or "unknown")
    if user_id == "unknown":
        return True
    return user_id == current_user.username or current_user.role == "admin"


def _incident_matches_filters(
    incident: dict[str, object],
    *,
    handled: bool | None = None,
    archived: bool | None = None,
) -> bool:
    incident_handled = bool(incident.get("handled"))
    incident_archived = bool(incident.get("archived"))
    if handled is not None and incident_handled != handled:
        return False
    if archived is not None and incident_archived != archived:
        return False
    return True


def _normalize_incident_updates(payload: dict[str, object]) -> dict[str, object]:
    now = int(time.time())
    updates: dict[str, object] = {}
    if "handled" in payload:
        handled = bool(payload.get("handled"))
        updates["handled"] = handled
        updates["handled_at"] = now if handled else None
    if "archived" in payload:
        archived = bool(payload.get("archived"))
        updates["archived"] = archived
        updates["archived_at"] = now if archived else None
    return updates


@router.get("/tasks/summary")
async def get_task_summary(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    redis = get_redis()
    status_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    slow_tasks: list[dict[str, object]] = []
    total = 0

    async for key in redis.scan_iter(match="task:*"):
        task = dict(await redis.hgetall(key))
        if not task or not _task_visible_to_user(task, current_user):
            continue
        total += 1
        status = str(task.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

        diagnostics = _build_diagnostics(task)
        error_code = str(diagnostics.get("error_code") or "")
        if error_code:
            error_counts[error_code] = error_counts.get(error_code, 0) + 1

        if diagnostics.get("timeout_exceeded"):
            slow_tasks.append(
                {
                    "task_id": task.get("task_id") or key.split(":", 1)[1],
                    "status": status,
                    "stage": diagnostics.get("stage"),
                    "age_seconds": diagnostics.get("age_seconds"),
                    "user_id": task.get("user_id") or "unknown",
                }
            )

    slow_tasks.sort(key=lambda item: int(item.get("age_seconds") or 0), reverse=True)
    top_errors = [
        {"error_code": code, "count": count, "title": ERROR_CATALOG.get(code, {}).get("title", "任务状态")}
        for code, count in sorted(error_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    recent_incidents = [
        incident
        for incident in await list_task_incidents(limit=20)
        if _incident_visible_to_user(incident, current_user)
        and _incident_matches_filters(incident, archived=False)
    ]

    return {
        "total": total,
        "status_counts": status_counts,
        "top_errors": top_errors,
        "suspected_timeouts": slow_tasks[:10],
        "recent_incidents": recent_incidents[:10],
    }


@router.get("/tasks/incidents")
async def get_task_incidents(
    current_user: Annotated[User, Depends(get_current_active_user)],
    limit: int = 20,
    handled: bool | None = None,
    archived: bool | None = None,
):
    safe_limit = max(1, min(limit, 100))
    incidents = [
        incident
        for incident in await list_task_incidents(limit=safe_limit)
        if _incident_visible_to_user(incident, current_user)
        and _incident_matches_filters(incident, handled=handled, archived=archived)
    ]
    return {"incidents": incidents}


@router.patch("/tasks/incidents/{incident_id}")
async def update_incident_status(
    incident_id: str,
    payload: Annotated[dict[str, object], Body(...)],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    incidents = await list_task_incidents(limit=200)
    incident = next(
        (
            item
            for item in incidents
            if str(item.get("incident_id") or "") == incident_id
        ),
        None,
    )
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    if not _incident_visible_to_user(incident, current_user):
        raise HTTPException(status_code=403, detail="Not authorized to update this incident")

    updates = _normalize_incident_updates(payload or {})
    if not updates:
        raise HTTPException(status_code=400, detail="No supported incident fields provided")

    updated = await update_task_incident(incident_id, updates)
    if updated is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return {"incident": updated}


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str, current_user: Annotated[User, Depends(get_current_active_user)]
):
    task = await get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Check isolation
    # If task has user_id, ensure it matches current_user or admin
    task_user_id = task.get("user_id")
    if task_user_id and task_user_id != "unknown":
        if task_user_id != current_user.username and current_user.role != "admin":
            raise HTTPException(
                status_code=403, detail="Not authorized to view this task"
            )

    if task.get("status") == "failed":
        task["can_retry"] = "true"
    diagnostics = _build_diagnostics(task)
    if diagnostics["timeout_exceeded"] and task.get("status") == "running":
        task["can_retry"] = "true"
    task["diagnostics"] = diagnostics

    return task


@router.post("/tasks/{task_id}/retry")
async def retry_task(
    task_id: str, current_user: Annotated[User, Depends(get_current_active_user)]
):
    task = await get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    task_user_id = task.get("user_id")
    if task_user_id and task_user_id != "unknown":
        if task_user_id != current_user.username and current_user.role != "admin":
            raise HTTPException(
                status_code=403, detail="Not authorized to retry this task"
            )

    if task.get("status") != "failed":
        raise HTTPException(status_code=400, detail="Only failed tasks can be retried")

    storage_reference = str(task.get("storage_uri") or task.get("file_path") or "").strip()
    if not storage_reference:
        raise HTTPException(status_code=400, detail="Task is missing file_path")

    new_task_id = str(uuid.uuid4())
    retry_count = int(task.get("retry_count") or 0) + 1
    await init_task(
        new_task_id,
        {
            "task_id": new_task_id,
            "status": "queued",
            "progress": 0,
            "step": "queued",
            "message": "已重新入队",
            "file_path": storage_reference,
            "storage_uri": storage_reference,
            "filename": task.get("filename") or "",
            "created_at": int(time.time()),
            "user_id": task_user_id or current_user.username,
            "retry_count": retry_count,
            "retried_from_task_id": task_id,
        },
    )
    await enqueue_ingest_pdf(new_task_id, storage_reference, user_id=task_user_id or current_user.username)
    return {
        "message": "Retried",
        "task_id": new_task_id,
        "retried_from_task_id": task_id,
        "retry_count": retry_count,
    }
