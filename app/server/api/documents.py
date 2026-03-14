import os
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.infrastructure.database.models import User
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.stores import MySQLDocStore
from app.infrastructure.queue.client import enqueue_ingest_pdf
from app.infrastructure.queue.redis_client import (
    claim_task_operation,
    get_task,
    init_task,
    release_task_operation,
)
from app.server.api.auth import get_current_active_user

router = APIRouter(prefix="/documents", tags=["documents"])


def _serialize_document(row: dict) -> dict:
    source_path = str(row.get("source_path") or "")
    return {
        "doc_id": row.get("doc_id"),
        "user_id": row.get("user_id"),
        "filename": os.path.basename(source_path),
        "source_path": source_path,
        "checksum": row.get("checksum"),
        "created_at": row.get("created_at"),
        "parent_chunk_count": row.get("parent_chunk_count", 0),
        "embedding_count": row.get("embedding_count", 0),
    }


@router.get("")
async def list_documents(
    current_user: Annotated[User, Depends(get_current_active_user)],
    q: str | None = Query(default=None),
):
    if not ensure_schema_if_possible():
        return {"documents": []}

    store = MySQLDocStore()
    docs = store.search_documents(
        current_user.username,
        include_all_users=current_user.role == "admin",
        filename_query=q,
    )
    return {"documents": [_serialize_document(doc) for doc in docs]}


@router.get("/{doc_id}")
async def get_document(
    doc_id: int,
    current_user: Annotated[User, Depends(get_current_active_user)],
    preview_limit: int = Query(default=3, ge=1, le=20),
):
    if not ensure_schema_if_possible():
        raise HTTPException(status_code=404, detail="Document not found")

    store = MySQLDocStore()
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if current_user.role != "admin" and doc.get("user_id") != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this document")
    out = _serialize_document(doc)
    out["preview"] = store.get_document_preview(doc_id, limit=preview_limit)
    return out


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: int,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        raise HTTPException(status_code=404, detail="Document not found")

    store = MySQLDocStore()
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if current_user.role != "admin" and doc.get("user_id") != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to delete this document")

    deleted = store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    source_path = str(doc.get("source_path") or "").strip()
    file_deleted = False
    if source_path and os.path.exists(source_path):
        try:
            os.remove(source_path)
            file_deleted = True
        except OSError:
            file_deleted = False

    return {
        "message": "Deleted",
        "doc_id": doc_id,
        "file_deleted": file_deleted,
    }


@router.post("/{doc_id}/reindex")
async def reindex_document(
    doc_id: int,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        raise HTTPException(status_code=400, detail="Database is not available")

    store = MySQLDocStore()
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if current_user.role != "admin" and doc.get("user_id") != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to reindex this document")

    source_path = str(doc.get("source_path") or "").strip()
    if not source_path or not os.path.exists(source_path):
        raise HTTPException(status_code=400, detail="Document source file is missing")

    task_id = str(uuid.uuid4())
    operation_key = f"reindex_document:{doc_id}"
    claimed_task_id = await claim_task_operation(operation_key, task_id)
    if claimed_task_id != task_id:
        existing_task = await get_task(claimed_task_id)
        if existing_task.get("status") in {"queued", "running"}:
            return {
                "message": "Reindex already queued",
                "task_id": claimed_task_id,
                "doc_id": doc_id,
            }
        await release_task_operation(operation_key, expected_task_id=claimed_task_id)
        claimed_task_id = await claim_task_operation(operation_key, task_id)
        if claimed_task_id != task_id:
            return {
                "message": "Reindex already queued",
                "task_id": claimed_task_id,
                "doc_id": doc_id,
            }
    await init_task(
        task_id,
        {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "step": "queued",
            "message": "文档重建索引已入队",
            "file_path": source_path,
            "filename": os.path.basename(source_path),
            "created_at": int(time.time()),
            "user_id": doc.get("user_id") or current_user.username,
            "task_type": "reindex_document",
            "doc_id": doc_id,
            "retry_count": 0,
            "retryable": "false",
            "operation_key": operation_key,
        },
    )
    await enqueue_ingest_pdf(
        task_id,
        source_path,
        user_id=doc.get("user_id") or current_user.username,
    )
    return {
        "message": "Reindex queued",
        "task_id": task_id,
        "doc_id": doc_id,
    }
