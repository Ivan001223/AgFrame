import os
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.infrastructure.database.models import User
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.stores import KnowledgeBaseDocumentStore, KnowledgeBaseStore, MySQLDocStore
from app.infrastructure.queue.client import enqueue_ingest_pdf
from app.infrastructure.queue.redis_client import (
    claim_task_operation,
    get_task,
    init_task,
    release_task_operation,
)
from app.infrastructure.storage.object_store import (
    get_object_store,
    storage_display_label,
    storage_filename,
    storage_media_type,
)
from app.server.api.auth import get_current_active_user

router = APIRouter(prefix="/documents", tags=["documents"])


def _serialize_document(row: dict) -> dict:
    source_reference = str(row.get("source_path") or "")
    filename = storage_filename(source_reference) or os.path.basename(source_reference)
    return {
        "doc_id": row.get("doc_id"),
        "user_id": row.get("user_id"),
        "filename": filename,
        "source_path": storage_display_label(source_reference),
        "download_url": f"/documents/{row.get('doc_id')}/download" if row.get("doc_id") is not None else None,
        "checksum": row.get("checksum"),
        "created_at": row.get("created_at"),
        "parent_chunk_count": row.get("parent_chunk_count", 0),
        "embedding_count": row.get("embedding_count", 0),
        "knowledge_base_id": row.get("knowledge_base_id"),
        "knowledge_base_name": row.get("knowledge_base_name"),
    }


def _build_download_response(source_reference: str, *, filename: str):
    storage = get_object_store()
    media_type = storage_media_type(filename)
    local_path = storage.get_local_path(source_reference)
    if local_path:
        if not os.path.isfile(local_path):
            raise HTTPException(status_code=404, detail="Document source file is missing")
        return FileResponse(local_path, filename=filename, media_type=media_type)

    if not storage.exists(source_reference):
        raise HTTPException(status_code=404, detail="Document source file is missing")
    return StreamingResponse(
        storage.iter_bytes(source_reference),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


class DocumentKnowledgeBaseAssignmentPayload(BaseModel):
    knowledge_base_id: str | None = None


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


@router.get("/{doc_id}/download")
async def download_document(
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
        raise HTTPException(status_code=403, detail="Not authorized to access this document")

    source_reference = str(doc.get("source_path") or "").strip()
    if not source_reference:
        raise HTTPException(status_code=404, detail="Document source file is missing")

    return _build_download_response(
        source_reference,
        filename=storage_filename(source_reference) or f"document-{doc_id}",
    )


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

    source_reference = str(doc.get("source_path") or "").strip()
    file_deleted = False
    if source_reference:
        storage = get_object_store()
        if storage.exists(source_reference):
            storage.delete(source_reference)
            file_deleted = True

    deleted = store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "message": "Deleted",
        "doc_id": doc_id,
        "file_deleted": file_deleted,
    }


@router.put("/{doc_id}/knowledge-base")
async def assign_document_knowledge_base(
    doc_id: int,
    payload: DocumentKnowledgeBaseAssignmentPayload,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if not ensure_schema_if_possible():
        raise HTTPException(status_code=400, detail="Database is not available")

    doc_store = MySQLDocStore()
    document = doc_store.get_document(doc_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if current_user.role != "admin" and document.get("user_id") != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to edit this document")

    knowledge_base_id = str(payload.knowledge_base_id or "").strip() or None
    if knowledge_base_id:
        knowledge_base = KnowledgeBaseStore().get_knowledge_base(knowledge_base_id)
        if knowledge_base is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        if current_user.role != "admin" and str(knowledge_base.get("user_id") or "") != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to use this knowledge base")
        KnowledgeBaseDocumentStore().assign_document(doc_id=doc_id, knowledge_base_id=knowledge_base_id)
    else:
        KnowledgeBaseDocumentStore().clear_document(doc_id=doc_id)

    refreshed = doc_store.get_document(doc_id)
    if refreshed is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _serialize_document(refreshed)


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

    source_reference = str(doc.get("source_path") or "").strip()
    if not source_reference or not get_object_store().exists(source_reference):
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
            "message": "鏂囨。閲嶅缓绱㈠紩宸插叆闃?",
            "file_path": source_reference,
            "storage_uri": source_reference,
            "filename": storage_filename(source_reference),
            "created_at": int(time.time()),
            "user_id": doc.get("user_id") or current_user.username,
            "task_type": "reindex_document",
            "doc_id": doc_id,
            "knowledge_base_id": doc.get("knowledge_base_id"),
            "retry_count": 0,
            "retryable": "false",
            "operation_key": operation_key,
        },
    )
    await enqueue_ingest_pdf(
        task_id,
        source_reference,
        user_id=doc.get("user_id") or current_user.username,
        knowledge_base_id=doc.get("knowledge_base_id"),
    )
    return {
        "message": "Reindex queued",
        "task_id": task_id,
        "doc_id": doc_id,
    }
