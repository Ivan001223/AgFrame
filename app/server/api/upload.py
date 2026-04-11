import hashlib
import logging
import os
import tempfile
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.infrastructure.config.settings import settings
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
    build_document_storage_key,
    build_upload_storage_key,
    get_object_store,
)
from app.infrastructure.utils.files import sha256_file
from app.server.api.auth import get_current_active_user

router = APIRouter()
logger = logging.getLogger(__name__)
UPLOAD_CHUNK_SIZE = 1024 * 1024


def _get_ocr_engine():
    from app.skills.ocr.ocr_engine import ocr_engine

    return ocr_engine


def _extract_uploaded_image_text(file_path: str) -> str:
    try:
        return _get_ocr_engine().process_file(file_path) or ""
    except Exception as exc:
        logger.warning("image OCR failed path=%s error=%s", file_path, exc)
        return ""


def _safe_uploaded_name(original_name: str | None, *, fallback: str) -> str:
    base_name = os.path.basename(original_name or "").strip() or fallback
    return f"{uuid.uuid4()}_{base_name}"


async def _write_upload_to_temp_file(file: UploadFile, *, suffix: str = "") -> tuple[str, str]:
    hasher = hashlib.sha256()
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                temp_file.write(chunk)
    finally:
        await file.close()
    return temp_path, hasher.hexdigest()


def _remove_temp_file(path: str | None) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


async def _safe_release_task_operation(operation_key: str, task_id: str) -> None:
    if not operation_key or not task_id:
        return
    try:
        await release_task_operation(operation_key, expected_task_id=task_id)
    except Exception as exc:
        logger.warning("failed to release task operation key=%s error=%s", operation_key, exc)


@router.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    knowledge_base_id: str | None = Form(default=None),
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
):
    user_id = current_user.username if current_user else "unknown"
    selected_knowledge_base_id = str(knowledge_base_id or "").strip() or None

    if selected_knowledge_base_id and ensure_schema_if_possible():
        selected_knowledge_base = KnowledgeBaseStore().get_knowledge_base(selected_knowledge_base_id)
        if selected_knowledge_base is None:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        if current_user and current_user.role != "admin" and str(selected_knowledge_base.get("user_id") or "") != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to use this knowledge base")

    results = []

    for file in files:
        temp_path = ""
        original_name = os.path.basename(file.filename or "")
        if not original_name.lower().endswith(".pdf"):
            results.append(
                {
                    "filename": file.filename,
                    "status": "skipped",
                    "message": "Only PDF supported",
                }
            )
            continue

        try:
            task_id = ""
            operation_key = ""
            safe_name = _safe_uploaded_name(original_name, fallback=f"{uuid.uuid4()}.pdf")
            temp_path, checksum = await _write_upload_to_temp_file(file, suffix=".pdf")

            if ensure_schema_if_possible():
                existing = MySQLDocStore().find_by_checksum(
                    user_id=user_id,
                    checksum=checksum,
                )
                if existing:
                    if selected_knowledge_base_id:
                        KnowledgeBaseDocumentStore().assign_document(
                            doc_id=int(existing["doc_id"]),
                            knowledge_base_id=selected_knowledge_base_id,
                        )
                    _remove_temp_file(temp_path)
                    results.append(
                        {
                            "filename": safe_name,
                            "status": "duplicate",
                            "message": "Document already exists",
                            "existing_doc_id": existing["doc_id"],
                            "knowledge_base_id": selected_knowledge_base_id,
                        }
                    )
                    continue
                operation_key = f"upload_pdf:{user_id}:{checksum}"
            else:
                operation_key = f"upload_pdf:{user_id}:{safe_name}"

            task_id = str(uuid.uuid4())
            claimed_task_id = await claim_task_operation(operation_key, task_id)
            if claimed_task_id != task_id:
                existing_task = await get_task(claimed_task_id)
                if existing_task.get("status") in {"queued", "running"}:
                    _remove_temp_file(temp_path)
                    results.append(
                        {
                            "filename": safe_name,
                            "status": "already_queued",
                            "task_id": claimed_task_id,
                        }
                    )
                    continue
                await release_task_operation(
                    operation_key,
                    expected_task_id=claimed_task_id,
                )
                claimed_task_id = await claim_task_operation(operation_key, task_id)
                if claimed_task_id != task_id:
                    _remove_temp_file(temp_path)
                    results.append(
                        {
                            "filename": safe_name,
                            "status": "already_queued",
                            "task_id": claimed_task_id,
                        }
                    )
                    continue

            storage_uri = get_object_store().store_file(
                source_path=temp_path,
                key=build_document_storage_key(owner=user_id, filename=safe_name),
            )
            _remove_temp_file(temp_path)
            temp_path = ""

            await init_task(
                task_id,
                {
                    "task_id": task_id,
                    "status": "queued",
                    "progress": 0,
                    "step": "queued",
                    "message": "宸插叆闃?",
                    "file_path": storage_uri,
                    "storage_uri": storage_uri,
                    "filename": safe_name,
                    "created_at": int(time.time()),
                    "user_id": user_id,
                    "retry_count": 0,
                    "retryable": "false",
                    "operation_key": operation_key,
                    "knowledge_base_id": selected_knowledge_base_id,
                },
            )
            await enqueue_ingest_pdf(
                task_id,
                storage_uri,
                user_id=user_id,
                knowledge_base_id=selected_knowledge_base_id,
            )
            results.append(
                {
                    "filename": safe_name,
                    "status": "queued",
                    "task_id": task_id,
                    "knowledge_base_id": selected_knowledge_base_id,
                }
            )
        except Exception as exc:
            _remove_temp_file(temp_path)
            if operation_key and task_id:
                await _safe_release_task_operation(operation_key, task_id)
            results.append(
                {"filename": file.filename, "status": "error", "message": str(exc)}
            )

    return {"results": results}


@router.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
):
    user_id = current_user.username if current_user else "unknown"
    original_name = os.path.basename(file.filename or "")
    safe_name = _safe_uploaded_name(original_name, fallback="upload.bin")
    suffix = os.path.splitext(original_name)[1]
    temp_path, _ = await _write_upload_to_temp_file(file, suffix=suffix)

    try:
        text = _extract_uploaded_image_text(temp_path)
        storage_uri = get_object_store().store_file(
            source_path=temp_path,
            key=build_upload_storage_key(owner=user_id, filename=safe_name),
        )
    finally:
        _remove_temp_file(temp_path)

    return {
        "url": f"/uploads/{user_id}/{safe_name}",
        "text": text,
        "storage_uri": storage_uri,
    }
