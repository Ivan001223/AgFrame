import logging
import os
import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.infrastructure.config.settings import settings
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.stores import MySQLDocStore
from app.infrastructure.database.models import User
from app.infrastructure.queue.client import enqueue_ingest_pdf
from app.infrastructure.queue.redis_client import (
    claim_task_operation,
    get_task,
    init_task,
    release_task_operation,
)
from app.infrastructure.utils.files import sha256_file
from app.server.api.auth import get_current_active_user

router = APIRouter()
logger = logging.getLogger(__name__)


def _get_ocr_engine():
    from app.skills.ocr.ocr_engine import ocr_engine

    return ocr_engine


def _extract_uploaded_image_text(file_path: str) -> str:
    try:
        return _get_ocr_engine().process_file(file_path) or ""
    except Exception as exc:
        logger.warning("image OCR failed path=%s error=%s", file_path, exc)
        return ""


# 上传（RAG）
@router.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
):
    user_id = current_user.username if current_user else "unknown"

    # 物理路径隔离：data/documents/{user_id}/
    upload_dir = os.path.join(settings.storage_local.documents_dir, user_id)
    os.makedirs(upload_dir, exist_ok=True)

    results = []

    for file in files:
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
            safe_name = original_name or f"{uuid.uuid4()}.pdf"
            safe_name = f"{uuid.uuid4()}_{safe_name}"
            file_path = os.path.join(upload_dir, safe_name)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            if ensure_schema_if_possible():
                checksum = sha256_file(file_path)
                existing = MySQLDocStore().find_by_checksum(
                    user_id=user_id,
                    checksum=checksum,
                )
                if existing:
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                    results.append(
                        {
                            "filename": safe_name,
                            "status": "duplicate",
                            "message": "Document already exists",
                            "existing_doc_id": existing["doc_id"],
                        }
                    )
                    continue
                operation_key = f"upload_pdf:{user_id}:{checksum}"
            else:
                checksum = None
                operation_key = f"upload_pdf:{user_id}:{safe_name}"

            task_id = str(uuid.uuid4())
            claimed_task_id = await claim_task_operation(operation_key, task_id)
            if claimed_task_id != task_id:
                existing_task = await get_task(claimed_task_id)
                try:
                    os.remove(file_path)
                except OSError:
                    pass
                if existing_task.get("status") in {"queued", "running"}:
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
                    results.append(
                        {
                            "filename": safe_name,
                            "status": "already_queued",
                            "task_id": claimed_task_id,
                        }
                    )
                    continue
            await init_task(
                task_id,
                {
                    "task_id": task_id,
                    "status": "queued",
                    "progress": 0,
                    "step": "queued",
                    "message": "已入队",
                    "file_path": file_path,
                    "filename": safe_name,
                    "created_at": int(time.time()),
                    "user_id": user_id,  # 绑定用户 ID
                    "retry_count": 0,
                    "retryable": "false",
                    "operation_key": operation_key,
                },
            )
            # 传 user_id 给队列任务，以便写入 Document 表时关联用户
            await enqueue_ingest_pdf(task_id, file_path, user_id=user_id)
            results.append(
                {"filename": safe_name, "status": "queued", "task_id": task_id}
            )
        except Exception as e:
            results.append(
                {"filename": file.filename, "status": "error", "message": str(e)}
            )

    return {"results": results}


# 上传（OCR）
@router.post("/upload/image")
async def upload_image(
    file: UploadFile = File(...),
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
):
    # Image upload logic usually for quick OCR or multimodal,
    # not necessarily RAG ingestion. But could verify user.
    user_id = current_user.username if current_user else "unknown"
    uploads_dir = os.path.join(settings.storage_local.uploads_dir, user_id)
    os.makedirs(uploads_dir, exist_ok=True)

    original_name = os.path.basename(file.filename or "")
    safe_name = original_name or "upload.bin"
    safe_name = f"{uuid.uuid4()}_{safe_name}"
    file_path = os.path.join(uploads_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text = _extract_uploaded_image_text(file_path)
    return {"url": f"/uploads/{user_id}/{safe_name}", "text": text}
