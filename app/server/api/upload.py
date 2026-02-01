import os
import uuid
import time
from typing import List, Annotated

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.infrastructure.queue.client import enqueue_ingest_pdf
from app.infrastructure.queue.redis_client import init_task
from app.skills.ocr.ocr_engine import ocr_engine
from app.server.api.auth import get_current_active_user
from app.infrastructure.database.models import User

router = APIRouter()


# 上传（RAG）
@router.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
):
    upload_dir = "data/documents"
    os.makedirs(upload_dir, exist_ok=True)

    results = []

    # 允许普通用户上传，因为数据现在是隔离的
    # 如果希望限制 admin，在 main.py 或这里加 Depends(get_current_admin_user)
    # 根据用户需求：“上传的文件等都应该由用户来管理” -> 普通用户可以上传

    user_id = current_user.username if current_user else "unknown"

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

            task_id = str(uuid.uuid4())
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
    uploads_dir = "data/uploads"
    os.makedirs(uploads_dir, exist_ok=True)

    original_name = os.path.basename(file.filename or "")
    safe_name = original_name or "upload.bin"
    safe_name = f"{uuid.uuid4()}_{safe_name}"
    file_path = os.path.join(uploads_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # text = ocr_engine.process_file(file_path)  # 若已配置 OCR 引擎可取消注释启用
    return {"url": f"/uploads/{safe_name}", "text": "OCR Placeholder"}
