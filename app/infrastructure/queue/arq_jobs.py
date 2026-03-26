from __future__ import annotations

import time
from typing import Any

import anyio

from app.harness.runtime.run_service import build_run_service
from app.harness.runtime.verification_service import VerificationService
from app.infrastructure.queue.redis_client import (
    append_task_incident,
    get_task,
    release_task_operation,
    update_task,
)
from app.infrastructure.utils.logging import bind_logger, get_logger
from app.skills.rag.rag_engine import get_rag_engine

_log = get_logger("task_queue.arq_jobs")


def _normalize_ingest_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if result is True:
        return {"ok": True}
    return {
        "ok": False,
        "error_code": "ingest_returned_false",
        "error_message": "add_knowledge_base 返回 False",
    }


async def ingest_pdf(
    ctx: dict[str, Any], task_id: str, file_path: str, user_id: str = None
) -> bool:
    logger = bind_logger(_log, session_id=task_id, node="ingest_pdf")
    started_at = int(time.time())
    existing_task = await get_task(task_id)
    operation_key = str(existing_task.get("operation_key") or "")
    await update_task(
        task_id,
        {
            "status": "running",
            "progress": 1,
            "step": "start",
            "started_at": started_at,
            "message": "开始处理",
            "error": "",
            "user_id": user_id or "unknown",
        },
    )

    try:
        await update_task(
            task_id, {"progress": 10, "step": "validating", "message": "校验文件"}
        )
        await update_task(
            task_id, {"progress": 25, "step": "ingest", "message": "开始摄取"}
        )
        await update_task(
            task_id, {"progress": 60, "step": "indexing", "message": "构建索引"}
        )
        await update_task(
            task_id, {"progress": 85, "step": "finalizing", "message": "写入结果"}
        )
        # 传递 user_id 给 RAG 引擎
        result = await anyio.to_thread.run_sync(
            lambda: _normalize_ingest_result(
                get_rag_engine().add_knowledge_base(file_path, user_id=user_id)
            )
        )
        finished_at = int(time.time())
        if result.get("ok"):
            await update_task(
                task_id,
                {
                    "status": "succeeded",
                    "progress": 100,
                    "step": "done",
                    "finished_at": finished_at,
                    "message": "处理完成",
                    "retryable": "false",
                    "result_stage": str(result.get("stage") or ""),
                },
            )
            logger.info("task succeeded file_path=%s", file_path)
            await release_task_operation(operation_key, expected_task_id=task_id)
            return True

        await update_task(
            task_id,
            {
                "status": "failed",
                "progress": 100,
                "step": "failed",
                "finished_at": finished_at,
                "error": str(result.get("error_message") or "add_knowledge_base 返回 False"),
                "error_code": str(result.get("error_code") or "ingest_returned_false"),
                "result_stage": str(result.get("stage") or ""),
                "retryable": "true",
            },
        )
        await append_task_incident(
            {
                "task_id": task_id,
                "user_id": user_id or "unknown",
                "status": "failed",
                "error_code": str(result.get("error_code") or "ingest_returned_false"),
                "error_message": str(result.get("error_message") or "add_knowledge_base 返回 False"),
                "stage": str(result.get("stage") or ""),
                "file_path": file_path,
                "timestamp": finished_at,
            }
        )
        logger.info(
            "task failed file_path=%s error_code=%s stage=%s",
            file_path,
            result.get("error_code"),
            result.get("stage"),
        )
        await release_task_operation(operation_key, expected_task_id=task_id)
        return False
    except Exception as e:
        finished_at = int(time.time())
        await update_task(
            task_id,
            {
                "status": "failed",
                "progress": 100,
                "step": "exception",
                "finished_at": finished_at,
                "error": str(e),
                "retryable": "true",
            },
        )
        await append_task_incident(
            {
                "task_id": task_id,
                "user_id": user_id or "unknown",
                "status": "failed",
                "error_code": "task_exception",
                "error_message": str(e),
                "stage": "exception",
                "file_path": file_path,
                "timestamp": finished_at,
            }
        )
        logger.exception("task exception file_path=%s", file_path)
        await release_task_operation(operation_key, expected_task_id=task_id)
        return False


async def run_harness_task(ctx: dict[str, Any], run_id: str) -> bool:
    service = build_run_service()
    run = service.get_run(run_id)
    if not run:
        return False

    service.mark_running(run_id)

    try:
        if run.get("task_type") != "document_ingest":
            verification = VerificationService().build_document_ingest_result(
                ok=False,
                stage="unsupported_task_type",
                error_code="unsupported_task_type",
                error_message="unsupported harness task type",
            )
            service.complete_with_verification(run_id, verification)
            return False

        input_json = run.get("input_json") or {}
        file_path = str(input_json.get("file_path") or "")
        user_id = str(run.get("user_id") or "") or None

        result = await anyio.to_thread.run_sync(
            lambda: _normalize_ingest_result(get_rag_engine().add_knowledge_base(file_path, user_id=user_id))
        )
        verification = VerificationService().build_document_ingest_result(
            ok=bool(result.get("ok")),
            stage=str(result.get("stage") or "") or None,
            error_code=str(result.get("error_code") or "") or None,
            error_message=str(result.get("error_message") or "") or None,
        )
        service.complete_with_verification(run_id, verification)
        return bool(result.get("ok"))
    except Exception as exc:
        verification = VerificationService().build_document_ingest_result(
            ok=False,
            stage="exception",
            error_code="task_exception",
            error_message=str(exc),
        )
        service.complete_with_verification(run_id, verification)
        return False


async def resume_harness_task(ctx: dict[str, Any], run_id: str) -> bool:
    return await run_harness_task(ctx, run_id)
