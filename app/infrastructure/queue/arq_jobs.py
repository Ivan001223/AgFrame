from __future__ import annotations

import time
from typing import Any, Dict

import anyio

from app.skills.rag.rag_engine import get_rag_engine
from app.infrastructure.queue.redis_client import update_task
from app.infrastructure.utils.logging import bind_logger, get_logger


_log = get_logger("task_queue.arq_jobs")


async def ingest_pdf(ctx: Dict[str, Any], task_id: str, file_path: str) -> bool:
    logger = bind_logger(_log, session_id=task_id, node="ingest_pdf")
    started_at = int(time.time())
    await update_task(
        task_id,
        {
            "status": "running",
            "progress": 1,
            "step": "start",
            "started_at": started_at,
            "message": "开始处理",
            "error": "",
        },
    )

    try:
        await update_task(task_id, {"progress": 5, "step": "ingest", "message": "开始摄取"})
        ok = await anyio.to_thread.run_sync(lambda: bool(get_rag_engine().add_knowledge_base(file_path)))
        finished_at = int(time.time())
        if ok:
            await update_task(
                task_id,
                {
                    "status": "succeeded",
                    "progress": 100,
                    "step": "done",
                    "finished_at": finished_at,
                    "message": "处理完成",
                },
            )
            logger.info("task succeeded file_path=%s", file_path)
            return True

        await update_task(
            task_id,
            {
                "status": "failed",
                "progress": 100,
                "step": "failed",
                "finished_at": finished_at,
                "error": "add_knowledge_base 返回 False",
            },
        )
        logger.info("task failed(return_false) file_path=%s", file_path)
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
            },
        )
        logger.exception("task exception file_path=%s", file_path)
        return False

