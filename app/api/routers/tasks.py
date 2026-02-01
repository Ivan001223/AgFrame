from fastapi import APIRouter, HTTPException
from app.core.task_queue.redis_client import get_task

router = APIRouter()


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = await get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
