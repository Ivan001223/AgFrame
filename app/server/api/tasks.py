from typing import Annotated
from fastapi import APIRouter, HTTPException, Depends
from app.infrastructure.queue.redis_client import get_task
from app.server.api.auth import get_current_active_user, get_current_admin_user
from app.infrastructure.database.models import User

router = APIRouter()


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

    return task
