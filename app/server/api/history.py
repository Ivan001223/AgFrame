import uuid
from typing import Dict, Any, Annotated
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.infrastructure.database.history_manager import history_manager
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.stores import MySQLConversationStore
from app.memory.long_term.memory_update_service import memory_update_service
from app.infrastructure.database.models import User
from app.server.api.auth import get_current_active_user

router = APIRouter()


# 历史记录
@router.get("/history/{user_id}")
async def get_history(
    user_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    # Enforce data isolation
    if user_id != current_user.username:
        raise HTTPException(
            status_code=403, detail="Not authorized to access this history"
        )

    if ensure_schema_if_possible():
        store = MySQLConversationStore()
        return {"history": store.list_sessions(user_id)}
    return {"history": history_manager.get_history(user_id)}


@router.post("/history/{user_id}/save")
async def save_history(
    user_id: str,
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if user_id != current_user.username:
        raise HTTPException(
            status_code=403, detail="Not authorized to save to this history"
        )

    session_id = payload.get("session_id") or str(uuid.uuid4())
    messages = payload.get("messages") or []
    title = payload.get("title")

    if not ensure_schema_if_possible():
        return history_manager.save_session(user_id, session_id, messages, title)

    store = MySQLConversationStore()
    saved = store.save_session(user_id, session_id, messages, title)
    background_tasks.add_task(
        memory_update_service.update_after_save, user_id, session_id, messages
    )

    return saved


@router.delete("/history/{user_id}/{session_id}")
async def delete_history(
    user_id: str,
    session_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if user_id != current_user.username:
        raise HTTPException(
            status_code=403, detail="Not authorized to delete this history"
        )

    if ensure_schema_if_possible():
        store = MySQLConversationStore()
        store.delete_session(user_id, session_id)
    else:
        history_manager.delete_session(user_id, session_id)
    return {"message": "Deleted"}
