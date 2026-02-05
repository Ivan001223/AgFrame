from typing import Annotated, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from datetime import datetime

from app.server.api.auth import get_current_active_user
from app.infrastructure.database.models import User
from app.infrastructure.checkpoint.redis_store import checkpoint_store
from app.runtime.graph.state import ActionRequired


router = APIRouter(prefix="/interrupt", tags=["human-in-the-loop"])


class ApproveRequest(BaseModel):
    approved: bool = True
    comment: Optional[str] = None


class InterruptStatusResponse(BaseModel):
    session_id: str
    interrupted: bool
    action_required: Optional[ActionRequired] = None
    checkpoint_saved_at: Optional[str] = None


class ApproveResponse(BaseModel):
    session_id: str
    approved: bool
    action_type: str
    approved_by: str
    approved_at: str


@router.get("/{session_id}", response_model=InterruptStatusResponse)
async def get_interrupt_status(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    checkpoint = await checkpoint_store.load(session_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Session not found or no interrupt")

    checkpoint_data = checkpoint.get("checkpoint", {})
    action_required = checkpoint_data.get("action_required")
    interrupted = checkpoint_data.get("interrupted", False)

    return {
        "session_id": session_id,
        "interrupted": interrupted,
        "action_required": action_required,
        "checkpoint_saved_at": checkpoint.get("updated_at"),
    }


@router.post("/{session_id}/approve", response_model=ApproveResponse)
async def approve_action(
    session_id: str,
    request: ApproveRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    checkpoint = await checkpoint_store.load(session_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Session not found or no interrupt")

    checkpoint_data = checkpoint.get("checkpoint", {})
    action_required = checkpoint_data.get("action_required")

    if not action_required:
        raise HTTPException(status_code=400, detail="No pending action to approve")

    if not request.approved:
        action_required["approved"] = False
        action_required["approved_by"] = current_user.username
        action_required["approved_at"] = datetime.utcnow().isoformat()
        await checkpoint_store.save(session_id, checkpoint_data)

        return {
            "session_id": session_id,
            "approved": False,
            "action_type": action_required.get("action_type", "unknown"),
            "approved_by": current_user.username,
            "approved_at": action_required["approved_at"],
        }

    action_required["approved"] = True
    action_required["approved_by"] = current_user.username
    action_required["approved_at"] = datetime.utcnow().isoformat()

    checkpoint_data["action_required"] = action_required
    await checkpoint_store.save(session_id, checkpoint_data)

    return {
        "session_id": session_id,
        "approved": True,
        "action_type": action_required.get("action_type", "unknown"),
        "approved_by": current_user.username,
        "approved_at": action_required["approved_at"],
    }


@router.get("/{session_id}/resume")
async def get_resume_command(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    checkpoint = await checkpoint_store.load(session_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Session not found")

    checkpoint_data = checkpoint.get("checkpoint", {})
    action_required = checkpoint_data.get("action_required")

    if action_required and not action_required.get("approved"):
        raise HTTPException(status_code=400, detail="Action not yet approved")

    return {
        "session_id": session_id,
        "can_resume": True,
        "resume_payload": {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": "default",
                "checkpoint_id": checkpoint_data.get("checkpoint_id"),
            }
        },
    }
