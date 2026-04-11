from datetime import datetime
from typing import Annotated, Any, cast

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from app.harness.runtime.event_service import HarnessEventService
from app.infrastructure.checkpoint.redis_store import checkpoint_store
from app.infrastructure.database.history_manager import history_manager
from app.infrastructure.database.models import User
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.database.stores import MySQLConversationStore
from app.runtime.graph.resume_service import GraphResumeService
from app.runtime.graph.state import ActionRequired
from app.server.api.auth import get_current_active_user

router = APIRouter(prefix="/interrupt", tags=["human-in-the-loop"])


class ApproveRequest(BaseModel):
    approved: bool = True
    comment: str | None = None


class InterruptStatusResponse(BaseModel):
    session_id: str
    interrupted: bool
    action_required: ActionRequired | None = None
    checkpoint_saved_at: str | None = None


class ApproveResponse(BaseModel):
    session_id: str
    approved: bool
    action_type: str
    approved_by: str
    approved_at: str


class ResumeExecutionResponse(BaseModel):
    session_id: str
    resumed: bool
    interrupted: bool | None = None
    reply: str | None = None
    messages: list[dict[str, str]] = []
    context: dict[str, Any] | None = None


def get_graph_resume_service():
    return GraphResumeService()


def get_interrupt_event_service():
    return HarnessEventService(database_optional=True)


def _normalize_resume_messages(messages: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(messages, list):
        return normalized
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        if role and content:
            normalized.append({"role": role, "content": content})
    return normalized


def _persist_interrupt_session_messages(
    *,
    user_id: str,
    session_id: str,
    messages: list[dict[str, Any]],
    background_tasks: BackgroundTasks | None = None,
) -> Any:
    if ensure_schema_if_possible():
        saved = MySQLConversationStore().save_session(user_id, session_id, messages, None)
        if background_tasks is not None:
            from app.server.session_history import update_memory_after_save

            background_tasks.add_task(update_memory_after_save, user_id, session_id, messages)
        return saved
    return history_manager.save_session(user_id, session_id, messages, None)


def _checkpoint_owner(checkpoint_data: dict[str, Any]) -> str | None:
    candidates: list[Any] = [
        checkpoint_data.get("user_id"),
        (checkpoint_data.get("context") or {}).get("user_id") if isinstance(checkpoint_data.get("context"), dict) else None,
    ]
    channel_values = checkpoint_data.get("channel_values")
    if isinstance(channel_values, dict):
        candidates.extend(
            [
                channel_values.get("user_id"),
                (channel_values.get("context") or {}).get("user_id")
                if isinstance(channel_values.get("context"), dict)
                else None,
            ]
        )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _history_visible_to_user(session_id: str, current_user: User) -> bool:
    if current_user.role == "admin":
        return True
    if ensure_schema_if_possible():
        return MySQLConversationStore().get_session_detail(current_user.username, session_id) is not None
    return history_manager.get_session(current_user.username, session_id) is not None


def _interrupt_event_user_id(checkpoint_data: dict[str, Any], current_user: User) -> str:
    owner = _checkpoint_owner(checkpoint_data)
    return owner or current_user.username


def _record_interrupt_event(
    *,
    session_id: str,
    checkpoint_data: dict[str, Any],
    current_user: User,
    event_type: str,
    details: dict[str, Any] | None = None,
    actor: str | None = None,
) -> dict[str, object] | None:
    return cast(
        dict[str, object] | None,
        get_interrupt_event_service().record(
        event_type=event_type,
        event_source="interrupt",
        user_id=_interrupt_event_user_id(checkpoint_data, current_user),
        session_id=session_id,
        actor=actor or current_user.username,
        details=dict(details or {}),
        ),
    )


def _ensure_interrupt_visible_to_user(
    *,
    session_id: str,
    checkpoint: dict[str, Any],
    current_user: User,
) -> dict[str, Any]:
    checkpoint_data = dict(checkpoint.get("checkpoint") or {})
    if current_user.role == "admin":
        return checkpoint_data
    owner = _checkpoint_owner(checkpoint_data)
    if owner:
        if owner != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this session interrupt")
        return checkpoint_data
    if not _history_visible_to_user(session_id, current_user):
        raise HTTPException(status_code=403, detail="Not authorized to access this session interrupt")
    return checkpoint_data


def _is_explicit_rejection(action_required: dict[str, Any] | None) -> bool:
    if not isinstance(action_required, dict):
        return False
    if bool(action_required.get("approved")):
        return False
    return bool(action_required.get("approved_by") or action_required.get("approved_at"))


def _is_approval_granted(action_required: dict[str, Any] | None) -> bool:
    return isinstance(action_required, dict) and bool(action_required.get("approved"))


def _resume_block_reason(action_required: dict[str, Any] | None) -> str:
    return "approval_not_granted" if _is_explicit_rejection(action_required) else "approval_pending"


@router.get("/{session_id}", response_model=InterruptStatusResponse)
async def get_interrupt_status(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    checkpoint = await checkpoint_store.load(session_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Session not found or no interrupt")

    checkpoint_data = _ensure_interrupt_visible_to_user(
        session_id=session_id,
        checkpoint=checkpoint,
        current_user=current_user,
    )
    action_required = checkpoint_data.get("action_required")
    interrupted = checkpoint_data.get("interrupted", False)

    return {
        "session_id": session_id,
        "interrupted": interrupted,
        "action_required": action_required,
        "checkpoint_saved_at": checkpoint.get("updated_at"),
    }


@router.get("/{session_id}/events")
async def list_interrupt_events(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    checkpoint = await checkpoint_store.load(session_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Session not found or no interrupt")

    checkpoint_data = _ensure_interrupt_visible_to_user(
        session_id=session_id,
        checkpoint=checkpoint,
        current_user=current_user,
    )
    user_id = None if current_user.role == "admin" else _interrupt_event_user_id(checkpoint_data, current_user)
    events = get_interrupt_event_service().list_for_session(
        session_id=session_id,
        user_id=user_id,
    )
    return {"events": events}


@router.post("/{session_id}/approve", response_model=ApproveResponse)
async def approve_action(
    session_id: str,
    request: ApproveRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    checkpoint = await checkpoint_store.load(session_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Session not found or no interrupt")

    checkpoint_data = _ensure_interrupt_visible_to_user(
        session_id=session_id,
        checkpoint=checkpoint,
        current_user=current_user,
    )
    action_required = checkpoint_data.get("action_required")

    if not action_required:
        raise HTTPException(status_code=400, detail="No pending action to approve")

    if not request.approved:
        action_required["approved"] = False
        action_required["approved_by"] = current_user.username
        action_required["approved_at"] = datetime.utcnow().isoformat()
        await checkpoint_store.save(session_id, checkpoint_data)
        _record_interrupt_event(
            session_id=session_id,
            checkpoint_data=checkpoint_data,
            current_user=current_user,
            event_type="interrupt.rejected",
            details={
                "action_type": str(action_required.get("action_type") or "unknown"),
                "comment": request.comment,
            },
        )

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
    _record_interrupt_event(
        session_id=session_id,
        checkpoint_data=checkpoint_data,
        current_user=current_user,
        event_type="interrupt.approved",
        details={
            "action_type": str(action_required.get("action_type") or "unknown"),
            "comment": request.comment,
        },
    )

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

    checkpoint_data = _ensure_interrupt_visible_to_user(
        session_id=session_id,
        checkpoint=checkpoint,
        current_user=current_user,
    )
    action_required = checkpoint_data.get("action_required")
    action_required_payload = action_required if isinstance(action_required, dict) else None

    if not _is_approval_granted(action_required_payload):
        _record_interrupt_event(
            session_id=session_id,
            checkpoint_data=checkpoint_data,
            current_user=current_user,
            event_type="interrupt.resume_blocked",
            details={
                "reason": _resume_block_reason(action_required_payload),
                "action_type": str((action_required_payload or {}).get("action_type") or "unknown"),
            },
        )
        raise HTTPException(status_code=400, detail="Action not yet approved")

    return {
        "session_id": session_id,
        "can_resume": True,
        "resume_payload": {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": checkpoint_data.get("checkpoint_ns", ""),
                "checkpoint_id": checkpoint_data.get("checkpoint_id"),
            }
        },
    }


@router.post("/{session_id}/resume", response_model=ResumeExecutionResponse)
async def resume_session(
    session_id: str,
    background_tasks: BackgroundTasks,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    checkpoint = await checkpoint_store.load(session_id)
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Session not found")

    _ensure_interrupt_visible_to_user(
        session_id=session_id,
        checkpoint=checkpoint,
        current_user=current_user,
    )
    checkpoint_data = dict(checkpoint.get("checkpoint") or {})
    checkpoint = await checkpoint_store.load(session_id) or {
        "checkpoint": checkpoint_data,
        "updated_at": checkpoint.get("updated_at"),
    }
    action_required = checkpoint_data.get("action_required")
    action_required_payload = action_required if isinstance(action_required, dict) else None
    if not _is_approval_granted(action_required_payload):
        _record_interrupt_event(
            session_id=session_id,
            checkpoint_data=checkpoint_data,
            current_user=current_user,
            event_type="interrupt.resume_blocked",
            details={
                "reason": _resume_block_reason(action_required_payload),
                "action_type": str((action_required_payload or {}).get("action_type") or "unknown"),
            },
        )
        raise HTTPException(status_code=400, detail="Action not yet approved")
    _record_interrupt_event(
        session_id=session_id,
        checkpoint_data=checkpoint_data,
        current_user=current_user,
        event_type="interrupt.resume_requested",
        details={},
    )
    result = await get_graph_resume_service().resume_approved_session(
        session_id=session_id,
        checkpoint=checkpoint,
    )
    if not bool(result.get("ok")):
        _record_interrupt_event(
            session_id=session_id,
            checkpoint_data=checkpoint_data,
            current_user=current_user,
            event_type="interrupt.resume_failed",
            details={
                "error_code": str(result.get("error_code") or ""),
                "error_message": str(result.get("error_message") or ""),
            },
        )
        raise HTTPException(
            status_code=400,
            detail=result.get("error_message") or "Session resume failed",
        )

    messages = _normalize_resume_messages(result.get("messages"))
    if messages:
        _persist_interrupt_session_messages(
            user_id=current_user.username,
            session_id=session_id,
            messages=messages,
            background_tasks=background_tasks,
        )
    _record_interrupt_event(
        session_id=session_id,
        checkpoint_data=checkpoint_data,
        current_user=current_user,
        event_type="interrupt.resumed",
        details={
            "interrupted": result.get("interrupted"),
            "message_count": len(messages),
            "reply_present": bool(result.get("reply")),
        },
    )

    return {
        "session_id": session_id,
        "resumed": True,
        "interrupted": result.get("interrupted"),
        "reply": result.get("reply"),
        "messages": messages,
        "context": result.get("context"),
    }
