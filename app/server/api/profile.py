from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.infrastructure.database.models import User
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.server.api.auth import get_current_active_user

router = APIRouter()


@router.get("/profile/{user_id}")
async def get_profile(
    user_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    if current_user.role != "admin" and current_user.username != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to access this profile")
    if not ensure_schema_if_possible():
        return {"user_id": user_id, "profile": None}
    from app.skills.profile.profile_engine import UserProfileEngine

    engine = UserProfileEngine()
    return {"user_id": user_id, "profile": engine.get_profile(user_id)}
