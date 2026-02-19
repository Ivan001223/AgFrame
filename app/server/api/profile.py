from fastapi import APIRouter

from app.infrastructure.database.schema import ensure_schema_if_possible
from app.skills.profile.profile_engine import UserProfileEngine

router = APIRouter()


@router.get("/profile/{user_id}")
async def get_profile(user_id: str):
    if not ensure_schema_if_possible():
        return {"user_id": user_id, "profile": None}
    engine = UserProfileEngine()
    return {"user_id": user_id, "profile": engine.get_profile(user_id)}
