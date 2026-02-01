from typing import Dict, Any, Annotated
from fastapi import APIRouter, Depends
from app.infrastructure.config.config_manager import config_manager
from app.server.api.auth import get_current_active_user, get_current_admin_user
from app.infrastructure.database.models import User, UserProfile
from app.infrastructure.database.orm import get_session
from sqlalchemy import select
import time

router = APIRouter()


# 系统全局配置（仅 Admin）
@router.get("/settings", dependencies=[Depends(get_current_admin_user)])
async def get_settings():
    return config_manager.get_config()


@router.post("/settings", dependencies=[Depends(get_current_admin_user)])
async def update_settings(config: Dict[str, Any]):
    return config_manager.update_config(config)


# 用户个性化配置（隔离）
@router.get("/settings/user")
async def get_user_settings(
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    # 从 UserProfile 中读取 settings 字段（如果没有，可以新增或存在 profile_json 中）
    # 这里假设存在 profile_json["settings"] 中
    with get_session() as session:
        stmt = select(UserProfile).where(UserProfile.user_id == current_user.username)
        profile = session.execute(stmt).scalar_one_or_none()
        if profile and profile.profile_json:
            return profile.profile_json.get("settings", {})
    return {}


@router.post("/settings/user")
async def update_user_settings(
    settings: Dict[str, Any],
    current_user: Annotated[User, Depends(get_current_active_user)],
):
    with get_session() as session:
        stmt = select(UserProfile).where(UserProfile.user_id == current_user.username)
        profile = session.execute(stmt).scalar_one_or_none()

        if not profile:
            # Create new profile if not exists
            profile_data = {"settings": settings}
            new_profile = UserProfile(
                user_id=current_user.username,
                profile_json=profile_data,
                updated_at=int(time.time()),
            )
            session.add(new_profile)
        else:
            # Update existing
            current_data = dict(profile.profile_json or {})
            current_data["settings"] = settings

            # 手动更新字段
            profile.profile_json = current_data
            profile.updated_at = int(time.time())
            session.add(profile)

            # Explicitly merge/add ensures session tracks it
            # In some cases with SQLite JSON, we might need flag_modified
            from sqlalchemy.orm.attributes import flag_modified

            flag_modified(profile, "profile_json")

    return {"message": "User settings updated"}
