from typing import Dict, Any
from fastapi import APIRouter
from app.infrastructure.config.config_manager import config_manager

router = APIRouter()


# 配置
@router.get("/settings")
async def get_settings():
    return config_manager.get_config()


@router.post("/settings")
async def update_settings(config: Dict[str, Any]):
    return config_manager.update_config(config)
