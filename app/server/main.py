import os
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from app.runtime.graph.graph import app as graph_app
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.utils.logging import init_logging
from app.infrastructure.observability import get_langfuse_callback
from app.infrastructure.queue.redis_client import get_redis

# Import routers
from app.server.api import upload, tasks, settings, history, profile, vectorstore, auth
from app.server.api.auth import get_current_active_user, get_current_admin_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动钩子
    init_logging()
    print("后端脚手架已启动")
    ensure_schema_if_possible()

    # Init Rate Limiter
    redis = get_redis()
    await FastAPILimiter.init(redis)

    yield


app = FastAPI(title="Agent Scaffold API", version="1.0", lifespan=lifespan)

# 跨域（CORS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def per_req_config_modifier(config: Dict[str, Any], request: Any) -> Dict[str, Any]:
    """
    Injects Langfuse callback into the config for every request.
    """
    handler = get_langfuse_callback()
    if handler:
        existing_callbacks = config.get("callbacks", [])
        if isinstance(existing_callbacks, list):
            config["callbacks"] = existing_callbacks + [handler]
        else:
            config["callbacks"] = [handler]
    return config


# 路由（LangServe）
# 保护 /chat 接口：需要登录 + 限流 (10次/60秒)
add_routes(
    app,
    graph_app,
    path="/chat",
    enable_feedback_endpoint=True,
    per_req_config_modifier=per_req_config_modifier,
    dependencies=[
        Depends(get_current_active_user),
        Depends(RateLimiter(times=10, seconds=60)),
    ],
)

# 静态文件
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)
app.mount("/files", StaticFiles(directory="data/documents"), name="files")
app.mount("/uploads", StaticFiles(directory="data/uploads"), name="uploads")

# Include Routers
app.include_router(auth.router)
app.include_router(
    upload.router, dependencies=[Depends(get_current_admin_user)]
)  # 只有 Admin 能上传
app.include_router(tasks.router)
app.include_router(
    settings.router, dependencies=[Depends(get_current_admin_user)]
)  # Admin 配置
app.include_router(history.router, dependencies=[Depends(get_current_active_user)])
app.include_router(profile.router, dependencies=[Depends(get_current_active_user)])
app.include_router(vectorstore.router, dependencies=[Depends(get_current_admin_user)])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
