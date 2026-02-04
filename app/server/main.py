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

from app.runtime.graph.graph import run_app
from app.infrastructure.checkpoint.redis_store import checkpoint_store
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.utils.logging import init_logging
from app.infrastructure.observability import get_langfuse_callback
from app.infrastructure.queue.redis_client import get_redis

# Import routers
from app.server.api import upload, tasks, settings, history, profile, vectorstore, auth, interrupt
from app.server.api.auth import get_current_active_user, get_current_admin_user


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_logging()
    print("后端脚手架已启动")
    ensure_schema_if_possible()

    redis = get_redis()
    await FastAPILimiter.init(redis)

    await checkpoint_store.get_saver()
    print(f"Checkpoint store initialized: {type(checkpoint_store)}")

    yield


app = FastAPI(title="Agent Scaffold API", version="1.0", lifespan=lifespan)


def per_req_config_modifier(config: Dict[str, Any], request: Any) -> Dict[str, Any]:
    """
    Injects Langfuse callback into the config for every request.
    Also injects user_id and thread_id into configurable params if needed.
    """
    # 自动生成 thread_id 如果不存在
    config.setdefault("configurable", {})
    if "thread_id" not in config["configurable"]:
        config["configurable"]["thread_id"] = str(uuid.uuid4())

    # 注入 user_id 到 configurable
    user = getattr(request.state, "user", None)
    if user:
        config["configurable"]["user_id"] = user.username

    handler = get_langfuse_callback()
    if handler:
        existing_callbacks = config.get("callbacks", [])
        if isinstance(existing_callbacks, list):
            config["callbacks"] = existing_callbacks + [handler]
        else:
            config["callbacks"] = [handler]
    return config


# 自定义中间件用于将 current_user 注入到 request.state，以便 per_req_config_modifier 使用
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from app.server.api.auth import decode_access_token
from app.infrastructure.database.orm import get_sessionmaker
from app.infrastructure.database.models import User
from sqlalchemy import select


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                payload = decode_access_token(token)
                if payload:
                    username = payload.get("sub")

                    # 简单起见，这里不查库验证 active，只解析 username
                    # 严谨的鉴权在 Depends 中做。这里只是为了传参给 LangGraph。
                    # 或者我们可以只传 username。
                    class SimpleUser:
                        def __init__(self, u):
                            self.username = u

                    request.state.user = SimpleUser(username)
            except Exception:
                pass
        response = await call_next(request)
        return response


app = FastAPI(title="Agent Scaffold API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuthMiddleware)

# 路由（LangServe）
# 保护 /chat 接口：需要登录 + 限流 (10次/60秒)
graph_app = run_app(checkpointer=checkpoint_store)

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
app.include_router(interrupt.router, dependencies=[Depends(get_current_active_user)])
app.include_router(upload.router)  # 移除 admin 限制，内部已根据 user 隔离
app.include_router(tasks.router)
app.include_router(settings.router)  # 内部已处理 Admin 限制
app.include_router(history.router, dependencies=[Depends(get_current_active_user)])
app.include_router(profile.router, dependencies=[Depends(get_current_active_user)])
app.include_router(vectorstore.router, dependencies=[Depends(get_current_admin_user)])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
