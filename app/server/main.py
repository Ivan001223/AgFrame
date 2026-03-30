import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.infrastructure.checkpoint.redis_store import checkpoint_store
from app.infrastructure.config.settings import settings
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.queue.redis_client import get_redis
from app.infrastructure.utils.logging import get_logger, init_logging
from app.server.api import (
    auth,
    chat,
    documents,
    file_access,
    harness,
    health,
    history,
    interrupt,
    memory,
    profile,
    tasks,
    upload,
    vectorstore,
)
from app.server.api import (
    settings as api_settings,
)
from app.server.api.auth import (
    decode_access_token,
    get_current_active_user,
    get_current_admin_user,
)
from app.server.chat_runtime import apply_request_runtime_config, get_chat_graph_app
from app.server.cors_policy import build_cors_options
from app.server.error_handlers import register_exception_handlers
from app.server.rate_limit import build_rate_limiter, init_rate_limiter

logger = get_logger("exception_handler")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_logging()
    logger.info("Backend scaffold started")
    settings.validate_security()
    ensure_schema_if_possible()

    redis = get_redis()
    await init_rate_limiter(redis)

    await checkpoint_store.get_saver()
    logger.info(f"Checkpoint store initialized: {type(checkpoint_store)}")

    yield


def per_req_config_modifier(config: dict[str, Any], request: Any) -> dict[str, Any]:
    """
    Injects Langfuse callback into the config for every request.
    Also injects user_id and thread_id into configurable params if needed.
    """
    return apply_request_runtime_config(config, request)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID")
        request.state.request_id = request_id if request_id else str(uuid.uuid4())
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
            except Exception as e:
                logger.debug(f"Auth middleware error: {e}")
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response


app = FastAPI(title="Agent Scaffold API", version="0.3.1", lifespan=lifespan)

server_config = settings.server
storage_config = settings.storage_local

cors_options = build_cors_options(
    cors_origins=server_config.cors_origins,
    cors_allow_credentials=server_config.cors_allow_credentials,
)
app.add_middleware(
    CORSMiddleware,
    **cors_options,
)
app.add_middleware(AuthMiddleware)

register_exception_handlers(app)


# 路由（LangServe）
# 保护 /chat 接口：需要登录 + 限流 (10次/60秒)
graph_app = get_chat_graph_app()

add_routes(
    app,
    graph_app,
    path="/chat",
    enable_feedback_endpoint=True,
    per_req_config_modifier=per_req_config_modifier,
    dependencies=[
        Depends(get_current_active_user),
        Depends(build_rate_limiter(times=10, seconds=60)),
    ],
)

# 文件目录
documents_dir = storage_config.documents_dir
uploads_dir = storage_config.uploads_dir

os.makedirs(documents_dir, exist_ok=True)
os.makedirs(uploads_dir, exist_ok=True)

# Include Routers
# 健康检查路由（无需认证）
app.include_router(health.router)

# 认证路由
app.include_router(auth.router)
app.include_router(chat.router, dependencies=[Depends(build_rate_limiter(times=10, seconds=60))])
app.include_router(interrupt.router, dependencies=[Depends(get_current_active_user)])
app.include_router(upload.router)  # 移除 admin 限制，内部已根据 user 隔离
app.include_router(tasks.router)
app.include_router(harness.router, dependencies=[Depends(get_current_active_user)])
app.include_router(api_settings.router)  # 内部已处理 Admin 限制
app.include_router(history.router, dependencies=[Depends(get_current_active_user)])
app.include_router(memory.router, dependencies=[Depends(get_current_active_user)])
app.include_router(profile.router, dependencies=[Depends(get_current_active_user)])
app.include_router(documents.router, dependencies=[Depends(get_current_active_user)])
app.include_router(file_access.router, dependencies=[Depends(get_current_active_user)])
app.include_router(vectorstore.router, dependencies=[Depends(get_current_admin_user)])

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
    )
