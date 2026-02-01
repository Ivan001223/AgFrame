import os
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

from app.runtime.graph.graph import app as graph_app
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.utils.logging import init_logging
from app.infrastructure.observability import get_langfuse_callback

# Import routers
from app.server.api import upload, tasks, settings, history, profile, vectorstore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动钩子（例如模型下载）可放在这里
    init_logging()
    print("后端脚手架已启动")
    ensure_schema_if_possible()
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
add_routes(
    app,
    graph_app,
    path="/chat",
    enable_feedback_endpoint=True,
    per_req_config_modifier=per_req_config_modifier,
)

# 静态文件
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)
app.mount("/files", StaticFiles(directory="data/documents"), name="files")
app.mount("/uploads", StaticFiles(directory="data/uploads"), name="uploads")

# Include Routers
app.include_router(upload.router)
app.include_router(tasks.router)
app.include_router(settings.router)
app.include_router(history.router)
app.include_router(profile.router)
app.include_router(vectorstore.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
