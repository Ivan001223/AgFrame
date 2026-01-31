import os
import uvicorn
from contextlib import asynccontextmanager
from typing import List, Dict, Any
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# 导入通用图
from app.core.workflow.graph import app as graph_app
from app.core.config.config_manager import config_manager
from app.core.services.ocr_engine import ocr_engine
from app.core.services.rag_engine import get_rag_engine
from app.core.database.history_manager import history_manager
from app.core.database.schema import ensure_schema_if_possible
from app.core.database.stores import MySQLConversationStore
from app.core.services.memory_update_service import memory_update_service
from app.core.services.profile_engine import UserProfileEngine
from app.core.utils.logging import init_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动钩子（例如模型下载）可放在这里
    init_logging()
    print("后端脚手架已启动")
    ensure_schema_if_possible()
    yield

app = FastAPI(
    title="Agent Scaffold API",
    version="1.0",
    lifespan=lifespan
)

# 跨域（CORS）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由（LangServe）
add_routes(
    app,
    graph_app,
    path="/chat",
    enable_feedback_endpoint=True,
)

# 静态文件
os.makedirs("data/documents", exist_ok=True)
os.makedirs("data/uploads", exist_ok=True)
app.mount("/files", StaticFiles(directory="data/documents"), name="files")
app.mount("/uploads", StaticFiles(directory="data/uploads"), name="uploads")

# 上传（RAG）
@app.post("/upload")
async def upload_documents(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    upload_dir = "data/documents"
    results = []
    
    for file in files:
        original_name = os.path.basename(file.filename or "")
        if not original_name.lower().endswith(".pdf"):
            results.append({"filename": file.filename, "status": "skipped", "message": "Only PDF supported"})
            continue
        try:
            safe_name = original_name or f"{uuid.uuid4()}.pdf"
            safe_name = f"{uuid.uuid4()}_{safe_name}"
            file_path = os.path.join(upload_dir, safe_name)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            background_tasks.add_task(get_rag_engine().add_knowledge_base, file_path)
            results.append({"filename": safe_name, "status": "uploaded"})
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "message": str(e)})
            
    return {"results": results}

# 上传（OCR）
@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    uploads_dir = "data/uploads"
    original_name = os.path.basename(file.filename or "")
    safe_name = original_name or "upload.bin"
    safe_name = f"{uuid.uuid4()}_{safe_name}"
    file_path = os.path.join(uploads_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    # text = ocr_engine.process_file(file_path)  # 若已配置 OCR 引擎可取消注释启用
    return {"url": f"/uploads/{safe_name}", "text": "OCR Placeholder"}

# 配置
@app.get("/settings")
async def get_settings():
    return config_manager.get_config()

@app.post("/settings")
async def update_settings(config: Dict[str, Any]):
    return config_manager.update_config(config)

# 历史记录
@app.get("/history/{user_id}")
async def get_history(user_id: str):
    if ensure_schema_if_possible():
        store = MySQLConversationStore()
        return {"history": store.list_sessions(user_id)}
    return {"history": history_manager.get_history(user_id)}

@app.post("/history/{user_id}/save")
async def save_history(user_id: str, payload: Dict[str, Any], background_tasks: BackgroundTasks):
    session_id = payload.get("session_id") or str(uuid.uuid4())
    messages = payload.get("messages") or []
    title = payload.get("title")

    if not ensure_schema_if_possible():
        return history_manager.save_session(user_id, session_id, messages, title)

    store = MySQLConversationStore()
    saved = store.save_session(user_id, session_id, messages, title)
    background_tasks.add_task(memory_update_service.update_after_save, user_id, session_id, messages)

    return saved

@app.delete("/history/{user_id}/{session_id}")
async def delete_history(user_id: str, session_id: str):
    if ensure_schema_if_possible():
        store = MySQLConversationStore()
        store.delete_session(user_id, session_id)
    else:
        history_manager.delete_session(user_id, session_id)
    return {"message": "Deleted"}


@app.get("/profile/{user_id}")
async def get_profile(user_id: str):
    if not ensure_schema_if_possible():
        return {"user_id": user_id, "profile": None}
    engine = UserProfileEngine()
    return {"user_id": user_id, "profile": engine.get_profile(user_id)}


@app.post("/vectorstore/docs/clear")
async def clear_docs_vectorstore():
    get_rag_engine().clear()
    return {"message": "cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
