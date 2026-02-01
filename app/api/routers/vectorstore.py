from fastapi import APIRouter
from app.core.services.rag_engine import get_rag_engine

router = APIRouter()


@router.post("/vectorstore/docs/clear")
async def clear_docs_vectorstore():
    get_rag_engine().clear()
    return {"message": "cleared"}
