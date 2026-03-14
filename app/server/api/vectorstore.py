from fastapi import APIRouter

router = APIRouter()


@router.post("/vectorstore/docs/clear")
async def clear_docs_vectorstore():
    from app.skills.rag.rag_engine import get_rag_engine

    get_rag_engine().clear()
    return {"message": "cleared"}
