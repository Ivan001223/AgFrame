import os
from typing import Optional, Any

from langchain_community.vectorstores import FAISS


def load_faiss(
    persist_directory: str,
    embeddings: Any,
    *,
    allow_dangerous_deserialization: bool = True,
) -> Optional[FAISS]:
    if not (
        os.path.exists(persist_directory)
        and os.path.exists(os.path.join(persist_directory, "index.faiss"))
    ):
        return None
    try:
        return FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
        )
    except Exception:
        return None


def save_faiss(persist_directory: str, store: Optional[FAISS]) -> bool:
    if store is None:
        return False
    try:
        os.makedirs(persist_directory, exist_ok=True)
        store.save_local(persist_directory)
        return True
    except Exception:
        return False

