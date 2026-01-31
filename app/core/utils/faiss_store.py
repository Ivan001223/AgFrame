import os
from typing import Optional, Any

from langchain_community.vectorstores import FAISS


def load_faiss(
    persist_directory: str,
    embeddings: Any,
    *,
    allow_dangerous_deserialization: bool = True,
) -> Optional[FAISS]:
    """
    加载本地持久化的 FAISS 索引。
    
    Args:
        persist_directory: 索引存储目录
        embeddings: Embeddings 模型实例
        allow_dangerous_deserialization: 是否允许反序列化 (默认 True)
        
    Returns:
        Optional[FAISS]: 加载成功的 FAISS 实例，失败返回 None
    """
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
    """
    将 FAISS 索引保存到本地。
    
    Args:
        persist_directory: 保存目录
        store: FAISS 实例
        
    Returns:
        bool: 保存是否成功
    """
    if store is None:
        return False
    try:
        os.makedirs(persist_directory, exist_ok=True)
        store.save_local(persist_directory)
        return True
    except Exception:
        return False

