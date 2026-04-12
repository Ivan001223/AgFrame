from __future__ import annotations

from functools import lru_cache

from app.infrastructure.checkpoint.redis_store import checkpoint_store
from app.runtime.graph.graph import run_app


@lru_cache(maxsize=1)
def get_chat_graph_app():
    return run_app(checkpointer=checkpoint_store)
