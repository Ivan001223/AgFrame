from __future__ import annotations

from app.infrastructure.checkpoint.redis_store import checkpoint_store


class CheckpointAdapter:
    async def load(self, session_id: str):
        return await checkpoint_store.load(session_id)

    async def save(self, session_id: str, checkpoint: dict[str, object]):
        await checkpoint_store.save(session_id, checkpoint)
