
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.infrastructure.config.settings import settings


def _get_redis_url() -> str:
    queue_cfg = settings.queue
    url = queue_cfg.redis_url or "redis://localhost:6379/0"
    return str(url)


class AsyncRedisSaverWrapper(BaseCheckpointSaver):
    def __init__(self):
        self._saver: Any = None

    async def get_saver(self):
        if self._saver is None:
            from langgraph.checkpoint.redis import AsyncRedisSaver

            self._saver = AsyncRedisSaver(redis_url=_get_redis_url())
            await self._saver.setup()
        return self._saver

    async def aget_tuple(self, config):
        saver = await self.get_saver()
        return await saver.aget_tuple(config)

    async def aput(self, config, checkpoint, metadata, new_version):
        saver = await self.get_saver()
        return await saver.aput(config, checkpoint, metadata, new_version)

    async def aput_writes(self, config, writes, task_id, task_path=''):
        saver = await self.get_saver()
        return await saver.aput_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id):
        saver = await self.get_saver()
        return await saver.adelete_thread(thread_id)

    async def alist(self, config, limit, before, filter=None):
        saver = await self.get_saver()
        return await saver.alist(config, limit, before, filter)


checkpoint_store = AsyncRedisSaverWrapper()
