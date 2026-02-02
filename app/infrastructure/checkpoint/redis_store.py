import json
import redis
from typing import Optional, Any, Dict
from datetime import datetime


class CheckpointStore:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _make_key(self, thread_id: str, checkpoint_ns: str = "default") -> str:
        return f"agframe:checkpoint:{checkpoint_ns}:{thread_id}"

    async def save(self, thread_id: str, checkpoint: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> None:
        key = self._make_key(thread_id)
        data = {
            "checkpoint": json.dumps(checkpoint),
            "metadata": json.dumps(metadata or {}),
            "updated_at": datetime.utcnow().isoformat(),
        }
        self.client.hset(key, mapping=data)
        self.client.expire(key, 86400 * 7)

    async def load(self, thread_id: str) -> Optional[Dict[str, Any]]:
        key = self._make_key(thread_id)
        data = self.client.hgetall(key)
        if not data:
            return None
        return {
            "checkpoint": json.loads(data.get("checkpoint", "{}")),
            "metadata": json.loads(data.get("metadata", "{}")),
            "updated_at": data.get("updated_at"),
        }

    async def delete(self, thread_id: str) -> None:
        key = self._make_key(thread_id)
        self.client.delete(key)

    async def list_threads(self, namespace: str = "default") -> list:
        pattern = f"agframe:checkpoint:{namespace}:*"
        keys = self.client.keys(pattern)
        return [key.split(":")[-1] for key in keys]


checkpoint_store = CheckpointStore()


async def get_checkpoint_store() -> CheckpointStore:
    return checkpoint_store
