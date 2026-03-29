from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver, empty_checkpoint

from app.infrastructure.config.settings import settings

_DEFAULT_CHECKPOINT_NS = ""
_LEGACY_CHECKPOINT_NS = "default"
_COMPAT_CHECKPOINT_KEYS = ("interrupted", "action_required")
_REQUIRED_CHECKPOINT_KEYS = (
    "id",
    "ts",
    "channel_values",
    "channel_versions",
    "versions_seen",
)


def _get_redis_url() -> str:
    queue_cfg = settings.queue
    url = queue_cfg.redis_url or "redis://:redissecret@localhost:6379/0"
    return str(url)



def _checkpoint_config(
    session_id: str,
    checkpoint_ns: str = _DEFAULT_CHECKPOINT_NS,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    configurable: dict[str, Any] = {
        "thread_id": session_id,
        "checkpoint_ns": checkpoint_ns,
    }
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}



def _checkpoint_namespaces(checkpoint_ns: str) -> tuple[str, ...]:
    if checkpoint_ns == _DEFAULT_CHECKPOINT_NS:
        return (_DEFAULT_CHECKPOINT_NS, _LEGACY_CHECKPOINT_NS)
    return (checkpoint_ns,)



def _looks_like_checkpoint(checkpoint: dict[str, Any]) -> bool:
    return all(key in checkpoint for key in _REQUIRED_CHECKPOINT_KEYS)



def _extract_compat_value(checkpoint: dict[str, Any], key: str) -> tuple[bool, Any]:
    if key in checkpoint:
        return True, checkpoint[key]

    channel_values = checkpoint.get("channel_values")
    if isinstance(channel_values, dict) and key in channel_values:
        return True, channel_values[key]

    return False, None



def _adapt_checkpoint_for_interrupt(
    checkpoint: dict[str, Any],
    checkpoint_id: str | None,
    checkpoint_ns: str,
) -> dict[str, Any]:
    checkpoint_view = deepcopy(checkpoint)
    channel_values = checkpoint_view.get("channel_values")
    if not isinstance(channel_values, dict):
        channel_values = {}
        checkpoint_view["channel_values"] = channel_values

    for key in _COMPAT_CHECKPOINT_KEYS:
        if key in channel_values:
            checkpoint_view[key] = channel_values[key]
        elif key in checkpoint_view:
            channel_values[key] = checkpoint_view[key]
            checkpoint_view[key] = channel_values[key]

    checkpoint_view["checkpoint_id"] = checkpoint_id or checkpoint_view.get("id")
    checkpoint_view["checkpoint_ns"] = checkpoint_ns
    return checkpoint_view


class AsyncRedisSaverWrapper(BaseCheckpointSaver):
    def __init__(self):
        self._saver: Any = None

    def get_next_version(self, current: Any, channel: Any) -> Any:
        saver = self._saver
        getter = getattr(saver, "get_next_version", None) if saver is not None else None
        if callable(getter):
            return getter(current, channel)

        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        return f"{current_v + 1:032}.0000000000000000"

    async def get_saver(self):
        if self._saver is None:
            from langgraph.checkpoint.redis import AsyncRedisSaver

            self._saver = AsyncRedisSaver(redis_url=_get_redis_url())
            setup = getattr(self._saver, "setup", None)
            if callable(setup):
                await setup()
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

    async def load(self, session_id: str) -> dict[str, Any] | None:
        saver = await self.get_saver()
        checkpoint_tuple = None
        for checkpoint_ns in _checkpoint_namespaces(_DEFAULT_CHECKPOINT_NS):
            checkpoint_tuple = await saver.aget_tuple(_checkpoint_config(session_id, checkpoint_ns))
            if checkpoint_tuple is not None:
                break
        if checkpoint_tuple is None:
            return None

        checkpoint_config = checkpoint_tuple.config.get("configurable", {})
        checkpoint = _adapt_checkpoint_for_interrupt(
            checkpoint_tuple.checkpoint,
            checkpoint_config.get("checkpoint_id"),
            checkpoint_config.get("checkpoint_ns", _DEFAULT_CHECKPOINT_NS),
        )

        updated_at = checkpoint_tuple.metadata.get("saved_at") or checkpoint.get("ts")
        if not updated_at:
            updated_at = datetime.now(timezone.utc).isoformat()

        return {
            "checkpoint": checkpoint,
            "updated_at": updated_at,
        }

    async def save(self, session_id: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
        saver = await self.get_saver()
        request_checkpoint = deepcopy(checkpoint)
        checkpoint_ns_value = request_checkpoint.get("checkpoint_ns")
        requested_checkpoint_ns = (
            checkpoint_ns_value if isinstance(checkpoint_ns_value, str) else _DEFAULT_CHECKPOINT_NS
        )

        existing_tuple = None
        checkpoint_ns = requested_checkpoint_ns
        for candidate_ns in _checkpoint_namespaces(requested_checkpoint_ns):
            existing_tuple = await saver.aget_tuple(_checkpoint_config(session_id, candidate_ns))
            if existing_tuple is not None:
                checkpoint_ns = candidate_ns
                break

        existing_checkpoint = deepcopy(existing_tuple.checkpoint) if existing_tuple else None
        existing_checkpoint_id = (
            existing_tuple.config.get("configurable", {}).get("checkpoint_id")
            if existing_tuple
            else None
        )

        input_is_checkpoint = _looks_like_checkpoint(request_checkpoint)
        if existing_checkpoint is not None:
            checkpoint_payload = deepcopy(existing_checkpoint)
        elif input_is_checkpoint:
            checkpoint_payload = deepcopy(request_checkpoint)
        else:
            checkpoint_payload = empty_checkpoint()

        if not isinstance(checkpoint_payload.get("channel_values"), dict):
            checkpoint_payload["channel_values"] = {}
        if not isinstance(checkpoint_payload.get("channel_versions"), dict):
            checkpoint_payload["channel_versions"] = {}
        if not isinstance(checkpoint_payload.get("versions_seen"), dict):
            checkpoint_payload["versions_seen"] = {}

        channel_values = checkpoint_payload["channel_values"]
        channel_versions = checkpoint_payload["channel_versions"]
        baseline_channel_values = (
            existing_checkpoint.get("channel_values", {})
            if isinstance(existing_checkpoint, dict)
            and isinstance(existing_checkpoint.get("channel_values"), dict)
            else {}
        )

        version_getter = getattr(saver, "get_next_version", self.get_next_version)
        new_versions: dict[str, Any] = {}
        updated_channels: list[str] = []

        if existing_tuple is None and input_is_checkpoint:
            for key, value in channel_values.items():
                if key not in channel_versions:
                    channel_versions[key] = version_getter(None, None)
                new_versions[key] = channel_versions[key]
                updated_channels.append(key)

        for key in _COMPAT_CHECKPOINT_KEYS:
            has_value, value = _extract_compat_value(request_checkpoint, key)
            if not has_value:
                continue

            previous_value = baseline_channel_values.get(key)
            if value is None:
                if key in channel_values:
                    channel_values.pop(key, None)
                    next_version = version_getter(channel_versions.get(key), None)
                    channel_versions[key] = next_version
                    new_versions[key] = next_version
                    if key not in updated_channels:
                        updated_channels.append(key)
                continue

            channel_values[key] = deepcopy(value)
            if key not in baseline_channel_values or previous_value != value:
                next_version = version_getter(channel_versions.get(key), None)
                channel_versions[key] = next_version
                new_versions[key] = next_version
                if key not in updated_channels:
                    updated_channels.append(key)

        checkpoint_payload["updated_channels"] = updated_channels or None

        parent_checkpoint_id = (
            request_checkpoint.get("checkpoint_id")
            or existing_checkpoint_id
            or request_checkpoint.get("id")
        )

        fresh_checkpoint = empty_checkpoint()
        checkpoint_payload["id"] = str(fresh_checkpoint["id"])
        checkpoint_payload["ts"] = str(fresh_checkpoint["ts"])
        if "v" not in checkpoint_payload and existing_checkpoint is not None:
            checkpoint_payload["v"] = existing_checkpoint.get("v", fresh_checkpoint["v"])

        for key in (*_COMPAT_CHECKPOINT_KEYS, "checkpoint_id", "checkpoint_ns"):
            checkpoint_payload.pop(key, None)

        config = _checkpoint_config(
            session_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=parent_checkpoint_id,
        )
        saved_at = datetime.now(timezone.utc).isoformat()
        metadata = deepcopy(existing_tuple.metadata) if existing_tuple else {}
        metadata["source"] = "update"
        if "step" not in metadata:
            metadata["step"] = 0
        metadata["saved_at"] = saved_at

        next_config = await saver.aput(config, checkpoint_payload, metadata, new_versions)
        saved_checkpoint_id = next_config["configurable"].get(
            "checkpoint_id",
            checkpoint_payload["id"],
        )
        return {
            "session_id": session_id,
            "checkpoint_id": saved_checkpoint_id,
            "updated_at": saved_at,
        }


checkpoint_store = AsyncRedisSaverWrapper()
