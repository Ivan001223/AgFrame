from __future__ import annotations

import logging
import os
from collections.abc import MutableMapping
from typing import Any


class DefaultContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for field in ("trace_id", "user_id", "session_id", "node"):
            if not hasattr(record, field):
                setattr(record, field, "-")
        return True


class ContextLogger(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: MutableMapping[str, Any]):
        extra = dict(self.extra or {})
        extra.update(kwargs.get("extra") or {})
        kwargs["extra"] = extra
        return msg, kwargs


def init_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        for handler in root.handlers:
            handler.addFilter(DefaultContextFilter())
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s trace_id=%(trace_id)s user_id=%(user_id)s session_id=%(session_id)s node=%(node)s %(message)s",
    )
    for handler in root.handlers:
        handler.addFilter(DefaultContextFilter())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def bind_logger(
    logger: logging.Logger,
    *,
    trace_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    node: str | None = None,
) -> ContextLogger:
    return ContextLogger(
        logger,
        {
            "trace_id": trace_id or "-",
            "user_id": user_id or "-",
            "session_id": session_id or "-",
            "node": node or "-",
        },
    )
