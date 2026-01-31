from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional


class ContextLogger(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: Dict[str, Any]):
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
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s trace_id=%(trace_id)s user_id=%(user_id)s session_id=%(session_id)s node=%(node)s %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def bind_logger(
    logger: logging.Logger,
    *,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    node: Optional[str] = None,
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

