import importlib
import logging
import os
from typing import Any, cast

_callback_handler_type: Any | None = None
try:
    _langfuse_callback_module = importlib.import_module("langfuse.callback")
except ImportError:
    _callback_handler_type = None
else:
    _callback_handler_type = getattr(_langfuse_callback_module, "CallbackHandler", None)

logger = logging.getLogger(__name__)


def get_langfuse_callback() -> object | None:
    """
    Returns a LangfuseCallbackHandler if credentials are set.
    Returns None otherwise to avoid crashing.
    """
    if not _callback_handler_type:
        logger.warning("Langfuse not installed.")
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com") # Default to Cloud

    if public_key and secret_key:
        try:
            return cast(
                object,
                _callback_handler_type(
                public_key=public_key,
                secret_key=secret_key,
                host=host
                ),
            )
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse callback: {e}")
            return None
    else:
        logger.info("Langfuse credentials not found. Tracing disabled.")
        return None
