import os
import logging
from typing import Optional

try:
    from langfuse.callback import CallbackHandler
    from langfuse import Langfuse
except ImportError:
    CallbackHandler = None
    Langfuse = None

logger = logging.getLogger(__name__)

def get_langfuse_callback() -> Optional[object]:
    """
    Returns a LangfuseCallbackHandler if credentials are set.
    Returns None otherwise to avoid crashing.
    """
    if not CallbackHandler:
        logger.warning("Langfuse not installed.")
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com") # Default to Cloud

    if public_key and secret_key:
        try:
            return CallbackHandler(
                public_key=public_key,
                secret_key=secret_key,
                host=host
            )
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse callback: {e}")
            return None
    else:
        logger.info("Langfuse credentials not found. Tracing disabled.")
        return None
