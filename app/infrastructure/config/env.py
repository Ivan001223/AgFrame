from __future__ import annotations

from dotenv import load_dotenv

_loaded = False


def init_env() -> None:
    global _loaded
    if _loaded:
        return
    load_dotenv()
    _loaded = True

