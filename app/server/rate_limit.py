from __future__ import annotations

from typing import Any

from fastapi_limiter.depends import RateLimiter as _PackageRateLimiter

from app.infrastructure.utils.logging import get_logger

log = get_logger("server.rate_limit")

_legacy_fastapi_limiter: Any | None = None
try:
    from fastapi_limiter import FastAPILimiter as _ImportedFastAPILimiter
except ImportError:
    _legacy_fastapi_limiter = None
else:
    _legacy_fastapi_limiter = _ImportedFastAPILimiter


async def init_rate_limiter(redis: Any) -> None:
    """
    Initialize rate limiting when the installed fastapi-limiter version requires it.

    fastapi-limiter 0.1.x exposed FastAPILimiter.init(redis), while 0.2.x moved to
    per-route pyrate-limiter instances and no global init step.
    """
    if _legacy_fastapi_limiter is not None:
        await _legacy_fastapi_limiter.init(redis)
        return
    log.warning("fastapi-limiter legacy initializer not available; using local limiter compatibility mode")


def build_rate_limiter(*, times: int, seconds: int):
    """
    Build a RateLimiter dependency that works across fastapi-limiter API versions.
    """
    if _legacy_fastapi_limiter is not None:
        return _PackageRateLimiter(times=times, seconds=seconds)

    from pyrate_limiter import Duration, Limiter, Rate  # type: ignore[import-not-found]

    return _PackageRateLimiter(  # type: ignore[call-arg]
        limiter=Limiter(Rate(times, Duration.SECOND * seconds))
    )
