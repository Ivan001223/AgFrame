from __future__ import annotations

import base64
from hashlib import sha256

from cryptography.fernet import Fernet, InvalidToken

from app.infrastructure.config.settings import settings

_FERNET_PREFIX = "fernet:"


def _build_fernet() -> Fernet:
    secret_key = str(settings.auth.secret_key or "").strip()
    if not secret_key:
        raise RuntimeError("AUTH_SECRET_KEY must be configured before encrypting provider secrets.")
    derived_key = base64.urlsafe_b64encode(sha256(secret_key.encode("utf-8")).digest())
    return Fernet(derived_key)


def encrypt_secret(secret: str) -> str:
    if not secret:
        return ""
    encrypted = _build_fernet().encrypt(secret.encode("utf-8")).decode("ascii")
    return f"{_FERNET_PREFIX}{encrypted}"


def _decode_legacy_secret(stored: str) -> str:
    try:
        return base64.b64decode(stored.encode("ascii")).decode("utf-8")
    except Exception:
        return stored


def decrypt_secret(stored: str) -> str:
    normalized = str(stored or "").strip()
    if not normalized:
        return ""
    if not normalized.startswith(_FERNET_PREFIX):
        return _decode_legacy_secret(normalized)
    token = normalized[len(_FERNET_PREFIX) :]
    try:
        return _build_fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("Stored secret could not be decrypted.") from exc
