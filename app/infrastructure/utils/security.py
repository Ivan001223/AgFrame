import warnings
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.infrastructure.config.settings import AuthConfig, settings

_default_secret_warning_shown = False
_DEFAULT_AUTH_SECRET = AuthConfig().secret_key


def get_auth_config() -> AuthConfig:
    return settings.auth


def _check_default_secret(secret_key: str) -> None:
    """Warn once when the default JWT secret is still in use."""
    global _default_secret_warning_shown
    if not _default_secret_warning_shown and secret_key == _DEFAULT_AUTH_SECRET:
        warnings.warn(
            "WARNING: The application is still using the default JWT secret key. "
            "Set AUTH_SECRET_KEY in your environment or config before production use.",
            UserWarning,
            stacklevel=3,
        )
        _default_secret_warning_shown = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    plain_password_bytes = plain_password.encode("utf-8")
    hashed_password_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(plain_password_bytes, hashed_password_bytes)


def get_password_hash(password: str) -> str:
    password_bytes = password.encode("utf-8")
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    auth_config = get_auth_config()
    secret_key = getattr(auth_config, "secret_key", _DEFAULT_AUTH_SECRET)
    _check_default_secret(secret_key)
    algorithm = getattr(auth_config, "algorithm", "HS256")

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


def decode_access_token(token: str) -> dict | None:
    auth_config = get_auth_config()
    secret_key = getattr(auth_config, "secret_key", _DEFAULT_AUTH_SECRET)
    algorithm = getattr(auth_config, "algorithm", "HS256")

    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except jwt.PyJWTError:
        return None
