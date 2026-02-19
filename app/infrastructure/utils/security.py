import warnings
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.infrastructure.config.settings import settings

_default_secret_warning_shown = False


def get_auth_config():
    return settings.auth


def _check_default_secret(secret_key: str):
    """检查是否使用了默认密钥，如果是则发出警告"""
    global _default_secret_warning_shown
    if not _default_secret_warning_shown and secret_key == "secret":
        warnings.warn(
            "WARNING: 使用默认的 JWT secret_key ('secret')！"
            "这在生产环境中非常不安全。请在环境变量或配置文件中设置 AUTH_SECRET_KEY。",
            UserWarning,
            stacklevel=3
        )
        _default_secret_warning_shown = True


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if isinstance(plain_password, str):
        plain_password = plain_password.encode("utf-8")
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode("utf-8")
    return bcrypt.checkpw(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    if isinstance(password, str):
        password = password.encode("utf-8")
    # gensalt() generates a salt and returns bytes. hashpw returns bytes.
    return bcrypt.hashpw(password, bcrypt.gensalt()).decode("utf-8")


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    auth_config = get_auth_config()
    secret_key = auth_config.get("secret_key", "secret")
    _check_default_secret(secret_key)
    algorithm = auth_config.get("algorithm", "HS256")

    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    auth_config = get_auth_config()
    secret_key = auth_config.get("secret_key", "secret")
    algorithm = auth_config.get("algorithm", "HS256")

    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except jwt.PyJWTError:
        return None
