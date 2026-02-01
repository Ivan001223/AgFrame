from datetime import datetime, timedelta, timezone
from typing import Optional, Union, Any
import bcrypt
import jwt
from app.infrastructure.config.config_manager import config_manager


def get_auth_config():
    return config_manager.get_config().get("auth", {})


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


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    auth_config = get_auth_config()
    secret_key = auth_config.get("secret_key", "secret")
    algorithm = auth_config.get("algorithm", "HS256")

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)

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
