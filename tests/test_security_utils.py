from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pytest

from app.infrastructure.utils import security


@dataclass(frozen=True)
class _AuthCfg:
    secret_key: str = "test_secret_key_012345678901234567890123456789"
    algorithm: str = "HS256"


def test_password_hash_roundtrip():
    hashed = security.get_password_hash("p@ssw0rd")
    assert isinstance(hashed, str)
    assert security.verify_password("p@ssw0rd", hashed) is True
    assert security.verify_password("wrong", hashed) is False


def test_get_auth_config_smoke():
    cfg = security.get_auth_config()
    assert hasattr(cfg, "secret_key")
    assert hasattr(cfg, "algorithm")


def test_access_token_roundtrip(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(security, "get_auth_config", lambda: _AuthCfg())
    token = security.create_access_token({"sub": "u1", "role": "user"})
    payload = security.decode_access_token(token)
    assert payload is not None
    assert payload["sub"] == "u1"
    assert payload["role"] == "user"


def test_access_token_expired_returns_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(security, "get_auth_config", lambda: _AuthCfg())
    token = security.create_access_token({"sub": "u1"}, expires_delta=timedelta(minutes=-1))
    assert security.decode_access_token(token) is None


def test_default_secret_emits_warning_once(monkeypatch: pytest.MonkeyPatch):
    warnings_called: list[str] = []

    def _warn(msg: str, *args, **kwargs):
        s = str(msg)
        if "使用默认的 JWT secret_key" in s:
            warnings_called.append(s)

    monkeypatch.setattr(security, "get_auth_config", lambda: _AuthCfg(secret_key="secret"))
    monkeypatch.setattr(security.warnings, "warn", _warn)
    security.create_access_token({"sub": "u1"})
    security.create_access_token({"sub": "u1"})
    assert len(warnings_called) == 1
