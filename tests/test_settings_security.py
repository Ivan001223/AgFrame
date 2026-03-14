from __future__ import annotations

import pytest

from app.infrastructure.config.settings import AuthConfig, DatabaseConfig, LLMConfig, Settings


def test_validate_security_raises_on_insecure_defaults():
    s = Settings()
    with pytest.raises(ValueError):
        s.validate_security()


def test_validate_security_passes_with_safe_values():
    s = Settings(
        auth=AuthConfig(AUTH_SECRET_KEY="x" * 40),
        database=DatabaseConfig(DB_PASSWORD="StrongPassw0rd!"),
        llm=LLMConfig(LLM_API_KEY="k" * 40),
    )
    s.validate_security()
