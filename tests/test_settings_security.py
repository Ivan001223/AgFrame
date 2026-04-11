from __future__ import annotations

import warnings

import pytest

from app.infrastructure.config.settings import AuthConfig, DatabaseConfig, LLMConfig, Settings


def test_validate_security_raises_on_insecure_defaults():
    s = Settings()
    with pytest.raises(ValueError, match="auth.secret_key uses an insecure default"):
        s.validate_security()


def test_validate_security_passes_with_safe_values():
    s = Settings(
        auth=AuthConfig(AUTH_SECRET_KEY="x" * 40),
        database=DatabaseConfig(DB_PASSWORD="StrongPassw0rd!"),
        llm=LLMConfig(LLM_API_KEY="k" * 40),
    )
    s.validate_security()


def test_validate_security_uses_database_url_password_when_present():
    s = Settings(
        auth=AuthConfig(AUTH_SECRET_KEY="x" * 40),
        database=DatabaseConfig(
            DATABASE_URL="postgresql+psycopg://agframe:StrongPassw0rd%21@127.0.0.1:5432/agframe",
            DB_PASSWORD="password",
        ),
        llm=LLMConfig(LLM_API_KEY="k" * 40),
    )
    s.validate_security()


def test_validate_security_rejects_insecure_database_url_password():
    s = Settings(
        auth=AuthConfig(AUTH_SECRET_KEY="x" * 40),
        database=DatabaseConfig(
            DATABASE_URL="postgresql+psycopg://agframe:password@127.0.0.1:5432/agframe",
            DB_PASSWORD="StrongPassw0rd!",
        ),
        llm=LLMConfig(LLM_API_KEY="k" * 40),
    )
    with pytest.raises(ValueError, match="database.password uses an insecure default"):
        s.validate_security()


def test_validate_security_skips_llm_key_warning_for_dev_stub():
    s = Settings(
        auth=AuthConfig(AUTH_SECRET_KEY="x" * 40),
        database=DatabaseConfig(DB_PASSWORD="StrongPassw0rd!"),
        llm=LLMConfig(LLM_MODEL="dev-stub", LLM_API_KEY=""),
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        s.validate_security()
    assert not record


def test_validate_security_warns_for_cloud_model_without_api_key():
    s = Settings(
        auth=AuthConfig(AUTH_SECRET_KEY="x" * 40),
        database=DatabaseConfig(DB_PASSWORD="StrongPassw0rd!"),
        llm=LLMConfig(LLM_MODEL="gpt-4o-mini", LLM_API_KEY=""),
    )
    with pytest.warns(UserWarning, match="llm.api_key is not configured"):
        s.validate_security()
