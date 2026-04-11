import base64

import pytest

from app.infrastructure.utils.secrets import encrypt_secret
from app.runtime.llm.provider_registry import ModelProviderRegistry, RegisteredProvider


def test_registry_fallback_resolution():
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="primary",
            name="Primary",
            base_url="https://primary.example",
            api_key="primary_key",
            models=["gpt-5"],
            is_default=True,
        )
    )
    registry.register(
        RegisteredProvider(
            provider_id="secondary",
            name="Secondary",
            base_url="https://secondary.example",
            api_key="secondary_key",
            models=["gpt-5", "claude-3"],
        )
    )

    res = registry.resolve("gpt-5", preferred_provider_id="primary")
    assert res.provider_id == "primary"

    res2 = registry.resolve("claude-3", preferred_provider_id="primary", fallback_provider_id="secondary")
    assert res2.provider_id == "secondary"

    res3 = registry.resolve("gpt-5")
    assert res3.provider_id == "primary"

def test_registry_system_fallback():
    registry = ModelProviderRegistry()
    registry.register(
        RegisteredProvider(
            provider_id="third",
            name="Third",
            base_url="https://third.example",
            api_key="third_key",
            models=["gpt-4o"],
            is_default=False,
        )
    )
    res = registry.resolve("unknown-model", preferred_provider_id="third")
    assert res.provider_id is None
    assert res.model == "unknown-model"


def test_registry_loads_legacy_base64_provider_secrets():
    registry = ModelProviderRegistry()

    loaded = registry.load_from_store_rows(
        [
            {
                "provider_id": "legacy",
                "name": "Legacy",
                "base_url": "https://legacy.example",
                "api_key_encrypted": base64.b64encode(b"legacy-key").decode("ascii"),
                "models_json": ["gpt-4o"],
                "is_default": True,
                "enabled": True,
            }
        ]
    )

    assert loaded == 1
    assert registry.get_provider("legacy").api_key == "legacy-key"


def test_registry_loads_encrypted_provider_secrets():
    pytest.importorskip("cryptography.fernet")
    registry = ModelProviderRegistry()

    loaded = registry.load_from_store_rows(
        [
            {
                "provider_id": "encrypted",
                "name": "Encrypted",
                "base_url": "https://encrypted.example",
                "api_key_encrypted": encrypt_secret("encrypted-key"),
                "models_json": ["gpt-4o"],
                "is_default": True,
                "enabled": True,
            }
        ]
    )

    assert loaded == 1
    assert registry.get_provider("encrypted").api_key == "encrypted-key"
