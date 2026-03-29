import pytest
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
