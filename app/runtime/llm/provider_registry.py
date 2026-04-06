from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any

from app.infrastructure.config.settings import settings

_log = logging.getLogger("runtime.llm.provider_registry")


@dataclass(frozen=True)
class ResolvedModel:
    """解析后的模型连接参数"""
    model: str
    base_url: str
    api_key: str
    provider_id: str | None = None


@dataclass(frozen=True)
class RegisteredProvider:
    """注册的模型提供商"""
    provider_id: str
    name: str
    base_url: str
    api_key: str
    models: list[str]
    is_default: bool = False
    enabled: bool = True


def _normalize_model_name(model: str) -> str:
    return str(model or "").strip().lower()


def _normalize_base_url(base_url: str) -> str:
    return str(base_url or "").strip().lower().rstrip("/")


def infer_tool_calling_support(
    *,
    model: str,
    base_url: str = "",
    provider_id: str | None = None,
) -> tuple[str, str]:
    normalized_model = _normalize_model_name(model)
    normalized_base_url = _normalize_base_url(base_url)
    normalized_provider = str(provider_id or "").strip().lower()

    if normalized_model == "local-qwen3-vl":
        return "unsupported", "The local Qwen runtime does not support native tool calling."

    if normalized_model in {"dev-stub", "dev_stub"}:
        return "unknown", "The dev-stub adapter is test-oriented, so tool calling depends on the injected runtime."

    if normalized_model.startswith(("gpt-", "o1", "o3", "o4")):
        return "supported", "This model family supports OpenAI-style tool calling."

    if normalized_base_url and any(
        marker in normalized_base_url for marker in ("api.openai.com", "openai.azure.com", "/openai/", "/openai/v1")
    ):
        return "supported", "This provider route appears to be OpenAI-compatible for tool calling."

    if normalized_provider.startswith("openai"):
        return "supported", "This provider route is labeled as OpenAI-compatible for tool calling."

    return "unknown", "Tool calling support is not verified for this provider route yet."


def _obfuscate_key(api_key: str) -> str:
    """简单 base64 混淆（非加密），用于 DB 存储。生产环境应替换为 KMS。"""
    if not api_key:
        return ""
    return base64.b64encode(api_key.encode("utf-8")).decode("ascii")


def _deobfuscate_key(stored: str) -> str:
    """反混淆 base64"""
    if not stored:
        return ""
    try:
        return base64.b64decode(stored.encode("ascii")).decode("utf-8")
    except Exception:
        return stored


class ModelProviderRegistry:
    """
    内存注册表，管理多个 LLM 提供商。
    支持从 DB store 加载，也支持手动注册。
    """

    def __init__(self) -> None:
        self._providers: dict[str, RegisteredProvider] = {}

    def register(self, provider: RegisteredProvider) -> None:
        self._providers[provider.provider_id] = provider

    def unregister(self, provider_id: str) -> bool:
        return self._providers.pop(provider_id, None) is not None

    def list_providers(self) -> list[RegisteredProvider]:
        return [p for p in self._providers.values() if p.enabled]

    def get_provider(self, provider_id: str) -> RegisteredProvider | None:
        provider = self._providers.get(provider_id)
        if provider is not None and provider.enabled:
            return provider
        return None

    def get_default(self) -> RegisteredProvider | None:
        for provider in self._providers.values():
            if provider.is_default and provider.enabled:
                return provider
        return None

    def _try_provider(self, provider: RegisteredProvider, model: str) -> ResolvedModel | None:
        if not provider.enabled:
            return None
        if provider.models and model not in provider.models:
            return None
        return ResolvedModel(
            model=model,
            base_url=provider.base_url,
            api_key=provider.api_key,
            provider_id=provider.provider_id,
        )

    def resolve(
        self,
        model: str,
        *,
        preferred_provider_id: str | None = None,
        fallback_provider_id: str | None = None,
    ) -> ResolvedModel:
        """
        解析模型到实际连接参数。
        优先级：preferred → fallback → default provider → settings.llm
        """
        if preferred_provider_id:
            provider = self.get_provider(preferred_provider_id)
            if provider is not None:
                resolved = self._try_provider(provider, model)
                if resolved is not None:
                    return resolved
                _log.info(
                    "preferred provider %s does not support model %s, trying fallback",
                    preferred_provider_id, model,
                )

        if fallback_provider_id:
            provider = self.get_provider(fallback_provider_id)
            if provider is not None:
                resolved = self._try_provider(provider, model)
                if resolved is not None:
                    _log.info(
                        "resolved model %s via fallback provider %s",
                        model, fallback_provider_id,
                    )
                    return resolved
                _log.info(
                    "fallback provider %s does not support model %s, trying default",
                    fallback_provider_id, model,
                )

        default_provider = self.get_default()
        if default_provider is not None:
            resolved = self._try_provider(default_provider, model)
            if resolved is not None:
                return resolved
            if default_provider.enabled:
                return ResolvedModel(
                    model=model,
                    base_url=default_provider.base_url,
                    api_key=default_provider.api_key,
                    provider_id=default_provider.provider_id,
                )

        llm_config = settings.llm
        return ResolvedModel(
            model=model,
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
            provider_id=None,
        )

    def load_from_store_rows(self, rows: list[dict[str, Any]]) -> int:
        """从 DB store 的行列表加载 providers 到内存。返回加载数量。"""
        count = 0
        for row in rows:
            provider = RegisteredProvider(
                provider_id=str(row.get("provider_id") or ""),
                name=str(row.get("name") or ""),
                base_url=str(row.get("base_url") or ""),
                api_key=_deobfuscate_key(str(row.get("api_key_encrypted") or "")),
                models=list(row.get("models_json") or []),
                is_default=bool(row.get("is_default", False)),
                enabled=bool(row.get("enabled", True)),
            )
            if provider.provider_id:
                self.register(provider)
                count += 1
        return count


_global_registry: ModelProviderRegistry | None = None


def get_provider_registry() -> ModelProviderRegistry:
    """获取全局 provider registry 单例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ModelProviderRegistry()
    return _global_registry


def reset_provider_registry() -> None:
    """重置全局 registry（测试用）"""
    global _global_registry
    _global_registry = None
