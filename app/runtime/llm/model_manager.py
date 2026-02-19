from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoProcessor

from app.runtime.llm.model_importer import resolve_pretrained_source


def get_best_device() -> str:
    """获取当前环境可用的最佳计算设备 (cuda > mps > cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_dtype_for_device(device: str) -> torch.dtype:
    """根据设备类型选择合适的 torch 数据类型"""
    return torch.float32 if device == "cpu" else torch.float16


def get_config_value(config: dict, path: tuple[str, ...]) -> Any:
    """从嵌套字典中获取配置值"""
    cur: Any = config
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def resolve_model_ref(
    *,
    env_var: str,
    config: dict,
    config_path: tuple[str, ...],
    explicit: str | None,
    default: str,
) -> str:
    """
    解析模型引用路径或 ID。
    优先级：环境变量 > 配置文件 > 显式参数 > 默认值
    """
    env_path = os.getenv(env_var)
    if env_path and os.path.exists(env_path):
        return env_path

    config_model = get_config_value(config, config_path)
    if config_model:
        return str(config_model)

    if explicit:
        return explicit

    return default


def resolve_provider(config: dict, component_key: str) -> str:
    """
    解析模型提供方 (provider)。
    优先级：组件配置 > 全局 model_manager 配置 > 默认 'hf'
    """
    component = config.get(component_key) or {}
    provider = component.get("provider")
    if provider:
        return str(provider)
    manager = config.get("model_manager") or {}
    provider = manager.get("provider")
    return str(provider) if provider else "hf"


def resolve_modelscope_cache_dir(config: dict, component_key: str) -> str | None:
    """解析 ModelScope 缓存目录配置"""
    component = config.get(component_key) or {}
    if component.get("cache_dir"):
        return str(component.get("cache_dir"))
    manager = config.get("model_manager") or {}
    if manager.get("cache_dir"):
        return str(manager.get("cache_dir"))
    return None


@dataclass(frozen=True)
class ModelSpec:
    """模型规格描述符，包含加载所需的所有元数据"""
    provider: str
    model_ref: str
    revision: str | None = None
    cache_dir: str | None = None
    trust_remote_code: bool = True
    modelscope_fallback_to_hf: bool = True


def build_model_spec(
    *,
    config: dict,
    component_key: str,
    env_var: str,
    config_path: tuple[str, ...],
    explicit: str | None,
    default: str,
) -> ModelSpec:
    """
    构建统一的模型规格对象 (ModelSpec)。
    整合配置、环境变量和默认值。
    """
    provider = resolve_provider(config, component_key)
    component = config.get(component_key) or {}
    manager = config.get("model_manager") or {}

    model_ref = resolve_model_ref(
        env_var=env_var,
        config=config,
        config_path=config_path,
        explicit=explicit,
        default=default,
    )
    revision = component.get("revision") or manager.get("revision")
    cache_dir = resolve_modelscope_cache_dir(config, component_key)
    trust_remote_code = component.get("trust_remote_code")
    if trust_remote_code is None:
        trust_remote_code = manager.get("trust_remote_code")
    trust_remote_code = True if trust_remote_code is None else bool(trust_remote_code)
    fallback = manager.get("modelscope_fallback_to_hf")
    fallback = True if fallback is None else bool(fallback)

    return ModelSpec(
        provider=str(provider),
        model_ref=str(model_ref),
        revision=str(revision) if revision is not None else None,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        trust_remote_code=trust_remote_code,
        modelscope_fallback_to_hf=fallback,
    )


def load_model_and_processor(
    *,
    spec: ModelSpec,
    device: str,
    model_cls: type[Any] = AutoModel,
    processor_cls: type[Any] = AutoProcessor,
    require_processor: bool = True,
) -> tuple[Any, Any | None]:
    """
    通用模型加载函数。
    加载模型和处理器（Processor），并移动到指定设备。
    """
    imported = resolve_pretrained_source(
        provider=spec.provider,
        model_ref=spec.model_ref,
        cache_dir=spec.cache_dir,
        revision=spec.revision,
        modelscope_fallback_to_hf=spec.modelscope_fallback_to_hf,
    )
    source = imported.pretrained_source
    model = model_cls.from_pretrained(
        source,
        trust_remote_code=spec.trust_remote_code,
        torch_dtype=torch_dtype_for_device(device),
    ).to(device)
    model.eval()

    processor = None
    try:
        processor = processor_cls.from_pretrained(source, trust_remote_code=spec.trust_remote_code)
    except Exception:
        if require_processor:
            raise
        processor = None
    return model, processor
