from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_model_manager(monkeypatch):
    module_path = Path(__file__).resolve().parent.parent / "app" / "runtime" / "llm" / "model_manager.py"
    spec = importlib.util.spec_from_file_location("model_manager_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None

    torch_stub = ModuleType("torch")
    torch_stub.float16 = "float16"
    torch_stub.float32 = "float32"
    torch_stub.dtype = object
    torch_stub.cuda = ModuleType("torch.cuda")
    torch_stub.cuda.is_available = lambda: False
    torch_stub.backends = ModuleType("torch.backends")
    torch_stub.backends.mps = ModuleType("torch.backends.mps")
    torch_stub.backends.mps.is_available = lambda: False

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("not used in tests")

    class _AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("not used in tests")

    transformers_stub = ModuleType("transformers")
    transformers_stub.AutoModel = _AutoModel
    transformers_stub.AutoProcessor = _AutoProcessor

    importer_stub = ModuleType("app.runtime.llm.model_importer")
    importer_stub.resolve_pretrained_source = lambda **kwargs: None

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setitem(sys.modules, "app.runtime.llm.model_importer", importer_stub)

    module = importlib.util.module_from_spec(spec)
    sys.modules["model_manager_test_module"] = module
    spec.loader.exec_module(module)
    return module


def test_model_manager_resolve_model_ref_and_dtype(monkeypatch, tmp_path: Path):
    model_manager = _load_model_manager(monkeypatch)
    model_file = tmp_path / "model.bin"
    model_file.write_text("ok", encoding="utf-8")

    monkeypatch.setenv("TEST_MODEL_PATH", str(model_file))
    from_env = model_manager.resolve_model_ref(
        env_var="TEST_MODEL_PATH",
        config={"llm": {"model": "cfg-model"}},
        config_path=("llm", "model"),
        explicit="explicit-model",
        default="default-model",
    )
    assert from_env == str(model_file)

    monkeypatch.delenv("TEST_MODEL_PATH")
    from_config = model_manager.resolve_model_ref(
        env_var="TEST_MODEL_PATH",
        config={"llm": {"model": "cfg-model"}},
        config_path=("llm", "model"),
        explicit="explicit-model",
        default="default-model",
    )
    assert from_config == "cfg-model"

    fallback = model_manager.resolve_model_ref(
        env_var="TEST_MODEL_PATH",
        config={},
        config_path=("llm", "model"),
        explicit=None,
        default="default-model",
    )
    assert fallback == "default-model"
    assert model_manager.torch_dtype_for_device("cpu") == "float32"
    assert model_manager.torch_dtype_for_device("cuda") == "float16"


def test_model_manager_build_model_spec(monkeypatch):
    model_manager = _load_model_manager(monkeypatch)
    config = {
        "embedder": {
            "provider": "modelscope",
            "model": "embed-model",
            "revision": "v1",
            "cache_dir": "/tmp/embed",
            "trust_remote_code": False,
        },
        "model_manager": {
            "provider": "hf",
            "revision": "global-rev",
            "cache_dir": "/tmp/global",
            "modelscope_fallback_to_hf": False,
        },
    }
    spec = model_manager.build_model_spec(
        config=config,
        component_key="embedder",
        env_var="EMBED_MODEL_PATH",
        config_path=("embedder", "model"),
        explicit=None,
        default="default-model",
    )
    assert spec.provider == "modelscope"
    assert spec.model_ref == "embed-model"
    assert spec.revision == "v1"
    assert spec.cache_dir == "/tmp/embed"
    assert spec.trust_remote_code is False
    assert spec.modelscope_fallback_to_hf is False


def test_model_manager_provider_and_cache_fallbacks(monkeypatch):
    model_manager = _load_model_manager(monkeypatch)

    assert model_manager.resolve_provider({}, "embedder") == "hf"
    assert (
        model_manager.resolve_provider(
            {"model_manager": {"provider": "modelscope"}},
            "embedder",
        )
        == "modelscope"
    )
    assert model_manager.resolve_modelscope_cache_dir({}, "embedder") is None
    assert (
        model_manager.resolve_modelscope_cache_dir(
            {"model_manager": {"cache_dir": "/tmp/global-cache"}},
            "embedder",
        )
        == "/tmp/global-cache"
    )

    explicit = model_manager.resolve_model_ref(
        env_var="MISSING_MODEL_PATH",
        config={},
        config_path=("embedder", "model"),
        explicit="explicit-model",
        default="default-model",
    )
    assert explicit == "explicit-model"


def test_model_manager_device_selection(monkeypatch):
    model_manager = _load_model_manager(monkeypatch)

    monkeypatch.setattr(model_manager.torch.cuda, "is_available", lambda: True)
    assert model_manager.get_best_device() == "cuda"

    monkeypatch.setattr(model_manager.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(model_manager.torch.backends.mps, "is_available", lambda: True)
    assert model_manager.get_best_device() == "mps"

    monkeypatch.setattr(model_manager.torch.backends.mps, "is_available", lambda: False)
    assert model_manager.get_best_device() == "cpu"


def test_model_manager_load_model_and_processor_success(monkeypatch):
    model_manager = _load_model_manager(monkeypatch)

    class _Loaded:
        def __init__(self):
            self.device = None
            self.eval_called = False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True

    loaded = _Loaded()

    class _ModelCls:
        calls = []

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            return loaded

    class _ProcessorCls:
        calls = []

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            return {"source": source, "kwargs": kwargs}

    monkeypatch.setattr(
        model_manager,
        "resolve_pretrained_source",
        lambda **kwargs: type("Imported", (), {"pretrained_source": "/tmp/model"})(),
    )

    spec = model_manager.ModelSpec(
        provider="hf",
        model_ref="repo/model",
        revision="rev-123",
        cache_dir="/tmp/cache",
        trust_remote_code=False,
        modelscope_fallback_to_hf=False,
    )
    model, processor = model_manager.load_model_and_processor(
        spec=spec,
        device="cpu",
        model_cls=_ModelCls,
        processor_cls=_ProcessorCls,
    )

    assert model is loaded
    assert loaded.device == "cpu"
    assert loaded.eval_called is True
    assert processor["source"] == "/tmp/model"
    assert _ModelCls.calls[0][0] == "/tmp/model"
    assert _ModelCls.calls[0][1]["trust_remote_code"] is False


def test_model_manager_load_model_and_processor_processor_failure(monkeypatch):
    model_manager = _load_model_manager(monkeypatch)

    class _Loaded:
        def to(self, device):
            return self

        def eval(self):
            return None

    class _ModelCls:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return _Loaded()

    class _ProcessorCls:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            raise RuntimeError("processor failed")

    monkeypatch.setattr(
        model_manager,
        "resolve_pretrained_source",
        lambda **kwargs: type("Imported", (), {"pretrained_source": "/tmp/model"})(),
    )

    spec = model_manager.ModelSpec(provider="hf", model_ref="repo/model")

    model, processor = model_manager.load_model_and_processor(
        spec=spec,
        device="cpu",
        model_cls=_ModelCls,
        processor_cls=_ProcessorCls,
        require_processor=False,
    )
    assert model is not None
    assert processor is None

    with pytest.raises(RuntimeError, match="processor failed"):
        model_manager.load_model_and_processor(
            spec=spec,
            device="cpu",
            model_cls=_ModelCls,
            processor_cls=_ProcessorCls,
            require_processor=True,
        )


def test_init_env_is_idempotent(monkeypatch):
    module_path = Path(__file__).resolve().parent.parent / "app" / "infrastructure" / "config" / "env.py"
    spec = importlib.util.spec_from_file_location("env_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None

    dotenv_stub = ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: None
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_stub)

    env = importlib.util.module_from_spec(spec)
    sys.modules["env_test_module"] = env
    spec.loader.exec_module(env)

    calls = {"count": 0}

    def _fake_load_dotenv():
        calls["count"] += 1

    monkeypatch.setattr(env, "_loaded", False)
    monkeypatch.setattr(env, "load_dotenv", _fake_load_dotenv)

    env.init_env()
    env.init_env()

    assert calls["count"] == 1
