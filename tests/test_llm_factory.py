from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.infrastructure.config.settings import settings
from app.runtime.llm.dev_stub import DevStubChatModel
from app.runtime.llm import llm_factory


def test_get_llm_returns_dev_stub_when_requested(monkeypatch):
    monkeypatch.setattr(settings.llm, "model", "dev-stub", raising=False)

    llm = llm_factory.get_llm(streaming=False)

    assert isinstance(llm, DevStubChatModel)
    reply = llm.invoke([HumanMessage(content="Reply briefly for smoke")])
    assert "[dev-stub]" in reply.content


def test_get_llm_uses_chat_openai_for_remote_models(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(settings.llm, "model", "gpt-4o-mini", raising=False)
    monkeypatch.setattr(settings.llm, "base_url", "https://example.invalid/v1", raising=False)
    monkeypatch.setattr(settings.llm, "api_key", "sk-test", raising=False)
    monkeypatch.setattr(llm_factory, "ChatOpenAI", _FakeChatOpenAI)

    llm = llm_factory.get_llm(temperature=0.2, streaming=False, json_mode=True)

    assert isinstance(llm, _FakeChatOpenAI)
    assert captured == {
        "model": "gpt-4o-mini",
        "temperature": 0.2,
        "base_url": "https://example.invalid/v1",
        "api_key": "sk-test",
        "streaming": False,
        "model_kwargs": {},
    }
