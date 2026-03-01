from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from app.infrastructure.utils.files import sha256_file
from app.infrastructure.utils.image_handler import (
    convert_url_to_base64,
    is_local_url,
    process_multimodal_content,
)
from app.infrastructure.utils.json_parser import parse_json_from_llm
from app.infrastructure.utils.logging import bind_logger, init_logging
from app.infrastructure.utils.message_utils import sanitize_messages_for_routing


def test_sha256_file(tmp_path: Any):
    p = tmp_path / "a.bin"
    p.write_bytes(b"abc")
    assert sha256_file(str(p)) == hashlib.sha256(b"abc").hexdigest()


def test_parse_json_from_llm_removes_think_and_code_fence():
    s = "<think>xxx</think>\n```json\n{\"a\": 1}\n```"
    assert parse_json_from_llm(s) == {"a": 1}


def test_parse_json_from_llm_extracts_embedded_object():
    s = "prefix {\"a\": 1, \"b\": 2} suffix"
    assert parse_json_from_llm(s) == {"a": 1, "b": 2}


def test_parse_json_from_llm_fixes_bad_backslash_escape():
    s = "prefix {\"path\": \"c:\\q\"} suffix"
    assert parse_json_from_llm(s)["path"] == "c:\\q"


def test_parse_json_from_llm_extracts_list():
    s = "result: [1, 2, 3]"
    assert parse_json_from_llm(s) == [1, 2, 3]


def test_bind_logger_and_process_merges_extra():
    lg = logging.getLogger("t")
    adapter = bind_logger(lg, trace_id="t1", user_id="u1", session_id="s1", node="n1")
    msg, kwargs = adapter.process("m", {"extra": {"k": "v"}})
    assert msg == "m"
    assert kwargs["extra"]["trace_id"] == "t1"
    assert kwargs["extra"]["k"] == "v"


def test_init_logging_handles_existing_handlers():
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    try:
        root.handlers = []
        init_logging()
        init_logging()
    finally:
        root.handlers = old_handlers


def test_sanitize_messages_for_routing_strips_multimodal():
    from langchain_core.messages import AIMessage, HumanMessage

    @dataclass
    class _M:
        type: str = "system"
        content: Any = "sys"

    msgs = [
        HumanMessage(content=[{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "x"}}]),
        AIMessage(content="ok"),
        _M(),
    ]
    out = sanitize_messages_for_routing(msgs)  # type: ignore[arg-type]
    assert out[0].type == "human"
    assert out[0].content == "hi"
    assert out[1].type == "ai"
    assert out[1].content == "ok"
    assert out[2].type == "human"
    assert "[system]" in out[2].content


def test_is_local_url():
    assert is_local_url("http://localhost:8000/x") is True
    assert is_local_url("http://127.0.0.1:8000/x") is True
    assert is_local_url("https://example.com/x") is False


def test_convert_url_to_base64_success(monkeypatch: pytest.MonkeyPatch):
    class _R:
        status_code = 200
        content = b"abc"

    monkeypatch.setattr("app.infrastructure.utils.image_handler.requests.get", lambda *a, **k: _R())
    out = convert_url_to_base64("http://localhost:8000/a.jpg")
    assert out.startswith("data:image/jpeg;base64,")
    assert out.split(",", 1)[1] == base64.b64encode(b"abc").decode("utf-8")


def test_convert_url_to_base64_failure(monkeypatch: pytest.MonkeyPatch):
    def _boom(*args: Any, **kwargs: Any):
        raise RuntimeError("x")

    monkeypatch.setattr("app.infrastructure.utils.image_handler.requests.get", _boom)
    assert convert_url_to_base64("http://localhost:8000/a.jpg") is None


def test_process_multimodal_content_rewrites_local_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "app.infrastructure.utils.image_handler.convert_url_to_base64",
        lambda url: "data:image/jpeg;base64,xxx",
    )
    raw = [
        {"type": "text", "text": "hi"},
        {"type": "image_url", "image_url": {"url": "http://localhost:8000/a.jpg"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
    ]
    out = process_multimodal_content(raw)
    assert out[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert out[2]["image_url"]["url"].startswith("https://")

