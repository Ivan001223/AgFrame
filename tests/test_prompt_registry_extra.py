from __future__ import annotations

from app.runtime.prompts.prompt_registry import PromptRegistry, PromptTemplate


def _reset_registry() -> PromptRegistry:
    r = PromptRegistry()
    r._prompts = {}
    r._ab_tests = {}
    return r


def test_prompt_template_dict_roundtrip():
    t = PromptTemplate(
        name="n",
        template="Hello {x}",
        variables=["x"],
        version="2.0.0",
        description="d",
        tags=["a"],
    )
    t2 = PromptTemplate.from_dict(t.to_dict())
    assert t2.name == "n"
    assert t2.version == "2.0.0"
    assert t2.tags == ["a"]


def test_registry_get_latest_and_specific_version():
    r = _reset_registry()
    r.register(PromptTemplate(name="p", template="v1", variables=[], version="1.0.0"))
    r.register(PromptTemplate(name="p", template="v2", variables=[], version="2.0.0"))
    latest = r.get("p", "latest")
    assert latest is not None
    assert latest.version == "2.0.0"
    v1 = r.get("p", "1.0.0")
    assert v1 is not None
    assert v1.template == "v1"
    assert r.get("not_exist", "latest") is None


def test_registry_export_import():
    r1 = _reset_registry()
    r1.register(PromptTemplate(name="x", template="t", variables=["a"], version="3.0.0"))
    test_id = r1.create_ab_test("x", "3.0.0", "3.0.0", traffic_split=0.5)
    exported = r1.export_prompts()

    r2 = _reset_registry()
    r2.import_prompts(exported)

    assert r2.get("x", "3.0.0") is not None
    assert test_id in r2._ab_tests
