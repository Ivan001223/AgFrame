import pytest
import sys
import os
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEvaluationFramework:
    """评估框架集成测试"""

    def test_golden_dataset_loading(self):
        """测试 Golden Dataset 加载"""
        fixture_path = os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "golden_cases.json"
        )

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "cases" in data
        assert len(data["cases"]) >= 5
        assert "metadata" in data

        first_case = data["cases"][0]
        assert "id" in first_case
        assert "input" in first_case
        assert "expected_tool" in first_case
        assert "expected_keywords" in first_case

    def test_golden_case_structure(self):
        """测试 Golden Case 结构完整性"""
        fixture_path = os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "golden_cases.json"
        )

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for case in data["cases"]:
            assert "id" in case
            assert "input" in case
            assert "expected_tool" in case
            assert "expected_keywords" in case
            assert "description" in case
            assert "eval_criteria" in case

            assert "answer_relevancy_min" in case["eval_criteria"]
            assert "faithfulness_min" in case["eval_criteria"]

    def test_deepeval_import(self):
        """测试 DeepEval 导入"""
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.dataset import Golden, EvaluationDataset

        assert LLMTestCase is not None
        assert AnswerRelevancyMetric is not None
        assert FaithfulnessMetric is not None
        assert Golden is not None
        assert EvaluationDataset is not None

    def test_ragas_import(self):
        """测试 Ragas 导入"""
        try:
            from ragas import EvaluationDataset
            from ragas.metrics import AnswerRelevancy, Faithfulness

            assert AnswerRelevancy is not None
            assert Faithfulness is not None
        except ImportError:
            pytest.skip("Ragas not installed")


class TestEvaluationRunner:
    """评估运行器测试"""

    def test_eval_criteria_compliance(self):
        """测试评估标准合规性"""
        fixture_path = os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "golden_cases.json"
        )

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for case in data["cases"]:
            criteria = case.get("eval_criteria", {})
            assert "answer_relevancy_min" in criteria
            assert "faithfulness_min" in criteria

            assert 0 <= criteria["answer_relevancy_min"] <= 1
            assert 0 <= criteria["faithfulness_min"] <= 1

    def test_golden_case_coverage(self):
        """测试 Golden Case 覆盖场景"""
        fixture_path = os.path.join(
            os.path.dirname(__file__),
            "fixtures",
            "golden_cases.json"
        )

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_tools = set()
        for case in data["cases"]:
            expected_tools.add(case["expected_tool"])

        assert len(expected_tools) >= 3


class TestPromptA_B:
    """Prompt A/B 测试测试"""

    def test_prompt_registry_import(self):
        """测试 Prompt 注册表导入"""
        from app.runtime.prompts.prompt_registry import PromptRegistry, get_prompt_registry

        registry = get_prompt_registry()
        assert registry is not None

    def test_prompt_template_structure(self):
        """测试 Prompt 模板结构"""
        from app.runtime.prompts.prompt_registry import PromptTemplate

        template = PromptTemplate(
            name="test_prompt",
            template="You are a {role} assistant.",
            variables=["role"],
        )

        assert template.name == "test_prompt"
        assert "{role}" in template.template
        assert "role" in template.variables

    def test_prompt_rendering(self):
        """测试 Prompt 渲染"""
        from app.runtime.prompts.prompt_registry import PromptTemplate

        template = PromptTemplate(
            name="greeting",
            template="Hello, {name}! Welcome to {system}.",
            variables=["name", "system"],
        )

        rendered = template.render(name="User", system="AgFrame")

        assert "Hello, User!" in rendered
        assert "AgFrame" in rendered

    def test_prompt_registration(self):
        """测试 Prompt 注册"""
        from app.runtime.prompts.prompt_registry import PromptTemplate, PromptRegistry

        registry = PromptRegistry()
        initial_count = len(registry._prompts)

        new_prompt = PromptTemplate(
            name="custom_prompt",
            template="Custom: {task}",
            variables=["task"],
        )
        registry.register(new_prompt)

        assert len(registry._prompts) > initial_count

    def test_ab_test_creation(self):
        """测试 A/B 测试创建"""
        from app.runtime.prompts.prompt_registry import PromptRegistry

        registry = PromptRegistry()
        test_id = registry.create_ab_test(
            name="orchestrator",
            variant_a_version="1.0.0",
            variant_b_version="1.1.0",
            traffic_split=0.5,
        )

        assert test_id.startswith("ab_")
        assert test_id in registry._ab_tests

    def test_ab_variant_selection(self):
        """测试 A/B 变体选择"""
        from app.runtime.prompts.prompt_registry import PromptRegistry, PromptVariant

        registry = PromptRegistry()
        registry.create_ab_test(
            name="test_prompt",
            variant_a_version="1.0.0",
            variant_b_version="1.1.0",
            traffic_split=0.5,
        )

        variant_a_count = 0
        variant_b_count = 0
        for i in range(100):
            variant = registry.get_ab_variant("ab_test_prompt", f"user_{i}")
            if variant == PromptVariant.A:
                variant_a_count += 1
            else:
                variant_b_count += 1

        assert variant_a_count > 0
        assert variant_b_count > 0

    def test_prompt_version_listing(self):
        """测试 Prompt 版本列表"""
        from app.runtime.prompts.prompt_registry import PromptRegistry

        registry = PromptRegistry()
        versions = registry.list_versions("orchestrator")

        assert len(versions) >= 1
