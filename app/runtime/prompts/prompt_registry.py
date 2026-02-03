import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class PromptVariant(Enum):
    A = "variant_a"
    B = "variant_b"


@dataclass
class PromptTemplate:
    name: str
    template: str
    variables: List[str]
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def render(self, **kwargs: str) -> str:
        rendered = self.template
        for var in self.variables:
            value = kwargs.get(var, f"{{{var}}}")
            rendered = rendered.replace(f"{{{var}}}", str(value))
        return rendered

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        return cls(
            name=data["name"],
            template=data["template"],
            variables=data["variables"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )


class PromptRegistry:
    _instance: Optional["PromptRegistry"] = None
    _prompts: Dict[str, PromptTemplate] = {}
    _ab_tests: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._prompts:
            self._load_default_prompts()

    def _load_default_prompts(self):
        default_prompts = [
            PromptTemplate(
                name="orchestrator",
                template="""你是监督者（Supervisor）Agent。
你的目标是将用户请求路由到合适的子 Agent；当任务已完成时返回 FINISH。

### 可用 Agent：
1. **general**：处理一般聊天与不需要特定工具的请求。
2. **rag**：处理需要检索文档的请求。
3. **web_search**：处理需要实时信息的请求。
4. **python_executor**：处理需要执行代码的请求。
5. **ocr**：处理需要分析图像的请求。
6. **memory**：处理需要检索历史记忆的请求。

### 路由规则：
- 若用户意图清晰且匹配某个 Agent，则路由到该 Agent。
- 若请求已完成或对话应结束，则返回 "FINISH"。

### 输出要求：
- 仅输出符合 schema 的合法 JSON（不要输出额外文本、Markdown 代码块或解释）。""",
                variables=[],
                version="1.0.0",
                description="编排器系统提示词",
                tags=["router", "supervisor"],
            ),
            PromptTemplate(
                name="general_chat",
                template="你是一个友好的 AI 助手。请用简洁清晰的语言回答用户的问题。",
                variables=[],
                version="1.0.0",
                description="通用聊天提示词",
                tags=["chat", "general"],
            ),
            PromptTemplate(
                name="rag_answer",
                template="""基于以下上下文信息回答用户的问题。如果上下文中没有相关信息，请明确说明。

<context>
{context}
</context>

用户问题：{question}

请提供准确、有帮助的回答，并在适当的地方引用信息来源。""",
                variables=["context", "question"],
                version="1.0.0",
                description="RAG 回答提示词",
                tags=["rag", "retrieval"],
            ),
            PromptTemplate(
                name="code_generation",
                template="""你是一个专业程序员。请为用户的需求提供清晰的代码解决方案。

要求：
- 使用最佳实践
- 添加必要的注释
- 考虑边界情况

用户需求：{task}

请提供完整的代码实现。""",
                variables=["task"],
                version="1.0.0",
                description="代码生成提示词",
                tags=["code", "programming"],
            ),
        ]

        for prompt in default_prompts:
            self.register(prompt)

    def register(self, template: PromptTemplate) -> None:
        key = f"{template.name}:{template.version}"
        self._prompts[key] = template
        logger.info(f"Registered prompt: {key}")

    def get(self, name: str, version: str = "latest") -> Optional[PromptTemplate]:
        if version == "latest":
            matching = [k for k in self._prompts.keys() if k.startswith(f"{name}:")]
            if not matching:
                return None
            matching.sort(key=lambda x: self._prompts[x].version, reverse=True)
            return self._prompts[matching[0]]

        key = f"{name}:{version}"
        return self._prompts.get(key)

    def list_versions(self, name: str) -> List[str]:
        matching = [k for k in self._prompts.keys() if k.startswith(f"{name}:")]
        return [k.split(":")[1] for k in matching]

    def create_ab_test(
        self,
        name: str,
        variant_a_version: str,
        variant_b_version: str,
        traffic_split: float = 0.5,
    ) -> str:
        test_id = f"ab_{name}"
        self._ab_tests[test_id] = {
            "name": name,
            "variant_a": variant_a_version,
            "variant_b": variant_b_version,
            "traffic_split": traffic_split,
            "active": True,
        }
        logger.info(f"Created A/B test: {test_id}")
        return test_id

    def get_ab_variant(
        self, test_id: str, user_id: str
    ) -> Optional[PromptVariant]:
        test = self._ab_tests.get(test_id)
        if not test or not test.get("active"):
            return None

        import hashlib
        hash_val = int(hashlib.md5(f"{user_id}".encode()).hexdigest(), 16)
        if hash_val % 100 < test["traffic_split"] * 100:
            return PromptVariant.A
        return PromptVariant.B

    def get_ab_prompt(
        self, test_id: str, user_id: str
    ) -> Optional[PromptTemplate]:
        variant = self.get_ab_variant(test_id, user_id)
        if not variant:
            return None

        test = self._ab_tests[test_id]
        version = test[f"variant_{variant.value}"]
        return self.get(test["name"], version)

    def export_prompts(self) -> Dict[str, Any]:
        return {
            "prompts": {k: v.to_dict() for k, v in self._prompts.items()},
            "ab_tests": self._ab_tests,
        }

    def import_prompts(self, data: Dict[str, Any]) -> None:
        for key, val in data.get("prompts", {}).items():
            self.register(PromptTemplate.from_dict(val))
        self._ab_tests = data.get("ab_tests", {})


def get_prompt_registry() -> PromptRegistry:
    return PromptRegistry()
