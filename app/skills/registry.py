from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillCapabilityDescriptor:
    skill_id: str
    title: str
    description: str
    prompt_hint: str = ""
    suggested_tool_ids: tuple[str, ...] = ()
    suggested_mcp_server_ids: tuple[str, ...] = ()


def _normalize_skill_key(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip()).strip("_")


def _humanize_identifier(value: str) -> str:
    parts = [part for part in value.replace("-", "_").split("_") if part]
    return " ".join(part.capitalize() for part in parts) or value


SKILL_CAPABILITY_REGISTRY: dict[str, SkillCapabilityDescriptor] = {
    "common": SkillCapabilityDescriptor(
        skill_id="common",
        title="Common",
        description="Shared prompting, grading, and orchestration helpers used by the runtime.",
        prompt_hint=(
            "Reuse shared runtime conventions for structured reasoning, concise summaries, and clean task decomposition."
        ),
    ),
    "memory": SkillCapabilityDescriptor(
        skill_id="memory",
        title="Memory",
        description="Reuse stored user and project memory to keep work consistent across runs.",
        prompt_hint=(
            "Treat prior user or project memory as a constraint source. Reuse remembered decisions when relevant, "
            "but call out uncertainty if the stored memory may be stale."
        ),
    ),
    "ocr": SkillCapabilityDescriptor(
        skill_id="ocr",
        title="OCR",
        description="Extract text from PDFs, images, and office files before downstream analysis.",
        prompt_hint=(
            "When working from files or screenshots, extract the text carefully first and preserve uncertainty for low-confidence regions."
        ),
        suggested_tool_ids=("read_document",),
        suggested_mcp_server_ids=("filesystem",),
    ),
    "profile": SkillCapabilityDescriptor(
        skill_id="profile",
        title="Profile",
        description="Adapt work to stored profile preferences, role context, and collaboration habits.",
        prompt_hint=(
            "Use available profile context to adapt tone, level of detail, and constraints to the user or team you are serving."
        ),
    ),
    "rag": SkillCapabilityDescriptor(
        skill_id="rag",
        title="RAG",
        description="Retrieve and ground answers with project knowledge-base evidence.",
        prompt_hint=(
            "Prefer knowledge-base evidence when the project context matters. Preserve source grounding, distinguish retrieved facts from inference, "
            "and say when retrieval coverage is weak."
        ),
        suggested_tool_ids=("knowledge_retriever", "read_document"),
        suggested_mcp_server_ids=("filesystem",),
    ),
    "research": SkillCapabilityDescriptor(
        skill_id="research",
        title="Research",
        description="Gather current external evidence, compare sources, and separate facts from hypotheses.",
        prompt_hint=(
            "Collect recent external evidence, compare multiple sources when possible, and separate verified facts from working hypotheses."
        ),
        suggested_tool_ids=("web_search", "get_current_time", "read_document"),
        suggested_mcp_server_ids=("fetch", "browser"),
    ),
    "tools": SkillCapabilityDescriptor(
        skill_id="tools",
        title="Tools",
        description="Use approved tools deliberately and safely when they materially improve execution quality.",
        prompt_hint=(
            "Use tools only when they materially advance the task. Prefer the smallest safe tool, fold the result back into your reasoning, "
            "and avoid assuming a tool is available unless the capability map says it is enabled."
        ),
        suggested_tool_ids=(
            "web_search",
            "read_document",
            "calculator",
            "write_file",
            "python_executor",
            "get_current_time",
        ),
        suggested_mcp_server_ids=("filesystem", "browser", "github"),
    ),
}


def get_skill_descriptor(skill_id: str) -> SkillCapabilityDescriptor | None:
    normalized = _normalize_skill_key(skill_id)
    if not normalized:
        return None
    return SKILL_CAPABILITY_REGISTRY.get(normalized)


def build_fallback_skill_descriptor(skill_id: str) -> SkillCapabilityDescriptor:
    normalized = _normalize_skill_key(skill_id)
    title = _humanize_identifier(normalized or skill_id)
    return SkillCapabilityDescriptor(
        skill_id=normalized or skill_id.strip(),
        title=title,
        description=f"Local skill package from app/skills/{normalized or skill_id.strip()}",
    )
