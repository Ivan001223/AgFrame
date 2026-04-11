from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.runtime.prompts.context_pruner import (
    ContextPruningConfig,
    PromptPruningSummary,
    build_prompt_pruning_summary,
    prune_documents,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptBudget:
    max_recent_history_lines: int = 10
    max_docs: int = 3
    max_memories: int = 3
    max_doc_chars_total: int = 6000
    max_memory_chars_total: int = 3000
    max_profile_chars_total: int = 2500
    max_item_chars: int = 2000


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _take_with_budget(items: Sequence[str], *, max_total_chars: int) -> list[str]:
    out: list[str] = []
    remaining = max_total_chars
    for it in items:
        if remaining <= 0:
            break
        if len(it) <= remaining:
            out.append(it)
            remaining -= len(it)
        else:
            out.append(_truncate(it, remaining))
            break
    return out


def _get_meta_str(meta: dict[str, Any], key: str) -> str | None:
    val = meta.get(key)
    if val is None:
        return None
    return str(val)


def _get_meta_int(meta: dict[str, Any], key: str) -> int | None:
    val = meta.get(key)
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse {key} as int: {e}")
        return None


def build_citations(*, docs: Sequence[Document], memories: Sequence[Document]) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for i, d in enumerate(docs, start=1):
        meta = dict(getattr(d, "metadata", {}) or {})
        citations.append(
            {
                "kind": "doc",
                "label": f"Doc {i}",
                "doc_id": _get_meta_str(meta, "doc_id") or _get_meta_str(meta, "source"),
                "page": _get_meta_int(meta, "page_num"),
                "source": _get_meta_str(meta, "source"),
            }
        )
    for i, m in enumerate(memories, start=1):
        meta = dict(getattr(m, "metadata", {}) or {})
        citations.append(
            {
                "kind": "memory",
                "label": f"Memory {i}",
                "session_id": _get_meta_str(meta, "session_id"),
                "source": _get_meta_str(meta, "source"),
            }
        )
    return citations


def build_system_prompt(
    *,
    profile: Any,
    recent_history_lines: Sequence[str],
    docs: Sequence[Document],
    memories: Sequence[Document],
    query: str = "",
    focus_hint: str | None = None,
    web_search: dict[str, Any] | None = None,
    self_correction: str | None = None,
    budget: PromptBudget | None = None,
    pruning: ContextPruningConfig | None = None,
) -> tuple[str, list[dict[str, Any]], PromptPruningSummary]:
    b = budget or PromptBudget()
    pruning_cfg = pruning or ContextPruningConfig()

    recent_lines = list(recent_history_lines)[-b.max_recent_history_lines :]
    profile_block = _truncate(str(profile), b.max_profile_chars_total)

    pruned_docs, doc_pruning = prune_documents(
        list(docs)[: b.max_docs],
        query=query,
        focus_hint=focus_hint,
        config=pruning_cfg,
    )
    pruned_memories, memory_pruning = prune_documents(
        list(memories)[: b.max_memories],
        query=query,
        focus_hint=focus_hint,
        config=pruning_cfg,
    )

    doc_items: list[str] = []
    for i, d in enumerate(pruned_docs, start=1):
        meta = dict(getattr(d, "metadata", {}) or {})
        ref = (
            f"doc_id={meta.get('doc_id')}, parent_chunk_id={meta.get('parent_chunk_id')}, "
            f"page={meta.get('page_num')}"
        )
        content = _truncate(str(getattr(d, "page_content", "") or ""), b.max_item_chars)
        doc_items.append(f"[Doc {i}] ({ref})\n{content}")

    mem_items: list[str] = []
    for i, m in enumerate(pruned_memories, start=1):
        meta = dict(getattr(m, "metadata", {}) or {})
        ref = (
            f"session_id={meta.get('session_id')}, "
            f"msg_range={meta.get('start_msg_id')}..{meta.get('end_msg_id')}"
        )
        content = _truncate(str(getattr(m, "page_content", "") or ""), b.max_item_chars)
        mem_items.append(f"[Memory {i}] ({ref})\n{content}")

    doc_block = "\n".join(_take_with_budget(doc_items, max_total_chars=b.max_doc_chars_total))
    mem_block = "\n".join(_take_with_budget(mem_items, max_total_chars=b.max_memory_chars_total))

    web_search_block = ""
    if web_search:
        query = _truncate(str(web_search.get("query") or ""), 200)
        result = _truncate(str(web_search.get("result") or ""), b.max_item_chars)
        web_search_block = f"\n\n<web_search query={query!r}>\n{result}\n</web_search>"

    self_correction_block = ""
    if self_correction:
        self_correction_block = f"\n\n<self_correction>\n{_truncate(str(self_correction), b.max_item_chars)}\n</self_correction>"

    system_prompt = (
        "你是一个严谨的助理。回答时优先使用提供的上下文与用户画像。\n"
        "当引用文档内容时，尽量给出对应 Doc 编号；当引用历史记忆时，尽量给出 Memory 编号。\n"
        "如果上下文不足以回答细节，明确说明缺失点并给出下一步需要的信息。\n\n"
        f"<context_focus_hint>\n{_truncate(str(focus_hint or query), 300)}\n</context_focus_hint>\n\n"
        f"<user_profile>\n{profile_block}\n</user_profile>\n\n"
        f"<recent_history>\n{chr(10).join(recent_lines) if recent_lines else ''}\n</recent_history>\n\n"
        f"<retrieved_docs>\n{doc_block}\n</retrieved_docs>\n\n"
        f"<retrieved_memories>\n{mem_block}\n</retrieved_memories>\n"
        f"{web_search_block}{self_correction_block}"
    )

    citations = build_citations(
        docs=pruned_docs, memories=pruned_memories
    )
    pruning_summary = build_prompt_pruning_summary(
        focus_hint=focus_hint or query,
        docs=doc_pruning,
        memories=memory_pruning,
    )
    return system_prompt, citations, pruning_summary
