from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document


@dataclass(frozen=True)
class PromptBudget:
    max_recent_history_lines: int = 10
    max_docs: int = 3
    max_memories: int = 3
    max_doc_chars_total: int = 6000
    max_memory_chars_total: int = 3000
    max_item_chars: int = 2000


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _take_with_budget(items: Sequence[str], *, max_total_chars: int) -> List[str]:
    out: List[str] = []
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


def _get_meta_str(meta: Dict[str, Any], key: str) -> Optional[str]:
    val = meta.get(key)
    if val is None:
        return None
    return str(val)


def _get_meta_int(meta: Dict[str, Any], key: str) -> Optional[int]:
    val = meta.get(key)
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        return None


def build_citations(*, docs: Sequence[Document], memories: Sequence[Document]) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
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
    budget: Optional[PromptBudget] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    b = budget or PromptBudget()

    recent_lines = list(recent_history_lines)[-b.max_recent_history_lines :]

    doc_items: List[str] = []
    for i, d in enumerate(list(docs)[: b.max_docs], start=1):
        meta = dict(getattr(d, "metadata", {}) or {})
        ref = (
            f"doc_id={meta.get('doc_id')}, parent_chunk_id={meta.get('parent_chunk_id')}, "
            f"page={meta.get('page_num')}"
        )
        content = _truncate(str(getattr(d, "page_content", "") or ""), b.max_item_chars)
        doc_items.append(f"[Doc {i}] ({ref})\n{content}")

    mem_items: List[str] = []
    for i, m in enumerate(list(memories)[: b.max_memories], start=1):
        meta = dict(getattr(m, "metadata", {}) or {})
        ref = (
            f"session_id={meta.get('session_id')}, "
            f"msg_range={meta.get('start_msg_id')}..{meta.get('end_msg_id')}"
        )
        content = _truncate(str(getattr(m, "page_content", "") or ""), b.max_item_chars)
        mem_items.append(f"[Memory {i}] ({ref})\n{content}")

    doc_block = "\n".join(_take_with_budget(doc_items, max_total_chars=b.max_doc_chars_total))
    mem_block = "\n".join(_take_with_budget(mem_items, max_total_chars=b.max_memory_chars_total))

    system_prompt = (
        "你是一个严谨的助理。回答时优先使用提供的上下文与用户画像。\n"
        "当引用文档内容时，尽量给出对应 Doc 编号；当引用历史记忆时，尽量给出 Memory 编号。\n"
        "如果上下文不足以回答细节，明确说明缺失点并给出下一步需要的信息。\n\n"
        f"<user_profile>\n{profile}\n</user_profile>\n\n"
        f"<recent_history>\n{chr(10).join(recent_lines) if recent_lines else ''}\n</recent_history>\n\n"
        f"<retrieved_docs>\n{doc_block}\n</retrieved_docs>\n\n"
        f"<retrieved_memories>\n{mem_block}\n</retrieved_memories>\n"
    )

    citations = build_citations(
        docs=list(docs)[: b.max_docs], memories=list(memories)[: b.max_memories]
    )
    return system_prompt, citations

