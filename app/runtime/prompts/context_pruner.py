from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, NotRequired, Sequence, TypedDict

from langchain_core.documents import Document

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")

_SEMANTIC_ALIASES: dict[str, tuple[str, ...]] = {
    "renew": ("refresh", "rotate", "extend", "prolong"),
    "renews": ("refresh", "rotate", "extend", "prolong"),
    "refresh": ("renew", "fresh", "reissue"),
    "token": ("credential", "credentials", "session", "secret", "lease", "grant"),
    "credential": ("identity", "grant", "secret"),
    "credentials": ("identity", "grant", "secret"),
    "grant": ("credential", "credentials", "session", "browser"),
    "identity": ("credential", "credentials", "grant", "artifact"),
    "artifact": ("grant", "token", "credential"),
    "login": ("oauth", "browser", "session"),
    "rotate": ("renew", "replace", "extend", "reissue"),
    "issue": ("grant", "token", "session"),
    "extend": ("renew", "refresh", "prolong", "lease"),
    "prolong": ("extend", "renew", "refresh", "lease"),
    "oauth": ("provider", "login", "callback"),
    "callback": ("return", "redirect", "provider"),
    "retry": ("requeue", "retry", "backoff", "jitter"),
    "backoff": ("jitter", "delay", "retry"),
    "revoke": ("discard", "purge", "clear", "invalidate", "drop"),
    "revokes": ("discard", "purge", "clear", "invalidate", "drop"),
    "stale": ("persisted", "cached", "expired"),
}

_NOISE_TOKENS = {
    "banner",
    "badge",
    "button",
    "color",
    "copy",
    "icon",
    "label",
    "theme",
    "toast",
    "ui",
}


@dataclass(frozen=True)
class ContextPruningConfig:
    enabled: bool = True
    method: str = "heuristic"
    auto_reranker_min_lines: int = 40
    auto_reranker_min_chars: int = 2500
    min_keywords: int = 2
    min_keep_lines: int = 4
    max_keep_ratio: float = 0.45
    neighbor_window: int = 1
    reranker_window_radius: int = 1
    max_lines_per_item: int = 24
    score_threshold: float = 0.18


class SavingsSummary(TypedDict):
    saved: int
    saved_ratio: float


class DocumentPruningStats(TypedDict):
    enabled: bool
    method: str
    scoring_source: str
    line_count_before: int
    line_count_after: int
    char_count_before: int
    char_count_after: int
    ratio: float
    char_savings: SavingsSummary
    line_savings: SavingsSummary
    kept_indexes: NotRequired[list[int]]


class AggregatePruningSummary(TypedDict):
    items: int
    items_pruned: int
    char_count_before: int
    char_count_after: int
    line_count_before: int
    line_count_after: int
    ratio: float
    focus_hint: str
    enabled: bool
    method: str
    scoring_source: str
    char_savings: SavingsSummary
    line_savings: SavingsSummary


class PromptPruningTraceBlock(TypedDict):
    method: str
    scoring_source: str
    char_savings: SavingsSummary
    line_savings: SavingsSummary
    items: int
    items_pruned: int


class PromptPruningTrace(TypedDict):
    focus_hint: str
    method: str
    scoring_source: str
    docs: PromptPruningTraceBlock
    memories: PromptPruningTraceBlock


class CandidatePruningTrace(TypedDict):
    focus_hint: str
    method: str
    scoring_source: str
    items: int
    items_pruned: int
    char_savings: SavingsSummary
    line_savings: SavingsSummary


class PromptPruningSummary(TypedDict):
    focus_hint: str
    method: str
    scoring_source: str
    docs: AggregatePruningSummary
    memories: AggregatePruningSummary


def build_prompt_pruning_summary(
    *,
    focus_hint: str,
    docs: AggregatePruningSummary,
    memories: AggregatePruningSummary,
) -> PromptPruningSummary:
    return {
        "focus_hint": focus_hint,
        "method": docs["method"] if docs["method"] == memories["method"] else "mixed",
        "scoring_source": (
            docs["scoring_source"]
            if docs["scoring_source"] == memories["scoring_source"]
            else "mixed"
        ),
        "docs": docs,
        "memories": memories,
    }


def build_pruning_config(source: Any | None = None, **overrides: Any) -> ContextPruningConfig:
    if source is None:
        values: dict[str, Any] = {}
    elif isinstance(source, ContextPruningConfig):
        values = source.__dict__.copy()
    else:
        values = {
            "enabled": getattr(source, "enabled", True),
            "method": getattr(source, "method", "heuristic"),
            "auto_reranker_min_lines": getattr(source, "auto_reranker_min_lines", 40),
            "auto_reranker_min_chars": getattr(source, "auto_reranker_min_chars", 2500),
            "min_keywords": getattr(source, "min_keywords", 2),
            "min_keep_lines": getattr(source, "min_keep_lines", 4),
            "max_keep_ratio": getattr(source, "max_keep_ratio", 0.45),
            "neighbor_window": getattr(source, "neighbor_window", 1),
            "reranker_window_radius": getattr(source, "reranker_window_radius", 1),
            "max_lines_per_item": getattr(source, "max_lines_per_item", 24),
            "score_threshold": getattr(source, "score_threshold", 0.18),
        }
    values.update(overrides)
    return ContextPruningConfig(**values)


def _tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _unique_keywords(*texts: str, min_keywords: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for token in _tokenize(text):
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
            for alias in _SEMANTIC_ALIASES.get(token, ()):
                if alias in seen:
                    continue
                seen.add(alias)
                out.append(alias)
    if len(out) <= min_keywords:
        return out
    return out[: max(min_keywords, len(out))]


def _line_score(line: str, keywords: Sequence[str]) -> float:
    content = (line or "").strip().lower()
    if not content:
        return 0.0
    line_tokens = set(_tokenize(content))
    if not line_tokens:
        return 0.0
    overlap = sum(1 for keyword in keywords if keyword in line_tokens)
    if overlap == 0:
        return 0.0
    density = overlap / max(len(line_tokens), 1)
    coverage = overlap / max(len(keywords), 1)
    exact_bonus = sum(0.25 for keyword in keywords if keyword in content)
    noise_penalty = 0.2 if line_tokens & _NOISE_TOKENS else 0.0
    return max(0.0, density + coverage + exact_bonus - noise_penalty)


def _seed_keyword_coverage(
    lines: Sequence[str],
    scores: Sequence[float],
    keywords: Sequence[str],
) -> list[int]:
    token_sets = [set(_tokenize(line)) for line in lines]
    seeded: set[int] = set()
    for keyword in keywords:
        best_index = None
        best_score = -1.0
        for index, token_set in enumerate(token_sets):
            if keyword not in token_set:
                continue
            score = float(scores[index])
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is not None:
            seeded.add(best_index)
    return sorted(seeded)


def _line_keyword_sets(lines: Sequence[str], keywords: Sequence[str]) -> list[set[str]]:
    keyword_set = set(keywords)
    return [set(_tokenize(line)) & keyword_set for line in lines]


def _resolve_reranker_source() -> str:
    return "lightweight_ranker"


def _ngrams(tokens: Sequence[str], size: int) -> set[str]:
    if len(tokens) < size:
        return set()
    return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def _score_candidates_with_local_ranker(
    candidates: Sequence[str],
    query: str,
    keywords: Sequence[str],
) -> list[float]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0 for _ in candidates]
    query_token_counts = Counter(query_tokens)
    query_token_set = set(query_tokens)
    query_bigrams = _ngrams(query_tokens, 2)
    query_trigrams = _ngrams(query_tokens, 3)
    keyword_set = set(keywords)
    scores: list[float] = []
    for candidate in candidates:
        candidate_tokens = _tokenize(candidate)
        if not candidate_tokens:
            scores.append(0.0)
            continue
        token_set = set(candidate_tokens)
        bigrams = _ngrams(candidate_tokens, 2)
        trigrams = _ngrams(candidate_tokens, 3)
        weighted_overlap = sum(1.0 / query_token_counts[token] for token in token_set & query_token_set)
        keyword_overlap = len(token_set & keyword_set)
        bigram_overlap = len(bigrams & query_bigrams)
        trigram_overlap = len(trigrams & query_trigrams)
        bridge_bonus = 0.35 if "\n" in candidate and keyword_overlap >= 3 else 0.0
        noise_penalty = 0.15 if token_set & _NOISE_TOKENS else 0.0
        score = (
            (weighted_overlap / max(len(query_token_set), 1))
            + (keyword_overlap / max(len(keyword_set), 1))
            + (bigram_overlap * 1.4)
            + (trigram_overlap * 2.0)
            + bridge_bonus
            - noise_penalty
        )
        scores.append(max(0.0, score))
    return scores


def _score_lines_with_reranker(lines: list[str], query: str, keywords: Sequence[str]) -> list[float]:
    scored_indexes = [index for index, line in enumerate(lines) if line.strip()]
    if not scored_indexes:
        return [0.0 for _ in lines]
    scores = [0.0 for _ in lines]
    candidates = [lines[index] for index in scored_indexes]
    ranked_scores = _score_candidates_with_local_ranker(candidates, query, keywords)
    for local_index, score in enumerate(ranked_scores):
        source_index = scored_indexes[local_index]
        scores[source_index] = float(score)
    return scores


def _build_window_candidates(lines: Sequence[str], radius: int) -> tuple[list[str], list[tuple[int, int, int]]]:
    candidates: list[str] = []
    windows: list[tuple[int, int, int]] = []
    if radius <= 0:
        return [str(line) for line in lines], [(index, index, index) for index in range(len(lines))]
    centers = range(radius, max(radius, len(lines) - radius))
    for index in centers:
        if not str(lines[index]).strip():
            continue
        start = index - radius
        end = index + radius + 1
        window = "\n".join(str(part) for part in lines[start:end] if str(part).strip())
        if not window.strip():
            continue
        candidates.append(window)
        windows.append((start, end, index))
    if candidates:
        return candidates, windows
    for index, line in enumerate(lines):
        if not str(line).strip():
            continue
        start = max(0, index - radius)
        end = min(len(lines), index + radius + 1)
        window = "\n".join(str(part) for part in lines[start:end] if str(part).strip())
        if not window.strip():
            continue
        candidates.append(window)
        windows.append((start, end, index))
    return candidates, windows


def _score_line_windows_with_reranker(
    lines: list[str],
    query: str,
    radius: int,
    keywords: Sequence[str],
) -> list[float]:
    candidates, windows = _build_window_candidates(lines, radius)
    if not candidates:
        return [0.0 for _ in lines]
    scores = [0.0 for _ in lines]
    ranked_scores = _score_candidates_with_local_ranker(candidates, query, keywords)
    for local_index, score in enumerate(ranked_scores):
        start, end, center = windows[local_index]
        for cursor in range(start, end):
            distance = abs(cursor - center)
            weight = 1.0 / ((distance + 1.0) ** 2)
            scores[cursor] = max(scores[cursor], float(score) * weight)
    return scores


def _expand_query(query: str, keywords: Sequence[str]) -> str:
    additions = [keyword for keyword in keywords if keyword not in query.lower()]
    if not additions:
        return query
    return f"{query}\nRelated terms: {' '.join(additions[:16])}"


def _resolve_method(config: ContextPruningConfig, *, line_count: int, char_count: int) -> str:
    method = (config.method or "heuristic").strip().lower()
    if method != "auto":
        return method
    if line_count >= config.auto_reranker_min_lines or char_count >= config.auto_reranker_min_chars:
        return "reranker"
    return "heuristic"


def _build_savings(*, before: int, after: int) -> SavingsSummary:
    saved = max(0, before - after)
    return {
        "saved": saved,
        "saved_ratio": round(saved / before, 4) if before > 0 else 0.0,
    }


def _pick_line_indexes(
    scores: list[float],
    config: ContextPruningConfig,
    *,
    seed_indexes: Sequence[int] | None = None,
    keyword_sets: Sequence[set[str]] | None = None,
) -> list[int]:
    if not scores:
        return []
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    max_keep = max(config.min_keep_lines, int(len(scores) * config.max_keep_ratio))
    max_keep = min(max_keep, config.max_lines_per_item, len(scores))

    protected_ranked = sorted(set(seed_indexes or []), key=lambda index: scores[index], reverse=True)
    protected = set(protected_ranked)
    selected: set[int] = set()
    selected.update({
        index
        for index, score in ranked
        if score >= config.score_threshold
    })
    if len(selected) < min(config.min_keep_lines, len(scores)):
        for index, _ in ranked[: min(config.min_keep_lines, len(scores))]:
            selected.add(index)
    if len(selected) > max_keep:
        selected = {index for index, _ in ranked[:max_keep]}
    for protected_index in protected_ranked:
        if protected_index in selected:
            continue
        if keyword_sets:
            covered_keywords: set[str] = set()
            for selected_index in selected:
                covered_keywords.update(keyword_sets[selected_index])
            new_keywords = keyword_sets[protected_index] - covered_keywords
            if new_keywords:
                replaceable = sorted(
                    selected,
                    key=lambda index: scores[index],
                )
                for candidate_index in replaceable:
                    if scores[protected_index] < scores[candidate_index] * 0.5:
                        continue
                    selected.remove(candidate_index)
                    selected.add(protected_index)
                    break
                if protected_index in selected:
                    continue
        if len(selected) < max_keep:
            selected.add(protected_index)
            continue
        replaceable = sorted(
            (
                index
                for index in selected
                if index not in protected and scores[index] < scores[protected_index]
            ),
            key=lambda index: scores[index],
        )
        if not replaceable:
            continue
        selected.remove(replaceable[0])
        selected.add(protected_index)

    expanded: set[int] = set()
    for index in selected:
        start = max(0, index - config.neighbor_window)
        end = min(len(scores), index + config.neighbor_window + 1)
        for cursor in range(start, end):
            expanded.add(cursor)
    if len(expanded) > max_keep:
        expanded = set(selected)
        if len(expanded) < max_keep and config.neighbor_window > 0:
            neighbor_candidates: list[tuple[int, float, int]] = []
            for index in selected:
                start = max(0, index - config.neighbor_window)
                end = min(len(scores), index + config.neighbor_window + 1)
                for cursor in range(start, end):
                    if cursor in expanded:
                        continue
                    neighbor_candidates.append((abs(cursor - index), -scores[cursor], cursor))
            for _, _, cursor in sorted(set(neighbor_candidates)):
                if len(expanded) >= max_keep:
                    break
                expanded.add(cursor)
    return sorted(expanded)


def prune_document_content(
    content: str,
    *,
    query: str,
    focus_hint: str | None,
    config: ContextPruningConfig,
) -> tuple[str, DocumentPruningStats]:
    lines = content.splitlines()
    raw_chars = len(content)
    requested_method = _resolve_method(config, line_count=len(lines), char_count=raw_chars)
    effective_method = requested_method
    scoring_source = "heuristic"
    if not config.enabled or len(lines) <= config.min_keep_lines:
        stats: DocumentPruningStats = {
            "enabled": config.enabled,
            "method": effective_method,
            "scoring_source": scoring_source,
            "line_count_before": len(lines),
            "line_count_after": len(lines),
            "char_count_before": raw_chars,
            "char_count_after": raw_chars,
            "ratio": 1.0,
            "char_savings": _build_savings(before=raw_chars, after=raw_chars),
            "line_savings": _build_savings(before=len(lines), after=len(lines)),
        }
        return content, stats

    keywords = _unique_keywords(focus_hint or "", query, min_keywords=config.min_keywords)
    if not keywords:
        stats: DocumentPruningStats = {
            "enabled": config.enabled,
            "method": effective_method,
            "scoring_source": scoring_source,
            "line_count_before": len(lines),
            "line_count_after": len(lines),
            "char_count_before": raw_chars,
            "char_count_after": raw_chars,
            "ratio": 1.0,
            "char_savings": _build_savings(before=raw_chars, after=raw_chars),
            "line_savings": _build_savings(before=len(lines), after=len(lines)),
        }
        return content, stats

    scoring_query = (focus_hint or query).strip() or query
    scores = [_line_score(line, keywords) for line in lines]
    if requested_method == "reranker" and scoring_query.strip():
        try:
            scoring_source = _resolve_reranker_source()
            expanded_query = _expand_query(scoring_query, keywords)
            reranker_scores = _score_lines_with_reranker(
                lines,
                expanded_query,
                keywords,
            )
            window_scores = _score_line_windows_with_reranker(
                lines,
                expanded_query,
                config.reranker_window_radius,
                keywords,
            )
            scores = [
                max(heuristic_score, window_score) + reranker_score
                for heuristic_score, reranker_score, window_score in zip(
                    scores,
                    reranker_scores,
                    window_scores,
                )
            ]
        except Exception:
            effective_method = "heuristic"
            scoring_source = "heuristic"
    else:
        effective_method = "heuristic"
    seed_indexes = _seed_keyword_coverage(lines, scores, keywords)
    keep_indexes = _pick_line_indexes(
        scores,
        config,
        seed_indexes=seed_indexes,
        keyword_sets=_line_keyword_sets(lines, keywords),
    )
    if not keep_indexes:
        stats: DocumentPruningStats = {
            "enabled": config.enabled,
            "method": effective_method,
            "scoring_source": scoring_source,
            "line_count_before": len(lines),
            "line_count_after": len(lines),
            "char_count_before": raw_chars,
            "char_count_after": raw_chars,
            "ratio": 1.0,
            "char_savings": _build_savings(before=raw_chars, after=raw_chars),
            "line_savings": _build_savings(before=len(lines), after=len(lines)),
        }
        return content, stats

    kept_lines: list[str] = []
    previous = None
    for index in keep_indexes:
        if previous is not None and index - previous > 1:
            kept_lines.append("[...]")
        kept_lines.append(lines[index])
        previous = index
    pruned = "\n".join(kept_lines).strip()
    if not pruned:
        pruned = content
    stats: DocumentPruningStats = {
        "enabled": config.enabled,
        "method": effective_method,
        "scoring_source": scoring_source,
        "line_count_before": len(lines),
        "line_count_after": len(pruned.splitlines()),
        "char_count_before": raw_chars,
        "char_count_after": len(pruned),
        "ratio": round(len(pruned) / raw_chars, 4) if raw_chars else 1.0,
        "char_savings": _build_savings(before=raw_chars, after=len(pruned)),
        "line_savings": _build_savings(before=len(lines), after=len(pruned.splitlines())),
        "kept_indexes": keep_indexes,
    }
    return pruned, stats


def prune_documents(
    documents: Sequence[Document],
    *,
    query: str,
    focus_hint: str | None,
    config: ContextPruningConfig,
) -> tuple[list[Document], AggregatePruningSummary]:
    totals: dict[str, Any] = {
        "items": len(documents),
        "items_pruned": 0,
        "char_count_before": 0,
        "char_count_after": 0,
        "line_count_before": 0,
        "line_count_after": 0,
    }
    pruned_documents: list[Document] = []
    for document in documents:
        original = str(getattr(document, "page_content", "") or "")
        pruned_content, stats = prune_document_content(
            original,
            query=query,
            focus_hint=focus_hint,
            config=config,
        )
        metadata = dict(getattr(document, "metadata", {}) or {})
        metadata["context_pruning"] = stats
        metadata["context_focus_hint"] = focus_hint or query
        if pruned_content != original:
            totals["items_pruned"] += 1
        totals["char_count_before"] += stats["char_count_before"]
        totals["char_count_after"] += stats["char_count_after"]
        totals["line_count_before"] += stats["line_count_before"]
        totals["line_count_after"] += stats["line_count_after"]
        pruned_documents.append(Document(page_content=pruned_content, metadata=metadata))

    before = totals["char_count_before"]
    totals["ratio"] = round(totals["char_count_after"] / before, 4) if before else 1.0
    totals["focus_hint"] = focus_hint or query
    totals["enabled"] = config.enabled
    methods = {
        str((getattr(document, "metadata", {}) or {}).get("context_pruning", {}).get("method") or "")
        for document in pruned_documents
    }
    methods.discard("")
    totals["method"] = methods.pop() if len(methods) == 1 else (config.method if methods else config.method)
    sources = {
        str((getattr(document, "metadata", {}) or {}).get("context_pruning", {}).get("scoring_source") or "")
        for document in pruned_documents
    }
    sources.discard("")
    if len(sources) == 1:
        totals["scoring_source"] = sources.pop()
    else:
        totals["scoring_source"] = "mixed" if sources else "heuristic"
    totals["char_savings"] = _build_savings(
        before=totals["char_count_before"],
        after=totals["char_count_after"],
    )
    totals["line_savings"] = _build_savings(
        before=totals["line_count_before"],
        after=totals["line_count_after"],
    )
    summary: AggregatePruningSummary = totals
    return pruned_documents, summary


def build_candidate_pruning_trace(summary: AggregatePruningSummary) -> CandidatePruningTrace:
    return {
        "focus_hint": summary["focus_hint"],
        "method": summary["method"],
        "scoring_source": summary["scoring_source"],
        "items": summary["items"],
        "items_pruned": summary["items_pruned"],
        "char_savings": summary["char_savings"],
        "line_savings": summary["line_savings"],
    }


def build_prompt_pruning_trace(
    *,
    focus_hint: str,
    method: str,
    scoring_source: str,
    docs: AggregatePruningSummary,
    memories: AggregatePruningSummary,
) -> PromptPruningTrace:
    return {
        "focus_hint": focus_hint,
        "method": method,
        "scoring_source": scoring_source,
        "docs": {
            "method": docs["method"],
            "scoring_source": docs["scoring_source"],
            "char_savings": docs["char_savings"],
            "line_savings": docs["line_savings"],
            "items": docs["items"],
            "items_pruned": docs["items_pruned"],
        },
        "memories": {
            "method": memories["method"],
            "scoring_source": memories["scoring_source"],
            "char_savings": memories["char_savings"],
            "line_savings": memories["line_savings"],
            "items": memories["items"],
            "items_pruned": memories["items_pruned"],
        },
    }
