from __future__ import annotations

import re
from collections import Counter
from typing import Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]


def _ngrams(tokens: Sequence[str], size: int) -> set[str]:
    if len(tokens) < size:
        return set()
    return {
        " ".join(tokens[index : index + size])
        for index in range(len(tokens) - size + 1)
    }


def score_text_candidates(query: str, candidates: Sequence[str]) -> list[float]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return [0.0 for _ in candidates]

    query_counts = Counter(query_tokens)
    query_token_set = set(query_tokens)
    query_bigrams = _ngrams(query_tokens, 2)
    query_trigrams = _ngrams(query_tokens, 3)

    scores: list[float] = []
    for candidate in candidates:
        candidate_tokens = tokenize(candidate)
        if not candidate_tokens:
            scores.append(0.0)
            continue
        token_set = set(candidate_tokens)
        bigrams = _ngrams(candidate_tokens, 2)
        trigrams = _ngrams(candidate_tokens, 3)
        weighted_overlap = sum(
            1.0 / query_counts[token]
            for token in token_set & query_token_set
        )
        phrase_bonus = (len(bigrams & query_bigrams) * 1.5) + (
            len(trigrams & query_trigrams) * 2.0
        )
        length_penalty = min(len(candidate_tokens) / 200.0, 0.25)
        score = (
            weighted_overlap / max(len(query_token_set), 1)
            + phrase_bonus
            - length_penalty
        )
        scores.append(max(0.0, float(score)))
    return scores


def rank_text_candidates(
    query: str,
    candidates: Sequence[str],
    *,
    top_k: int,
) -> list[tuple[int, float]]:
    scores = score_text_candidates(query, candidates)
    ranked = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return [(index, float(score)) for index, score in ranked[: max(0, int(top_k))]]
