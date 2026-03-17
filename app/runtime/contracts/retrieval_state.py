from __future__ import annotations

from typing import Any, Mapping

from langchain_core.documents import Document
from typing_extensions import TypedDict

RETRIEVAL_ARTIFACT_KEYS = (
    "retrieved_docs",
    "retrieved_docs_candidates",
    "retrieved_docs_candidates_raw",
    "retrieved_memories",
    "retrieved_profile_items",
    "citations",
)


class RetrievalArtifactsPayload(TypedDict, total=False):
    retrieved_docs: list[Document]
    retrieved_docs_candidates: list[Document]
    retrieved_docs_candidates_raw: list[Document]
    retrieved_memories: list[Document]
    retrieved_profile_items: list[dict[str, Any]]
    citations: list[dict[str, Any]]


def build_retrieval_artifacts_payload(
    *,
    current: Mapping[str, Any] | None = None,
    retrieved_docs: list[Document] | None = None,
    retrieved_docs_candidates: list[Document] | None = None,
    retrieved_docs_candidates_raw: list[Document] | None = None,
    retrieved_memories: list[Document] | None = None,
    retrieved_profile_items: list[dict[str, Any]] | None = None,
    citations: list[dict[str, Any]] | None = None,
    clear_all: bool = False,
) -> RetrievalArtifactsPayload:
    payload: RetrievalArtifactsPayload = {}
    if not clear_all and current:
        for key in RETRIEVAL_ARTIFACT_KEYS:
            value = current.get(key)
            if isinstance(value, list):
                payload[key] = value

    if retrieved_docs is not None:
        payload["retrieved_docs"] = retrieved_docs
    if retrieved_docs_candidates is not None:
        payload["retrieved_docs_candidates"] = retrieved_docs_candidates
    if retrieved_docs_candidates_raw is not None:
        payload["retrieved_docs_candidates_raw"] = retrieved_docs_candidates_raw
    if retrieved_memories is not None:
        payload["retrieved_memories"] = retrieved_memories
    if retrieved_profile_items is not None:
        payload["retrieved_profile_items"] = retrieved_profile_items
    if citations is not None:
        payload["citations"] = citations
    return payload


def clear_retrieval_artifacts_inplace(target: dict[str, Any]) -> None:
    for key in RETRIEVAL_ARTIFACT_KEYS:
        target.pop(key, None)
