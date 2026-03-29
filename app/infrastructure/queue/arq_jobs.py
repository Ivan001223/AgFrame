from __future__ import annotations

import re
import time
from typing import Any

import anyio

from app.harness.contracts.run import HarnessTaskType
from app.harness.persistence.stores import HarnessModelProviderStore
from app.harness.runtime.checkpoint_adapter import CheckpointAdapter
from app.harness.runtime.run_service import build_run_service
from app.harness.runtime.verification_service import VerificationService
from app.infrastructure.queue.redis_client import (
    append_task_incident,
    get_task,
    release_task_operation,
    update_task,
)
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.infrastructure.utils.logging import bind_logger, get_logger
from app.memory.long_term.user_memory_engine import UserMemoryEngine
from app.runtime.graph.orchestration_graph import (
    OutputGuardrailTrip,
    build_orchestration_execution_plan,
    compile_orchestration_graph,
    invoke_orchestration_step,
    review_orchestration_output,
)
from app.runtime.graph.resume_service import GraphResumeService
from app.runtime.llm.provider_registry import ModelProviderRegistry
from app.server.session_history import persist_session_messages
from app.skills.research.enhanced_search import (
    enhanced_search_response,
    enhanced_web_search,
    fetch_browser_previews,
)
from app.skills.rag.rag_engine import get_rag_engine

_log = get_logger("task_queue.arq_jobs")


def _maybe_call(service: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(service, method_name, None)
    if callable(method):
        return method(*args, **kwargs)
    return None


def _normalize_ingest_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if result is True:
        return {"ok": True}
    return {
        "ok": False,
        "error_code": "ingest_returned_false",
        "error_message": "add_knowledge_base 返回 False",
    }


def _normalize_skill_key(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip()).strip("_")


def _filter_graph_for_execution(graph: dict[str, Any], selected_agent_ids: list[str]) -> dict[str, Any]:
    if not selected_agent_ids:
        return graph
    selected = {str(agent_id) for agent_id in selected_agent_ids}
    filtered = dict(graph)
    filtered["agents"] = [
        agent
        for agent in graph.get("agents") or []
        if isinstance(agent, dict) and str(agent.get("agent_id") or "") in selected
    ]
    filtered["edges"] = [
        edge
        for edge in graph.get("edges") or []
        if isinstance(edge, dict)
        and str(edge.get("source_agent_id") or "") in selected
        and str(edge.get("target_agent_id") or "") in selected
    ]
    return filtered


def _build_provider_registry(*, user_id: str | None) -> ModelProviderRegistry:
    registry = ModelProviderRegistry()
    rows = HarnessModelProviderStore().list_providers(user_id=user_id)
    registry.load_from_store_rows(rows)
    return registry


def _serialize_orchestration_resume_state(state: dict[str, Any], next_step_index: int) -> dict[str, Any]:
    return {
        "next_step_index": next_step_index,
        "state": {
            "task": str(state.get("task") or ""),
            "agent_outputs": dict(state.get("agent_outputs") or {}),
            "output_artifacts": dict(state.get("output_artifacts") or {}),
            "current_agent": str(state.get("current_agent") or ""),
            "loop_index": int(state.get("loop_index") or 0),
            "errors": list(state.get("errors") or []),
        },
    }


def _window_text_for_stream_review(
    text: str,
    *,
    window_chars: int,
    end_char: int | None = None,
) -> tuple[str, int, int]:
    content = str(text or "")
    if not content:
        return "", 0, 0
    normalized_window = max(int(window_chars or 0), 1)
    end = min(len(content), max(int(end_char if end_char is not None else len(content)), 0))
    start = max(0, end - normalized_window)
    return content[start:end], start, end


async def _review_incremental_stream_output(
    *,
    review_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    task: str,
    agent_conf: dict[str, Any],
    stream_review_state: dict[str, int],
    chunk_text: str,
    accumulated_text: str,
    chunk_index: int,
) -> dict[str, Any] | None:
    review_cursor = int(stream_review_state.get("last_reviewed_chars") or 0)
    trigger_chars = max(int(review_config.get("stream_review_trigger_chars") or 24), 1)
    window_chars = int(review_config.get("stream_review_window_chars") or 1200)
    check_count = int(stream_review_state.get("check_count") or 0)
    should_force_trailing_check = bool(str(chunk_text or "").strip())

    while review_cursor < len(accumulated_text):
        remaining = len(accumulated_text) - review_cursor
        if remaining < trigger_chars and review_cursor > 0 and not should_force_trailing_check:
            break
        review_end = min(len(accumulated_text), review_cursor + trigger_chars)
        window_text, start_char, end_char = _window_text_for_stream_review(
            accumulated_text,
            window_chars=window_chars,
            end_char=review_end,
        )
        if not window_text.strip():
            stream_review_state["last_reviewed_chars"] = review_end
            review_cursor = review_end
            should_force_trailing_check = False
            continue
        check_count += 1
        review_result = await review_orchestration_output(
            review_config=review_config,
            provider_config=provider_config,
            default_timeout=default_timeout,
            provider_registry=provider_registry,
            task=task,
            agent_id=f"{str(agent_conf.get('cluster_agent_id') or agent_conf.get('agent_id') or '')}::stream::{check_count}",
            agent_name=f"{str(agent_conf.get('cluster_name') or agent_conf.get('name') or 'agent')} live stream guard",
            output=window_text,
        )
        if not bool(review_result.get("approved")):
            return {
                "blocked": True,
                "review_stage": "agent_output_stream",
                "review_output": str(review_result.get("review_output") or ""),
                "chunk_index": int(chunk_index),
                "check_count": int(check_count),
                "segment_index": int(check_count) - 1,
                "segment_count": int(check_count),
                "segment_start_char": int(start_char),
                "segment_end_char": int(end_char),
                "last_reviewed_char": int(end_char),
                "review_cursor_char": int(end_char),
                "segment_preview": window_text[-400:],
                "partial_output": accumulated_text[:end_char],
            }
        stream_review_state["last_reviewed_chars"] = review_end
        stream_review_state["check_count"] = check_count
        review_cursor = review_end
        should_force_trailing_check = False

    return None


def _restore_orchestration_state(task: str, resume_payload: dict[str, Any] | None) -> tuple[dict[str, Any], int]:
    payload = dict(resume_payload or {})
    state_payload = dict(payload.get("state") or {})
    next_step_index = int(payload.get("next_step_index") or 0)
    restored = {
        "messages": [],
        "task": str(state_payload.get("task") or task),
        "agent_outputs": dict(state_payload.get("agent_outputs") or {}),
        "output_artifacts": dict(state_payload.get("output_artifacts") or {}),
        "current_agent": str(state_payload.get("current_agent") or ""),
        "loop_index": int(state_payload.get("loop_index") or 0),
        "errors": list(state_payload.get("errors") or []),
    }
    continuation = payload.get("continuation")
    if isinstance(continuation, dict) and continuation:
        restored["continuation"] = dict(continuation)
    return restored, next_step_index


def _build_cluster_research_plan(*, task: str, artifact: dict[str, Any]) -> dict[str, list[str]]:
    cluster_name = str(artifact.get("cluster_name") or artifact.get("cluster_agent_id") or "cluster").strip()
    winning_strategy = str(artifact.get("winning_strategy") or artifact.get("winning_vote") or "").strip()
    next_step = str(artifact.get("next_step") or "").strip()
    base_task = str(task or "").strip()
    paper_prompts = [
        f"{base_task} {winning_strategy} benchmark evaluation research paper".strip(),
        f"{base_task} {next_step or cluster_name} recent advances arxiv".strip(),
    ]
    web_prompts = [
        f"{base_task} {winning_strategy} latest progress".strip(),
        f"{base_task} {next_step or cluster_name} latest industry developments".strip(),
    ]

    def _dedupe(values: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        items: list[str] = []
        for value in values:
            normalized = " ".join(str(value or "").split())
            if normalized and normalized not in seen:
                seen.add(normalized)
                items.append(normalized)
        return items[:limit]

    paper_queries = _dedupe(paper_prompts, 2)
    web_queries = _dedupe(web_prompts, 2)
    return {
        "paper_queries": paper_queries,
        "web_queries": web_queries,
        "queries": paper_queries + web_queries,
    }


def _build_cluster_research_queries(*, task: str, artifact: dict[str, Any]) -> list[str]:
    return _build_cluster_research_plan(task=task, artifact=artifact)["queries"]


def _parse_search_result_items(result: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    pattern = re.compile(r"^\d+\.\s+\[(?P<title>[^\]]+)\]\((?P<url>[^)]+)\)\s*$")
    lines = [line.rstrip() for line in str(result or "").splitlines()]
    current: dict[str, str] | None = None
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            if current:
                items.append(current)
            current = {
                "title": match.group("title").strip(),
                "url": match.group("url").strip(),
                "snippet": "",
            }
            continue
        if current and line.strip():
            snippet = line.strip().removeprefix("-").strip()
            current["snippet"] = f"{current['snippet']} {snippet}".strip()
    if current:
        items.append(current)
    return items


def _search_response_to_items(response: Any) -> list[dict[str, str]]:
    results = getattr(response, "results", None)
    if not isinstance(results, list):
        return []
    items: list[dict[str, str]] = []
    for result in results:
        title = str(getattr(result, "title", "") or "").strip()
        url = str(getattr(result, "url", "") or "").strip()
        snippet = str(getattr(result, "snippet", "") or "").strip()
        provider = str(getattr(result, "provider", "") or "").strip()
        if not url:
            continue
        items.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "provider": provider,
            }
        )
    return items


def _chunk_output_for_review(
    output: str,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    text = str(output or "")
    normalized_chunk = max(int(chunk_chars or 0), 1)
    normalized_overlap = max(0, min(int(overlap_chars or 0), normalized_chunk - 1))
    if not text.strip():
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    total_length = len(text)
    while start < total_length:
        end = min(total_length, start + normalized_chunk)
        chunks.append(
            {
                "segment_index": len(chunks),
                "start_char": start,
                "end_char": end,
                "content": text[start:end],
            }
        )
        if end >= total_length:
            break
        start = max(end - normalized_overlap, start + 1)
    return chunks


async def _run_segmented_output_review(
    *,
    review_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    task: str,
    agent_id: str,
    agent_name: str,
    output: str,
) -> dict[str, Any]:
    review_enabled = bool(review_config.get("enabled", True))
    pipeline_enabled = bool(review_config.get("pipeline_review_enabled", True))
    if not review_enabled or not pipeline_enabled:
        return {
            "approved": True,
            "mode": "disabled",
            "segments_reviewed": 0,
            "segment_count": 0,
            "blocked_segment": None,
            "results": [],
        }

    chunks = _chunk_output_for_review(
        output,
        chunk_chars=int(review_config.get("pipeline_chunk_chars") or 1200),
        overlap_chars=int(review_config.get("pipeline_chunk_overlap_chars") or 150),
    )
    if len(chunks) <= 1:
        return {
            "approved": True,
            "mode": "single_segment_bypass",
            "segments_reviewed": len(chunks),
            "segment_count": len(chunks),
            "blocked_segment": None,
            "results": [],
        }

    results: list[dict[str, Any]] = []
    total = len(chunks)
    for chunk in chunks:
        segment_index = int(chunk["segment_index"])
        review_result = await review_orchestration_output(
            review_config=review_config,
            provider_config=provider_config,
            default_timeout=default_timeout,
            provider_registry=provider_registry,
            task=task,
            agent_id=f"{agent_id}::segment::{segment_index + 1}",
            agent_name=f"{agent_name} segment {segment_index + 1}/{total}",
            output=str(chunk["content"] or ""),
        )
        chunk_result = {
            **chunk,
            "approved": bool(review_result.get("approved")),
            "review_output": str(review_result.get("review_output") or ""),
            "segment_count": total,
        }
        results.append(chunk_result)
        if not chunk_result["approved"]:
            return {
                "approved": False,
                "mode": "segmented",
                "segments_reviewed": len(results),
                "segment_count": total,
                "blocked_segment": chunk_result,
                "results": results,
            }

    return {
        "approved": True,
        "mode": "segmented",
        "segments_reviewed": len(results),
        "segment_count": total,
        "blocked_segment": None,
        "results": results,
    }


def _classify_research_source(item: dict[str, str]) -> str:
    haystack = f"{item.get('title', '')} {item.get('url', '')} {item.get('snippet', '')}".lower()
    if any(token in haystack for token in ["arxiv", "paper", "doi", "ieee", "acm", "springer"]):
        return "paper"
    return "web"


def _summarize_research_payload(searches: list[dict[str, Any]], *, browser_previews: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    sources: list[dict[str, str]] = []
    citations: list[dict[str, str]] = []
    latest_progress: list[str] = []
    papers: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for search in searches:
        query = str(search.get("query") or "").strip()
        for item in search.get("items") or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            source_kind = _classify_research_source(item)
            normalized = {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "snippet": str(item.get("snippet") or "").strip(),
                "query": query,
                "kind": source_kind,
            }
            sources.append(normalized)
            citations.append(
                {
                    "label": normalized["title"] or normalized["url"],
                    "url": normalized["url"],
                    "kind": source_kind,
                }
            )
            if normalized["snippet"]:
                latest_progress.append(normalized["snippet"])
            if source_kind == "paper":
                papers.append(normalized)

    latest_progress = latest_progress[:5]
    papers = papers[:5]
    normalized_previews = [
        {
            "url": str(item.get("url") or ""),
            "final_url": str(item.get("final_url") or ""),
            "title": str(item.get("title") or ""),
            "description": str(item.get("description") or ""),
            "status_code": item.get("status_code"),
            "content_type": str(item.get("content_type") or "") or None,
        }
        for item in (browser_previews or [])
        if isinstance(item, dict) and str(item.get("url") or "").strip()
    ][:3]
    digest_sections: list[str] = []
    if papers:
        digest_sections.append(
            "Paper-First Findings:\n"
            + "\n".join(f"- {item['title']} ({item['url']})" for item in papers)
        )
    if latest_progress:
        digest_sections.append("Latest Progress:\n" + "\n".join(f"- {item}" for item in latest_progress))
    if normalized_previews:
        digest_sections.append(
            "Browser Previews:\n"
            + "\n".join(
                f"- {item['title'] or item['url']}: {item['description'] or 'No preview description.'}"
                for item in normalized_previews
            )
        )
    if not digest_sections and sources:
        digest_sections.append(
            "Sources:\n"
            + "\n".join(f"- {item['title'] or item['url']} ({item['url']})" for item in sources[:5])
        )
    return {
        "sources": sources[:8],
        "citations": citations[:8],
        "latest_progress": latest_progress,
        "papers": papers,
        "browser_previews": normalized_previews,
        "digest": "\n\n".join(digest_sections).strip(),
    }


async def _run_cluster_auto_research(
    *,
    task: str,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    plan = _build_cluster_research_plan(task=task, artifact=artifact)
    searches: list[dict[str, Any]] = []
    provider_runs: list[dict[str, Any]] = []

    for query in plan["paper_queries"]:
        response = await enhanced_search_response(
            query=query,
            provider="arxiv",
            use_cache=True,
            max_results=4,
        )
        items = _search_response_to_items(response)
        searches.append(
            {
                "query": query,
                "provider": "arxiv",
                "items": items,
                "result_count": len(items),
            }
        )
        provider_runs.append(
            {
                "provider": "arxiv",
                "query": query,
                "result_count": len(items),
                "cached": bool(getattr(response, "cached", False)),
            }
        )

    for query in plan["web_queries"]:
        result = await enhanced_web_search(
            query=query,
            provider=None,
            use_cache=True,
            max_results=5,
        )
        items = _parse_search_result_items(result)
        searches.append(
            {
                "query": query,
                "provider": "web",
                "result": result,
                "items": items,
                "result_count": len(items),
            }
        )
        provider_runs.append(
            {
                "provider": "web",
                "query": query,
                "result_count": len(items),
                "cached": True,
            }
        )

    preview_urls: list[str] = []
    for search in searches:
        prioritized_items = search.get("items") or []
        if not isinstance(prioritized_items, list):
            continue
        for item in prioritized_items:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url or url in preview_urls:
                continue
            preview_urls.append(url)
            if len(preview_urls) >= 3:
                break
        if len(preview_urls) >= 3:
            break

    browser_previews = await anyio.to_thread.run_sync(
        lambda: [
            preview.__dict__
            for preview in fetch_browser_previews(preview_urls, max_previews=3, timeout_seconds=8)
        ]
    ) if preview_urls else []

    summary = _summarize_research_payload(searches, browser_previews=browser_previews)
    return {
        "research_mode": "paper_first_browser_preview",
        "queries": plan["queries"],
        "paper_queries": plan["paper_queries"],
        "web_queries": plan["web_queries"],
        "result_count": len(searches),
        "searches": searches,
        "provider_runs": provider_runs,
        "sources": summary["sources"],
        "citations": summary["citations"],
        "latest_progress": summary["latest_progress"],
        "papers": summary["papers"],
        "browser_previews": summary["browser_previews"],
        "digest": summary["digest"],
    }


async def _review_cluster_research_evidence(
    *,
    review_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    task: str,
    agent_id: str,
    agent_name: str,
    research_payload: dict[str, Any],
) -> dict[str, object]:
    digest = str(research_payload.get("digest") or "").strip()
    latest_progress = list(research_payload.get("latest_progress") or [])
    papers = list(research_payload.get("papers") or [])
    browser_previews = list(research_payload.get("browser_previews") or [])
    evidence_lines = [
        f"Task: {task}",
        f"Cluster node id: {agent_id}",
        f"Cluster node name: {agent_name}",
        "",
        "Structured research evidence:",
        f"- Queries: {', '.join(str(query) for query in research_payload.get('queries') or []) or 'none'}",
        f"- Latest progress count: {len(latest_progress)}",
        f"- Paper count: {len(papers)}",
        f"- Browser preview count: {len(browser_previews)}",
    ]
    if latest_progress:
        evidence_lines.append("Latest progress:")
        evidence_lines.extend(f"- {str(item)}" for item in latest_progress[:5])
    if papers:
        evidence_lines.append("Papers:")
        evidence_lines.extend(
            f"- {str(item.get('title') or 'Untitled')} ({str(item.get('url') or 'no-url')})"
            for item in papers[:5]
            if isinstance(item, dict)
        )
    if browser_previews:
        evidence_lines.append("Browser previews:")
        evidence_lines.extend(
            f"- {str(item.get('title') or item.get('url') or 'Untitled')}: {str(item.get('description') or 'no description')}"
            for item in browser_previews[:3]
            if isinstance(item, dict)
        )
    if digest:
        evidence_lines.extend(["", "Digest:", digest])
    evidence_text = "\n".join(evidence_lines).strip()
    return await review_orchestration_output(
        review_config=review_config,
        provider_config=provider_config,
        default_timeout=default_timeout,
        provider_registry=provider_registry,
        task=task,
        agent_id=f"{agent_id}::research",
        agent_name=f"{agent_name} research evidence",
        output=evidence_text,
    )


def _persist_cluster_research_memory(
    *,
    user_id: str | None,
    run_id: str,
    cluster_id: str,
    research_digest: str,
) -> dict[str, Any]:
    if not user_id or not research_digest.strip():
        return {"stored": False, "reason": "missing_user_or_digest"}
    if not ensure_schema_if_possible():
        return {"stored": False, "reason": "schema_unavailable"}
    try:
        UserMemoryEngine().add_chat_summary(
            user_id=user_id,
            session_id=f"harness_run:{run_id}:{cluster_id}",
            summary_text=research_digest,
        )
        return {"stored": True, "reason": None}
    except Exception as exc:
        return {"stored": False, "reason": str(exc)}


async def ingest_pdf(
    ctx: dict[str, Any], task_id: str, file_path: str, user_id: str = None
) -> bool:
    logger = bind_logger(_log, session_id=task_id, node="ingest_pdf")
    started_at = int(time.time())
    existing_task = await get_task(task_id)
    operation_key = str(existing_task.get("operation_key") or "")
    await update_task(
        task_id,
        {
            "status": "running",
            "progress": 1,
            "step": "start",
            "started_at": started_at,
            "message": "开始处理",
            "error": "",
            "user_id": user_id or "unknown",
        },
    )

    try:
        await update_task(
            task_id, {"progress": 10, "step": "validating", "message": "校验文件"}
        )
        await update_task(
            task_id, {"progress": 25, "step": "ingest", "message": "开始摄取"}
        )
        await update_task(
            task_id, {"progress": 60, "step": "indexing", "message": "构建索引"}
        )
        await update_task(
            task_id, {"progress": 85, "step": "finalizing", "message": "写入结果"}
        )
        # 传递 user_id 给 RAG 引擎
        result = await anyio.to_thread.run_sync(
            lambda: _normalize_ingest_result(
                get_rag_engine().add_knowledge_base(file_path, user_id=user_id)
            )
        )
        finished_at = int(time.time())
        if result.get("ok"):
            await update_task(
                task_id,
                {
                    "status": "succeeded",
                    "progress": 100,
                    "step": "done",
                    "finished_at": finished_at,
                    "message": "处理完成",
                    "retryable": "false",
                    "result_stage": str(result.get("stage") or ""),
                },
            )
            logger.info("task succeeded file_path=%s", file_path)
            await release_task_operation(operation_key, expected_task_id=task_id)
            return True

        await update_task(
            task_id,
            {
                "status": "failed",
                "progress": 100,
                "step": "failed",
                "finished_at": finished_at,
                "error": str(result.get("error_message") or "add_knowledge_base 返回 False"),
                "error_code": str(result.get("error_code") or "ingest_returned_false"),
                "result_stage": str(result.get("stage") or ""),
                "retryable": "true",
            },
        )
        await append_task_incident(
            {
                "task_id": task_id,
                "user_id": user_id or "unknown",
                "status": "failed",
                "error_code": str(result.get("error_code") or "ingest_returned_false"),
                "error_message": str(result.get("error_message") or "add_knowledge_base 返回 False"),
                "stage": str(result.get("stage") or ""),
                "file_path": file_path,
                "timestamp": finished_at,
            }
        )
        logger.info(
            "task failed file_path=%s error_code=%s stage=%s",
            file_path,
            result.get("error_code"),
            result.get("stage"),
        )
        await release_task_operation(operation_key, expected_task_id=task_id)
        return False
    except Exception as e:
        finished_at = int(time.time())
        await update_task(
            task_id,
            {
                "status": "failed",
                "progress": 100,
                "step": "exception",
                "finished_at": finished_at,
                "error": str(e),
                "retryable": "true",
            },
        )
        await append_task_incident(
            {
                "task_id": task_id,
                "user_id": user_id or "unknown",
                "status": "failed",
                "error_code": "task_exception",
                "error_message": str(e),
                "stage": "exception",
                "file_path": file_path,
                "timestamp": finished_at,
            }
        )
        logger.exception("task exception file_path=%s", file_path)
        await release_task_operation(operation_key, expected_task_id=task_id)
        return False


async def run_harness_task(ctx: dict[str, Any], run_id: str) -> bool:
    service = build_run_service()
    run = service.get_run(run_id)
    if not run:
        return False

    verification_service = VerificationService()
    task_type = str(run.get("task_type") or "")
    if task_type != HarnessTaskType.SESSION_RESUME_APPROVAL.value:
        _maybe_call(service, "mark_running", run_id)

    try:
        if task_type == HarnessTaskType.DOCUMENT_INGEST.value:
            _maybe_call(service, "set_current_step", run_id, "ingest_document")
            input_json = run.get("input_json") or {}
            file_path = str(input_json.get("file_path") or "")
            user_id = str(run.get("user_id") or "") or None

            result = await anyio.to_thread.run_sync(
                lambda: _normalize_ingest_result(get_rag_engine().add_knowledge_base(file_path, user_id=user_id))
            )
            verification = verification_service.build_document_ingest_result(
                ok=bool(result.get("ok")),
                stage=str(result.get("stage") or "") or None,
                error_code=str(result.get("error_code") or "") or None,
                error_message=str(result.get("error_message") or "") or None,
            )
            service.complete_with_verification(run_id, verification)
            return bool(result.get("ok"))

        if task_type == HarnessTaskType.AGENT_ORCHESTRATION.value:
            input_json = run.get("input_json") or {}
            metadata_json = run.get("metadata_json") if isinstance(run, dict) else {}
            metadata_json = metadata_json if isinstance(metadata_json, dict) else {}
            graph = input_json.get("graph") if isinstance(input_json, dict) else {}
            graph = graph if isinstance(graph, dict) else {}
            agents = graph.get("agents") if isinstance(graph.get("agents"), list) else []
            selected_agent_ids = input_json.get("selected_agent_ids") if isinstance(input_json, dict) else []
            selected_agent_ids = selected_agent_ids if isinstance(selected_agent_ids, list) else []
            user_id = str(run.get("user_id") or "") or None
            loop_count = int(input_json.get("loop_count") or 1) if isinstance(input_json, dict) else 1
            review_agent = graph.get("review_agent") if isinstance(graph.get("review_agent"), dict) else {}
            review_agent_enabled = bool(review_agent.get("enabled", True))
            loaded_skill_ids = {
                _normalize_skill_key(str(item.get("skill_id") or ""))
                for item in graph.get("skill_pool") or []
                if isinstance(item, dict)
            }

            active_agents = [
                agent
                for agent in agents
                if isinstance(agent, dict)
                and (
                    not selected_agent_ids
                    or str(agent.get("agent_id") or "") in {str(agent_id) for agent_id in selected_agent_ids}
                )
            ]
            active_agent_ids = [str(agent.get("agent_id") or "") for agent in active_agents]
            if not active_agents:
                verification = verification_service.build_agent_orchestration_result(
                    ok=False,
                    active_agent_ids=[],
                    blocked_agents=[],
                    loop_count=loop_count,
                    review_agent_enabled=review_agent_enabled,
                    error_code="missing_active_agents",
                    error_message="no agents selected for orchestration",
                )
                service.complete_with_verification(run_id, verification)
                return False

            _maybe_call(service, "set_current_step", run_id, "prepare_orchestration")
            _maybe_call(
                service,
                "record_event",
                run_id,
                event_type="orchestration.review_agent_attached",
                details={
                    "enabled": review_agent_enabled,
                    "hidden": bool(review_agent.get("hidden", True)),
                    "name": str(review_agent.get("name") or "Compliance reviewer"),
                    "model": str(review_agent.get("model") or "gpt-5.1-codex-mini"),
                },
            )

            blocked_agents: list[dict[str, object]] = []
            for agent in active_agents:
                agent_id = str(agent.get("agent_id") or "")
                required_skills = [
                    _normalize_skill_key(str(skill_id))
                    for skill_id in (agent.get("skill_ids") or [])
                    if str(skill_id).strip()
                ]
                missing_skills = [skill_id for skill_id in required_skills if skill_id not in loaded_skill_ids]
                _maybe_call(
                    service,
                    "record_event",
                    run_id,
                    event_type="orchestration.agent_ready",
                    details={
                        "agent_id": agent_id,
                        "agent_name": str(agent.get("name") or agent_id),
                        "role": str(agent.get("role") or "specialist"),
                        "skill_ids": required_skills,
                        "missing_skills": missing_skills,
                    },
                )
                if missing_skills:
                    blocked_agents.append(
                        {
                            "agent_id": agent_id,
                            "missing_skills": missing_skills,
                        }
                    )

            if blocked_agents:
                verification = verification_service.build_agent_orchestration_result(
                    ok=False,
                    active_agent_ids=active_agent_ids,
                    blocked_agents=blocked_agents,
                    loop_count=loop_count,
                    review_agent_enabled=review_agent_enabled,
                    error_code="missing_skill_approval",
                    error_message="one or more agents are blocked by missing skills",
                )
                service.complete_with_verification(run_id, verification)
                return False

            task_str = str(input_json.get("task") or "") or str(input_json.get("project_name") or "")
            default_timeout = int(input_json.get("timeout_seconds") or 60)
            recovery_mode = str(metadata_json.get("review_recovery_mode") or "").strip() or None
            execution_graph = _filter_graph_for_execution(graph, selected_agent_ids)
            provider_registry = _build_provider_registry(user_id=user_id)
            ordered_agents, provider_config, review_config = build_orchestration_execution_plan(execution_graph)
            review_enabled = bool(review_config.get("enabled", True))
            resume_payload = input_json.get("orchestration_resume") if isinstance(input_json, dict) else None
            approved_resume = bool(isinstance(resume_payload, dict) and resume_payload.get("review_decision") == "approved")
            resume_continue_mode = (
                str(resume_payload.get("continue_mode") or "").strip()
                if isinstance(resume_payload, dict)
                else ""
            )
            stream_continuation_pending = approved_resume and resume_continue_mode == "accept_partial_stream_output"
            initial_state, next_step_index = _restore_orchestration_state(task_str, resume_payload if isinstance(resume_payload, dict) else None)
            if approved_resume and not stream_continuation_pending:
                input_json = dict(input_json)
                input_json.pop("orchestration_resume", None)
                _maybe_call(service, "update_run_input_json", run_id, input_json)
            elif stream_continuation_pending:
                continuation = initial_state.get("continuation") if isinstance(initial_state, dict) else None
                _maybe_call(
                    service,
                    "record_event",
                    run_id,
                    event_type="orchestration.stream_continuation_resumed",
                    details={
                        "agent_id": str((continuation or {}).get("agent_id") or ""),
                        "agent_name": str((continuation or {}).get("agent_name") or ""),
                        "partial_length": len(str((continuation or {}).get("partial_output") or "")),
                        "next_step_index": next_step_index,
                    },
                )
                _maybe_call(
                    service,
                    "patch_runtime_state",
                    run_id,
                    continuation={
                        "enabled": True,
                        "mode": "continue_with_partial_stream_output",
                        "status": "resumed",
                        "agent_id": str((continuation or {}).get("agent_id") or ""),
                        "agent_name": str((continuation or {}).get("agent_name") or ""),
                        "step_index": int(next_step_index),
                        "prefix_length": len(str((continuation or {}).get("partial_output") or "")),
                        "resumed_at": int(time.time() * 1000),
                    },
                )
            execution_steps = [
                {"loop_number": loop_number, "agent_config": agent_conf}
                for loop_number in range(1, max(loop_count, 1) + 1)
                for agent_conf in ordered_agents
            ]

            _maybe_call(service, "set_current_step", run_id, "executing_graph")
            final_state = initial_state
            for step_index in range(next_step_index, len(execution_steps)):
                step = execution_steps[step_index]
                agent_conf = step["agent_config"]
                final_state["loop_index"] = int(step["loop_number"]) - 1
                previous_state = {
                    "messages": [],
                    "task": str(final_state.get("task") or task_str),
                    "agent_outputs": dict(final_state.get("agent_outputs") or {}),
                    "output_artifacts": dict(final_state.get("output_artifacts") or {}),
                    "current_agent": str(final_state.get("current_agent") or ""),
                    "loop_index": int(final_state.get("loop_index") or 0),
                    "errors": list(final_state.get("errors") or []),
                }
                stream_review_state = {
                    "last_reviewed_chars": 0,
                    "check_count": 0,
                }

                async def _review_stream_chunk(chunk_text: str, accumulated_text: str, chunk_index: int):
                    if not review_enabled:
                        return None
                    blocked = await _review_incremental_stream_output(
                        review_config=review_config,
                        provider_config=provider_config,
                        default_timeout=default_timeout,
                        provider_registry=provider_registry,
                        task=task_str,
                        agent_conf=agent_conf,
                        stream_review_state=stream_review_state,
                        chunk_text=chunk_text,
                        accumulated_text=accumulated_text,
                        chunk_index=chunk_index,
                    )
                    if blocked:
                        _maybe_call(
                            service,
                            "patch_runtime_state",
                            run_id,
                            review={
                                "stage": "agent_output_stream",
                                "status": "pending",
                                "agent_id": str(agent_conf.get("cluster_agent_id") or agent_conf.get("agent_id") or ""),
                                "agent_name": str(agent_conf.get("cluster_name") or agent_conf.get("name") or "agent"),
                                "review_output": str(blocked.get("review_output") or ""),
                                "check_count": int(blocked.get("check_count") or 0),
                                "segment_index": int(blocked.get("segment_index") or 0),
                                "segment_count": int(blocked.get("segment_count") or 0),
                                "segment_start_char": int(blocked.get("segment_start_char") or 0),
                                "segment_end_char": int(blocked.get("segment_end_char") or 0),
                                "last_reviewed_char": int(blocked.get("last_reviewed_char") or 0),
                            },
                            continuation={
                                "enabled": True,
                                "mode": "continue_with_partial_stream_output",
                                "status": "pending",
                                "agent_id": str(agent_conf.get("cluster_agent_id") or agent_conf.get("agent_id") or ""),
                                "agent_name": str(agent_conf.get("cluster_name") or agent_conf.get("name") or "agent"),
                                "step_index": int(step_index),
                                "prefix_length": len(str(blocked.get("partial_output") or "")),
                            },
                        )
                    return blocked

                try:
                    final_state.update(
                        await invoke_orchestration_step(
                            agent_config=agent_conf,
                            provider_config=provider_config,
                            default_timeout=default_timeout,
                            provider_registry=provider_registry,
                            state=final_state,
                            on_stream_chunk=_review_stream_chunk,
                        )
                    )
                except OutputGuardrailTrip as exc:
                    agent_output_key = str(agent_conf.get("cluster_agent_id") or agent_conf.get("agent_id") or "")
                    agent_name = str(agent_conf.get("cluster_name") or agent_conf.get("name") or agent_output_key)
                    blocked_payload = dict(exc.payload or {})
                    partial_content = str(exc.partial_content or "")
                    serialized_resume = _serialize_orchestration_resume_state(previous_state, step_index)
                    serialized_resume["rollback_state"] = _serialize_orchestration_resume_state(previous_state, step_index)["state"]
                    serialized_resume["continuation"] = {
                        "resume_agent_id": str(agent_conf.get("agent_id") or agent_output_key),
                        "agent_id": agent_output_key,
                        "agent_name": agent_name,
                        "partial_output": partial_content,
                        "review_output": str(blocked_payload.get("review_output") or ""),
                        "review_stage": "agent_output_stream",
                        "step_index": step_index,
                        "loop_number": int(step["loop_number"]),
                    }
                    input_json = dict(input_json)
                    input_json["orchestration_resume"] = serialized_resume
                    _maybe_call(service, "update_run_input_json", run_id, input_json)
                    _maybe_call(
                        service,
                        "record_event",
                        run_id,
                        event_type="orchestration.review_stream_blocked",
                        details={
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "review_output": str(blocked_payload.get("review_output") or ""),
                            "chunk_index": int(blocked_payload.get("chunk_index") or 0),
                            "check_count": int(blocked_payload.get("check_count") or 0),
                            "segment_start_char": int(blocked_payload.get("segment_start_char") or 0),
                            "segment_end_char": int(blocked_payload.get("segment_end_char") or 0),
                            "last_reviewed_char": int(blocked_payload.get("last_reviewed_char") or 0),
                            "partial_length": len(partial_content),
                        },
                    )
                    _maybe_call(
                        service,
                        "create_approval_request",
                        run_id=run_id,
                        action_type="orchestration_review",
                        reason=f"Review agent blocked streaming output from {agent_name}",
                        payload_json={
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "review_output": str(blocked_payload.get("review_output") or ""),
                            "step_index": step_index,
                            "loop_number": int(step["loop_number"]),
                            "review_stage": "agent_output_stream",
                            "check_count": int(blocked_payload.get("check_count") or 0),
                            "segment_index": int(blocked_payload.get("segment_index") or 0),
                            "segment_count": int(blocked_payload.get("segment_count") or 0),
                            "segment_start_char": int(blocked_payload.get("segment_start_char") or 0),
                            "segment_end_char": int(blocked_payload.get("segment_end_char") or 0),
                            "last_reviewed_char": int(blocked_payload.get("last_reviewed_char") or 0),
                            "segment_preview": str(blocked_payload.get("segment_preview") or ""),
                            "partial_output": partial_content[-1000:],
                        },
                        requested_by=user_id,
                    )
                    return False
                agent_output_key = str(agent_conf.get("cluster_agent_id") or agent_conf.get("agent_id") or "")
                agent_name = str(agent_conf.get("cluster_name") or agent_conf.get("name") or agent_output_key)
                if stream_continuation_pending:
                    continuation = final_state.get("continuation") if isinstance(final_state, dict) else None
                    continuation_resume_agent_id = str(
                        (continuation or {}).get("resume_agent_id")
                        or (continuation or {}).get("agent_id")
                        or ""
                    )
                    if continuation_resume_agent_id and continuation_resume_agent_id == str(agent_conf.get("agent_id") or agent_output_key):
                        final_state.pop("continuation", None)
                        stream_continuation_pending = False
                        input_json = dict(input_json)
                        input_json.pop("orchestration_resume", None)
                        _maybe_call(service, "update_run_input_json", run_id, input_json)
                        _maybe_call(
                            service,
                            "record_event",
                            run_id,
                            event_type="orchestration.stream_continuation_completed",
                            details={
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                            },
                        )
                        _maybe_call(
                            service,
                            "patch_runtime_state",
                            run_id,
                            continuation={
                                "enabled": True,
                                "status": "completed",
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "step_index": int(step_index),
                                "completed_at": int(time.time() * 1000),
                            },
                        )
                latest_output = str((final_state.get("agent_outputs") or {}).get(agent_output_key) or "")
                agent_artifact = dict((final_state.get("output_artifacts") or {}).get(agent_output_key) or {})
                if (
                    str(agent_conf.get("cluster_strategy") or "") == "brainstorm"
                    and bool(agent_conf.get("cluster_auto_research", False))
                    and agent_artifact
                ):
                    research_payload: dict[str, Any]
                    try:
                        research_payload = await _run_cluster_auto_research(
                            task=task_str,
                            artifact=agent_artifact,
                        )
                    except Exception as exc:
                        research_payload = {
                            "queries": _build_cluster_research_queries(task=task_str, artifact=agent_artifact),
                            "result_count": 0,
                            "searches": [],
                            "digest": "",
                            "error": str(exc),
                        }
                    memory_status = _persist_cluster_research_memory(
                        user_id=user_id,
                        run_id=run_id,
                        cluster_id=agent_output_key,
                        research_digest=str(research_payload.get("digest") or ""),
                    )
                    research_payload["memory"] = memory_status
                    updated_artifacts = dict(final_state.get("output_artifacts") or {})
                    current_cluster_artifact = dict(updated_artifacts.get(agent_output_key) or {})
                    current_cluster_artifact["research"] = research_payload
                    updated_artifacts[agent_output_key] = current_cluster_artifact
                    final_state["output_artifacts"] = updated_artifacts
                    if str(research_payload.get("digest") or "").strip():
                        updated_outputs = dict(final_state.get("agent_outputs") or {})
                        updated_outputs[agent_output_key] = (
                            f"{latest_output}\n\nResearch Digest:\n{str(research_payload.get('digest') or '').strip()}"
                        ).strip()
                        final_state["agent_outputs"] = updated_outputs
                        latest_output = str(updated_outputs.get(agent_output_key) or latest_output)
                    _maybe_call(
                        service,
                        "record_event",
                        run_id,
                        event_type="orchestration.cluster_research_completed",
                        details={
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "queries": list(research_payload.get("queries") or []),
                            "result_count": int(research_payload.get("result_count") or 0),
                            "paper_count": len(list(research_payload.get("papers") or [])),
                            "browser_preview_count": len(list(research_payload.get("browser_previews") or [])),
                            "memory_stored": bool(memory_status.get("stored")),
                            "error": str(research_payload.get("error") or "") or None,
                        },
                    )
                    current_runtime_state = _maybe_call(service, "_load_persisted_runtime_state", run_id) or {}
                    current_cluster_ids = list(
                        ((current_runtime_state.get("research") or {}) if isinstance(current_runtime_state, dict) else {}).get("cluster_ids") or []
                    )
                    _maybe_call(
                        service,
                        "patch_runtime_state",
                        run_id,
                        research={
                            "enabled": True,
                            "mode": str(research_payload.get("research_mode") or "paper_first"),
                            "paper_count": len(list(research_payload.get("papers") or [])),
                            "browser_preview_count": len(list(research_payload.get("browser_previews") or [])),
                            "source_count": len(list(research_payload.get("sources") or [])),
                            "cluster_ids": sorted({*current_cluster_ids, agent_output_key}),
                        },
                    )
                    if review_enabled and (
                        str(research_payload.get("digest") or "").strip()
                        or list(research_payload.get("papers") or [])
                        or list(research_payload.get("latest_progress") or [])
                    ):
                        research_review_result = await _review_cluster_research_evidence(
                            review_config=review_config,
                            provider_config=provider_config,
                            default_timeout=default_timeout,
                            provider_registry=provider_registry,
                            task=task_str,
                            agent_id=agent_output_key,
                            agent_name=agent_name,
                            research_payload=research_payload,
                        )
                        _maybe_call(
                            service,
                            "record_event",
                            run_id,
                            event_type="orchestration.cluster_research_review_completed",
                            details={
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "approved": bool(research_review_result.get("approved")),
                                "review_output": str(research_review_result.get("review_output") or ""),
                            },
                        )
                        if not bool(research_review_result.get("approved")):
                            rollback_state = _serialize_orchestration_resume_state(previous_state, step_index)["state"]
                            rollback_artifacts = dict(rollback_state.get("output_artifacts") or {})
                            rollback_artifacts[agent_output_key] = {
                                **agent_artifact,
                                "research": {
                                    **research_payload,
                                    "blocked": True,
                                    "review_output": str(research_review_result.get("review_output") or ""),
                                },
                            }
                            rollback_outputs = dict(rollback_state.get("agent_outputs") or {})
                            rollback_outputs[agent_output_key] = latest_output
                            rollback_state["output_artifacts"] = rollback_artifacts
                            rollback_state["agent_outputs"] = rollback_outputs
                            serialized_resume = _serialize_orchestration_resume_state(final_state, step_index + 1)
                            serialized_resume["rollback_state"] = rollback_state
                            input_json = dict(input_json)
                            input_json["orchestration_resume"] = serialized_resume
                            _maybe_call(service, "update_run_input_json", run_id, input_json)
                            _maybe_call(
                                service,
                                "create_approval_request",
                                run_id=run_id,
                                action_type="orchestration_review",
                                reason=f"Review agent blocked research evidence from {agent_name}",
                                payload_json={
                                    "agent_id": agent_output_key,
                                    "agent_name": agent_name,
                                    "review_output": str(research_review_result.get("review_output") or ""),
                                    "step_index": step_index,
                                    "loop_number": int(step["loop_number"]),
                                    "review_stage": "cluster_research",
                                    "research_queries": list(research_payload.get("queries") or []),
                                },
                                requested_by=user_id,
                            )
                            _maybe_call(
                                service,
                                "patch_runtime_state",
                                run_id,
                                review={
                                    "stage": "cluster_research",
                                    "status": "pending",
                                    "agent_id": agent_output_key,
                                    "agent_name": agent_name,
                                    "review_output": str(research_review_result.get("review_output") or ""),
                                },
                            )
                            return False
                _maybe_call(
                    service,
                    "record_event",
                    run_id,
                    event_type="orchestration.step_completed",
                    details={
                        "agent_id": agent_output_key,
                        "agent_name": agent_name,
                        "step_index": step_index,
                        "loop_number": int(step["loop_number"]),
                    },
                )

                if review_enabled and latest_output:
                    segmented_review = await _run_segmented_output_review(
                        review_config=review_config,
                        provider_config=provider_config,
                        default_timeout=default_timeout,
                        provider_registry=provider_registry,
                        task=task_str,
                        agent_id=agent_output_key,
                        agent_name=agent_name,
                        output=latest_output,
                    )
                    if segmented_review["mode"] == "segmented":
                        _maybe_call(
                            service,
                            "record_event",
                            run_id,
                            event_type="orchestration.review_segment_scan_completed",
                            details={
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "approved": bool(segmented_review.get("approved")),
                                "segment_count": int(segmented_review.get("segment_count") or 0),
                                "segments_reviewed": int(segmented_review.get("segments_reviewed") or 0),
                                "blocked_segment_index": (
                                    int((segmented_review.get("blocked_segment") or {}).get("segment_index"))
                                    if isinstance(segmented_review.get("blocked_segment"), dict)
                                    and segmented_review.get("blocked_segment") is not None
                                    else None
                                ),
                            },
                        )
                    if not bool(segmented_review.get("approved")):
                        blocked_segment = dict(segmented_review.get("blocked_segment") or {})
                        serialized_resume = _serialize_orchestration_resume_state(final_state, step_index + 1)
                        serialized_resume["rollback_state"] = _serialize_orchestration_resume_state(previous_state, step_index)["state"]
                        input_json = dict(input_json)
                        input_json["orchestration_resume"] = serialized_resume
                        _maybe_call(service, "update_run_input_json", run_id, input_json)
                        _maybe_call(
                            service,
                            "create_approval_request",
                            run_id=run_id,
                            action_type="orchestration_review",
                            reason=f"Review agent blocked streamed output from {agent_name}",
                            payload_json={
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "review_output": str(blocked_segment.get("review_output") or ""),
                                "step_index": step_index,
                                "loop_number": int(step["loop_number"]),
                                "review_stage": "agent_output_segment",
                                "check_count": int(segmented_review.get("segments_reviewed") or 0),
                                "segment_index": int(blocked_segment.get("segment_index") or 0),
                                "segment_count": int(segmented_review.get("segment_count") or 0),
                                "segment_start_char": int(blocked_segment.get("start_char") or 0),
                                "segment_end_char": int(blocked_segment.get("end_char") or 0),
                                "last_reviewed_char": int(blocked_segment.get("end_char") or 0),
                                "segment_preview": str(blocked_segment.get("content") or "")[:400],
                            },
                            requested_by=user_id,
                        )
                        _maybe_call(
                            service,
                            "patch_runtime_state",
                            run_id,
                            review={
                                "stage": "agent_output_segment",
                                "status": "pending",
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "review_output": str(blocked_segment.get("review_output") or ""),
                                "check_count": int(segmented_review.get("segments_reviewed") or 0),
                                "segment_index": int(blocked_segment.get("segment_index") or 0),
                                "segment_count": int(segmented_review.get("segment_count") or 0),
                                "segment_start_char": int(blocked_segment.get("start_char") or 0),
                                "segment_end_char": int(blocked_segment.get("end_char") or 0),
                                "last_reviewed_char": int(blocked_segment.get("end_char") or 0),
                            },
                        )
                        return False

                    review_result = await review_orchestration_output(
                        review_config=review_config,
                        provider_config=provider_config,
                        default_timeout=default_timeout,
                        provider_registry=provider_registry,
                        task=task_str,
                        agent_id=agent_output_key,
                        agent_name=agent_name,
                        output=latest_output,
                    )
                    _maybe_call(
                        service,
                        "record_event",
                        run_id,
                        event_type="orchestration.review_completed",
                        details={
                            "agent_id": agent_output_key,
                            "agent_name": agent_name,
                            "approved": bool(review_result.get("approved")),
                            "review_output": str(review_result.get("review_output") or ""),
                        },
                    )
                    if not bool(review_result.get("approved")):
                        serialized_resume = _serialize_orchestration_resume_state(final_state, step_index + 1)
                        serialized_resume["rollback_state"] = _serialize_orchestration_resume_state(previous_state, step_index)["state"]
                        input_json = dict(input_json)
                        input_json["orchestration_resume"] = serialized_resume
                        _maybe_call(service, "update_run_input_json", run_id, input_json)
                        _maybe_call(
                            service,
                            "create_approval_request",
                            run_id=run_id,
                            action_type="orchestration_review",
                            reason=f"Review agent blocked output from {agent_name}",
                            payload_json={
                                "agent_id": agent_output_key,
                                "agent_name": agent_name,
                                "review_output": str(review_result.get("review_output") or ""),
                                "step_index": step_index,
                                "loop_number": int(step["loop_number"]),
                            },
                            requested_by=user_id,
                        )
                        return False

            agent_outputs = final_state.get("agent_outputs", {})
            output_artifacts = final_state.get("output_artifacts", {})
            errors = final_state.get("errors", [])
            
            _maybe_call(
                service,
                "record_event",
                run_id,
                event_type="orchestration.completed",
                details={
                    "agent_outputs": agent_outputs,
                    "output_artifacts": output_artifacts,
                    "errors": errors,
                    "review_agent_enabled": review_agent_enabled,
                    "recovery_mode": recovery_mode,
                },
            )

            verification = verification_service.build_agent_orchestration_result(
                ok=not bool(errors),
                active_agent_ids=active_agent_ids,
                blocked_agents=[],
                loop_count=loop_count,
                review_agent_enabled=review_agent_enabled,
                error_code="execution_errors" if errors else None,
                error_message="; ".join(errors) if errors else None,
                agent_outputs=agent_outputs,
                output_artifacts=output_artifacts,
                recovery_mode=recovery_mode,
            )
            service.complete_with_verification(run_id, verification)
            return not bool(errors)

        if task_type == HarnessTaskType.SESSION_RESUME_APPROVAL.value:
            _maybe_call(service, "set_current_step", run_id, "load_checkpoint")
            session_id = str(run.get("session_id") or "") or None
            if not session_id:
                verification = verification_service.build_session_resume_result(
                    ok=False,
                    session_id=None,
                    interrupted=None,
                    error_code="missing_session_id",
                    error_message="missing session_id for session resume execution",
                )
                service.complete_with_verification(run_id, verification)
                return False

            checkpoint = await CheckpointAdapter().load(session_id)
            if checkpoint is None:
                verification = verification_service.build_session_resume_result(
                    ok=False,
                    session_id=session_id,
                    interrupted=None,
                    error_code="checkpoint_missing",
                    error_message="resume checkpoint not found",
                )
                service.complete_with_verification(run_id, verification)
                return False

            _maybe_call(service, "set_current_step", run_id, "resume_graph")
            _maybe_call(service, "mark_resumed", run_id)
            resume_result = await GraphResumeService().resume_approved_session(
                session_id=session_id,
                checkpoint=checkpoint,
            )
            ok = bool(resume_result.get("ok"))
            messages = resume_result.get("messages")
            normalized_messages = messages if isinstance(messages, list) else []
            user_id = str(run.get("user_id") or "") or None
            if ok and user_id and normalized_messages:
                persist_session_messages(
                    user_id=user_id,
                    session_id=session_id,
                    messages=normalized_messages,
                )
            verification = verification_service.build_session_resume_result(
                ok=ok,
                session_id=session_id,
                interrupted=resume_result.get("interrupted") if "interrupted" in resume_result else None,
                error_code=str(resume_result.get("error_code") or "") or None,
                error_message=str(resume_result.get("error_message") or "") or None,
            )
            service.complete_with_verification(run_id, verification)
            return ok

        verification = verification_service.build_document_ingest_result(
            ok=False,
            stage="unsupported_task_type",
            error_code="unsupported_task_type",
            error_message="unsupported harness task type",
        )
        service.complete_with_verification(run_id, verification)
        return False
    except Exception as exc:
        if task_type == HarnessTaskType.SESSION_RESUME_APPROVAL.value:
            verification = verification_service.build_session_resume_result(
                ok=False,
                session_id=str(run.get("session_id") or "") or None,
                interrupted=None,
                error_code="task_exception",
                error_message=str(exc),
            )
        elif task_type == HarnessTaskType.AGENT_ORCHESTRATION.value:
            input_json = run.get("input_json") or {}
            selected_agent_ids = input_json.get("selected_agent_ids") if isinstance(input_json, dict) else []
            selected_agent_ids = selected_agent_ids if isinstance(selected_agent_ids, list) else []
            graph = input_json.get("graph") if isinstance(input_json, dict) else {}
            graph = graph if isinstance(graph, dict) else {}
            review_agent = graph.get("review_agent") if isinstance(graph.get("review_agent"), dict) else {}
            verification = verification_service.build_agent_orchestration_result(
                ok=False,
                active_agent_ids=[str(agent_id) for agent_id in selected_agent_ids],
                blocked_agents=[],
                loop_count=int(input_json.get("loop_count") or 1) if isinstance(input_json, dict) else 1,
                review_agent_enabled=bool(review_agent.get("enabled", True)),
                error_code="task_exception",
                error_message=str(exc),
            )
        else:
            verification = verification_service.build_document_ingest_result(
                ok=False,
                stage="exception",
                error_code="task_exception",
                error_message=str(exc),
            )
        service.complete_with_verification(run_id, verification)
        return False


async def resume_harness_task(ctx: dict[str, Any], run_id: str) -> bool:
    return await run_harness_task(ctx, run_id)
