from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from inspect import isawaitable
from typing import Annotated, Any, Callable, NotRequired, TypedDict

import anyio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.runtime.llm.stream_adapter import coerce_stream_text, stream_llm_events
from app.runtime.llm.llm_factory import get_llm_for_provider
from app.runtime.llm.provider_registry import ModelProviderRegistry, get_provider_registry

_log = logging.getLogger("runtime.graph.orchestration")

FIRST_PRINCIPLES_DIRECTIVE = (
    "Non-negotiable operating rule: always reason from first principles. "
    "Reduce the task to fundamental constraints, verify assumptions explicitly, "
    "and build conclusions from the most basic truths available rather than habit, imitation, or unsupported analogy."
)

GAME_THEORY_DIRECTIVE = (
    "For brainstorm clusters, also reason through a game-theoretic lens. "
    "Model the relevant actors, incentives, payoffs, information asymmetries, commitments, cooperation failures, "
    "competitive responses, and likely equilibrium outcomes before choosing a strategy."
)


class OrchestrationState(TypedDict):
    """
    Studio canvas orchestration multi-agent shared state.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    agent_outputs: dict[str, str]
    output_artifacts: dict[str, dict[str, Any]]
    current_agent: str
    loop_index: int
    errors: list[str]
    continuation: NotRequired[dict[str, Any]]


class OutputGuardrailTrip(Exception):
    def __init__(self, *, payload: dict[str, Any], partial_content: str):
        super().__init__(str(payload.get("review_output") or "stream output blocked"))
        self.payload = payload
        self.partial_content = partial_content


def _coerce_timeout(value: Any, default_timeout: int) -> int:
    try:
        return int(value) if value is not None else default_timeout
    except (TypeError, ValueError):
        return default_timeout


async def _invoke_llm_with_streaming_fallback(
    llm: Any,
    messages: list[BaseMessage],
    *,
    on_stream_chunk: Callable[[str, str, int], Any] | None = None,
    on_stream_event: Callable[[dict[str, Any]], Any] | None = None,
) -> tuple[str, str]:
    stream_method = getattr(llm, "astream", None)
    stream_events_method = getattr(llm, "astream_events", None)
    if callable(stream_method) or callable(stream_events_method):
        collected: list[str] = []
        chunk_index = 0
        async for stream_event in stream_llm_events(llm, messages, on_event=on_stream_event):
            if str(stream_event.get("type") or "") != "token":
                continue
            text = coerce_stream_text(stream_event.get("text"))
            if text:
                collected.append(text)
                if on_stream_chunk is not None:
                    callback_result = on_stream_chunk(text, "".join(collected), chunk_index)
                    if isawaitable(callback_result):
                        callback_result = await callback_result
                    if isinstance(callback_result, dict) and bool(callback_result.get("blocked")):
                        partial_content = str(callback_result.get("partial_output") or "".join(collected))
                        raise OutputGuardrailTrip(
                            payload=callback_result,
                            partial_content=partial_content,
                        )
                chunk_index += 1
        content = "".join(collected).strip()
        if content:
            return content, "astream"

    response = await llm.ainvoke(messages)
    return str(response.content) if response.content else "", "ainvoke"


def _build_continuation_context(agent_id: str, state: OrchestrationState) -> dict[str, Any] | None:
    continuation = state.get("continuation")
    if not isinstance(continuation, dict):
        return None
    target_agent_id = str(continuation.get("resume_agent_id") or continuation.get("agent_id") or "").strip()
    if not target_agent_id or target_agent_id != agent_id:
        return None
    partial_output = str(continuation.get("partial_output") or "")
    if not partial_output.strip():
        return None
    return {
        "partial_output": partial_output,
        "review_output": str(continuation.get("review_output") or ""),
        "agent_name": str(continuation.get("agent_name") or agent_id),
        "loop_number": continuation.get("loop_number"),
    }


def _merge_continuation_output(prefix: str, generated: str) -> str:
    approved_prefix = str(prefix or "")
    generated_text = str(generated or "")
    if not approved_prefix:
        return generated_text
    if not generated_text:
        return approved_prefix
    if generated_text.startswith(approved_prefix):
        return generated_text

    max_overlap = min(len(approved_prefix), len(generated_text))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if approved_prefix.endswith(generated_text[:size]):
            overlap = size
            break
    return approved_prefix + generated_text[overlap:]


def _topological_sort(agents: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    """
    Kahn's algorithm for topological sorting of agents based on edges.
    Returns agent IDs in execution order.
    """
    in_degree: dict[str, int] = {str(a["agent_id"]): 0 for a in agents}
    adj: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        source = str(edge.get("source_agent_id", ""))
        target = str(edge.get("target_agent_id", ""))
        if source in in_degree and target in in_degree:
            adj[source].append(target)
            in_degree[target] += 1

    queue = deque([node for node, deg in in_degree.items() if deg == 0])
    ordered: list[str] = []

    while queue:
        node = queue.popleft()
        ordered.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # In case of cycles, some nodes will be left with in_degree > 0
    # For now, just append the rest in their original order
    for node, deg in in_degree.items():
        if deg > 0 and node not in ordered:
            ordered.append(node)

    return ordered


def _build_cluster_member_prompt(cluster_config: dict[str, Any], member_config: dict[str, Any]) -> str:
    cluster_name = str(cluster_config.get("name") or "cluster")
    cluster_strategy = str(cluster_config.get("cluster_strategy") or "custom")
    cluster_description = str(cluster_config.get("description") or "")
    cluster_prompt = str(cluster_config.get("system_prompt") or "")
    member_prompt = str(member_config.get("system_prompt") or "")
    base = [
        f"You are part of the cluster '{cluster_name}'.",
        f"Cluster strategy: {cluster_strategy}.",
    ]
    if cluster_description:
        base.append(f"Cluster brief: {cluster_description}")
    if cluster_prompt:
        base.append(f"Cluster instructions: {cluster_prompt}")
    if member_prompt:
        base.append(f"Your member-specific instructions: {member_prompt}")
    if cluster_strategy == "brainstorm":
        base.extend(
            [
                "For brainstorm clusters, push for energetic disagreement when useful and stress-test assumptions.",
                "Use game theory explicitly: identify players, incentives, payoff tradeoffs, strategic moves, and equilibrium risks before recommending a path.",
                "Before your vote line, include this compact analysis block exactly once when possible:",
                "PLAYERS:",
                "INCENTIVES:",
                "RISKS:",
                "EQUILIBRIUM:",
                "Each response must end with a single explicit vote line in the exact format: VOTE: CONSERVATIVE, VOTE: BALANCED, or VOTE: AGGRESSIVE.",
                "When you vote, choose the strategy you believe should advance after this round.",
            ]
        )
    return "\n".join(base)


def _extract_brainstorm_votes(prior_sections: list[str]) -> dict[str, int]:
    counts = {"conservative": 0, "balanced": 0, "aggressive": 0}
    pattern = re.compile(r"VOTE:\s*(CONSERVATIVE|BALANCED|AGGRESSIVE)", re.IGNORECASE)
    for section in prior_sections:
        match = pattern.search(section)
        if not match:
            continue
        counts[match.group(1).lower()] += 1
    return counts


def _extract_member_analysis_block(content: str) -> dict[str, str]:
    matches = list(
        re.finditer(
            r"^(PLAYERS|INCENTIVES|RISKS|EQUILIBRIUM|VOTE):(?:[ \t]*.*)?$",
            content,
            re.MULTILINE,
        )
    )
    sections = {"PLAYERS": "", "INCENTIVES": "", "RISKS": "", "EQUILIBRIUM": ""}
    for index, match in enumerate(matches):
        label = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        sections[label] = content[start:end].strip()
    return sections


def _merge_round_analysis(entries: list[dict[str, str]]) -> dict[str, str]:
    buckets = {
        "key_players": [],
        "incentive_map": [],
        "dominant_risks": [],
        "expected_equilibrium": [],
    }
    seen = {key: set() for key in buckets}
    for entry in entries:
        parsed = _extract_member_analysis_block(entry.get("output", ""))
        mapping = {
            "key_players": parsed["PLAYERS"],
            "incentive_map": parsed["INCENTIVES"],
            "dominant_risks": parsed["RISKS"],
            "expected_equilibrium": parsed["EQUILIBRIUM"],
        }
        for key, value in mapping.items():
            normalized = value.strip()
            if normalized and normalized not in seen[key]:
                seen[key].add(normalized)
                buckets[key].append(normalized)
    return {key: " | ".join(values) for key, values in buckets.items()}


def _build_equilibrium_shift_details(
    previous_round: dict[str, Any] | None,
    current_round: dict[str, Any],
) -> dict[str, Any]:
    if previous_round is None:
        return {
            "shift_type": "initial",
            "changed_fields": [],
            "vote_shift": None,
            "summary": "Initial round baseline.",
        }

    changes: list[str] = []
    changed_fields: list[str] = []
    vote_shift: dict[str, str] | None = None
    if str(previous_round.get("winning_vote") or "") != str(current_round.get("winning_vote") or ""):
        changed_fields.append("winning_vote")
        vote_shift = {
            "from": str(previous_round.get("winning_vote") or "unknown"),
            "to": str(current_round.get("winning_vote") or "unknown"),
        }
        changes.append(
            f"Winning vote shifted from {previous_round.get('winning_vote') or 'unknown'} to {current_round.get('winning_vote') or 'unknown'}."
        )

    field_labels = {
        "key_players": "Player set",
        "incentive_map": "Incentive map",
        "dominant_risks": "Dominant risks",
        "expected_equilibrium": "Expected equilibrium",
    }
    for field, label in field_labels.items():
        previous_value = str(previous_round.get(field) or "").strip()
        current_value = str(current_round.get(field) or "").strip()
        if previous_value != current_value and current_value:
            changed_fields.append(field)
            if previous_value:
                changes.append(f"{label} changed from '{previous_value}' to '{current_value}'.")
            else:
                changes.append(f"{label} emerged as '{current_value}'.")

    shift_type = "none"
    if vote_shift is not None:
        shift_type = "vote_change"
    elif changed_fields:
        shift_type = "state_change"

    return {
        "shift_type": shift_type,
        "changed_fields": changed_fields,
        "vote_shift": vote_shift,
        "summary": " ".join(changes) if changes else "No major equilibrium shift from the previous round.",
    }


def _build_brainstorm_round_history(
    *,
    member_node_ids: list[str],
    outputs: dict[str, str],
) -> list[dict[str, Any]]:
    rounds: dict[int, list[dict[str, str]]] = defaultdict(list)
    for node_id in member_node_ids:
        content = str(outputs.get(node_id) or "").strip()
        if not content:
            continue
        match = re.search(r"__round_(\d+)__", node_id)
        round_index = int(match.group(1)) if match else 1
        rounds[round_index].append(
            {
                "member_node_id": node_id,
                "output": content,
            }
        )

    history: list[dict[str, Any]] = []
    previous_round: dict[str, Any] | None = None
    for round_index in sorted(rounds):
        entries = rounds[round_index]
        prior_sections = [entry["output"] for entry in entries]
        vote_tally = _extract_brainstorm_votes(prior_sections)
        round_analysis = _merge_round_analysis(entries)
        current_round = {
            "round_index": round_index,
            "member_outputs": entries,
            "vote_tally": vote_tally,
            "winning_vote": _winning_brainstorm_vote(vote_tally),
            **round_analysis,
        }
        shift_details = _build_equilibrium_shift_details(previous_round, current_round)
        current_round["equilibrium_shift"] = str(shift_details.get("summary") or "")
        current_round["equilibrium_shift_details"] = shift_details
        history.append(current_round)
        previous_round = current_round
    return history


def _winning_brainstorm_vote(votes: dict[str, int]) -> str:
    order = ["balanced", "conservative", "aggressive"]
    return max(order, key=lambda key: (votes.get(key, 0), -order.index(key))).upper()


def _compose_system_instruction(*parts: str) -> str:
    normalized = [FIRST_PRINCIPLES_DIRECTIVE]
    normalized.extend(part.strip() for part in parts if part and part.strip())
    return "\n\n".join(normalized)


def _compose_brainstorm_instruction(*parts: str) -> str:
    normalized = [FIRST_PRINCIPLES_DIRECTIVE, GAME_THEORY_DIRECTIVE]
    normalized.extend(part.strip() for part in parts if part and part.strip())
    return "\n\n".join(normalized)


def _build_brainstorm_summary_prompt(
    *,
    cluster_name: str,
    task: str,
    prior_sections: list[str],
    rounds: int,
) -> list[BaseMessage]:
    context = "\n\n".join(prior_sections) if prior_sections else "No member output was captured."
    votes = _extract_brainstorm_votes(prior_sections)
    winning_vote = _winning_brainstorm_vote(votes)
    return [
        SystemMessage(
            content=_compose_brainstorm_instruction(
                f"You are the synthesis chair for the brainstorm cluster '{cluster_name}'.",
                "Summarize the roundtable debate into three options: conservative, balanced, and aggressive.",
                "Frame each option using strategic interaction, incentive alignment, and likely equilibrium behavior.",
                "Then declare a winning strategy with a short rationale and a clear next-step recommendation.",
                "You must preserve the three-option structure even if one option is weak.",
            )
        ),
        HumanMessage(
            content=(
                f"Overall task: {task}\n"
                f"Debate rounds completed: {rounds}\n\n"
                "Observed vote tally:\n"
                f"- Conservative: {votes['conservative']}\n"
                f"- Balanced: {votes['balanced']}\n"
                f"- Aggressive: {votes['aggressive']}\n"
                f"Current winning vote: {winning_vote}\n\n"
                "Respond using this exact section structure:\n"
                "Conservative Strategy:\n"
                "Balanced Strategy:\n"
                "Aggressive Strategy:\n"
                "Winning Strategy:\n"
                "Game Theory Rationale:\n"
                "Key Players:\n"
                "Incentive Map:\n"
                "Dominant Risks:\n"
                "Expected Equilibrium:\n"
                "Next Step:\n\n"
                "Cluster debate transcript:\n"
                f"{context}"
            )
        ),
    ]


def _build_brainstorm_round_context(*, agent_config: dict[str, Any], state: OrchestrationState) -> str:
    if str(agent_config.get("cluster_strategy") or "") != "brainstorm":
        return ""
    current_round = int(agent_config.get("cluster_round_index") or 1)
    if current_round <= 1:
        return ""

    cluster_agent_id = str(agent_config.get("cluster_agent_id") or "")
    prior_member_ids = [
        str(node_id)
        for node_id in agent_config.get("cluster_member_node_ids") or []
        if str(node_id) and f"__round_{current_round - 1}__" in str(node_id)
    ]
    outputs = state.get("agent_outputs", {})
    prior_sections = [f"{node_id}:\n{outputs.get(node_id, '')}" for node_id in prior_member_ids if outputs.get(node_id)]
    if not prior_sections:
        return f"This is round {current_round}. No prior round transcript was available, so rebuild from first principles."

    votes = _extract_brainstorm_votes(prior_sections)
    winner = _winning_brainstorm_vote(votes)
    return (
        f"This is brainstorm round {current_round} for cluster '{cluster_agent_id or agent_config.get('name') or 'cluster'}'.\n"
        f"Previous round vote tally: Conservative={votes['conservative']}, Balanced={votes['balanced']}, Aggressive={votes['aggressive']}.\n"
        f"Current leading direction from the previous round: {winner}.\n"
        "React to the leading direction explicitly: either strengthen it with better first-principles support, or overturn it with a stronger alternative.\n"
        "Use game theory in this round: identify the key players, incentives, strategic responses, and equilibrium shifts that support or weaken the current leading direction."
    )


def _build_prior_research_context(*, agent_id: str, state: OrchestrationState) -> str:
    output_artifacts = state.get("output_artifacts", {}) or {}
    sections: list[str] = []
    for artifact_agent_id, artifact in output_artifacts.items():
        if str(artifact_agent_id) == str(agent_id):
            continue
        if not isinstance(artifact, dict):
            continue
        research = artifact.get("research")
        if not isinstance(research, dict):
            continue
        cluster_name = str(artifact.get("cluster_name") or artifact_agent_id or "cluster")
        winning_strategy = str(artifact.get("winning_strategy") or "").strip()
        next_step = str(artifact.get("next_step") or "").strip()
        latest_progress = [
            str(item).strip()
            for item in (research.get("latest_progress") or [])
            if str(item).strip()
        ][:3]
        paper_titles = [
            str(item.get("title") or "").strip()
            for item in (research.get("papers") or [])
            if isinstance(item, dict) and str(item.get("title") or "").strip()
        ][:3]
        block_lines = [f"Evidence from prior cluster '{cluster_name}':"]
        if winning_strategy:
            block_lines.append(f"- Winning strategy: {winning_strategy}")
        game_theory_rationale = str(artifact.get("game_theory_rationale") or "").strip()
        if game_theory_rationale:
            block_lines.append(f"- Game-theoretic rationale: {game_theory_rationale}")
        key_players = str(artifact.get("key_players") or "").strip()
        if key_players:
            block_lines.append(f"- Key players: {key_players}")
        incentive_map = str(artifact.get("incentive_map") or "").strip()
        if incentive_map:
            block_lines.append(f"- Incentive map: {incentive_map}")
        dominant_risks = str(artifact.get("dominant_risks") or "").strip()
        if dominant_risks:
            block_lines.append(f"- Dominant risks: {dominant_risks}")
        expected_equilibrium = str(artifact.get("expected_equilibrium") or "").strip()
        if expected_equilibrium:
            block_lines.append(f"- Expected equilibrium: {expected_equilibrium}")
        if next_step:
            block_lines.append(f"- Recommended next step: {next_step}")
        if latest_progress:
            block_lines.append("- Latest progress signals:")
            block_lines.extend(f"  - {item}" for item in latest_progress)
        if paper_titles:
            block_lines.append("- Relevant papers:")
            block_lines.extend(f"  - {item}" for item in paper_titles)
        sections.append("\n".join(block_lines))
    return "\n\n".join(sections)


def _parse_labeled_sections(content: str, labels: list[str]) -> dict[str, str]:
    matches = list(
        re.finditer(
            r"^(Conservative Strategy|Balanced Strategy|Aggressive Strategy|Winning Strategy|Game Theory Rationale|Key Players|Incentive Map|Dominant Risks|Expected Equilibrium|Next Step):\s*$",
            content,
            re.MULTILINE,
        )
    )
    sections = {label: "" for label in labels}
    for index, match in enumerate(matches):
        label = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        if label in sections:
            sections[label] = content[start:end].strip()
    return sections


def _build_cluster_output_artifact(
    *,
    cluster_agent_id: str,
    cluster_name: str,
    cluster_strategy: str,
    member_node_ids: list[str],
    outputs: dict[str, str],
    rounds: int,
    summary: str,
    prior_sections: list[str],
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "node_kind": "cluster",
        "cluster_agent_id": cluster_agent_id,
        "cluster_name": cluster_name,
        "cluster_strategy": cluster_strategy,
        "member_node_ids": member_node_ids,
        "member_count": len(member_node_ids),
        "rounds_completed": rounds,
        "raw_summary": summary,
    }
    if cluster_strategy != "brainstorm":
        return artifact

    votes = _extract_brainstorm_votes(prior_sections)
    winning_vote = _winning_brainstorm_vote(votes)
    sections = _parse_labeled_sections(
        summary,
        [
            "Conservative Strategy",
            "Balanced Strategy",
            "Aggressive Strategy",
            "Winning Strategy",
            "Game Theory Rationale",
            "Key Players",
            "Incentive Map",
            "Dominant Risks",
            "Expected Equilibrium",
            "Next Step",
        ],
    )
    artifact.update(
        {
            "vote_tally": votes,
            "winning_vote": winning_vote,
            "round_history": _build_brainstorm_round_history(member_node_ids=member_node_ids, outputs=outputs),
            "strategies": {
                "conservative": sections["Conservative Strategy"],
                "balanced": sections["Balanced Strategy"],
                "aggressive": sections["Aggressive Strategy"],
            },
            "winning_strategy": sections["Winning Strategy"],
            "game_theory_rationale": sections["Game Theory Rationale"],
            "key_players": sections["Key Players"],
            "incentive_map": sections["Incentive Map"],
            "dominant_risks": sections["Dominant Risks"],
            "expected_equilibrium": sections["Expected Equilibrium"],
            "next_step": sections["Next Step"],
        }
    )
    return artifact


def _expand_cluster_agents(
    agents: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expanded_agents: list[dict[str, Any]] = []
    expanded_edges: list[dict[str, Any]] = []
    entry_map: dict[str, str] = {}
    exit_map: dict[str, str] = {}

    for agent in agents:
        agent_id = str(agent.get("agent_id") or "")
        if not agent_id:
            continue
        if str(agent.get("node_kind") or "agent") != "cluster":
            expanded_agents.append(agent)
            entry_map[agent_id] = agent_id
            exit_map[agent_id] = agent_id
            continue

        members = [member for member in agent.get("cluster_members") or [] if isinstance(member, dict)]
        cluster_name = str(agent.get("name") or agent_id)
        cluster_strategy = str(agent.get("cluster_strategy") or "custom")
        cluster_rounds = max(1, min(int(agent.get("brainstorm_rounds") or 1), 5))
        effective_rounds = cluster_rounds if cluster_strategy == "brainstorm" else 1

        if not members:
            members = [
                {
                    "member_id": f"{agent_id}_lead",
                    "name": f"{cluster_name} lead",
                    "role": "lead",
                    "system_prompt": "Propose the first workable plan for the cluster.",
                    "model": str(agent.get("model") or "gpt-5.2"),
                }
            ]

        member_node_ids: list[str] = []
        previous_member_node_id: str | None = None
        for round_index in range(1, effective_rounds + 1):
            for member_index, member in enumerate(members, start=1):
                member_node_id = (
                    f"{agent_id}__round_{round_index}__member_{member_index}"
                    if effective_rounds > 1
                    else f"{agent_id}__member_{member_index}"
                )
                member_node_ids.append(member_node_id)
                expanded_agents.append(
                    {
                        "agent_id": member_node_id,
                        "name": (
                            f"{cluster_name} / {member.get('name') or f'member {member_index}'}"
                            f"{f' / round {round_index}' if effective_rounds > 1 else ''}"
                        ),
                        "role": str(member.get("role") or "specialist"),
                        "description": str(agent.get("description") or ""),
                        "system_prompt": _build_cluster_member_prompt(agent, member),
                        "model": str(member.get("model") or agent.get("model") or "gpt-5.2"),
                        "preferred_provider_id": member.get("preferred_provider_id") or agent.get("preferred_provider_id"),
                        "fallback_provider_id": member.get("fallback_provider_id") or agent.get("fallback_provider_id"),
                        "temperature": float(member.get("temperature", agent.get("temperature", 0.2)) or 0.2),
                        "timeout_seconds": _coerce_timeout(
                            member.get("timeout_seconds", agent.get("timeout_seconds")),
                            _coerce_timeout(agent.get("timeout_seconds"), 60),
                        ),
                        "cluster_agent_id": agent_id,
                        "cluster_strategy": cluster_strategy,
                        "cluster_round_index": round_index,
                        "cluster_member_node_ids": member_node_ids,
                    }
                )
                if previous_member_node_id is not None:
                    expanded_edges.append(
                        {
                            "edge_id": f"{agent_id}__internal_{previous_member_node_id}__{member_node_id}",
                            "source_agent_id": previous_member_node_id,
                            "target_agent_id": member_node_id,
                            "interaction": "cluster_internal",
                        }
                    )
                previous_member_node_id = member_node_id

        summary_node_id = f"{agent_id}__summary"
        expanded_agents.append(
            {
                "agent_id": summary_node_id,
                "name": f"{cluster_name} / summary",
                "role": "cluster_summary",
                "cluster_summary": True,
                "cluster_agent_id": agent_id,
                "cluster_name": cluster_name,
                "cluster_strategy": cluster_strategy,
                "brainstorm_rounds": effective_rounds,
                "cluster_member_node_ids": member_node_ids,
                "cluster_member_count": len(members),
                "cluster_auto_research": bool(agent.get("cluster_auto_research", False)),
                "cluster_auto_review": bool(agent.get("cluster_auto_review", True)),
                "model": str(agent.get("model") or "gpt-5.2"),
                "preferred_provider_id": agent.get("preferred_provider_id"),
                "fallback_provider_id": agent.get("fallback_provider_id"),
                "temperature": float(agent.get("temperature", 0.2) or 0.2),
                "timeout_seconds": _coerce_timeout(agent.get("timeout_seconds"), 60),
            }
        )
        expanded_edges.append(
            {
                "edge_id": f"{agent_id}__summary_edge",
                "source_agent_id": member_node_ids[-1],
                "target_agent_id": summary_node_id,
                "interaction": "cluster_summary",
            }
        )
        entry_map[agent_id] = member_node_ids[0]
        exit_map[agent_id] = summary_node_id

    for edge in edges:
        source = str(edge.get("source_agent_id") or "")
        target = str(edge.get("target_agent_id") or "")
        mapped_source = exit_map.get(source)
        mapped_target = entry_map.get(target)
        if not mapped_source or not mapped_target:
            continue
        expanded_edges.append(
            {
                **edge,
                "source_agent_id": mapped_source,
                "target_agent_id": mapped_target,
            }
        )

    return expanded_agents, expanded_edges


def build_orchestration_execution_plan(graph_json: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    agents = graph_json.get("agents") or []
    edges = graph_json.get("edges") or []
    provider_config = graph_json.get("provider_config") or {}
    review_config = graph_json.get("review_agent") or {}
    expanded_agents, expanded_edges = _expand_cluster_agents(agents, edges)
    execution_order = _topological_sort(expanded_agents, expanded_edges)
    agent_by_id = {str(agent.get("agent_id") or ""): agent for agent in expanded_agents}
    ordered_agents = [agent_by_id[agent_id] for agent_id in execution_order if agent_id in agent_by_id]
    return ordered_agents, provider_config, review_config


async def invoke_orchestration_step(
    *,
    agent_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    state: OrchestrationState,
    is_review: bool = False,
    on_stream_chunk: Callable[[str, str, int], Any] | None = None,
    on_stream_event: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    node = _build_agent_node(
        agent_config,
        provider_config,
        default_timeout,
        provider_registry,
        is_review=is_review,
        on_stream_chunk=on_stream_chunk,
        on_stream_event=on_stream_event,
    )
    return await node(state)


async def review_orchestration_output(
    *,
    review_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    task: str,
    agent_id: str,
    agent_name: str,
    output: str,
) -> dict[str, object]:
    if not bool(review_config.get("enabled", True)):
        return {"approved": True, "review_output": "PASS: review disabled"}

    model = str(review_config.get("model") or "gpt-5.1-codex-mini")
    preferred = review_config.get("preferred_provider_id") or provider_config.get("preferred_provider_id")
    fallback = review_config.get("fallback_provider_id") or provider_config.get("fallback_provider_id")
    timeout = _coerce_timeout(review_config.get("timeout_seconds"), default_timeout)
    resolved = provider_registry.resolve(model, preferred_provider_id=preferred, fallback_provider_id=fallback)
    llm = get_llm_for_provider(
        model=resolved.model,
        base_url=resolved.base_url,
        api_key=resolved.api_key,
        temperature=float(review_config.get("temperature", 0) or 0),
        timeout_seconds=timeout,
    )
    messages = [
        SystemMessage(
            content=_compose_system_instruction(
                f"You are {str(review_config.get('name') or 'Reviewer')}, the runtime safety reviewer for a collaborative workflow.",
                str(review_config.get("system_prompt") or ""),
                "Review only the provided latest node output.",
                "Reply with exactly one leading decision token: PASS or BLOCK.",
                "Use BLOCK only when the output is risky, unsafe, policy-violating, or requires human review before downstream use.",
            )
        ),
        HumanMessage(
            content=(
                f"Task: {task}\n"
                f"Node id: {agent_id}\n"
                f"Node name: {agent_name}\n\n"
                "Latest node output to review:\n"
                f"{output}"
            )
        ),
    ]
    try:
        with anyio.fail_after(timeout):
            review_output, _ = await _invoke_llm_with_streaming_fallback(llm, messages)
    except Exception as exc:
        review_output = f"BLOCK: review agent failed and requires human confirmation. Error: {exc!s}"
    normalized = review_output.strip().upper()
    approved = normalized.startswith("PASS")
    if not approved and not normalized.startswith("BLOCK"):
        approved = True
    return {
        "approved": approved,
        "review_output": review_output or ("PASS: no issues detected" if approved else "BLOCK: human review required"),
    }


def _build_agent_node(
    agent_config: dict[str, Any],
    provider_config: dict[str, Any],
    default_timeout: int,
    provider_registry: ModelProviderRegistry,
    *,
    is_review: bool = False,
    on_stream_chunk: Callable[[str, str, int], Any] | None = None,
    on_stream_event: Callable[[dict[str, Any]], Any] | None = None,
) -> Callable[[OrchestrationState], dict[str, Any]]:
    agent_id = str(agent_config["agent_id"])
    name = str(agent_config.get("name", agent_id))
    is_cluster_summary = bool(agent_config.get("cluster_summary"))
    system_prompt = str(agent_config.get("system_prompt", ""))
    model = str(agent_config.get("model", "gpt-5.2"))
    temperature = float(agent_config.get("temperature", 0.2))
    agent_timeout = agent_config.get("timeout_seconds")
    timeout = int(agent_timeout) if agent_timeout else default_timeout

    preferred = agent_config.get("preferred_provider_id") or provider_config.get("preferred_provider_id")
    fallback = agent_config.get("fallback_provider_id") or provider_config.get("fallback_provider_id")

    async def _invoke_agent(state: OrchestrationState) -> dict[str, Any]:
        """LangGraph node representing a single canvas agent."""
        _log.info("Executing canvas agent node: %s (%s)", name, agent_id)
        continuation_context = _build_continuation_context(agent_id, state)

        if is_cluster_summary:
            cluster_agent_id = str(agent_config.get("cluster_agent_id") or agent_id)
            cluster_name = str(agent_config.get("cluster_name") or cluster_agent_id)
            cluster_strategy = str(agent_config.get("cluster_strategy") or "custom")
            member_node_ids = [str(node_id) for node_id in agent_config.get("cluster_member_node_ids") or [] if str(node_id)]
            outputs = state.get("agent_outputs", {})
            artifacts = state.get("output_artifacts", {}).copy()
            sections = [
                f"{member_id}:\n{outputs.get(member_id, '')}".strip()
                for member_id in member_node_ids
                if outputs.get(member_id)
            ]
            summary = "\n\n".join(section for section in sections if section) or f"{cluster_name} completed with no member output."
            if cluster_strategy == "brainstorm":
                resolved = provider_registry.resolve(model, preferred_provider_id=preferred, fallback_provider_id=fallback)
                llm = get_llm_for_provider(
                    model=resolved.model,
                    base_url=resolved.base_url,
                    api_key=resolved.api_key,
                    temperature=temperature,
                    timeout_seconds=timeout,
                )
                try:
                    prompt_messages = _build_brainstorm_summary_prompt(
                        cluster_name=cluster_name,
                        task=str(state.get("task") or ""),
                        prior_sections=sections,
                        rounds=int(agent_config.get("brainstorm_rounds") or 1),
                    )
                    if continuation_context:
                        prompt_messages = list(prompt_messages) + [
                            HumanMessage(
                                content=(
                                    "A previous approved partial summary for this same node already exists. "
                                    "Continue from that exact prefix without restarting or repeating it.\n\n"
                                    f"Approved partial summary:\n{continuation_context['partial_output']}\n\n"
                                    f"Review note:\n{continuation_context['review_output'] or 'none'}"
                                )
                            )
                        ]
                    with anyio.fail_after(timeout):
                        summary, _ = await _invoke_llm_with_streaming_fallback(
                            llm,
                            prompt_messages,
                            on_stream_chunk=on_stream_chunk,
                            on_stream_event=on_stream_event,
                        )
                        if continuation_context:
                            summary = _merge_continuation_output(
                                str(continuation_context.get("partial_output") or ""),
                                summary,
                            )
                except TimeoutError:
                    summary = f"{cluster_name} summary timed out.\n\n{summary}"
                except OutputGuardrailTrip:
                    raise
                except Exception as exc:
                    summary = f"{cluster_name} summary failed: {exc!s}\n\n{summary}"
            new_outputs = outputs.copy()
            new_outputs[cluster_agent_id] = summary
            artifacts[cluster_agent_id] = _build_cluster_output_artifact(
                cluster_agent_id=cluster_agent_id,
                cluster_name=cluster_name,
                cluster_strategy=cluster_strategy,
                member_node_ids=member_node_ids,
                outputs=outputs,
                rounds=int(agent_config.get("brainstorm_rounds") or 1),
                summary=summary,
                prior_sections=sections,
            )
            if "cluster_member_count" in agent_config:
                artifacts[cluster_agent_id]["member_count"] = int(agent_config.get("cluster_member_count") or 0)
            return {
                "current_agent": cluster_agent_id,
                "agent_outputs": new_outputs,
                "output_artifacts": artifacts,
                "errors": list(state.get("errors", [])),
                "messages": [AIMessage(content=f"[{cluster_name}] cluster summary:\n{summary}")],
            }

        resolved = provider_registry.resolve(model, preferred_provider_id=preferred, fallback_provider_id=fallback)
        llm = get_llm_for_provider(
            model=resolved.model,
            base_url=resolved.base_url,
            api_key=resolved.api_key,
            temperature=temperature,
            timeout_seconds=timeout,
        )

        if is_review:
            prior_outputs = [
                f"Agent {aid} output:\n{output}"
                for aid, output in state.get("agent_outputs", {}).items()
                if str(aid).strip()
            ]
            review_context = (
                f"The overall user task is: {state.get('task')}\n\n"
                "Review the completed agent collaboration for correctness, safety, and handoff quality.\n"
                "Return a concise review summary."
            )
            if prior_outputs:
                review_context += "\n\nCollected agent outputs:\n" + "\n\n".join(prior_outputs)
            messages = [
                SystemMessage(
                    content=_compose_system_instruction(
                        f"You are {name}, the review agent for a collaborative workflow.",
                        f"Your instructions: {system_prompt}",
                    )
                ),
                HumanMessage(content=review_context),
            ]
        else:
            instruction_builder = (
                _compose_brainstorm_instruction
                if str(agent_config.get("cluster_strategy") or "") == "brainstorm"
                else _compose_system_instruction
            )
            sys_msg = SystemMessage(
                content=instruction_builder(
                    f"You are {name}, a specialist agent in a collaborative workflow.",
                    f"Your instructions: {system_prompt}",
                )
            )
            task_context = f"The overall user task is: {state.get('task')}\n"
            brainstorm_round_context = _build_brainstorm_round_context(agent_config=agent_config, state=state)
            if brainstorm_round_context:
                task_context += f"\n{brainstorm_round_context}\n"
            prior_research_context = _build_prior_research_context(agent_id=agent_id, state=state)
            if prior_research_context:
                task_context += (
                    "\nExternal research evidence from earlier clusters is available. "
                    "Use it to refine your reasoning, but still verify assumptions from first principles.\n"
                    f"{prior_research_context}\n"
                )
            if continuation_context:
                task_context += (
                    "\nThis node is resuming from an approved partial output that was interrupted by live review. "
                    "Continue from the exact end of that approved prefix without restarting, repeating, or summarizing the prefix.\n"
                    f"\nApproved partial output:\n{continuation_context['partial_output']}\n"
                )
                if str(continuation_context.get("review_output") or "").strip():
                    task_context += f"\nReview note:\n{str(continuation_context.get('review_output') or '').strip()}\n"
            other_outputs = []
            for aid, output in state.get("agent_outputs", {}).items():
                if aid != agent_id:
                    other_outputs.append(f"Agent {aid} output:\n{output}")
            if other_outputs:
                task_context += "\nPrevious agents completed work:\n" + "\n".join(other_outputs)
            messages = [sys_msg, HumanMessage(content=task_context)] + state.get("messages", [])

        content = ""
        error_msg = None
        try:
            with anyio.fail_after(timeout):
                content, _ = await _invoke_llm_with_streaming_fallback(
                    llm,
                    messages,
                    on_stream_chunk=None if is_review else on_stream_chunk,
                    on_stream_event=None if is_review else on_stream_event,
                )
        except OutputGuardrailTrip:
            raise
        except TimeoutError:
            error_msg = f"Agent {name} ({agent_id}) timed out after {timeout} seconds."
            content = f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"Agent {name} ({agent_id}) execution failed: {e!s}"
            content = f"Error: {error_msg}"

        if continuation_context and not is_review and not is_cluster_summary and not error_msg:
            content = _merge_continuation_output(
                str(continuation_context.get("partial_output") or ""),
                content,
            )

        new_outputs = state.get("agent_outputs", {}).copy()
        if is_review:
            new_outputs["review_agent"] = content
        else:
            new_outputs[agent_id] = content
        errors = list(state.get("errors", []))
        if error_msg:
            _log.warning(error_msg)
            errors.append(error_msg)

        return {
            "current_agent": agent_id if not is_review else "review_agent",
            "agent_outputs": new_outputs,
            "output_artifacts": state.get("output_artifacts", {}).copy(),
            "errors": errors,
            "messages": [AIMessage(content=f"[{name}] says:\n{content}")],
        }

    return _invoke_agent


def compile_orchestration_graph(
    graph_json: dict[str, Any],
    *,
    task: str,
    loop_count: int = 1,
    default_timeout: int = 60,
    provider_registry: ModelProviderRegistry | None = None,
):
    """
    Compiles a Harness Studio JSON definition into an executable LangGraph StateGraph.
    Uses topological sorting to determine linear execution order.
    """
    builder = StateGraph(OrchestrationState)
    
    agents = graph_json.get("agents") or []
    edges = graph_json.get("edges") or []
    provider_config = graph_json.get("provider_config") or {}
    review_config = graph_json.get("review_agent") or {}
    registry = provider_registry or get_provider_registry()
    agents, edges = _expand_cluster_agents(agents, edges)

    if not agents:
        # Empty graph fallback
        async def empty_node(state: OrchestrationState):
            return {"errors": ["No visible agents on canvas"]}
        builder.add_node("empty", empty_node)
        builder.add_edge(START, "empty")
        builder.add_edge("empty", END)
        return builder.compile()

    execution_order = _topological_sort(agents, edges)
    agent_by_id = {str(a["agent_id"]): a for a in agents}
    
    # 1. Build Nodes
    node_names = []
    for aid in execution_order:
        agent_conf = agent_by_id.get(aid)
        if not agent_conf:
            continue
        node_name = f"agent_{aid}"
        node_names.append(node_name)
        builder.add_node(node_name, _build_agent_node(agent_conf, provider_config, default_timeout, registry))

    review_enabled = bool(review_config.get("enabled", True))
    if review_enabled:
        review_agent = {
            "agent_id": "review_agent",
            "name": str(review_config.get("name") or "Reviewer"),
            "system_prompt": str(review_config.get("system_prompt") or ""),
            "model": str(review_config.get("model") or "gpt-5.1-codex-mini"),
            "preferred_provider_id": review_config.get("preferred_provider_id"),
            "fallback_provider_id": review_config.get("fallback_provider_id"),
            "temperature": float(review_config.get("temperature", 0)),
            "timeout_seconds": review_config.get("timeout_seconds"),
        }
        builder.add_node(
            "review_agent",
            _build_agent_node(review_agent, provider_config, default_timeout, registry, is_review=True),
        )

    # 2. Build Linear Edges
    if node_names:
        builder.add_edge(START, node_names[0])
        for i in range(len(node_names) - 1):
            builder.add_edge(node_names[i], node_names[i+1])
            
        last_agent_node = node_names[-1]
        
        if review_enabled:
            builder.add_edge(last_agent_node, "review_agent")
            end_or_loop_node = "review_agent"
        else:
            end_or_loop_node = last_agent_node

        # 3. Add looping logic
        def get_loop_router(max_loops: int):
            def route(state: OrchestrationState) -> str:
                current_loop = int(state.get("loop_index", 0)) + 1
                if current_loop < max_loops:
                    return "loop_bump"
                return END
            return route

        if loop_count > 1:
            async def bump_loop(state: OrchestrationState):
                return {"loop_index": int(state.get("loop_index", 0)) + 1}
            builder.add_node("loop_bump", bump_loop)
            builder.add_conditional_edges(end_or_loop_node, get_loop_router(loop_count), ["loop_bump", END])
            builder.add_edge("loop_bump", node_names[0])
        else:
            builder.add_edge(end_or_loop_node, END)
    
    return builder.compile()
