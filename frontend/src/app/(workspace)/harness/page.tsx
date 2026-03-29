'use client';

import Link from 'next/link';
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import {
  CheckCircle2,
  Clock3,
  GitBranchPlus,
  Layers3,
  Link2,
  Play,
  PlusCircle,
  RefreshCw,
  RotateCcw,
  Save,
  ShieldCheck,
  ShieldX,
  Sparkles,
  Unplug,
  Waypoints,
  Workflow,
} from 'lucide-react';
import {
  HarnessApprovalDTO,
  HarnessCanvasAgentDTO,
  HarnessClusterMemberDTO,
  HarnessEventDTO,
  HarnessProjectDetailDTO,
  HarnessRunDetailDTO,
  HarnessRunSummaryDTO,
  useHarnessApprovalMutation,
  useHarnessCreateStudioProjectMutation,
  useHarnessCurrentStudioProjectQuery,
  useHarnessPoliciesQuery,
  useHarnessRetryRunMutation,
  useHarnessRunDetailQuery,
  useHarnessRunsQuery,
  useHarnessSkillDecisionMutation,
  useHarnessSkillRequestMutation,
  useHarnessStudioProjectQuery,
  useHarnessStudioProjectsQuery,
  useHarnessStudioRunMutation,
  useHarnessUpdateStudioProjectMutation,
  useHarnessModelProvidersQuery,
} from '@/domains/harness/hooks';

function formatTimestamp(value?: number | null) {
  if (!value) {
    return 'Not recorded';
  }
  const normalized = value > 1_000_000_000_000 ? value : value * 1000;
  return new Date(normalized).toLocaleString();
}

function statusTone(status?: string | null) {
  switch (status) {
    case 'completed':
    case 'approved':
    case 'pass':
    case 'loaded':
      return 'bg-emerald-100 text-emerald-900 ring-emerald-200';
    case 'failed':
    case 'rejected':
    case 'fail':
      return 'bg-rose-100 text-rose-900 ring-rose-200';
    case 'waiting_approval':
    case 'pending':
      return 'bg-amber-100 text-amber-900 ring-amber-200';
    case 'resumed':
    case 'running':
    case 'verifying':
    case 'queued':
      return 'bg-sky-100 text-sky-900 ring-sky-200';
    default:
      return 'bg-slate-100 text-slate-800 ring-slate-200';
  }
}

function StatusPill({ value }: { value?: string | null }) {
  const text = value || 'unknown';
  return (
    <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${statusTone(text)}`}>
      {text}
    </span>
  );
}

function JsonBlock({ value }: { value?: Record<string, unknown> | null }) {
  if (!value || Object.keys(value).length === 0) {
    return <div className="text-sm text-slate-500">No structured payload.</div>;
  }
  return (
    <pre className="overflow-x-auto rounded-2xl bg-slate-950/95 p-4 text-xs leading-6 text-slate-100">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

function makeId(prefix: string) {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `${prefix}_${crypto.randomUUID().slice(0, 8)}`;
  }
  return `${prefix}_${Date.now()}`;
}

function topologicalSortAgents(
  agents: HarnessCanvasAgentDTO[],
  edges: { source_agent_id: string; target_agent_id: string }[]
) {
  const inDegree = new Map<string, number>();
  const adjacency = new Map<string, string[]>();

  for (const agent of agents) {
    inDegree.set(agent.agent_id, 0);
    adjacency.set(agent.agent_id, []);
  }

  for (const edge of edges) {
    if (!inDegree.has(edge.source_agent_id) || !inDegree.has(edge.target_agent_id)) {
      continue;
    }
    adjacency.get(edge.source_agent_id)?.push(edge.target_agent_id);
    inDegree.set(edge.target_agent_id, (inDegree.get(edge.target_agent_id) ?? 0) + 1);
  }

  const queue = agents.filter((agent) => (inDegree.get(agent.agent_id) ?? 0) === 0).map((agent) => agent.agent_id);
  const ordered: string[] = [];

  while (queue.length > 0) {
    const current = queue.shift();
    if (!current) {
      continue;
    }
    ordered.push(current);
    for (const next of adjacency.get(current) ?? []) {
      const nextDegree = (inDegree.get(next) ?? 0) - 1;
      inDegree.set(next, nextDegree);
      if (nextDegree === 0) {
        queue.push(next);
      }
    }
  }

  for (const agent of agents) {
    if (!ordered.includes(agent.agent_id)) {
      ordered.push(agent.agent_id);
    }
  }

  return ordered;
}

function normalizeProject(project: HarnessProjectDetailDTO): HarnessProjectDetailDTO {
  return {
    ...project,
    graph_json: {
      version: project.graph_json?.version ?? 1,
      agents: project.graph_json?.agents ?? [],
      edges: project.graph_json?.edges ?? [],
      skill_pool: project.graph_json?.skill_pool ?? [],
      pending_skill_requests: project.graph_json?.pending_skill_requests ?? [],
      skill_catalog: project.graph_json?.skill_catalog ?? [],
      review_agent: project.graph_json?.review_agent ?? {
        enabled: true,
        hidden: true,
        name: 'Compliance reviewer',
        model: 'gpt-5.1-codex-mini',
        system_prompt: '',
      },
      canvas: project.graph_json?.canvas ?? { x: 0, y: 0, zoom: 1 },
    },
  };
}

function createAgentSeed(index: number): HarnessCanvasAgentDTO {
  return {
    agent_id: makeId('agent'),
    name: `New agent ${index + 1}`,
    node_kind: 'agent',
    role: 'specialist',
    description: 'Describe what this agent owns in the loop.',
    system_prompt: 'You are a specialist agent collaborating inside a harness-managed canvas.',
    model: 'gpt-5.2',
    temperature: 0.2,
    max_iterations: 3,
    position: {
      x: 96 + (index % 3) * 240,
      y: 72 + Math.floor(index / 3) * 180,
    },
    skill_ids: [],
    skill_intents: [],
    cluster_members: [],
    brainstorm_rounds: 3,
    cluster_auto_research: false,
    cluster_auto_review: true,
  };
}

function createClusterMemberSeed(name: string, role: string, model: string, systemPrompt: string): HarnessClusterMemberDTO {
  return {
    member_id: makeId('member'),
    name,
    role,
    model,
    system_prompt: systemPrompt,
    temperature: 0.2,
  };
}

function createBrainstormClusterSeed(index: number): HarnessCanvasAgentDTO {
  return {
    agent_id: makeId('cluster'),
    name: `Brainstorm cluster ${index + 1}`,
    node_kind: 'cluster',
    cluster_strategy: 'brainstorm',
    role: 'cluster',
    description: 'A multi-model roundtable that debates approaches and passes forward the strongest direction.',
    system_prompt:
      'Run a high-energy roundtable. The lead proposes, the others challenge or refine, and the cluster converges on clear conservative, balanced, and aggressive paths. Use game theory explicitly to model players, incentives, strategic moves, and likely equilibrium outcomes.',
    model: 'gpt-5.2',
    position: {
      x: 96 + (index % 3) * 240,
      y: 72 + Math.floor(index / 3) * 180,
    },
    cluster_members: [
      createClusterMemberSeed('Lead strategist', 'chair', 'gpt-5.2', 'Set the initial direction, map the core players and incentives, and anchor the debate.'),
      createClusterMemberSeed('Fast challenger', 'critic', 'gpt-5.1-codex-mini', 'Challenge weak assumptions quickly by exposing incentive conflicts, adversarial responses, and equilibrium risks.'),
      createClusterMemberSeed('Synthesis voice', 'synthesizer', 'gpt-5.1-codex-mini', 'Condense the debate into conservative, balanced, and aggressive options with explicit game-theoretic tradeoffs.'),
    ],
    brainstorm_rounds: 3,
    cluster_auto_research: true,
    cluster_auto_review: true,
    skill_ids: [],
    skill_intents: ['research'],
  };
}

function createCustomClusterSeed(index: number): HarnessCanvasAgentDTO {
  return {
    agent_id: makeId('cluster'),
    name: `Custom cluster ${index + 1}`,
    node_kind: 'cluster',
    cluster_strategy: 'custom',
    role: 'cluster',
    description: 'A user-defined cluster that can coordinate multiple internal specialists before handing off.',
    system_prompt: 'Coordinate the member agents inside this cluster and hand off a clean combined output.',
    model: 'gpt-5.2',
    position: {
      x: 96 + (index % 3) * 240,
      y: 72 + Math.floor(index / 3) * 180,
    },
    cluster_members: [
      createClusterMemberSeed('Planner', 'planner', 'gpt-5.2', 'Break down the local cluster task.'),
      createClusterMemberSeed('Builder', 'builder', 'gpt-5.1-codex-mini', 'Execute the strongest next step inside the cluster.'),
    ],
    brainstorm_rounds: 2,
    cluster_auto_research: false,
    cluster_auto_review: true,
    skill_ids: [],
    skill_intents: [],
  };
}

function getNodeExecutionLabels(node: HarnessCanvasAgentDTO) {
  if (node.node_kind !== 'cluster') {
    return [node.name];
  }
  const rounds = node.cluster_strategy === 'brainstorm' ? Math.max(1, Math.min(node.brainstorm_rounds ?? 1, 5)) : 1;
  const memberLabels = Array.from({ length: rounds }).flatMap((_, roundIndex) =>
    (node.cluster_members ?? []).map((member) =>
      `${node.name} / ${member.name}${rounds > 1 ? ` (round ${roundIndex + 1})` : ''}`
    )
  );
  return [...memberLabels, `${node.name} / summary`];
}

function ProviderSelect({
  label,
  value,
  onChange,
  providers,
  emptyLabel,
}: {
  label: string;
  value?: string | null;
  onChange: (nextValue: string | null) => void;
  providers?: { provider_id: string; name: string }[];
  emptyLabel: string;
}) {
  return (
    <label className="block text-sm font-medium text-slate-800">
      {label}
      <select
        value={value || ''}
        onChange={(event) => onChange(event.target.value || null)}
        className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
      >
        <option value="">{emptyLabel}</option>
        {providers?.map((provider) => (
          <option key={provider.provider_id} value={provider.provider_id}>
            {provider.name}
          </option>
        ))}
      </select>
    </label>
  );
}

function normalizeOutputArtifacts(value: unknown) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return [];
  }
  return Object.entries(value as Record<string, unknown>).filter(
    ([, artifact]) => artifact && typeof artifact === 'object' && !Array.isArray(artifact)
  ) as Array<[string, Record<string, unknown>]>;
}

function humanizeRecoveryMode(value: string | null | undefined) {
  switch (value) {
    case 'continue_with_research':
      return 'Continue With Research';
    case 'continue_without_research':
      return 'Continue Without Research';
    case 'continue_with_partial_stream_output':
      return 'Continue With Stream Prefix';
    case 'continue_from_stream_block':
      return 'Continue From Stream Block';
    default:
      return value || 'Standard execution';
  }
}

function getApprovalReviewStage(approval?: HarnessApprovalDTO | null) {
  const payload = approval?.payload_json;
  return typeof payload?.review_stage === 'string' ? payload.review_stage : null;
}

function findLatestEvent(events: HarnessEventDTO[] | undefined, eventType: string) {
  if (!events || events.length === 0) {
    return null;
  }
  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (events[index]?.event_type === eventType) {
      return events[index];
    }
  }
  return null;
}

function getRunContinuationLabel(run: HarnessRunSummaryDTO) {
  const continuation = run.runtime_state?.continuation;
  if (!continuation?.enabled) {
    return null;
  }
  if (continuation.status === 'completed' || run.status === 'completed') {
    return 'stream continued';
  }
  if (run.latest_approval?.status === 'pending') {
    return 'stream review pending';
  }
  if (run.latest_approval?.status === 'rejected') {
    return 'stream continuation rejected';
  }
  if (run.latest_approval?.status === 'approved' || run.status === 'approved' || run.status === 'resumed' || run.status === 'running') {
    return 'stream continuation';
  }
  return 'stream review';
}

function buildStreamContinuationSnapshot({
  approval,
  events,
  recoveryMode,
  runtimeState,
}: {
  approval?: HarnessApprovalDTO | null;
  events?: HarnessEventDTO[];
  recoveryMode?: string | null;
  runtimeState?: HarnessRunDetailDTO['runtime_state'] | null;
}) {
  const reviewStage = getApprovalReviewStage(approval);
  const payload = approval?.payload_json || null;
  const resumedEvent = findLatestEvent(events, 'orchestration.stream_continuation_resumed');
  const completedEvent = findLatestEvent(events, 'orchestration.stream_continuation_completed');
  const continuation = runtimeState?.continuation;
  const visible =
    reviewStage === 'agent_output_stream' ||
    recoveryMode === 'continue_with_partial_stream_output' ||
    recoveryMode === 'continue_from_stream_block' ||
    Boolean(continuation?.enabled) ||
    Boolean(resumedEvent) ||
    Boolean(completedEvent);
  if (!visible) {
    return null;
  }

  const partialOutput = typeof payload?.partial_output === 'string' ? payload.partial_output : null;
  const prefixLength =
    continuation?.prefix_length ??
    partialOutput?.length ??
    coerceNumber(resumedEvent?.details_json?.partial_length) ??
    0;
  const stepIndex =
    continuation?.step_index ??
    coerceNumber(payload?.step_index) ??
    coerceNumber(resumedEvent?.details_json?.next_step_index);
  let phase = continuation?.status ? `${continuation.status[0]?.toUpperCase() || ''}${continuation.status.slice(1)}` : 'Awaiting approval';
  let tone: string = continuation?.status || 'pending';
  if (approval?.status === 'rejected') {
    phase = 'Continuation rejected';
    tone = 'rejected';
  } else if (continuation?.status === 'completed' || completedEvent) {
    phase = 'Continuation completed';
    tone = 'completed';
  } else if (continuation?.status === 'resumed' || resumedEvent) {
    phase = 'Continuation resumed';
    tone = 'resumed';
  } else if (continuation?.status === 'approved' || approval?.status === 'approved') {
    phase = 'Prefix approved';
    tone = 'approved';
  }

  return {
    phase,
    tone,
    prefixLength,
    stepIndex,
    resumedAt: continuation?.resumed_at ?? resumedEvent?.created_at ?? null,
    completedAt: continuation?.completed_at ?? completedEvent?.created_at ?? null,
    recoveryMode: continuation?.mode || recoveryMode,
  };
}

function coerceNumber(value: unknown) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function renderSegmentWindow(startChar: number | null, endChar: number | null) {
  if (startChar === null && endChar === null) {
    return null;
  }
  const startLabel = startChar === null ? '?' : `${startChar}`;
  const endLabel = endChar === null ? '?' : `${endChar}`;
  return `${startLabel} -> ${endLabel}`;
}

function ApprovalSummary({ approval }: { approval?: HarnessApprovalDTO | null }) {
  if (!approval) {
    return <p className="text-sm text-slate-500">No approval record yet.</p>;
  }

  const payload = approval.payload_json || null;
  const blockedAgentName = typeof payload?.agent_name === 'string' ? payload.agent_name : null;
  const blockedAgentId = typeof payload?.agent_id === 'string' ? payload.agent_id : null;
  const reviewOutput = typeof payload?.review_output === 'string' ? payload.review_output : null;
  const reviewStage = typeof payload?.review_stage === 'string' ? payload.review_stage : null;
  const rollbackStepIndex =
    typeof payload?.step_index === 'number'
      ? payload.step_index
      : typeof payload?.step_index === 'string'
        ? Number(payload.step_index)
        : null;
  const loopNumber =
    coerceNumber(payload?.loop_number);
  const segmentIndex = coerceNumber(payload?.segment_index);
  const segmentCount = coerceNumber(payload?.segment_count);
  const segmentStartChar = coerceNumber(payload?.segment_start_char);
  const segmentEndChar = coerceNumber(payload?.segment_end_char);
  const segmentPreview = typeof payload?.segment_preview === 'string' ? payload.segment_preview : null;
  const partialOutput = typeof payload?.partial_output === 'string' ? payload.partial_output : null;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm font-semibold text-slate-900">{approval.action_type || 'approval'}</div>
        <StatusPill value={approval.status} />
      </div>
      <dl className="grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
        <div>
          <dt className="text-slate-500">Requested by</dt>
          <dd className="mt-1">{approval.requested_by || 'unknown'}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Resolved by</dt>
          <dd className="mt-1">{approval.resolved_by || 'Not resolved'}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Created</dt>
          <dd className="mt-1">{formatTimestamp(approval.created_at)}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Resolved</dt>
          <dd className="mt-1">{formatTimestamp(approval.resolved_at)}</dd>
        </div>
      </dl>
      {approval.reason ? <div className="rounded-2xl bg-amber-50 p-4 text-sm text-amber-900">{approval.reason}</div> : null}
      {approval.action_type === 'orchestration_review' ? (
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-950">
          <div className="font-semibold">Blocked orchestration output</div>
          <div className="mt-2 space-y-1">
            <div>Node: {blockedAgentName || blockedAgentId || 'unknown node'}</div>
            {reviewStage === 'cluster_research' ? <div>Stage: cluster research evidence</div> : null}
            {reviewStage === 'agent_output_segment' ? <div>Stage: pipeline output segment</div> : null}
            {reviewStage === 'agent_output_stream' ? <div>Stage: live streaming output guard</div> : null}
            {loopNumber ? <div>Loop: {loopNumber}</div> : null}
            {segmentIndex !== null ? (
              <div>
                Segment: {segmentIndex + 1}
                {segmentCount !== null ? ` / ${segmentCount}` : ''}
              </div>
            ) : null}
            {renderSegmentWindow(segmentStartChar, segmentEndChar) ? (
              <div>Character window: {renderSegmentWindow(segmentStartChar, segmentEndChar)}</div>
            ) : null}
            {typeof rollbackStepIndex === 'number' && Number.isFinite(rollbackStepIndex) ? (
              <div>Rollback target: previous safe state before step {rollbackStepIndex + 1}</div>
            ) : null}
          </div>
          {reviewOutput ? <div className="mt-3 rounded-2xl bg-white/80 p-3 text-sm text-rose-900">{reviewOutput}</div> : null}
          {reviewStage === 'agent_output_stream' ? (
            <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-950">
              Approval will rerun this node with the accepted partial output as a continuation prefix, then continue the workflow. It will not resume the exact same low-level model stream from the interruption point.
            </div>
          ) : null}
          {segmentPreview ? (
            <div className="mt-3 rounded-2xl border border-rose-200 bg-white/80 p-3 text-sm text-rose-950">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-rose-500">Blocked segment preview</div>
              <pre className="mt-2 whitespace-pre-wrap break-words font-mono text-xs">{segmentPreview}</pre>
            </div>
          ) : null}
          {partialOutput ? (
            <div className="mt-3 rounded-2xl border border-slate-200 bg-white/80 p-3 text-slate-950">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Accepted Partial Output</div>
              <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">{partialOutput}</pre>
            </div>
          ) : null}
        </div>
      ) : null}
      {approval.comment ? <div className="rounded-2xl bg-slate-100 p-4 text-sm text-slate-700">{approval.comment}</div> : null}
      {payload ? (
        <details className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <summary className="cursor-pointer text-sm font-medium text-slate-800">Approval payload</summary>
          <div className="mt-3">
            <JsonBlock value={payload} />
          </div>
        </details>
      ) : null}
    </div>
  );
}

function EventRow({ event }: { event: HarnessEventDTO }) {
  const details = event.details_json || null;
  const blockedSegmentIndex = coerceNumber(details?.blocked_segment_index);
  const segmentsReviewed = coerceNumber(details?.segments_reviewed);
  const segmentCount = coerceNumber(details?.segment_count);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <StatusPill value={event.event_type} />
          <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{event.event_source || 'harness'}</div>
        </div>
        <div className="text-xs text-slate-500">{formatTimestamp(event.created_at)}</div>
      </div>
      <div className="mt-3 text-sm text-slate-700">
        actor: <span className="font-medium text-slate-900">{event.actor || 'system'}</span>
      </div>
      {event.event_type === 'orchestration.review_segment_scan_completed' ? (
        <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-950">
          <div className="font-semibold">Pipeline review scan</div>
          <div className="mt-2 space-y-1 text-sm">
            <div>Node: {typeof details?.agent_name === 'string' ? details.agent_name : typeof details?.agent_id === 'string' ? details.agent_id : 'unknown node'}</div>
            {segmentsReviewed !== null || segmentCount !== null ? (
              <div>
                Segments reviewed: {segmentsReviewed ?? '?'}
                {segmentCount !== null ? ` / ${segmentCount}` : ''}
              </div>
            ) : null}
            {blockedSegmentIndex !== null ? <div>Blocked at segment: {blockedSegmentIndex + 1}</div> : <div>No risky segment detected</div>}
          </div>
        </div>
      ) : null}
      {event.event_type === 'orchestration.review_stream_blocked' ? (
        <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-950">
          <div className="font-semibold">Live stream guard blocked output</div>
          <div className="mt-2 space-y-1 text-sm">
            <div>Node: {typeof details?.agent_name === 'string' ? details.agent_name : typeof details?.agent_id === 'string' ? details.agent_id : 'unknown node'}</div>
            <div>
              Character window: {renderSegmentWindow(coerceNumber(details?.segment_start_char), coerceNumber(details?.segment_end_char)) || 'unknown'}
            </div>
            <div>Partial output length: {coerceNumber(details?.partial_length) ?? 0}</div>
          </div>
        </div>
      ) : null}
      {event.event_type === 'orchestration.stream_continuation_resumed' ? (
        <div className="mt-3 rounded-2xl border border-sky-200 bg-sky-50 p-3 text-sm text-sky-950">
          <div className="font-semibold">Stream continuation resumed</div>
          <div className="mt-2 space-y-1 text-sm">
            <div>Node: {typeof details?.agent_name === 'string' ? details.agent_name : typeof details?.agent_id === 'string' ? details.agent_id : 'unknown node'}</div>
            <div>Partial prefix length: {coerceNumber(details?.partial_length) ?? 0}</div>
            <div>Resumed at step index: {coerceNumber(details?.next_step_index) ?? 0}</div>
          </div>
        </div>
      ) : null}
      {event.event_type === 'orchestration.stream_continuation_completed' ? (
        <div className="mt-3 rounded-2xl border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-950">
          <div className="font-semibold">Stream continuation completed</div>
          <div className="mt-2">
            Node: {typeof details?.agent_name === 'string' ? details.agent_name : typeof details?.agent_id === 'string' ? details.agent_id : 'unknown node'}
          </div>
        </div>
      ) : null}
      {event.details_json && Object.keys(event.details_json).length > 0 ? (
        <div className="mt-3">
          <JsonBlock value={event.details_json} />
        </div>
      ) : null}
    </div>
  );
}

function RunRow({
  run,
  selected,
  onSelect,
}: {
  run: HarnessRunSummaryDTO;
  selected: boolean;
  onSelect: () => void;
}) {
  const continuationLabel = getRunContinuationLabel(run);
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full rounded-3xl border p-4 text-left transition ${
        selected
          ? 'border-cyan-300 bg-cyan-50 shadow-[0_18px_60px_-40px_rgba(8,145,178,0.65)]'
          : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-900">{run.task_type || 'unknown_task'}</div>
          <div className="mt-1 text-xs text-slate-500">{run.run_id}</div>
        </div>
        <StatusPill value={run.status} />
      </div>
      <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-slate-500">Step</dt>
          <dd className="mt-1 text-slate-900">{run.current_step || 'idle'}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Verification</dt>
          <dd className="mt-1 text-slate-900">{run.latest_verification?.status || run.verification_status || 'pending'}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Approval</dt>
          <dd className="mt-1 text-slate-900">{run.latest_approval?.status || 'none'}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Updated</dt>
          <dd className="mt-1 text-slate-900">{formatTimestamp(run.updated_at)}</dd>
        </div>
      </dl>
      {continuationLabel ? (
        <div className="mt-3 inline-flex rounded-full bg-sky-50 px-3 py-1 text-xs font-semibold text-sky-900 ring-1 ring-inset ring-sky-200">
          {continuationLabel}
        </div>
      ) : null}
    </button>
  );
}

export default function HarnessPage() {
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [localDraftProject, setLocalDraftProject] = useState<HarnessProjectDetailDTO | null>(null);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [connectionSourceId, setConnectionSourceId] = useState<string | null>(null);
  const [selectedAgentIdsForRun, setSelectedAgentIdsForRun] = useState<string[]>([]);
  const [approvalComment, setApprovalComment] = useState('');
  const [loopCount, setLoopCount] = useState(1);
  const [taskText, setTaskText] = useState('');
  const [timeoutValue, setTimeoutValue] = useState<string>('');
  const [editorNotice, setEditorNotice] = useState<string | null>(null);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ agentId: string; offsetX: number; offsetY: number } | null>(null);

  const projectsQuery = useHarnessStudioProjectsQuery();
  const currentProjectQuery = useHarnessCurrentStudioProjectQuery();
  const createProjectMutation = useHarnessCreateStudioProjectMutation();
  const updateProjectMutation = useHarnessUpdateStudioProjectMutation();
  const skillRequestMutation = useHarnessSkillRequestMutation();
  const skillDecisionMutation = useHarnessSkillDecisionMutation();
  const studioRunMutation = useHarnessStudioRunMutation();
  const runsQuery = useHarnessRunsQuery();
  const policiesQuery = useHarnessPoliciesQuery();
  const approvalMutation = useHarnessApprovalMutation();
  const retryRunMutation = useHarnessRetryRunMutation();
  const providersQuery = useHarnessModelProvidersQuery();

  const activeProjectId = useMemo(() => {
    if (selectedProjectId) {
      return selectedProjectId;
    }
    if (currentProjectQuery.data?.project_id) {
      return currentProjectQuery.data.project_id;
    }
    return projectsQuery.data?.projects?.[0]?.project_id ?? null;
  }, [currentProjectQuery.data?.project_id, projectsQuery.data?.projects, selectedProjectId]);

  const projectQuery = useHarnessStudioProjectQuery(activeProjectId);
  const runs = useMemo(() => runsQuery.data?.runs ?? [], [runsQuery.data?.runs]);
  const activeRunId = useMemo(() => {
    if (selectedRunId && runs.some((run) => run.run_id === selectedRunId)) {
      return selectedRunId;
    }
    return runs[0]?.run_id ?? null;
  }, [runs, selectedRunId]);
  const detailQuery = useHarnessRunDetailQuery(activeRunId);
  const selectedRun = detailQuery.data;

  const studioProject = useMemo(() => {
    if (projectQuery.data) {
      return projectQuery.data;
    }
    if (activeProjectId && currentProjectQuery.data?.project_id === activeProjectId) {
      return currentProjectQuery.data;
    }
    return null;
  }, [activeProjectId, currentProjectQuery.data, projectQuery.data]);
  const normalizedStudioProject = useMemo(() => (studioProject ? normalizeProject(studioProject) : null), [studioProject]);

  useEffect(() => {
    const handleMove = (event: PointerEvent) => {
      const drag = dragRef.current;
      const canvas = canvasRef.current;
      if (!drag || !canvas) {
        return;
      }
      const rect = canvas.getBoundingClientRect();
      const nextX = event.clientX - rect.left - drag.offsetX;
      const nextY = event.clientY - rect.top - drag.offsetY;
      setLocalDraftProject((current) => {
        const baseline = current?.project_id === normalizedStudioProject?.project_id ? current : normalizedStudioProject;
        if (!baseline) {
          return baseline;
        }
        const agents = baseline.graph_json.agents?.map((agent) =>
          agent.agent_id === drag.agentId
            ? {
                ...agent,
                position: {
                  x: Math.max(16, nextX),
                  y: Math.max(16, nextY),
                },
              }
            : agent
        );
        return {
          ...baseline,
          graph_json: {
            ...baseline.graph_json,
            agents,
          },
        };
      });
    };

    const handleUp = () => {
      dragRef.current = null;
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
    return () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };
  }, [normalizedStudioProject]);

  const draftProject = useMemo(() => {
    if (!normalizedStudioProject) {
      return null;
    }
    if (localDraftProject?.project_id === normalizedStudioProject.project_id) {
      return localDraftProject;
    }
    return normalizedStudioProject;
  }, [localDraftProject, normalizedStudioProject]);
  const graph = draftProject?.graph_json;
  const agents = useMemo(() => graph?.agents ?? [], [graph?.agents]);
  const edges = useMemo(() => graph?.edges ?? [], [graph?.edges]);
  const effectiveSelectedAgentId = useMemo(() => {
    if (selectedAgentId && agents.some((agent) => agent.agent_id === selectedAgentId)) {
      return selectedAgentId;
    }
    return agents[0]?.agent_id ?? null;
  }, [agents, selectedAgentId]);
  const effectiveSelectedAgentIdsForRun = useMemo(
    () => selectedAgentIdsForRun.filter((agentId) => agents.some((agent) => agent.agent_id === agentId)),
    [agents, selectedAgentIdsForRun]
  );
  const selectedAgent = useMemo(
    () => agents.find((agent) => agent.agent_id === effectiveSelectedAgentId) ?? null,
    [agents, effectiveSelectedAgentId]
  );
  const pendingApproval = useMemo(() => {
    if (!selectedRun?.latest_approval) {
      return null;
    }
    return selectedRun.latest_approval.status === 'pending' ? selectedRun.latest_approval : null;
  }, [selectedRun]);
  const pendingReviewStage = useMemo(() => {
    const payload = pendingApproval?.payload_json;
    return typeof payload?.review_stage === 'string' ? payload.review_stage : null;
  }, [pendingApproval]);
  const rejectedReviewApproval = useMemo(() => {
    if (!selectedRun?.latest_approval) {
      return null;
    }
    const approval = selectedRun.latest_approval;
    if (approval.status === 'rejected' && approval.action_type === 'orchestration_review') {
      return approval;
    }
    return null;
  }, [selectedRun]);
  const rejectedReviewStage = useMemo(() => {
    const payload = rejectedReviewApproval?.payload_json;
    return typeof payload?.review_stage === 'string' ? payload.review_stage : null;
  }, [rejectedReviewApproval]);
  const outputArtifacts = useMemo(
    () => normalizeOutputArtifacts(selectedRun?.latest_verification?.artifacts_json?.output_artifacts),
    [selectedRun]
  );
  const recoveryMode = useMemo(() => {
    const verificationArtifacts = selectedRun?.latest_verification?.artifacts_json;
    const fromVerification =
      verificationArtifacts && typeof verificationArtifacts === 'object'
        ? verificationArtifacts.recovery_mode
        : null;
    if (typeof fromVerification === 'string' && fromVerification) {
      return fromVerification;
    }
    const metadataMode = selectedRun?.metadata_json?.review_recovery_mode;
    return typeof metadataMode === 'string' && metadataMode ? metadataMode : null;
  }, [selectedRun]);
  const streamContinuationSnapshot = useMemo(
    () =>
      buildStreamContinuationSnapshot({
        approval: selectedRun?.latest_approval,
        events: selectedRun?.events,
        recoveryMode,
        runtimeState: selectedRun?.runtime_state,
      }),
    [recoveryMode, selectedRun]
  );
  const orchestrationPolicy = useMemo(
    () => policiesQuery.data?.policies?.find((policy) => policy.task_type === 'agent_orchestration') ?? null,
    [policiesQuery.data?.policies]
  );
  const providerOptions = useMemo(() => providersQuery.data?.providers ?? [], [providersQuery.data?.providers]);
  const providerNameById = useMemo(
    () => new Map(providerOptions.map((provider) => [provider.provider_id, provider.name])),
    [providerOptions]
  );
  const executionPreview = useMemo(() => {
    if (!agents.length) {
      return { all: [] as string[], selected: [] as string[] };
    }

    const buildPreview = (selectedIds: string[]) => {
      const selectedSet = new Set(selectedIds);
      const scopedAgents = selectedIds.length > 0 ? agents.filter((agent) => selectedSet.has(agent.agent_id)) : agents;
      const scopedEdges =
        selectedIds.length > 0
          ? edges.filter((edge) => selectedSet.has(edge.source_agent_id) && selectedSet.has(edge.target_agent_id))
          : edges;
      const orderedIds = topologicalSortAgents(scopedAgents, scopedEdges);
      const orderedNames = orderedIds.flatMap((agentId) => {
        const node = scopedAgents.find((agent) => agent.agent_id === agentId);
        return node ? getNodeExecutionLabels(node) : [agentId];
      });
      const withReview =
        graph?.review_agent?.enabled === false ? orderedNames : [...orderedNames, graph?.review_agent?.name || 'Compliance reviewer'];
      const repeated = Array.from({ length: Math.max(loopCount, 1) }).flatMap((_, index) =>
        withReview.map((name) => `${name}${loopCount > 1 ? ` (loop ${index + 1})` : ''}`)
      );
      return repeated;
    };

    return {
      all: buildPreview([]),
      selected: buildPreview(effectiveSelectedAgentIdsForRun),
    };
  }, [agents, edges, effectiveSelectedAgentIdsForRun, graph?.review_agent?.enabled, graph?.review_agent?.name, loopCount]);
  const runReady = !!draftProject?.project_id && agents.length > 0;

  const updateDraftProject = (updater: (current: HarnessProjectDetailDTO) => HarnessProjectDetailDTO) => {
    setLocalDraftProject((current) => {
      const baseline = current?.project_id === draftProject?.project_id ? current : draftProject;
      return baseline ? updater(baseline) : baseline;
    });
  };

  const handleProjectCreate = () => {
    const nextIndex = (projectsQuery.data?.projects?.length ?? 0) + 1;
    createProjectMutation.mutate(
      {
        name: `Agent studio ${nextIndex}`,
        description: 'Canvas workspace for harness-managed agent collaboration.',
      },
      {
        onSuccess: (payload) => {
          setSelectedProjectId(payload.project_id);
          setLocalDraftProject(normalizeProject(payload));
          setEditorNotice('New studio project created.');
        },
      }
    );
  };

  const handleSaveProject = () => {
    if (!draftProject) {
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: draftProject.graph_json,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload));
          setEditorNotice('Studio saved.');
        },
      }
    );
  };

  const handleAddAgent = () => {
    const nextAgent = createAgentSeed(agents.length);
    updateDraftProject((current) => ({
      ...current,
      graph_json: {
        ...current.graph_json,
        agents: [...(current.graph_json.agents ?? []), nextAgent],
      },
    }));
    setSelectedAgentId(nextAgent.agent_id);
    setEditorNotice('Added a new canvas agent.');
  };

  const handleAddCluster = (strategy: 'brainstorm' | 'custom') => {
    const nextNode =
      strategy === 'brainstorm' ? createBrainstormClusterSeed(agents.length) : createCustomClusterSeed(agents.length);
    updateDraftProject((current) => ({
      ...current,
      graph_json: {
        ...current.graph_json,
        agents: [...(current.graph_json.agents ?? []), nextNode],
      },
    }));
    setSelectedAgentId(nextNode.agent_id);
    setEditorNotice(strategy === 'brainstorm' ? 'Added a brainstorm cluster.' : 'Added a custom cluster.');
  };

  const handleRemoveAgent = (agentId: string) => {
    updateDraftProject((current) => ({
      ...current,
      graph_json: {
        ...current.graph_json,
        agents: (current.graph_json.agents ?? []).filter((agent) => agent.agent_id !== agentId),
        edges: (current.graph_json.edges ?? []).filter(
          (edge) => edge.source_agent_id !== agentId && edge.target_agent_id !== agentId
        ),
        pending_skill_requests: (current.graph_json.pending_skill_requests ?? []).filter((request) => request.agent_id !== agentId),
      },
    }));
    setSelectedAgentId((current) => (current === agentId ? null : current));
    setSelectedAgentIdsForRun((current) => current.filter((value) => value !== agentId));
    setConnectionSourceId((current) => (current === agentId ? null : current));
  };

  const updateSelectedAgent = (updates: Partial<HarnessCanvasAgentDTO>) => {
    if (!effectiveSelectedAgentId) {
      return;
    }
    updateDraftProject((current) => ({
      ...current,
      graph_json: {
        ...current.graph_json,
        agents: (current.graph_json.agents ?? []).map((agent) =>
          agent.agent_id === effectiveSelectedAgentId ? { ...agent, ...updates } : agent
        ),
      },
    }));
  };

  const updateClusterMember = (memberId: string, updates: Partial<HarnessClusterMemberDTO>) => {
    if (!selectedAgent || selectedAgent.node_kind !== 'cluster') {
      return;
    }
    updateSelectedAgent({
      cluster_members: (selectedAgent.cluster_members ?? []).map((member) =>
        member.member_id === memberId ? { ...member, ...updates } : member
      ),
    });
  };

  const handleAddClusterMember = () => {
    if (!selectedAgent || selectedAgent.node_kind !== 'cluster') {
      return;
    }
    const nextIndex = (selectedAgent.cluster_members?.length ?? 0) + 1;
    updateSelectedAgent({
      cluster_members: [
        ...(selectedAgent.cluster_members ?? []),
        createClusterMemberSeed(`Member ${nextIndex}`, 'specialist', 'gpt-5.1-codex-mini', 'Contribute a distinct perspective to the cluster.'),
      ],
    });
  };

  const toggleRunSelection = (agentId: string) => {
    setSelectedAgentIdsForRun((current) =>
      current.includes(agentId) ? current.filter((value) => value !== agentId) : [...current, agentId]
    );
  };

  const toggleSkillAssignment = (skillId: string) => {
    if (!selectedAgent) {
      return;
    }
    const currentSkillIds = selectedAgent.skill_ids ?? [];
    updateSelectedAgent({
      skill_ids: currentSkillIds.includes(skillId)
        ? currentSkillIds.filter((value) => value !== skillId)
        : [...currentSkillIds, skillId],
    });
  };

  const handleConnectAgents = (targetAgentId: string) => {
    if (!connectionSourceId || connectionSourceId === targetAgentId) {
      setConnectionSourceId(null);
      return;
    }
    const exists = edges.some(
      (edge) => edge.source_agent_id === connectionSourceId && edge.target_agent_id === targetAgentId
    );
    if (exists) {
      setConnectionSourceId(null);
      return;
    }
    updateDraftProject((current) => ({
      ...current,
      graph_json: {
        ...current.graph_json,
        edges: [
          ...(current.graph_json.edges ?? []),
          {
            edge_id: makeId('edge'),
            source_agent_id: connectionSourceId,
            target_agent_id: targetAgentId,
            interaction: 'handoff',
          },
        ],
      },
    }));
    setConnectionSourceId(null);
  };

  const handleRun = (scope: 'all' | 'selected') => {
    if (!draftProject?.project_id) {
      return;
    }
    studioRunMutation.mutate(
      {
        projectId: draftProject.project_id,
        runScope: scope,
        agentIds: scope === 'selected' ? effectiveSelectedAgentIdsForRun : [],
        loopCount,
        task: taskText.trim() || undefined,
        timeoutSeconds: timeoutValue ? parseInt(timeoutValue, 10) : undefined,
      },
      {
        onSuccess: (payload) => {
          setSelectedRunId(payload.run_id);
          setEditorNotice(scope === 'all' ? 'Started full collaboration loop.' : 'Started partial collaboration loop.');
        },
      }
    );
  };

  const handleRequestSkills = () => {
    if (!draftProject?.project_id || !selectedAgent) {
      return;
    }
    const requestedSkills = (selectedAgent.skill_intents ?? []).map((value) => value.trim()).filter(Boolean);
    if (!requestedSkills.length) {
      setEditorNotice('Add one or more skill intents before requesting skills.');
      return;
    }
    skillRequestMutation.mutate(
      {
        projectId: draftProject.project_id,
        agentId: selectedAgent.agent_id,
        requestedSkills,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload));
          setEditorNotice('Skill requests synced from the source catalog.');
        },
      }
    );
  };

  const handleResolveSkillRequest = (requestId: string, approved: boolean) => {
    if (!draftProject?.project_id) {
      return;
    }
    skillDecisionMutation.mutate(
      {
        projectId: draftProject.project_id,
        requestId,
        approved,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload));
          setEditorNotice(approved ? 'Skill approved into the shared pool.' : 'Skill request rejected.');
        },
      }
    );
  };

  const handleRetry = () => {
    if (!selectedRun) {
      return;
    }
    retryRunMutation.mutate(
      { runId: selectedRun.run_id },
      {
        onSuccess: (payload) => setSelectedRunId(payload.run_id),
      }
    );
  };

  const handleDecision = (approved: boolean) => {
    if (!activeRunId) {
      return;
    }
    approvalMutation.mutate({
      runId: activeRunId,
      approved,
      comment: approvalComment,
    });
  };

  const startDragging = (event: ReactPointerEvent<HTMLButtonElement>, agent: HarnessCanvasAgentDTO) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const position = agent.position ?? { x: 0, y: 0 };
    dragRef.current = {
      agentId: agent.agent_id,
      offsetX: event.clientX - rect.left - position.x,
      offsetY: event.clientY - rect.top - position.y,
    };
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.16),_transparent_26%),radial-gradient(circle_at_bottom_right,_rgba(250,204,21,0.18),_transparent_24%),linear-gradient(180deg,#f8fafc_0%,#ecfeff_42%,#fff7ed_100%)]">
      <div className="mx-auto max-w-[1600px] px-4 py-8 sm:px-6 lg:px-8">
        <div className="rounded-[32px] border border-white/70 bg-white/85 p-8 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.45)] backdrop-blur">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-cyan-200">
                <Workflow className="h-3.5 w-3.5" />
                Harness Agent Studio
              </div>
              <h1 className="mt-4 font-serif text-4xl text-slate-950">Create, wire, govern, and launch agents from one canvas.</h1>
              <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-600">
                This workspace adds a frontend control plane on top of the current harness runtime: canvas editing, partial or full orchestration runs,
                skill-pool approvals, and a hidden default review agent that can be configured without cluttering the canvas.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Link
                href="/chat"
                className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50"
              >
                Back to chat
              </Link>
              <button
                type="button"
                onClick={() => {
                  projectsQuery.refetch();
                  currentProjectQuery.refetch();
                  projectQuery.refetch();
                  runsQuery.refetch();
                  detailQuery.refetch();
                }}
                className="inline-flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-800 hover:bg-slate-50"
              >
                <RefreshCw className="h-4 w-4" />
                Refresh
              </button>
              <button
                type="button"
                onClick={handleSaveProject}
                disabled={!draftProject || updateProjectMutation.isPending}
                className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
              >
                <Save className="h-4 w-4" />
                {updateProjectMutation.isPending ? 'Saving...' : 'Save studio'}
              </button>
            </div>
          </div>

          {editorNotice ? (
            <div className="mt-5 rounded-2xl border border-cyan-200 bg-cyan-50 px-4 py-3 text-sm text-cyan-900">{editorNotice}</div>
          ) : null}

          <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950">
            <span className="font-semibold">Immutable runtime rule:</span> every agent, cluster member, and review agent is always forced to reason from first principles.
            This baseline is system-controlled and cannot be turned off or edited in Studio.
          </div>

          <div className="mt-8 grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)_360px]">
            <aside className="space-y-6">
              <section className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-slate-900">Studio projects</div>
                    <div className="mt-1 text-sm text-slate-500">Multiple canvases, same harness runtime.</div>
                  </div>
                  <button
                    type="button"
                    onClick={handleProjectCreate}
                    disabled={createProjectMutation.isPending}
                    className="inline-flex items-center gap-2 rounded-2xl bg-cyan-600 px-3 py-2 text-xs font-semibold text-white hover:bg-cyan-500 disabled:opacity-50"
                  >
                    <PlusCircle className="h-4 w-4" />
                    New
                  </button>
                </div>
                <div className="mt-4 space-y-3">
                  {(projectsQuery.data?.projects ?? []).map((project) => (
                    <button
                      key={project.project_id}
                      type="button"
                      onClick={() => setSelectedProjectId(project.project_id)}
                      className={`w-full rounded-2xl border px-4 py-3 text-left transition ${
                        project.project_id === activeProjectId
                          ? 'border-cyan-300 bg-cyan-50'
                          : 'border-slate-200 bg-slate-50 hover:border-slate-300 hover:bg-white'
                      }`}
                    >
                      <div className="text-sm font-semibold text-slate-900">{project.name}</div>
                      <div className="mt-1 text-xs text-slate-500">{project.agent_count ?? 0} agents</div>
                    </button>
                  ))}
                </div>
              </section>

              <section className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-slate-900">Nodes</div>
                    <div className="mt-1 text-sm text-slate-500">{agents.length} visible on canvas</div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={handleAddAgent}
                      disabled={!draftProject}
                      className="inline-flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-50 disabled:opacity-50"
                    >
                      <GitBranchPlus className="h-4 w-4" />
                      Add agent
                    </button>
                    <button
                      type="button"
                      onClick={() => handleAddCluster('brainstorm')}
                      disabled={!draftProject}
                      className="inline-flex items-center gap-2 rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-semibold text-amber-900 hover:bg-amber-100 disabled:opacity-50"
                    >
                      <Sparkles className="h-4 w-4" />
                      Add brainstorm cluster
                    </button>
                    <button
                      type="button"
                      onClick={() => handleAddCluster('custom')}
                      disabled={!draftProject}
                      className="inline-flex items-center gap-2 rounded-2xl border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-900 hover:bg-emerald-100 disabled:opacity-50"
                    >
                      <Layers3 className="h-4 w-4" />
                      Add custom cluster
                    </button>
                  </div>
                </div>
                <div className="mt-4 space-y-3">
                  {agents.map((agent) => (
                    <div
                      key={agent.agent_id}
                      className={`rounded-2xl border px-4 py-3 ${
                        effectiveSelectedAgentId === agent.agent_id
                          ? 'border-cyan-300 bg-cyan-50'
                          : 'border-slate-200 bg-slate-50'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <button type="button" onClick={() => setSelectedAgentId(agent.agent_id)} className="text-left">
                          <div className="text-sm font-semibold text-slate-900">{agent.name}</div>
                          <div className="mt-1 flex flex-wrap items-center gap-2">
                            <span className="text-xs uppercase tracking-[0.16em] text-slate-500">{agent.role || 'specialist'}</span>
                            <span className="rounded-full bg-slate-200 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-700">
                              {agent.node_kind === 'cluster' ? `${agent.cluster_strategy || 'cluster'} cluster` : 'agent'}
                            </span>
                          </div>
                        </button>
                        <input
                          type="checkbox"
                          checked={effectiveSelectedAgentIdsForRun.includes(agent.agent_id)}
                          onChange={() => toggleRunSelection(agent.agent_id)}
                          className="mt-1 h-4 w-4 rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                        />
                      </div>
                      <div className="mt-3 flex items-center justify-between gap-3 text-xs text-slate-500">
                        <span>
                          {agent.node_kind === 'cluster'
                            ? `${agent.cluster_members?.length ?? 0} internal members`
                            : `${(agent.skill_ids ?? []).length} loaded skills`}
                        </span>
                        <button type="button" onClick={() => handleRemoveAgent(agent.agent_id)} className="text-rose-600 hover:text-rose-500">
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              <section className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <ShieldCheck className="h-4 w-4 text-amber-600" />
                  Review agent
                </div>
                <p className="mt-2 text-sm text-slate-500">
                  Hidden from the canvas, attached after the visible collaboration path, and executed with its own model and provider preferences.
                </p>
                <div className="mt-4 space-y-3">
                  <label className="block text-sm font-medium text-slate-800">
                    Name
                    <input
                      value={graph?.review_agent?.name ?? ''}
                      onChange={(event) =>
                        updateDraftProject((current) => ({
                          ...current,
                          graph_json: {
                            ...current.graph_json,
                            review_agent: {
                              enabled: current.graph_json.review_agent?.enabled ?? true,
                              hidden: current.graph_json.review_agent?.hidden ?? true,
                              model: current.graph_json.review_agent?.model ?? 'gpt-5.1-codex-mini',
                              preferred_provider_id: current.graph_json.review_agent?.preferred_provider_id ?? null,
                              fallback_provider_id: current.graph_json.review_agent?.fallback_provider_id ?? null,
                              system_prompt: current.graph_json.review_agent?.system_prompt ?? '',
                              name: event.target.value,
                            },
                          },
                        }))
                      }
                      className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                    />
                  </label>
                  <label className="block text-sm font-medium text-slate-800">
                    Model
                    <input
                      value={graph?.review_agent?.model ?? ''}
                      onChange={(event) =>
                        updateDraftProject((current) => ({
                          ...current,
                          graph_json: {
                            ...current.graph_json,
                            review_agent: {
                              enabled: current.graph_json.review_agent?.enabled ?? true,
                              hidden: current.graph_json.review_agent?.hidden ?? true,
                              name: current.graph_json.review_agent?.name ?? 'Compliance reviewer',
                              preferred_provider_id: current.graph_json.review_agent?.preferred_provider_id ?? null,
                              fallback_provider_id: current.graph_json.review_agent?.fallback_provider_id ?? null,
                              system_prompt: current.graph_json.review_agent?.system_prompt ?? '',
                              model: event.target.value,
                            },
                          },
                        }))
                      }
                      className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                    />
                  </label>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <ProviderSelect
                      label="Preferred Provider"
                      value={graph?.review_agent?.preferred_provider_id}
                      onChange={(nextValue) =>
                        updateDraftProject((current) => ({
                          ...current,
                          graph_json: {
                            ...current.graph_json,
                            review_agent: {
                              enabled: current.graph_json.review_agent?.enabled ?? true,
                              hidden: current.graph_json.review_agent?.hidden ?? true,
                              name: current.graph_json.review_agent?.name ?? 'Compliance reviewer',
                              model: current.graph_json.review_agent?.model ?? 'gpt-5.1-codex-mini',
                              preferred_provider_id: nextValue,
                              fallback_provider_id: current.graph_json.review_agent?.fallback_provider_id ?? null,
                              system_prompt: current.graph_json.review_agent?.system_prompt ?? '',
                            },
                          },
                        }))
                      }
                      providers={providerOptions}
                      emptyLabel="(Project preference)"
                    />
                    <ProviderSelect
                      label="Fallback Provider"
                      value={graph?.review_agent?.fallback_provider_id}
                      onChange={(nextValue) =>
                        updateDraftProject((current) => ({
                          ...current,
                          graph_json: {
                            ...current.graph_json,
                            review_agent: {
                              enabled: current.graph_json.review_agent?.enabled ?? true,
                              hidden: current.graph_json.review_agent?.hidden ?? true,
                              name: current.graph_json.review_agent?.name ?? 'Compliance reviewer',
                              model: current.graph_json.review_agent?.model ?? 'gpt-5.1-codex-mini',
                              preferred_provider_id: current.graph_json.review_agent?.preferred_provider_id ?? null,
                              fallback_provider_id: nextValue,
                              system_prompt: current.graph_json.review_agent?.system_prompt ?? '',
                            },
                          },
                        }))
                      }
                      providers={providerOptions}
                      emptyLabel="(None)"
                    />
                  </div>
                  <label className="block text-sm font-medium text-slate-800">
                    System prompt
                    <textarea
                      rows={5}
                      value={graph?.review_agent?.system_prompt ?? ''}
                      onChange={(event) =>
                        updateDraftProject((current) => ({
                          ...current,
                          graph_json: {
                            ...current.graph_json,
                            review_agent: {
                              enabled: current.graph_json.review_agent?.enabled ?? true,
                              hidden: current.graph_json.review_agent?.hidden ?? true,
                              name: current.graph_json.review_agent?.name ?? 'Compliance reviewer',
                              model: current.graph_json.review_agent?.model ?? 'gpt-5.1-codex-mini',
                              preferred_provider_id: current.graph_json.review_agent?.preferred_provider_id ?? null,
                              fallback_provider_id: current.graph_json.review_agent?.fallback_provider_id ?? null,
                              system_prompt: event.target.value,
                            },
                          },
                        }))
                      }
                      className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                    />
                  </label>
                </div>
              </section>
            </aside>

            <section className="space-y-6">
              <div className="rounded-[28px] border border-slate-200 bg-white/90 p-6 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                  <div>
                    <div className="text-xs uppercase tracking-[0.24em] text-slate-400">Canvas orchestration</div>
                    <h2 className="mt-2 text-2xl font-semibold text-slate-950">{draftProject?.name || 'Loading studio...'}</h2>
                    <p className="mt-2 text-sm text-slate-500">
                      Drag agents to compose the collaboration graph, connect handoffs, and launch a selected subset or the full loop.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-3">
                    <label className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
                      Task
                      <input
                        type="text"
                        value={taskText}
                        onChange={(event) => setTaskText(event.target.value)}
                        placeholder="Inherit from project name..."
                        className="w-48 rounded-xl border border-slate-200 px-2 py-1 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none"
                      />
                    </label>
                    <label className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
                      Timeout (s)
                      <input
                        type="number"
                        min={5}
                        max={600}
                        value={timeoutValue}
                        onChange={(event) => setTimeoutValue(event.target.value)}
                        placeholder="60"
                        className="w-16 rounded-xl border border-slate-200 px-2 py-1 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none"
                      />
                    </label>
                    <label className="flex items-center gap-2 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700">
                      Loop count
                      <input
                        type="number"
                        min={1}
                        max={10}
                        value={loopCount}
                        onChange={(event) => setLoopCount(Number(event.target.value) || 1)}
                        className="w-16 rounded-xl border border-slate-200 px-2 py-1 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none"
                      />
                    </label>
                    <button
                      type="button"
                      onClick={() => handleRun('selected')}
                      disabled={!runReady || effectiveSelectedAgentIdsForRun.length === 0 || studioRunMutation.isPending}
                      className="inline-flex items-center gap-2 rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm font-semibold text-sky-900 hover:bg-sky-100 disabled:opacity-50"
                    >
                      <Play className="h-4 w-4" />
                      Run selected
                    </button>
                    <button
                      type="button"
                      onClick={() => handleRun('all')}
                      disabled={!runReady || studioRunMutation.isPending}
                      className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
                    >
                      <Workflow className="h-4 w-4" />
                      Run all
                    </button>
                  </div>
                </div>

                <div className="mt-5 grid gap-4 md:grid-cols-3">
                  <div className="rounded-2xl bg-slate-50 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Policy</div>
                    <div className="mt-2 text-sm font-semibold text-slate-900">{orchestrationPolicy?.policy_id || 'agent_orchestration:v1'}</div>
                    <div className="mt-1 text-sm text-slate-500">retry budget {orchestrationPolicy?.retry_budget ?? 1}</div>
                  </div>
                  <div className="rounded-2xl bg-slate-50 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Skill pool</div>
                    <div className="mt-2 text-sm font-semibold text-slate-900">{graph?.skill_pool?.length ?? 0} approved skills</div>
                    <div className="mt-1 text-sm text-slate-500">{graph?.pending_skill_requests?.filter((item) => item.status === 'pending').length ?? 0} pending approvals</div>
                  </div>
                  <div className="rounded-2xl bg-slate-50 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Review path</div>
                    <div className="mt-2 text-sm font-semibold text-slate-900">
                      {graph?.review_agent?.enabled ? graph.review_agent.name : 'disabled'}
                    </div>
                    <div className="mt-1 text-sm text-slate-500">hidden reviewer attached outside the canvas</div>
                    {graph?.review_agent?.enabled ? (
                      <div className="mt-2 space-y-1 text-xs text-slate-500">
                        <div>
                          preferred{' '}
                          {graph.review_agent.preferred_provider_id
                            ? providerNameById.get(graph.review_agent.preferred_provider_id) || graph.review_agent.preferred_provider_id
                            : 'project preference'}
                        </div>
                        <div>
                          fallback{' '}
                          {graph.review_agent.fallback_provider_id
                            ? providerNameById.get(graph.review_agent.fallback_provider_id) || graph.review_agent.fallback_provider_id
                            : 'none'}
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>

                <div className="mt-5 grid gap-4 lg:grid-cols-2">
                  <div className="rounded-2xl bg-slate-50 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Run All Preview</div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {executionPreview.all.length > 0 ? (
                        executionPreview.all.map((step, index) => (
                          <span key={`all-${index}-${step}`} className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700 ring-1 ring-slate-200">
                            {index + 1}. {step}
                          </span>
                        ))
                      ) : (
                        <span className="text-sm text-slate-500">Add agents to preview the compiled run order.</span>
                      )}
                    </div>
                  </div>
                  <div className="rounded-2xl bg-slate-50 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">Run Selected Preview</div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {executionPreview.selected.length > 0 ? (
                        executionPreview.selected.map((step, index) => (
                          <span key={`selected-${index}-${step}`} className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700 ring-1 ring-slate-200">
                            {index + 1}. {step}
                          </span>
                        ))
                      ) : (
                        <span className="text-sm text-slate-500">Select one or more agents to preview partial execution.</span>
                      )}
                    </div>
                  </div>
                </div>

                <div
                  ref={canvasRef}
                  className="relative mt-6 h-[560px] overflow-hidden rounded-[28px] border border-dashed border-slate-300 bg-[linear-gradient(90deg,rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(180deg,rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(180deg,#f8fafc_0%,#ffffff_100%)] bg-[size:32px_32px,32px_32px,100%_100%]"
                >
                  <svg className="pointer-events-none absolute inset-0 h-full w-full">
                    {edges.map((edge) => {
                      const source = agents.find((agent) => agent.agent_id === edge.source_agent_id);
                      const target = agents.find((agent) => agent.agent_id === edge.target_agent_id);
                      if (!source?.position || !target?.position) {
                        return null;
                      }
                      const x1 = source.position.x + 104;
                      const y1 = source.position.y + 40;
                      const x2 = target.position.x + 104;
                      const y2 = target.position.y + 40;
                      const bend = Math.max(48, Math.abs(x2 - x1) / 2);
                      const path = `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
                      return (
                        <g key={edge.edge_id}>
                          <path d={path} fill="none" stroke="rgba(14,116,144,0.55)" strokeWidth="3" strokeLinecap="round" />
                          <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 10} className="fill-slate-500 text-[11px] font-semibold">
                            {edge.interaction || 'handoff'}
                          </text>
                        </g>
                      );
                    })}
                  </svg>

                  {agents.map((agent) => (
                    <div
                      key={agent.agent_id}
                      className={`absolute w-52 rounded-[24px] border shadow-[0_16px_60px_-40px_rgba(15,23,42,0.5)] transition ${
                        effectiveSelectedAgentId === agent.agent_id
                          ? 'border-cyan-300 bg-white ring-4 ring-cyan-100'
                          : 'border-slate-200 bg-white/95'
                      }`}
                      style={{
                        left: agent.position?.x ?? 0,
                        top: agent.position?.y ?? 0,
                      }}
                    >
                      <button
                        type="button"
                        onPointerDown={(event) => startDragging(event, agent)}
                        onClick={() => setSelectedAgentId(agent.agent_id)}
                        className="w-full rounded-t-[24px] bg-[linear-gradient(135deg,#0f172a,#155e75)] px-4 py-3 text-left text-white"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <div className="text-sm font-semibold">{agent.name}</div>
                            <div className="mt-1 text-[11px] uppercase tracking-[0.18em] text-cyan-100">{agent.role || 'specialist'}</div>
                            <div className="mt-2 flex flex-wrap gap-2">
                              <span className="rounded-full bg-white/15 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-50">
                                {agent.node_kind === 'cluster' ? `${agent.cluster_strategy || 'cluster'} cluster` : 'agent'}
                              </span>
                              {agent.node_kind === 'cluster' ? (
                                <span className="rounded-full bg-white/15 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-50">
                                  {agent.cluster_members?.length ?? 0} members
                                </span>
                              ) : null}
                            </div>
                          </div>
                          <Waypoints className="h-4 w-4 text-cyan-200" />
                        </div>
                      </button>
                      <div className="space-y-3 p-4">
                        <p className="min-h-[40px] text-xs leading-5 text-slate-600">{agent.description || 'No description yet.'}</p>
                        <div className="flex flex-wrap gap-2">
                          {agent.node_kind === 'cluster' ? (
                            <StatusPill value={agent.cluster_auto_research ? 'cluster_research' : 'cluster'} />
                          ) : (agent.skill_ids ?? []).length === 0 ? (
                            <StatusPill value="no_skills" />
                          ) : (
                            (agent.skill_ids ?? []).slice(0, 2).map((skillId) => <StatusPill key={skillId} value={skillId} />)
                          )}
                        </div>
                        <div className="flex items-center justify-between gap-2">
                          <button
                            type="button"
                            onClick={() => setConnectionSourceId(agent.agent_id)}
                            className={`inline-flex items-center gap-1 rounded-2xl px-3 py-2 text-xs font-semibold ${
                              connectionSourceId === agent.agent_id
                                ? 'bg-cyan-600 text-white'
                                : 'border border-slate-200 bg-white text-slate-700'
                            }`}
                          >
                            <Link2 className="h-3.5 w-3.5" />
                            Link out
                          </button>
                          <button
                            type="button"
                            onClick={() => handleConnectAgents(agent.agent_id)}
                            disabled={!connectionSourceId || connectionSourceId === agent.agent_id}
                            className="inline-flex items-center gap-1 rounded-2xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 disabled:opacity-50"
                          >
                            <Layers3 className="h-3.5 w-3.5" />
                            Link in
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}

                  {agents.length === 0 ? (
                    <div className="flex h-full items-center justify-center px-8 text-center text-sm text-slate-500">
                      Add your first agent to start composing the collaboration graph.
                    </div>
                  ) : null}
                </div>

                <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-slate-900">Edges</div>
                    {connectionSourceId ? (
                      <div className="inline-flex items-center gap-2 rounded-full bg-cyan-100 px-3 py-1 text-xs font-semibold text-cyan-900">
                        linking from {connectionSourceId}
                      </div>
                    ) : null}
                  </div>
                  <div className="mt-3 space-y-2">
                    {edges.length === 0 ? (
                      <div className="text-sm text-slate-500">No edges yet. Choose “Link out” on one node, then “Link in” on another.</div>
                    ) : (
                      edges.map((edge) => (
                        <div key={edge.edge_id} className="flex items-center justify-between gap-3 rounded-2xl bg-white px-4 py-3 text-sm text-slate-700">
                          <div>
                            <span className="font-medium text-slate-900">{edge.source_agent_id}</span>
                            {' -> '}
                            <span className="font-medium text-slate-900">{edge.target_agent_id}</span>
                            <span className="ml-2 text-slate-500">({edge.interaction || 'handoff'})</span>
                          </div>
                          <button
                            type="button"
                            onClick={() =>
                              updateDraftProject((current) => ({
                                ...current,
                                graph_json: {
                                  ...current.graph_json,
                                  edges: (current.graph_json.edges ?? []).filter((item) => item.edge_id !== edge.edge_id),
                                },
                              }))
                            }
                            className="text-rose-600 hover:text-rose-500"
                          >
                            Remove
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>

              <div className="rounded-[28px] border border-slate-200 bg-white/90 p-6 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <Workflow className="h-4 w-4 text-cyan-600" />
                  Runs and evidence
                </div>
                <div className="mt-5 grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
                  <div className="space-y-3">
                    {runs.length === 0 ? (
                      <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-8 text-sm text-slate-500">
                        No harness runs yet. Launch a canvas orchestration to start generating evidence.
                      </div>
                    ) : (
                      runs.map((run) => (
                        <RunRow
                          key={run.run_id}
                          run={run}
                          selected={run.run_id === activeRunId}
                          onSelect={() => setSelectedRunId(run.run_id)}
                        />
                      ))
                    )}
                  </div>
                  <div className="space-y-6">
                    {!selectedRun ? (
                      <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-8 text-sm text-slate-500">
                        Select a run to inspect approval state, verification, and event evidence.
                      </div>
                    ) : (
                      <>
                        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
                          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                            <div>
                              <div className="text-xs uppercase tracking-[0.24em] text-slate-400">Run detail</div>
                              <h3 className="mt-2 text-xl font-semibold text-slate-950">{selectedRun.task_type || 'unknown_task'}</h3>
                              <div className="mt-2 text-sm text-slate-500">{selectedRun.run_id}</div>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              <StatusPill value={selectedRun.status} />
                              <StatusPill value={selectedRun.latest_verification?.status || selectedRun.verification_status} />
                              {selectedRun.latest_approval ? <StatusPill value={selectedRun.latest_approval.status} /> : null}
                            </div>
                          </div>
                          <dl className="mt-5 grid gap-3 text-sm text-slate-700 md:grid-cols-2">
                            <div className="rounded-2xl bg-white p-4">
                              <dt className="text-slate-500">Current step</dt>
                              <dd className="mt-1 font-medium text-slate-900">{selectedRun.current_step || 'idle'}</dd>
                            </div>
                            <div className="rounded-2xl bg-white p-4">
                              <dt className="text-slate-500">Retry budget</dt>
                              <dd className="mt-1 font-medium text-slate-900">{selectedRun.policy?.retry_budget ?? 0}</dd>
                            </div>
                            <div className="rounded-2xl bg-white p-4">
                              <dt className="text-slate-500">Recovery mode</dt>
                              <dd className="mt-1 font-medium text-slate-900">{humanizeRecoveryMode(recoveryMode)}</dd>
                            </div>
                          </dl>
                          {streamContinuationSnapshot ? (
                            <div className="mt-5 rounded-3xl border border-sky-200 bg-sky-50 p-4">
                              <div className="flex items-center justify-between gap-3">
                                <div className="text-sm font-semibold text-sky-950">Stream continuation</div>
                                <StatusPill value={streamContinuationSnapshot.tone} />
                              </div>
                              <dl className="mt-4 grid gap-3 text-sm text-sky-950 md:grid-cols-2">
                                <div className="rounded-2xl bg-white/80 p-4">
                                  <dt className="text-sky-700">Phase</dt>
                                  <dd className="mt-1 font-medium">{streamContinuationSnapshot.phase}</dd>
                                </div>
                                <div className="rounded-2xl bg-white/80 p-4">
                                  <dt className="text-sky-700">Approved prefix</dt>
                                  <dd className="mt-1 font-medium">{streamContinuationSnapshot.prefixLength} chars</dd>
                                </div>
                                <div className="rounded-2xl bg-white/80 p-4">
                                  <dt className="text-sky-700">Recovery mode</dt>
                                  <dd className="mt-1 font-medium">{humanizeRecoveryMode(streamContinuationSnapshot.recoveryMode)}</dd>
                                </div>
                                <div className="rounded-2xl bg-white/80 p-4">
                                  <dt className="text-sky-700">Continuation step</dt>
                                  <dd className="mt-1 font-medium">
                                    {streamContinuationSnapshot.stepIndex !== null && streamContinuationSnapshot.stepIndex !== undefined
                                      ? streamContinuationSnapshot.stepIndex + 1
                                      : 'unknown'}
                                  </dd>
                                </div>
                                <div className="rounded-2xl bg-white/80 p-4">
                                  <dt className="text-sky-700">Resumed</dt>
                                  <dd className="mt-1 font-medium">{formatTimestamp(streamContinuationSnapshot.resumedAt)}</dd>
                                </div>
                                <div className="rounded-2xl bg-white/80 p-4">
                                  <dt className="text-sky-700">Completed</dt>
                                  <dd className="mt-1 font-medium">{formatTimestamp(streamContinuationSnapshot.completedAt)}</dd>
                                </div>
                              </dl>
                              <div className="mt-4 rounded-2xl border border-sky-200 bg-white/80 p-3 text-xs text-sky-950">
                                Continuation replays the interrupted node with the approved partial output as a prefix. It approximates resuming the prior stream without requiring the same low-level model connection.
                              </div>
                            </div>
                          ) : null}
                          {selectedRun.can_retry ? (
                            <div className="mt-5">
                              <button
                                type="button"
                                onClick={handleRetry}
                                disabled={retryRunMutation.isPending}
                                className="inline-flex items-center gap-2 rounded-2xl bg-sky-600 px-4 py-3 text-sm font-semibold text-white hover:bg-sky-500 disabled:opacity-50"
                              >
                                <RotateCcw className="h-4 w-4" />
                                {retryRunMutation.isPending
                                  ? 'Starting...'
                                  : rejectedReviewApproval
                                    ? rejectedReviewStage === 'cluster_research'
                                      ? 'Continue Without Research'
                                      : 'Continue From Rollback'
                                    : 'Retry run'}
                              </button>
                              {rejectedReviewApproval?.comment ? (
                                <div className="mt-3 rounded-2xl border border-sky-200 bg-sky-50 p-3 text-sm text-sky-950">
                                  {rejectedReviewStage === 'cluster_research'
                                    ? 'Next run will keep the last safe node output, discard the blocked research evidence, and include this redirect:'
                                    : 'Next run will resume from the last safe orchestration state and include this redirect:'}
                                  <div className="mt-2 font-medium">{rejectedReviewApproval.comment}</div>
                                </div>
                              ) : null}
                            </div>
                          ) : null}
                        </div>

                        <div className="grid gap-6 lg:grid-cols-2">
                          <div className="rounded-3xl border border-slate-200 bg-white p-5">
                            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                              <ShieldCheck className="h-4 w-4 text-amber-600" />
                              Approval state
                            </div>
                            <div className="mt-5">
                              <ApprovalSummary approval={selectedRun.latest_approval} />
                            </div>
                            {pendingApproval ? (
                              <div className="mt-6 space-y-3 rounded-3xl border border-amber-200 bg-amber-50 p-4">
                                <label className="block text-sm font-medium text-amber-950">
                                  Reviewer comment
                                  <textarea
                                    value={approvalComment}
                                    onChange={(event) => setApprovalComment(event.target.value)}
                                    rows={3}
                                    placeholder="Optional context for the decision."
                                    className="mt-2 w-full rounded-2xl border border-amber-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-amber-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                                  />
                                </label>
                                <div className="flex gap-3">
                                  <button
                                    type="button"
                                    disabled={approvalMutation.isPending}
                                    onClick={() => handleDecision(true)}
                                    className="inline-flex flex-1 items-center justify-center gap-2 rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
                                  >
                                    <CheckCircle2 className="h-4 w-4" />
                                    {pendingReviewStage === 'cluster_research' ? 'Continue With Research' : 'Approve'}
                                  </button>
                                  <button
                                    type="button"
                                    disabled={approvalMutation.isPending}
                                    onClick={() => handleDecision(false)}
                                    className="inline-flex flex-1 items-center justify-center gap-2 rounded-2xl bg-rose-600 px-4 py-3 text-sm font-semibold text-white hover:bg-rose-500 disabled:opacity-50"
                                  >
                                    <ShieldX className="h-4 w-4" />
                                    {pendingReviewStage === 'cluster_research' ? 'Continue Without Research' : 'Reject'}
                                  </button>
                                </div>
                              </div>
                            ) : null}
                          </div>

                          <div className="rounded-3xl border border-slate-200 bg-white p-5">
                            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                              <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                              Verification summary
                            </div>
                            {!selectedRun.latest_verification ? (
                              <div className="mt-5 text-sm text-slate-500">No verification result recorded yet.</div>
                            ) : (
                              <div className="mt-5 space-y-4">
                                <div className="flex items-center justify-between gap-3">
                                  <div className="text-sm font-semibold text-slate-900">
                                    {selectedRun.latest_verification.summary || 'Verification recorded'}
                                  </div>
                                  <StatusPill value={selectedRun.latest_verification.status} />
                                </div>
                                <dl className="grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
                                  <div>
                                    <dt className="text-slate-500">Recorded</dt>
                                    <dd className="mt-1">{formatTimestamp(selectedRun.latest_verification.created_at)}</dd>
                                  </div>
                                  <div>
                                    <dt className="text-slate-500">Checks</dt>
                                    <dd className="mt-1">
                                      {selectedRun.latest_verification.checks_json?.checks_run?.join(', ') || 'none'}
                                    </dd>
                                  </div>
                                  <div>
                                    <dt className="text-slate-500">Recovery mode</dt>
                                    <dd className="mt-1">{humanizeRecoveryMode(recoveryMode)}</dd>
                                  </div>
                                </dl>
                                <JsonBlock value={selectedRun.latest_verification.artifacts_json} />
                              </div>
                            )}
                          </div>
                        </div>

                        {outputArtifacts.length > 0 ? (
                          <div className="rounded-3xl border border-slate-200 bg-white p-5">
                            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                              <Sparkles className="h-4 w-4 text-cyan-600" />
                              Cluster synthesis
                            </div>
                            <div className="mt-5 grid gap-4 xl:grid-cols-2">
                              {outputArtifacts.map(([artifactKey, artifact]) => {
                                const strategies =
                                  artifact.strategies && typeof artifact.strategies === 'object' && !Array.isArray(artifact.strategies)
                                    ? (artifact.strategies as Record<string, unknown>)
                                    : null;
                                const voteTally =
                                  artifact.vote_tally && typeof artifact.vote_tally === 'object' && !Array.isArray(artifact.vote_tally)
                                    ? (artifact.vote_tally as Record<string, unknown>)
                                    : null;
                                const research =
                                  artifact.research && typeof artifact.research === 'object' && !Array.isArray(artifact.research)
                                    ? (artifact.research as Record<string, unknown>)
                                    : null;
                                const researchQueries = Array.isArray(research?.queries)
                                  ? (research?.queries as unknown[]).filter((item): item is string => typeof item === 'string')
                                  : [];
                                const paperQueries = Array.isArray(research?.paper_queries)
                                  ? (research?.paper_queries as unknown[]).filter((item): item is string => typeof item === 'string')
                                  : [];
                                const webQueries = Array.isArray(research?.web_queries)
                                  ? (research?.web_queries as unknown[]).filter((item): item is string => typeof item === 'string')
                                  : [];
                                const researchPapers = Array.isArray(research?.papers)
                                  ? (research?.papers as unknown[]).filter(
                                      (item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item)
                                    )
                                  : [];
                                const latestProgress = Array.isArray(research?.latest_progress)
                                  ? (research?.latest_progress as unknown[]).filter((item): item is string => typeof item === 'string')
                                  : [];
                                const researchSources = Array.isArray(research?.sources)
                                  ? (research?.sources as unknown[]).filter(
                                      (item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item)
                                    )
                                  : [];
                                const browserPreviews = Array.isArray(research?.browser_previews)
                                  ? (research?.browser_previews as unknown[]).filter(
                                      (item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item)
                                    )
                                  : [];
                                const providerRuns = Array.isArray(research?.provider_runs)
                                  ? (research?.provider_runs as unknown[]).filter(
                                      (item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item)
                                    )
                                  : [];
                                const roundHistory = Array.isArray(artifact.round_history)
                                  ? (artifact.round_history as unknown[]).filter(
                                      (item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item)
                                    )
                                  : [];
                                return (
                                  <div key={artifactKey} className="rounded-3xl bg-slate-50 p-4">
                                    <div className="flex items-start justify-between gap-3">
                                      <div>
                                        <div className="text-sm font-semibold text-slate-900">
                                          {typeof artifact.cluster_name === 'string' ? artifact.cluster_name : artifactKey}
                                        </div>
                                        <div className="mt-1 text-xs uppercase tracking-[0.16em] text-slate-400">
                                          {typeof artifact.cluster_strategy === 'string' ? artifact.cluster_strategy : 'cluster'}
                                        </div>
                                      </div>
                                      <StatusPill
                                        value={typeof artifact.winning_vote === 'string' ? artifact.winning_vote.toLowerCase() : 'cluster'}
                                      />
                                    </div>
                                    {voteTally ? (
                                      <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-600">
                                        {Object.entries(voteTally).map(([key, count]) => (
                                          <span key={key} className="rounded-full bg-white px-3 py-1 ring-1 ring-slate-200">
                                            {key}: {String(count)}
                                          </span>
                                        ))}
                                      </div>
                                    ) : null}
                                    {strategies ? (
                                      <div className="mt-4 space-y-3">
                                        {Object.entries(strategies).map(([key, detail]) => (
                                          <div key={key} className="rounded-2xl bg-white p-3">
                                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">{key}</div>
                                            <div className="mt-1 text-sm leading-6 text-slate-700">
                                              {typeof detail === 'string' && detail ? detail : 'No detail captured.'}
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    ) : null}
                                    {typeof artifact.winning_strategy === 'string' && artifact.winning_strategy ? (
                                      <div className="mt-4 rounded-2xl border border-cyan-200 bg-cyan-50 p-3 text-sm text-cyan-950">
                                        <div className="font-semibold">Winning strategy</div>
                                        <div className="mt-1">{artifact.winning_strategy}</div>
                                      </div>
                                    ) : null}
                                    {typeof artifact.game_theory_rationale === 'string' && artifact.game_theory_rationale ? (
                                      <div className="mt-3 rounded-2xl border border-indigo-200 bg-indigo-50 p-3 text-sm text-indigo-950">
                                        <div className="font-semibold">Game-Theory Rationale</div>
                                        <div className="mt-1">{artifact.game_theory_rationale}</div>
                                      </div>
                                    ) : null}
                                    {(typeof artifact.key_players === 'string' && artifact.key_players) ||
                                    (typeof artifact.incentive_map === 'string' && artifact.incentive_map) ||
                                    (typeof artifact.dominant_risks === 'string' && artifact.dominant_risks) ||
                                    (typeof artifact.expected_equilibrium === 'string' && artifact.expected_equilibrium) ? (
                                      <div className="mt-3 space-y-3 rounded-2xl border border-violet-200 bg-violet-50 p-3 text-sm text-violet-950">
                                        <div className="font-semibold">Game-Theory State</div>
                                        {typeof artifact.key_players === 'string' && artifact.key_players ? (
                                          <div>
                                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700/80">Key Players</div>
                                            <div className="mt-1">{artifact.key_players}</div>
                                          </div>
                                        ) : null}
                                        {typeof artifact.incentive_map === 'string' && artifact.incentive_map ? (
                                          <div>
                                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700/80">Incentive Map</div>
                                            <div className="mt-1">{artifact.incentive_map}</div>
                                          </div>
                                        ) : null}
                                        {typeof artifact.dominant_risks === 'string' && artifact.dominant_risks ? (
                                          <div>
                                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700/80">Dominant Risks</div>
                                            <div className="mt-1">{artifact.dominant_risks}</div>
                                          </div>
                                        ) : null}
                                        {typeof artifact.expected_equilibrium === 'string' && artifact.expected_equilibrium ? (
                                          <div>
                                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700/80">Expected Equilibrium</div>
                                            <div className="mt-1">{artifact.expected_equilibrium}</div>
                                          </div>
                                        ) : null}
                                      </div>
                                    ) : null}
                                    {roundHistory.length > 0 ? (
                                      <details className="mt-3 rounded-2xl border border-slate-200 bg-white p-3">
                                        <summary className="cursor-pointer text-sm font-semibold text-slate-900">Round History</summary>
                                        <div className="mt-3 space-y-3">
                                          {roundHistory.map((round, roundIndex) => {
                                            const memberOutputs = Array.isArray(round.member_outputs)
                                              ? (round.member_outputs as unknown[]).filter(
                                                  (item): item is Record<string, unknown> => !!item && typeof item === 'object' && !Array.isArray(item)
                                                )
                                              : [];
                                            const roundVotes =
                                              round.vote_tally && typeof round.vote_tally === 'object' && !Array.isArray(round.vote_tally)
                                                ? (round.vote_tally as Record<string, unknown>)
                                                : null;
                                            const shiftDetails =
                                              round.equilibrium_shift_details &&
                                              typeof round.equilibrium_shift_details === 'object' &&
                                              !Array.isArray(round.equilibrium_shift_details)
                                                ? (round.equilibrium_shift_details as Record<string, unknown>)
                                                : null;
                                            const changedFields = Array.isArray(shiftDetails?.changed_fields)
                                              ? (shiftDetails?.changed_fields as unknown[]).filter((item): item is string => typeof item === 'string')
                                              : [];
                                            const voteShift =
                                              shiftDetails?.vote_shift && typeof shiftDetails.vote_shift === 'object' && !Array.isArray(shiftDetails.vote_shift)
                                                ? (shiftDetails.vote_shift as Record<string, unknown>)
                                                : null;
                                            return (
                                              <div key={`${artifactKey}-round-${roundIndex}`} className="rounded-2xl bg-slate-50 p-3">
                                                <div className="flex items-center justify-between gap-3">
                                                  <div className="text-sm font-semibold text-slate-900">
                                                    Round {typeof round.round_index === 'number' ? round.round_index : roundIndex + 1}
                                                  </div>
                                                  <StatusPill
                                                    value={typeof round.winning_vote === 'string' ? round.winning_vote.toLowerCase() : 'round'}
                                                  />
                                                </div>
                                                {roundVotes ? (
                                                  <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-600">
                                                    {Object.entries(roundVotes).map(([key, count]) => (
                                                      <span key={key} className="rounded-full bg-white px-3 py-1 ring-1 ring-slate-200">
                                                        {key}: {String(count)}
                                                      </span>
                                                    ))}
                                                  </div>
                                                ) : null}
                                                {(typeof round.key_players === 'string' && round.key_players) ||
                                                (typeof round.incentive_map === 'string' && round.incentive_map) ||
                                                (typeof round.dominant_risks === 'string' && round.dominant_risks) ||
                                                (typeof round.expected_equilibrium === 'string' && round.expected_equilibrium) ? (
                                                  <div className="mt-3 space-y-2 rounded-2xl border border-violet-200 bg-violet-50 p-3 text-sm text-violet-950">
                                                    {typeof round.key_players === 'string' && round.key_players ? (
                                                      <div>
                                                        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700/80">Key Players</div>
                                                        <div className="mt-1">{round.key_players}</div>
                                                      </div>
                                                    ) : null}
                                                    {typeof round.incentive_map === 'string' && round.incentive_map ? (
                                                      <div>
                                                        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700/80">Incentive Map</div>
                                                        <div className="mt-1">{round.incentive_map}</div>
                                                      </div>
                                                    ) : null}
                                                    {typeof round.dominant_risks === 'string' && round.dominant_risks ? (
                                                      <div>
                                                        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700/80">Dominant Risks</div>
                                                        <div className="mt-1">{round.dominant_risks}</div>
                                                      </div>
                                                    ) : null}
                                                    {typeof round.expected_equilibrium === 'string' && round.expected_equilibrium ? (
                                                      <div>
                                                        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700/80">Expected Equilibrium</div>
                                                        <div className="mt-1">{round.expected_equilibrium}</div>
                                                      </div>
                                                    ) : null}
                                                  </div>
                                                ) : null}
                                                {typeof round.equilibrium_shift === 'string' && round.equilibrium_shift ? (
                                                  <div className="mt-3 rounded-2xl border border-cyan-200 bg-cyan-50 p-3 text-sm text-cyan-950">
                                                    <div className="text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700/80">Equilibrium Shift</div>
                                                    <div className="mt-1">{round.equilibrium_shift}</div>
                                                    {typeof shiftDetails?.shift_type === 'string' && shiftDetails.shift_type ? (
                                                      <div className="mt-2 text-xs text-cyan-900/80">Type: {shiftDetails.shift_type}</div>
                                                    ) : null}
                                                    {voteShift && typeof voteShift.from === 'string' && typeof voteShift.to === 'string' ? (
                                                      <div className="mt-2 text-xs text-cyan-900/80">
                                                        Vote shift: {voteShift.from} {'->'} {voteShift.to}
                                                      </div>
                                                    ) : null}
                                                    {changedFields.length > 0 ? (
                                                      <div className="mt-2 flex flex-wrap gap-2">
                                                        {changedFields.map((field) => (
                                                          <span key={field} className="rounded-full bg-white px-3 py-1 text-xs ring-1 ring-cyan-200">
                                                            {field}
                                                          </span>
                                                        ))}
                                                      </div>
                                                    ) : null}
                                                  </div>
                                                ) : null}
                                                {memberOutputs.length > 0 ? (
                                                  <div className="mt-3 space-y-2">
                                                    {memberOutputs.map((entry, entryIndex) => (
                                                      <div key={`${artifactKey}-round-entry-${entryIndex}`} className="rounded-2xl bg-white p-3">
                                                        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                                                          {typeof entry.member_node_id === 'string' ? entry.member_node_id : 'member'}
                                                        </div>
                                                        <div className="mt-1 whitespace-pre-wrap text-sm leading-6 text-slate-700">
                                                          {typeof entry.output === 'string' ? entry.output : ''}
                                                        </div>
                                                      </div>
                                                    ))}
                                                  </div>
                                                ) : null}
                                              </div>
                                            );
                                          })}
                                        </div>
                                      </details>
                                    ) : null}
                                    {typeof artifact.next_step === 'string' && artifact.next_step ? (
                                      <div className="mt-3 rounded-2xl border border-slate-200 bg-white p-3 text-sm text-slate-700">
                                        <div className="font-semibold text-slate-900">Next step</div>
                                        <div className="mt-1">{artifact.next_step}</div>
                                      </div>
                                    ) : null}
                                    {research ? (
                                      <div className="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-950">
                                        <div className="flex items-center justify-between gap-3">
                                          <div className="font-semibold">Research sync</div>
                                          {typeof research.research_mode === 'string' && research.research_mode ? (
                                            <span className="rounded-full bg-white px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em] ring-1 ring-emerald-200">
                                              {research.research_mode}
                                            </span>
                                          ) : null}
                                        </div>
                                        {researchQueries.length > 0 ? (
                                          <div className="mt-2 flex flex-wrap gap-2">
                                            {researchQueries.map((query) => (
                                              <span key={query} className="rounded-full bg-white px-3 py-1 text-xs ring-1 ring-emerald-200">
                                                {query}
                                              </span>
                                            ))}
                                          </div>
                                        ) : null}
                                        {paperQueries.length > 0 ? (
                                          <div className="mt-3 rounded-2xl bg-white/90 p-3">
                                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-800/80">
                                              Paper-first queries
                                            </div>
                                            <div className="mt-2 flex flex-wrap gap-2">
                                              {paperQueries.map((query) => (
                                                <span key={`${artifactKey}-paper-query-${query}`} className="rounded-full bg-emerald-50 px-3 py-1 text-xs ring-1 ring-emerald-200">
                                                  {query}
                                                </span>
                                              ))}
                                            </div>
                                          </div>
                                        ) : null}
                                        {webQueries.length > 0 ? (
                                          <div className="mt-3 rounded-2xl bg-white/90 p-3">
                                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-800/80">
                                              Web follow-up queries
                                            </div>
                                            <div className="mt-2 flex flex-wrap gap-2">
                                              {webQueries.map((query) => (
                                                <span key={`${artifactKey}-web-query-${query}`} className="rounded-full bg-slate-50 px-3 py-1 text-xs ring-1 ring-slate-200">
                                                  {query}
                                                </span>
                                              ))}
                                            </div>
                                          </div>
                                        ) : null}
                                        {typeof research.digest === 'string' && research.digest ? (
                                          <div className="mt-3 rounded-2xl bg-white/90 p-3 text-sm leading-6 text-slate-700">
                                            {research.digest}
                                          </div>
                                        ) : (
                                          <div className="mt-2 text-xs text-emerald-900/80">No research digest captured.</div>
                                        )}
                                        {latestProgress.length > 0 ? (
                                          <div className="mt-3 rounded-2xl bg-white/90 p-3">
                                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-800/80">
                                              Latest Progress
                                            </div>
                                            <div className="mt-2 space-y-2 text-sm leading-6 text-slate-700">
                                              {latestProgress.map((item, index) => (
                                                <div key={`${artifactKey}-progress-${index}`}>{item}</div>
                                              ))}
                                            </div>
                                          </div>
                                        ) : null}
                                        {researchPapers.length > 0 ? (
                                          <div className="mt-3 rounded-2xl bg-white/90 p-3">
                                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-800/80">
                                              Papers
                                            </div>
                                            <div className="mt-2 space-y-2">
                                              {researchPapers.map((paper, index) => (
                                                <a
                                                  key={`${artifactKey}-paper-${index}`}
                                                  href={typeof paper.url === 'string' ? paper.url : '#'}
                                                  target="_blank"
                                                  rel="noreferrer"
                                                  className="block rounded-2xl border border-emerald-100 bg-emerald-50/60 p-3 text-sm text-slate-700 hover:border-emerald-300"
                                                >
                                                  <div className="font-semibold text-slate-900">
                                                    {typeof paper.title === 'string' && paper.title ? paper.title : 'Untitled source'}
                                                  </div>
                                                  {typeof paper.snippet === 'string' && paper.snippet ? (
                                                    <div className="mt-1 leading-6">{paper.snippet}</div>
                                                  ) : null}
                                                </a>
                                              ))}
                                            </div>
                                          </div>
                                        ) : null}
                                        {browserPreviews.length > 0 ? (
                                          <div className="mt-3 rounded-2xl bg-white/90 p-3">
                                            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-emerald-800/80">
                                              Browser previews
                                            </div>
                                            <div className="mt-2 space-y-2">
                                              {browserPreviews.map((preview, index) => (
                                                <a
                                                  key={`${artifactKey}-preview-${index}`}
                                                  href={typeof preview.final_url === 'string' && preview.final_url ? preview.final_url : typeof preview.url === 'string' ? preview.url : '#'}
                                                  target="_blank"
                                                  rel="noreferrer"
                                                  className="block rounded-2xl border border-cyan-100 bg-cyan-50/60 p-3 text-sm text-slate-700 hover:border-cyan-300"
                                                >
                                                  <div className="font-semibold text-slate-900">
                                                    {typeof preview.title === 'string' && preview.title ? preview.title : 'Untitled preview'}
                                                  </div>
                                                  {typeof preview.description === 'string' && preview.description ? (
                                                    <div className="mt-1 leading-6">{preview.description}</div>
                                                  ) : null}
                                                  <div className="mt-2 text-[11px] uppercase tracking-[0.16em] text-cyan-900/70">
                                                    {typeof preview.content_type === 'string' && preview.content_type ? preview.content_type : 'preview'}
                                                  </div>
                                                </a>
                                              ))}
                                            </div>
                                          </div>
                                        ) : null}
                                        {providerRuns.length > 0 ? (
                                          <details className="mt-3 rounded-2xl bg-white/90 p-3">
                                            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-[0.16em] text-emerald-800/80">
                                              Provider runs
                                            </summary>
                                            <div className="mt-3 space-y-2">
                                              {providerRuns.map((entry, index) => (
                                                <div key={`${artifactKey}-provider-run-${index}`} className="rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                                                  <div className="font-medium text-slate-900">
                                                    {typeof entry.provider === 'string' ? entry.provider : 'provider'} · {coerceNumber(entry.result_count) ?? 0} results
                                                  </div>
                                                  {typeof entry.query === 'string' && entry.query ? (
                                                    <div className="mt-1 text-xs leading-5 text-slate-600">{entry.query}</div>
                                                  ) : null}
                                                </div>
                                              ))}
                                            </div>
                                          </details>
                                        ) : null}
                                        {researchSources.length > 0 ? (
                                          <details className="mt-3 rounded-2xl bg-white/90 p-3">
                                            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-[0.16em] text-emerald-800/80">
                                              Sources
                                            </summary>
                                            <div className="mt-3 space-y-2">
                                              {researchSources.map((source, index) => (
                                                <a
                                                  key={`${artifactKey}-source-${index}`}
                                                  href={typeof source.url === 'string' ? source.url : '#'}
                                                  target="_blank"
                                                  rel="noreferrer"
                                                  className="block rounded-2xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700 hover:border-cyan-300"
                                                >
                                                  <div className="font-medium text-slate-900">
                                                    {typeof source.title === 'string' && source.title ? source.title : source.url as string}
                                                  </div>
                                                  {typeof source.snippet === 'string' && source.snippet ? (
                                                    <div className="mt-1 text-xs leading-5 text-slate-600">{source.snippet}</div>
                                                  ) : null}
                                                </a>
                                              ))}
                                            </div>
                                          </details>
                                        ) : null}
                                        <div className="mt-3 text-xs text-emerald-900/80">
                                          Memory sync:{' '}
                                          {research.memory && typeof research.memory === 'object' && (research.memory as Record<string, unknown>).stored
                                            ? 'stored'
                                            : 'skipped'}
                                        </div>
                                      </div>
                                    ) : null}
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        ) : null}

                        <div className="grid gap-6 lg:grid-cols-2">
                          <div className="rounded-3xl border border-slate-200 bg-white p-5">
                            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                              <Clock3 className="h-4 w-4 text-sky-600" />
                              Input payload
                            </div>
                            <div className="mt-5">
                              <JsonBlock value={selectedRun.input_json} />
                            </div>
                          </div>
                          <div className="rounded-3xl border border-slate-200 bg-white p-5">
                            <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                              <Sparkles className="h-4 w-4 text-fuchsia-600" />
                              Metadata
                            </div>
                            <div className="mt-5">
                              <JsonBlock value={selectedRun.metadata_json || null} />
                            </div>
                          </div>
                        </div>

                        <div className="rounded-3xl border border-slate-200 bg-white p-5">
                          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                            <Workflow className="h-4 w-4 text-cyan-600" />
                            Event timeline
                          </div>
                          {!selectedRun.events || selectedRun.events.length === 0 ? (
                            <div className="mt-5 text-sm text-slate-500">No event evidence recorded yet.</div>
                          ) : (
                            <div className="mt-5 space-y-3">
                              {selectedRun.events.map((event) => (
                                <EventRow key={event.event_id || `${event.event_type}-${event.created_at}`} event={event} />
                              ))}
                            </div>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </section>

            <aside className="space-y-6">
              <section className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <Sparkles className="h-4 w-4 text-cyan-600" />
                  Node editor
                </div>
                {!selectedAgent ? (
                  <div className="mt-4 text-sm text-slate-500">
                    <p>Select an agent or cluster on the canvas to edit its specific settings.</p>
                    <div className="mt-8 space-y-4 rounded-xl border border-slate-200 bg-white p-4">
                      <h4 className="text-sm font-semibold text-slate-900">Project Provider Configuration</h4>
                      <ProviderSelect
                        label="Preferred Provider"
                        value={draftProject?.graph_json?.provider_config?.preferred_provider_id}
                        onChange={(nextValue) =>
                          updateDraftProject((current) => ({
                            ...current,
                            graph_json: {
                              ...current.graph_json,
                              provider_config: {
                                ...current.graph_json?.provider_config,
                                preferred_provider_id: nextValue,
                              },
                            },
                          }))
                        }
                        providers={providerOptions}
                        emptyLabel="(Default Settings Model)"
                      />
                      <ProviderSelect
                        label="Fallback Provider"
                        value={draftProject?.graph_json?.provider_config?.fallback_provider_id}
                        onChange={(nextValue) =>
                          updateDraftProject((current) => ({
                            ...current,
                            graph_json: {
                              ...current.graph_json,
                              provider_config: {
                                ...current.graph_json?.provider_config,
                                fallback_provider_id: nextValue,
                              },
                            },
                          }))
                        }
                        providers={providerOptions}
                        emptyLabel="(None)"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="mt-4 space-y-4">
                    <label className="block text-sm font-medium text-slate-800">
                      Name
                      <input
                        value={selectedAgent.name}
                        onChange={(event) => updateSelectedAgent({ name: event.target.value })}
                        className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                      />
                    </label>
                    <label className="block text-sm font-medium text-slate-800">
                      Role
                      <input
                        value={selectedAgent.role || ''}
                        onChange={(event) => updateSelectedAgent({ role: event.target.value })}
                        className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                      />
                    </label>
                    <label className="block text-sm font-medium text-slate-800">
                      Description
                      <textarea
                        rows={3}
                        value={selectedAgent.description || ''}
                        onChange={(event) => updateSelectedAgent({ description: event.target.value })}
                        className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                      />
                    </label>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <label className="block text-sm font-medium text-slate-800">
                        Model
                        <input
                          value={selectedAgent.model || ''}
                          onChange={(event) => updateSelectedAgent({ model: event.target.value })}
                          className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                        />
                      </label>
                      <ProviderSelect
                        label="Preferred Provider"
                        value={selectedAgent.preferred_provider_id}
                        onChange={(nextValue) => updateSelectedAgent({ preferred_provider_id: nextValue })}
                        providers={providerOptions}
                        emptyLabel="(Project preference)"
                      />
                    </div>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <ProviderSelect
                        label="Fallback Provider"
                        value={selectedAgent.fallback_provider_id}
                        onChange={(nextValue) => updateSelectedAgent({ fallback_provider_id: nextValue })}
                        providers={providerOptions}
                        emptyLabel="(None)"
                      />
                      <label className="block text-sm font-medium text-slate-800">
                        Max iterations
                        <input
                          type="number"
                          min={1}
                          max={12}
                          value={selectedAgent.max_iterations || 1}
                          onChange={(event) => updateSelectedAgent({ max_iterations: Number(event.target.value) || 1 })}
                          className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                        />
                      </label>
                    </div>
                    {selectedAgent.node_kind === 'cluster' ? (
                      <div className="space-y-4 rounded-3xl border border-amber-200 bg-amber-50/70 p-4">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <div className="text-sm font-semibold text-slate-900">Cluster configuration</div>
                            <div className="mt-1 text-xs text-slate-600">These members execute as an expanded runtime subgraph.</div>
                          </div>
                          <button
                            type="button"
                            onClick={handleAddClusterMember}
                            className="inline-flex items-center gap-2 rounded-2xl border border-amber-200 bg-white px-3 py-2 text-xs font-semibold text-amber-900 hover:bg-amber-100"
                          >
                            <PlusCircle className="h-3.5 w-3.5" />
                            Add member
                          </button>
                        </div>
                        <div className="grid gap-3 sm:grid-cols-2">
                          <label className="block text-sm font-medium text-slate-800">
                            Cluster strategy
                            <select
                              value={selectedAgent.cluster_strategy || 'custom'}
                              onChange={(event) =>
                                updateSelectedAgent({
                                  cluster_strategy: (event.target.value as 'brainstorm' | 'custom') || 'custom',
                                })
                              }
                              className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                            >
                              <option value="brainstorm">Brainstorm</option>
                              <option value="custom">Custom</option>
                            </select>
                          </label>
                          <label className="block text-sm font-medium text-slate-800">
                            Debate rounds
                            <input
                              type="number"
                              min={1}
                              max={5}
                              value={selectedAgent.brainstorm_rounds || 1}
                              onChange={(event) => updateSelectedAgent({ brainstorm_rounds: Number(event.target.value) || 1 })}
                              className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                            />
                          </label>
                        </div>
                        <div className="grid gap-3 sm:grid-cols-2">
                          <label className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800">
                            <input
                              type="checkbox"
                              checked={selectedAgent.cluster_auto_research ?? false}
                              onChange={(event) => updateSelectedAgent({ cluster_auto_research: event.target.checked })}
                              className="h-4 w-4 rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                            />
                            Auto research pass
                          </label>
                          <label className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800">
                            <input
                              type="checkbox"
                              checked={selectedAgent.cluster_auto_review ?? true}
                              onChange={(event) => updateSelectedAgent({ cluster_auto_review: event.target.checked })}
                              className="h-4 w-4 rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                            />
                            Auto attach review agent
                          </label>
                        </div>
                        <div className="space-y-3">
                          {(selectedAgent.cluster_members ?? []).map((member) => (
                            <div key={member.member_id} className="rounded-2xl border border-slate-200 bg-white p-4">
                              <div className="grid gap-3 sm:grid-cols-2">
                                <label className="block text-sm font-medium text-slate-800">
                                  Member name
                                  <input
                                    value={member.name}
                                    onChange={(event) => updateClusterMember(member.member_id, { name: event.target.value })}
                                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                  />
                                </label>
                                <label className="block text-sm font-medium text-slate-800">
                                  Role
                                  <input
                                    value={member.role || ''}
                                    onChange={(event) => updateClusterMember(member.member_id, { role: event.target.value })}
                                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                  />
                                </label>
                              </div>
                              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                                <label className="block text-sm font-medium text-slate-800">
                                  Model
                                  <input
                                    value={member.model || ''}
                                    onChange={(event) => updateClusterMember(member.member_id, { model: event.target.value })}
                                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                  />
                                </label>
                                <ProviderSelect
                                  label="Preferred Provider"
                                  value={member.preferred_provider_id}
                                  onChange={(nextValue) => updateClusterMember(member.member_id, { preferred_provider_id: nextValue })}
                                  providers={providerOptions}
                                  emptyLabel="(Cluster or project preference)"
                                />
                              </div>
                              <div className="mt-3">
                                <label className="block text-sm font-medium text-slate-800">
                                  System prompt
                                  <textarea
                                    rows={3}
                                    value={member.system_prompt || ''}
                                    onChange={(event) => updateClusterMember(member.member_id, { system_prompt: event.target.value })}
                                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                  />
                                </label>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                    <label className="block text-sm font-medium text-slate-800">
                      System prompt
                      <textarea
                        rows={7}
                        value={selectedAgent.system_prompt || ''}
                        onChange={(event) => updateSelectedAgent({ system_prompt: event.target.value })}
                        className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                      />
                    </label>
                    {selectedAgent.node_kind !== 'cluster' ? (
                      <>
                        <label className="block text-sm font-medium text-slate-800">
                          Skill intents
                          <textarea
                            rows={3}
                            value={(selectedAgent.skill_intents ?? []).join(', ')}
                            onChange={(event) =>
                              updateSelectedAgent({
                                skill_intents: event.target.value
                                  .split(',')
                                  .map((value) => value.trim())
                                  .filter(Boolean),
                              })
                            }
                            placeholder="research, rag, tools"
                            className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                          />
                        </label>
                        <button
                          type="button"
                          onClick={handleRequestSkills}
                          disabled={skillRequestMutation.isPending}
                          className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-amber-500 px-4 py-3 text-sm font-semibold text-white hover:bg-amber-400 disabled:opacity-50"
                        >
                          <Sparkles className="h-4 w-4" />
                          {skillRequestMutation.isPending ? 'Requesting...' : 'Request skills from source'}
                        </button>
                      </>
                    ) : null}
                  </div>
                )}
              </section>

              <section className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <Layers3 className="h-4 w-4 text-emerald-600" />
                  Skill pool
                </div>
                <div className="mt-4 space-y-3">
                  {(graph?.skill_pool ?? []).map((skill) => (
                    <label key={skill.skill_id} className="flex items-start gap-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
                      <input
                        type="checkbox"
                        disabled={!selectedAgent}
                        checked={!!selectedAgent?.skill_ids?.includes(skill.skill_id)}
                        onChange={() => toggleSkillAssignment(skill.skill_id)}
                        className="mt-1 h-4 w-4 rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                      />
                      <span className="flex-1">
                        <span className="block text-sm font-semibold text-slate-900">{skill.title}</span>
                        <span className="mt-1 block text-xs text-slate-500">{skill.source}</span>
                      </span>
                      <StatusPill value={skill.status || 'loaded'} />
                    </label>
                  ))}
                  {(graph?.skill_pool?.length ?? 0) === 0 ? (
                    <div className="text-sm text-slate-500">No approved skills in the pool yet.</div>
                  ) : null}
                </div>
              </section>

              <section className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <Unplug className="h-4 w-4 text-amber-600" />
                  Pending skill approvals
                </div>
                <div className="mt-4 space-y-3">
                  {(graph?.pending_skill_requests ?? [])
                    .filter((request) => request.status === 'pending')
                    .map((request) => (
                      <div key={request.request_id} className="rounded-2xl border border-amber-200 bg-amber-50 p-4">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <div className="text-sm font-semibold text-slate-900">{request.title}</div>
                            <div className="mt-1 text-xs text-slate-500">{request.agent_id}</div>
                          </div>
                          <StatusPill value={request.status} />
                        </div>
                        <div className="mt-2 text-xs leading-5 text-slate-600">{request.reason || request.source}</div>
                        <div className="mt-3 flex gap-3">
                          <button
                            type="button"
                            onClick={() => handleResolveSkillRequest(request.request_id, true)}
                            disabled={skillDecisionMutation.isPending}
                            className="inline-flex flex-1 items-center justify-center gap-2 rounded-2xl bg-emerald-600 px-3 py-2 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
                          >
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            Approve
                          </button>
                          <button
                            type="button"
                            onClick={() => handleResolveSkillRequest(request.request_id, false)}
                            disabled={skillDecisionMutation.isPending}
                            className="inline-flex flex-1 items-center justify-center gap-2 rounded-2xl bg-rose-600 px-3 py-2 text-xs font-semibold text-white hover:bg-rose-500 disabled:opacity-50"
                          >
                            <ShieldX className="h-3.5 w-3.5" />
                            Reject
                          </button>
                        </div>
                      </div>
                    ))}
                  {(graph?.pending_skill_requests ?? []).filter((request) => request.status === 'pending').length === 0 ? (
                    <div className="text-sm text-slate-500">No pending requests. Agents will reuse approved pool skills automatically.</div>
                  ) : null}
                </div>
              </section>

              <section className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                  <Workflow className="h-4 w-4 text-fuchsia-600" />
                  Available source skills
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  {(graph?.skill_catalog ?? []).map((skill) => (
                    <span key={skill.skill_id} className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-700">
                      {skill.skill_id}
                    </span>
                  ))}
                </div>
              </section>
            </aside>
          </div>
        </div>
      </div>
    </div>
  );
}
