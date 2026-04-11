'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type WheelEvent as ReactWheelEvent } from 'react';
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
  Server,
  ShieldCheck,
  Sparkles,
  Waypoints,
  Workflow,
} from 'lucide-react';
import {
  HarnessAgentCapabilitySummaryDTO,
  HarnessCanvasAgentDTO,
  HarnessCanvasEdgeDTO,
  HarnessClusterMemberDTO,
  HarnessDelegationTargetFitDTO,
  HarnessExecutionChecklistItemDTO,
  HarnessEventDTO,
  HarnessProjectDetailDTO,
  HarnessRunChecklistSnapshotDTO,
  HarnessRunSummaryDTO,
  HarnessWorkflowProgressDTO,
  HarnessApprovalDTO,
  HarnessSkillPoolItemDTO,
  HarnessSkillRequestDTO,
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
import { useKnowledgeBasesQuery } from '@/domains/knowledge-bases/hooks';
import { OverlayDialog } from '@/components/ui/OverlayDialog';
import { useMessages } from '@/lib/i18n';
import {
  availabilityScopeToneClasses,
  buildAvailabilityScopeSummary,
  buildCapabilityCoverageSummary,
  buildCollaborationScopeSummary,
  buildCoordinationTopologySummary,
  buildOrchestrationBriefSummary,
  coerceOrchestrationBriefSummary,
  capabilityAvailabilityBadgeClass,
  capabilityReadinessBadgeClass,
  delegationFitBadgeClass,
  formatAvailabilityAgentPreview,
  formatCapabilityAvailabilityLabel,
  formatCapabilityReadinessLabel,
  formatDelegationFitLabel,
  normalizeDelegationFit,
  resolveCapabilityAvailabilityStatus,
  resolveCapabilityReadinessStatus,
  shouldOpenProjectMcpForDiagnostic,
  shouldOpenProjectProvidersForDiagnostic,
} from './diagnostics';
import { HARNESS_MESSAGES } from './messages';
import {
  applyAgentCapabilityPolicySuggestions,
  applyCapabilityPolicySuggestionsToGraph,
  applyRoleProfilesToGraph,
  buildPolicyRepairScopeSummary,
  buildRoleProfilePeerOverlapDiagnostics,
  buildRoleProfileScopeSummary,
  computeActionableMcpPolicySuggestionIds,
  computeActionableToolPolicySuggestionIds,
  computeCoordinatorMcpPolicyRestrictionIds,
  computeCoordinatorToolPolicyRestrictionIds,
} from './policy-repair';
import {
  AvailabilityPreflightCard,
  CapabilityAvailabilityEvidenceCard,
  CollaborationContractEvidenceCard,
  CapabilityExecutionContractEvidenceCard,
  CapabilityReadinessEvidenceCard,
  HandoffDiagnosticsCard,
  RunPreflightSummaryCard,
  RunRecoveryGuideCard,
} from './recovery-cards';
import { GraphCapabilityCoverageSummary } from './graph-capability-coverage-summary';
import { GraphCollaborationSummary } from './graph-collaboration-summary';
import { GraphOrchestrationBrief } from './graph-orchestration-brief';
import { SelectedAgentCapabilityPanel } from './selected-agent-capability-panel';
import {
  coerceNumber,
  coerceRecord,
  coerceRecordList,
  coerceStringList,
  formatSkillTitle,
  formatTemplate,
} from './utils';

function formatTimestamp(value?: number | null, emptyLabel = 'Not recorded') {
  if (!value) {
    return emptyLabel;
  }
  const normalized = value > 1_000_000_000_000 ? value : value * 1000;
  return new Date(normalized).toLocaleString();
}

function normalizeSkillKey(value: string | null | undefined) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function buildAgentSkillRequestKey(agentId: string | null | undefined, skillId: string | null | undefined) {
  const normalizedAgentId = String(agentId || '').trim();
  const normalizedSkillId = normalizeSkillKey(skillId);
  if (!normalizedAgentId || !normalizedSkillId) {
    return '';
  }
  return `${normalizedAgentId}::${normalizedSkillId}`;
}

function getSkillRequestTimestamp(request: HarnessSkillRequestDTO) {
  return Number(request.resolved_at ?? request.discovered_at ?? 0);
}

function mergeSkillRequests(
  currentRequests: HarnessSkillRequestDTO[],
  updates: HarnessSkillRequestDTO[]
) {
  const byId = new Map(currentRequests.map((request) => [request.request_id, request]));
  for (const request of updates) {
    byId.set(request.request_id, request);
  }
  return Array.from(byId.values()).sort((left, right) => getSkillRequestTimestamp(right) - getSkillRequestTimestamp(left));
}

function mergeSkillPoolItems(
  currentItems: HarnessSkillPoolItemDTO[],
  updates: HarnessSkillPoolItemDTO[]
) {
  const byId = new Map(currentItems.map((item) => [normalizeSkillKey(item.skill_id), item]));
  for (const item of updates) {
    const key = normalizeSkillKey(item.skill_id);
    if (!key) {
      continue;
    }
    byId.set(key, item);
  }
  return Array.from(byId.values()).sort((left, right) => left.title.localeCompare(right.title));
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
    case 'blocked':
      return 'bg-rose-100 text-rose-900 ring-rose-200';
    case 'waiting_approval':
    case 'pending':
      return 'bg-amber-100 text-amber-900 ring-amber-200';
    case 'in_progress':
    case 'resumed':
    case 'running':
    case 'verifying':
    case 'queued':
      return 'bg-sky-100 text-sky-900 ring-sky-200';
    default:
      return 'bg-slate-100 text-slate-800 ring-slate-200';
  }
}

function humanizeHarnessValue(value: string | null | undefined, text: Record<string, string>) {
  switch (value) {
    case 'approval':
      return text.approvalAction;
    case 'resume':
      return text.resumeAction;
    case 'orchestration_review':
      return text.orchestrationReviewAction;
    case 'harness':
      return text.harnessSource;
    case 'agent_orchestration':
      return text.agentOrchestrationTask;
    case 'unknown_task':
      return text.unknownTask;
    case 'idle':
      return text.idleState;
    case 'pending':
      return text.pendingState;
    case 'none':
      return text.none;
    case 'waiting_approval':
      return text.waitingApprovalState;
    case 'queued':
      return text.queuedState;
    case 'in_progress':
      return text.inProgressState;
    case 'running':
      return text.runningState;
    case 'resumed':
      return text.resumedState;
    case 'verifying':
      return text.verifyingState;
    case 'blocked':
      return text.blockedState;
    case 'completed':
      return text.completedState;
    case 'approved':
      return text.approvedState;
    case 'rejected':
      return text.rejectedState;
    case 'failed':
      return text.failedState;
    case 'pass':
      return text.passState;
    case 'fail':
      return text.failState;
    case 'loaded':
      return text.loadedState;
    case 'available':
      return text.availableState;
    case 'cluster':
      return text.clusterState;
    case 'cluster_research':
      return text.clusterResearchState;
    case 'no_skills':
      return text.noSkillsState;
    case 'handoff':
      return text.handoffLabel;
    case 'round':
      return text.roundState;
    case 'orchestration.review_segment_scan_completed':
      return text.pipelineReviewScan;
    case 'orchestration.checklist_loaded':
      return text.checklistLoaded;
    case 'orchestration.review_stream_blocked':
      return text.liveStreamGuardBlockedOutput;
    case 'orchestration.stream_continuation_resumed':
      return text.streamContinuationResumed;
    case 'orchestration.stream_continuation_completed':
      return text.streamContinuationCompleted;
    case 'run.notification_ready':
      return text.notificationReadyAction;
    default:
      return value || text.unknownLabel;
  }
}

function StatusPill({ value }: { value?: string | null }) {
  const text = useMessages(HARNESS_MESSAGES);
  const rawValue = value || 'unknown';
  return (
    <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${statusTone(rawValue)}`}>
      {humanizeHarnessValue(rawValue, text)}
    </span>
  );
}

function JsonBlock({ value }: { value?: Record<string, unknown> | null }) {
  const text = useMessages(HARNESS_MESSAGES);
  if (!value || Object.keys(value).length === 0) {
    return <div className="text-sm text-slate-500">{text.noStructuredPayload}</div>;
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

const NODE_WIDTH = 240;
const NODE_HEIGHT = 232;
const NODE_CENTER_X = NODE_WIDTH / 2;
const NODE_CENTER_Y = 112;
const CANVAS_WORLD_MIN_X = -4800;
const CANVAS_WORLD_MAX_X = 4800;
const CANVAS_WORLD_MIN_Y = -3200;
const CANVAS_WORLD_MAX_Y = 3200;
const CANVAS_WORLD_WIDTH = CANVAS_WORLD_MAX_X - CANVAS_WORLD_MIN_X;
const CANVAS_WORLD_HEIGHT = CANVAS_WORLD_MAX_Y - CANVAS_WORLD_MIN_Y;
const CANVAS_PADDING = 120;
const NODE_SPAWN_MARGIN = 24;
const MIN_CANVAS_ZOOM = 0.45;
const MAX_CANVAS_ZOOM = 1.8;
type CanvasPanel = 'control' | 'skills' | 'runs' | 'review';
type CanvasCreationKind = 'agent' | 'brainstorm' | 'custom';
type RunFilter = 'all' | 'attention' | 'active' | 'checklist' | 'completed';
type RunQueueGroup = 'approval' | 'blocked' | 'checklist' | 'active' | 'completed' | 'other';

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizeCanvasViewport(canvas?: { x?: number; y?: number; zoom?: number }) {
  return {
    x: Number.isFinite(canvas?.x) ? Number(canvas?.x) : 0,
    y: Number.isFinite(canvas?.y) ? Number(canvas?.y) : 0,
    zoom: clamp(Number.isFinite(canvas?.zoom) ? Number(canvas?.zoom) : 1, MIN_CANVAS_ZOOM, MAX_CANVAS_ZOOM),
  };
}

function getCanvasSpawnPosition(
  index: number,
  viewport: { x: number; y: number; zoom: number },
  canvas: HTMLDivElement | null
) {
  const width = canvas?.clientWidth ?? 1440;
  const height = canvas?.clientHeight ?? 900;
  const offsetColumn = (index % 3) - 1;
  const offsetRow = Math.floor(index / 3) % 2 === 0 ? -1 : 1;
  const baseX = (width / 2 - viewport.x) / viewport.zoom - NODE_WIDTH / 2;
  const baseY = (height / 2 - viewport.y) / viewport.zoom - NODE_HEIGHT / 2;

  return {
    x: Math.round(
      clamp(
        baseX + offsetColumn * 48,
        CANVAS_WORLD_MIN_X + NODE_SPAWN_MARGIN,
        CANVAS_WORLD_MAX_X - NODE_WIDTH - NODE_SPAWN_MARGIN
      )
    ),
    y: Math.round(
      clamp(
        baseY + offsetRow * 28,
        CANVAS_WORLD_MIN_Y + NODE_SPAWN_MARGIN,
        CANVAS_WORLD_MAX_Y - NODE_HEIGHT - NODE_SPAWN_MARGIN
      )
    ),
  };
}

function localizeStudioProjectName(name: string | undefined, text: Record<string, string>) {
  if (!name) {
    return name ?? '';
  }
  if (name === 'Default agent studio' || name === text.defaultStudioProjectName) {
    return text.defaultStudioProjectName;
  }
  return name;
}

function localizeStudioProjectDescription(description: string | null | undefined, text: Record<string, string>) {
  if (!description) {
    return description ?? null;
  }
  if (
    description === 'Canvas workspace for composing and running multi-agent collaborations.' ||
    description === text.defaultStudioProjectDescription
  ) {
    return text.defaultStudioProjectDescription;
  }
  return description;
}

function localizeDefaultCanvasAgent(agent: HarnessCanvasAgentDTO, text: Record<string, string>): HarnessCanvasAgentDTO {
  if (agent.agent_id === 'agent_planner') {
    return {
      ...agent,
      name: agent.name === 'Planner' || agent.name === text.plannerName ? text.plannerName : agent.name,
      role:
        agent.role === 'coordinator' || agent.role === text.plannerDefaultRole
          ? text.plannerDefaultRole
          : agent.role,
      description:
        agent.description === 'Breaks the task into delegable work and routes to specialists.' ||
        agent.description === text.plannerDefaultDescription
          ? text.plannerDefaultDescription
          : agent.description,
      system_prompt:
        agent.system_prompt === 'Plan the collaboration loop, decide handoffs, and keep the swarm aligned.' ||
        agent.system_prompt === text.plannerDefaultPrompt
          ? text.plannerDefaultPrompt
          : agent.system_prompt,
    };
  }
  if (agent.agent_id === 'agent_researcher') {
    return {
      ...agent,
      name: agent.name === 'Researcher' || agent.name === text.researcherName ? text.researcherName : agent.name,
      role:
        agent.role === 'research' || agent.role === text.researcherDefaultRole
          ? text.researcherDefaultRole
          : agent.role,
      description:
        agent.description === 'Looks up context, documents, and supporting evidence.' ||
        agent.description === text.researcherDefaultDescription
          ? text.researcherDefaultDescription
          : agent.description,
      system_prompt:
        agent.system_prompt === 'Gather evidence, summarize findings, and hand off clean context.' ||
        agent.system_prompt === text.researcherDefaultPrompt
          ? text.researcherDefaultPrompt
          : agent.system_prompt,
    };
  }
  if (agent.agent_id === 'agent_builder') {
    return {
      ...agent,
      name: agent.name === 'Builder' || agent.name === text.builderName ? text.builderName : agent.name,
      role:
        agent.role === 'implementation' || agent.role === text.builderDefaultRole
          ? text.builderDefaultRole
          : agent.role,
      description:
        agent.description === 'Turns the plan into code or structured output.' ||
        agent.description === text.builderDefaultDescription
          ? text.builderDefaultDescription
          : agent.description,
      system_prompt:
        agent.system_prompt === 'Implement the agreed solution and keep outputs production-oriented.' ||
        agent.system_prompt === text.builderDefaultPrompt
          ? text.builderDefaultPrompt
          : agent.system_prompt,
    };
  }
  return agent;
}

function humanizeEdgeInteraction(value: string | null | undefined, text: Record<string, string>) {
  if (!value || value === 'handoff') {
    return text.handoffLabel;
  }
  return humanizeHarnessValue(value, text);
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

function normalizeProject(
  project: HarnessProjectDetailDTO,
  text: Record<string, string>,
  preservedCanvas?: { x?: number; y?: number; zoom?: number } | null
): HarnessProjectDetailDTO {
  return {
    ...project,
    name: localizeStudioProjectName(project.name, text),
    description: localizeStudioProjectDescription(project.description, text),
    graph_json: {
      version: project.graph_json?.version ?? 1,
      agents: (project.graph_json?.agents ?? []).map((agent) => localizeDefaultCanvasAgent(agent, text)),
      edges: project.graph_json?.edges ?? [],
      knowledge_base_ids: project.graph_json?.knowledge_base_ids ?? [],
      execution_checklist: (project.graph_json?.execution_checklist ?? []).map((item, index) => ({
        item_id: item.item_id || `check_${index + 1}`,
        content: item.content || '',
        status: item.status || 'pending',
        active_form: item.active_form || null,
      })),
      skill_pool: project.graph_json?.skill_pool ?? [],
      pending_skill_requests: project.graph_json?.pending_skill_requests ?? [],
      skill_catalog: project.graph_json?.skill_catalog ?? [],
      tool_catalog: project.graph_json?.tool_catalog ?? [],
      mcp_server_catalog: project.graph_json?.mcp_server_catalog ?? [],
      agent_capability_summaries: project.graph_json?.agent_capability_summaries ?? [],
      graph_diagnostics: project.graph_json?.graph_diagnostics ?? {
        weak_downstream_edges: [],
        best_next_handoffs: [],
        weak_edge_count: 0,
        best_next_count: 0,
      },
      review_agent: {
        enabled: project.graph_json?.review_agent?.enabled ?? true,
        hidden: project.graph_json?.review_agent?.hidden ?? true,
        name:
          project.graph_json?.review_agent?.name === 'Compliance reviewer' ||
          project.graph_json?.review_agent?.name === text.defaultReviewAgentName ||
          !project.graph_json?.review_agent?.name
            ? text.defaultReviewAgentName
            : project.graph_json.review_agent.name,
        model: project.graph_json?.review_agent?.model ?? 'gpt-5.1-codex-mini',
        preferred_provider_id: project.graph_json?.review_agent?.preferred_provider_id ?? null,
        fallback_provider_id: project.graph_json?.review_agent?.fallback_provider_id ?? null,
        system_prompt: project.graph_json?.review_agent?.system_prompt ?? '',
      },
      canvas: normalizeCanvasViewport(preservedCanvas ?? project.graph_json?.canvas),
      provider_config: project.graph_json?.provider_config ?? {},
    },
  };
}

function projectDraftFingerprint(project: HarnessProjectDetailDTO | null | undefined) {
  if (!project) {
    return '';
  }
  return JSON.stringify({
    name: project.name ?? '',
    description: project.description ?? null,
    graph_json: project.graph_json ?? null,
  });
}

function createAgentSeed(
  index: number,
  text: Record<string, string>,
  position: { x: number; y: number }
): HarnessCanvasAgentDTO {
  return {
    agent_id: makeId('agent'),
    name: formatTemplate(text.newAgentName, { count: index + 1 }),
    node_kind: 'agent',
    role: text.specialistRoleSeed,
    description: text.newAgentDescription,
    system_prompt: text.specialistAgentPrompt,
    model: 'gpt-5.2',
    temperature: 0.2,
    max_iterations: 3,
    position,
    skill_ids: [],
    skill_intents: [],
    required_skill_ids: [],
    required_tool_ids: [],
    allowed_tool_ids: [],
    denied_tool_ids: [],
    requires_tool_calling: false,
    required_mcp_server_ids: [],
    allowed_mcp_server_ids: [],
    denied_mcp_server_ids: [],
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

function createBrainstormClusterSeed(
  index: number,
  text: Record<string, string>,
  position: { x: number; y: number }
): HarnessCanvasAgentDTO {
  return {
    agent_id: makeId('cluster'),
    name: formatTemplate(text.brainstormClusterName, { count: index + 1 }),
    node_kind: 'cluster',
    cluster_strategy: 'brainstorm',
    role: text.clusterRoleSeed,
    description: text.brainstormClusterDescription,
    system_prompt: text.brainstormClusterPrompt,
    model: 'gpt-5.2',
    position,
    cluster_members: [
      createClusterMemberSeed(text.leadStrategistName, text.chairRoleSeed, 'gpt-5.2', text.leadStrategistPrompt),
      createClusterMemberSeed(text.fastChallengerName, text.criticRoleSeed, 'gpt-5.1-codex-mini', text.fastChallengerPrompt),
      createClusterMemberSeed(text.synthesisVoiceName, text.synthesizerRoleSeed, 'gpt-5.1-codex-mini', text.synthesisVoicePrompt),
    ],
    brainstorm_rounds: 3,
    cluster_auto_research: true,
    cluster_auto_review: true,
    skill_ids: [],
    skill_intents: ['research'],
    required_skill_ids: [],
    required_tool_ids: [],
    allowed_tool_ids: [],
    denied_tool_ids: [],
    requires_tool_calling: false,
    required_mcp_server_ids: [],
    allowed_mcp_server_ids: [],
    denied_mcp_server_ids: [],
  };
}

function createCustomClusterSeed(
  index: number,
  text: Record<string, string>,
  position: { x: number; y: number }
): HarnessCanvasAgentDTO {
  return {
    agent_id: makeId('cluster'),
    name: formatTemplate(text.customClusterName, { count: index + 1 }),
    node_kind: 'cluster',
    cluster_strategy: 'custom',
    role: text.clusterRoleSeed,
    description: text.customClusterDescription,
    system_prompt: text.customClusterPrompt,
    model: 'gpt-5.2',
    position,
    cluster_members: [
      createClusterMemberSeed(text.plannerName, text.plannerRoleSeed, 'gpt-5.2', text.plannerPrompt),
      createClusterMemberSeed(text.builderName, text.builderRoleSeed, 'gpt-5.1-codex-mini', text.builderPrompt),
    ],
    brainstorm_rounds: 2,
    cluster_auto_research: false,
    cluster_auto_review: true,
    skill_ids: [],
    skill_intents: [],
    required_skill_ids: [],
    required_tool_ids: [],
    allowed_tool_ids: [],
    denied_tool_ids: [],
    requires_tool_calling: false,
    required_mcp_server_ids: [],
    allowed_mcp_server_ids: [],
    denied_mcp_server_ids: [],
  };
}

function createExecutionChecklistItem(): HarnessExecutionChecklistItemDTO {
  return {
    item_id: makeId('check'),
    content: '',
    status: 'pending',
    active_form: null,
  };
}

function normalizeChecklistStatus(value: unknown): HarnessExecutionChecklistItemDTO['status'] {
  if (value === 'completed' || value === 'in_progress') {
    return value;
  }
  return 'pending';
}

function normalizeChecklistItem(value: unknown, fallbackId: string): HarnessExecutionChecklistItemDTO | null {
  if (!value || typeof value !== 'object') {
    return null;
  }
  const item = value as Record<string, unknown>;
  const content = typeof item.content === 'string' ? item.content.trim() : '';
  if (!content) {
    return null;
  }
  const activeForm = typeof item.active_form === 'string' ? item.active_form.trim() || null : null;
  return {
    item_id: typeof item.item_id === 'string' && item.item_id.trim() ? item.item_id.trim() : fallbackId,
    content,
    status: normalizeChecklistStatus(item.status),
    active_form: activeForm,
  };
}

function buildChecklistSnapshot(items: HarnessExecutionChecklistItemDTO[]): HarnessRunChecklistSnapshotDTO {
  const completedItems = items.filter((item) => item.status === 'completed').length;
  return {
    enabled: items.length > 0,
    total_items: items.length,
    open_items: items.length - completedItems,
    completed_items: completedItems,
    items,
  };
}

function resolveRunChecklistSnapshot(
  run: (HarnessRunSummaryDTO & { input_json?: Record<string, unknown> }) | null | undefined
): HarnessRunChecklistSnapshotDTO | null {
  if (!run) {
    return null;
  }

  const snapshotItems = (run.checklist_snapshot?.items ?? [])
    .map((item, index) => normalizeChecklistItem(item, `check_${index + 1}`))
    .filter((item): item is HarnessExecutionChecklistItemDTO => Boolean(item));

  if (snapshotItems.length > 0 || run.checklist_snapshot?.enabled) {
    const completedItems =
      typeof run.checklist_snapshot?.completed_items === 'number'
        ? run.checklist_snapshot.completed_items
        : snapshotItems.filter((item) => item.status === 'completed').length;
    const totalItems =
      typeof run.checklist_snapshot?.total_items === 'number'
        ? run.checklist_snapshot.total_items
        : snapshotItems.length;
    const openItems =
      typeof run.checklist_snapshot?.open_items === 'number'
        ? run.checklist_snapshot.open_items
        : Math.max(totalItems - completedItems, 0);
    return {
      enabled: run.checklist_snapshot?.enabled ?? snapshotItems.length > 0,
      total_items: totalItems,
      open_items: openItems,
      completed_items: completedItems,
      items: snapshotItems,
    };
  }

  const rawChecklist = run.input_json?.['task_checklist'];
  if (!Array.isArray(rawChecklist)) {
    return null;
  }
  const inputItems = rawChecklist
    .map((item, index) => normalizeChecklistItem(item, `check_${index + 1}`))
    .filter((item): item is HarnessExecutionChecklistItemDTO => Boolean(item));
  return inputItems.length > 0 ? buildChecklistSnapshot(inputItems) : null;
}

function getChecklistFocusItem(checklist: HarnessRunChecklistSnapshotDTO | null | undefined) {
  if (!checklist?.items || checklist.items.length === 0) {
    return null;
  }
  const inProgressItem = checklist.items.find((item) => item.status === 'in_progress');
  if (inProgressItem) {
    return {
      status: 'in_progress' as const,
      content: inProgressItem.active_form || inProgressItem.content,
    };
  }
  const pendingItem = checklist.items.find((item) => item.status === 'pending');
  if (pendingItem) {
    return {
      status: 'pending' as const,
      content: pendingItem.content,
    };
  }
  return null;
}

function getRunChecklistOpenCount(run: HarnessRunSummaryDTO) {
  const checklist = resolveRunChecklistSnapshot(run);
  if (!checklist?.enabled) {
    return 0;
  }
  return checklist.open_items ?? Math.max((checklist.total_items ?? checklist.items?.length ?? 0) - (checklist.completed_items ?? 0), 0);
}

function isRunPendingApproval(run: HarnessRunSummaryDTO) {
  return run.status === 'waiting_approval' || run.latest_approval?.status === 'pending';
}

function isRunBlocked(run: HarnessRunSummaryDTO) {
  return (
    run.workflow_progress?.status === 'blocked' ||
    run.status === 'failed' ||
    run.latest_approval?.status === 'rejected'
  );
}

function isRunActive(run: HarnessRunSummaryDTO) {
  return ['created', 'queued', 'running', 'approved', 'resumed', 'verifying'].includes(run.status || '');
}

function getRunQueueGroup(run: HarnessRunSummaryDTO): RunQueueGroup {
  if (isRunPendingApproval(run)) {
    return 'approval';
  }
  if (isRunBlocked(run)) {
    return 'blocked';
  }
  if (getRunChecklistOpenCount(run) > 0) {
    return 'checklist';
  }
  if (isRunActive(run)) {
    return 'active';
  }
  if (run.status === 'completed') {
    return 'completed';
  }
  return 'other';
}

function matchesRunFilter(run: HarnessRunSummaryDTO, filter: RunFilter) {
  const group = getRunQueueGroup(run);
  switch (filter) {
    case 'attention':
      return group === 'approval' || group === 'blocked' || group === 'checklist';
    case 'active':
      return group === 'approval' || group === 'active';
    case 'checklist':
      return group === 'checklist';
    case 'completed':
      return group === 'completed';
    default:
      return true;
  }
}

function compareHarnessRuns(a: HarnessRunSummaryDTO, b: HarnessRunSummaryDTO) {
  const rankMap: Record<RunQueueGroup, number> = {
    approval: 0,
    blocked: 1,
    checklist: 2,
    active: 3,
    completed: 4,
    other: 5,
  };

  const rankDelta = rankMap[getRunQueueGroup(a)] - rankMap[getRunQueueGroup(b)];
  if (rankDelta !== 0) {
    return rankDelta;
  }

  const updatedDelta = (Number(b.updated_at) || 0) - (Number(a.updated_at) || 0);
  if (updatedDelta !== 0) {
    return updatedDelta;
  }

  return String(b.run_id || '').localeCompare(String(a.run_id || ''));
}

function getNodeExecutionLabels(node: HarnessCanvasAgentDTO, text: Record<string, string>) {
  if (node.node_kind !== 'cluster') {
    return [node.name];
  }
  const rounds = node.cluster_strategy === 'brainstorm' ? Math.max(1, Math.min(node.brainstorm_rounds ?? 1, 5)) : 1;
  const memberLabels = Array.from({ length: rounds }).flatMap((_, roundIndex) =>
    (node.cluster_members ?? []).map((member) =>
      `${node.name} / ${member.name}${rounds > 1 ? ` (${formatTemplate(text.roundLabel, { count: roundIndex + 1 })})` : ''}`
    )
  );
  return [...memberLabels, `${node.name} / ${text.summaryLabel}`];
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

function humanizeRecoveryMode(value: string | null | undefined, text: Record<string, string>) {
  switch (value) {
    case 'continue_with_research':
      return text.continueWithResearch;
    case 'continue_without_research':
      return text.continueWithoutResearch;
    case 'continue_with_partial_stream_output':
      return text.continueWithStreamPrefix;
    case 'continue_from_stream_block':
      return text.continueFromStreamBlock;
    default:
      return value || text.standardExecution;
  }
}

function getRunContinuationLabel(run: HarnessRunSummaryDTO, text: Record<string, string>) {
  const continuation = run.runtime_state?.continuation;
  if (!continuation?.enabled) {
    return null;
  }
  if (continuation.status === 'completed' || run.status === 'completed') {
    return text.streamContinued;
  }
  if (run.latest_approval?.status === 'pending') {
    return text.streamReviewPending;
  }
  if (run.latest_approval?.status === 'rejected') {
    return text.streamContinuationRejected;
  }
  if (run.latest_approval?.status === 'approved' || run.status === 'approved' || run.status === 'resumed' || run.status === 'running') {
    return text.streamContinuation;
  }
  return text.streamReview;
}

function getWorkflowProgressSummary(
  workflow: HarnessWorkflowProgressDTO | null | undefined,
  text: Record<string, string>
) {
  if (!workflow?.enabled || !workflow.total_steps) {
    return null;
  }
  const completed = workflow.completed_steps ?? 0;
  const total = workflow.total_steps ?? 0;
  return formatTemplate(text.progressCount, { completed, total });
}

function WorkflowProgressCard({
  workflow,
}: {
  workflow?: HarnessWorkflowProgressDTO | null;
}) {
  const text = useMessages(HARNESS_MESSAGES);

  if (!workflow?.enabled || !workflow.steps || workflow.steps.length === 0) {
    return <div className="mt-5 text-sm text-slate-500">{text.noWorkflowProgress}</div>;
  }

  const summary = getWorkflowProgressSummary(workflow, text);

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        {summary ? (
          <span className="inline-flex rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
            {summary}
          </span>
        ) : null}
        {workflow.blocking_step_index !== null && workflow.blocking_step_index !== undefined ? (
          <span className="inline-flex rounded-full bg-rose-50 px-3 py-1 text-xs font-semibold text-rose-900 ring-1 ring-rose-200">
            {formatTemplate(text.blockedAtStepCount, { count: workflow.blocking_step_index + 1 })}
          </span>
        ) : null}
        {workflow.current_step_label ? <StatusPill value={workflow.status} /> : null}
      </div>
      {workflow.current_step_label ? (
        <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700">
          <span className="font-semibold text-slate-900">{text.currentStep}:</span> {workflow.current_step_label}
        </div>
      ) : null}
      <div className="space-y-2">
        {workflow.steps.map((step) => (
          <div
            key={step.step_id}
            className={`flex items-center justify-between gap-3 rounded-2xl border px-4 py-3 ${
              step.status === 'completed'
                ? 'border-emerald-200 bg-emerald-50/60'
                : step.status === 'blocked'
                  ? 'border-rose-200 bg-rose-50/60'
                  : step.status === 'in_progress'
                    ? 'border-sky-200 bg-sky-50/70'
                    : 'border-slate-200 bg-white'
            }`}
          >
            <div className="min-w-0">
              <div className="text-xs uppercase tracking-[0.18em] text-slate-400">
                {text.stepLabel} {step.step_index + 1}
                {step.loop_number > 1 ? ` • ${formatTemplate(text.loopExecutionLabel, { count: step.loop_number })}` : ''}
              </div>
              <div className="mt-1 truncate text-sm font-medium text-slate-900">{step.label}</div>
            </div>
            <StatusPill value={step.status} />
          </div>
        ))}
      </div>
    </div>
  );
}

function ChecklistSnapshotCard({
  checklist,
}: {
  checklist?: HarnessRunChecklistSnapshotDTO | null;
}) {
  const text = useMessages(HARNESS_MESSAGES);

  if (!checklist?.enabled || !checklist.items || checklist.items.length === 0) {
    return <div className="mt-5 text-sm text-slate-500">{text.noChecklistSnapshot}</div>;
  }

  const totalItems = checklist.total_items ?? checklist.items.length;
  const completedItems = checklist.completed_items ?? checklist.items.filter((item) => item.status === 'completed').length;
  const openItems = checklist.open_items ?? Math.max(totalItems - completedItems, 0);

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
          {formatTemplate(text.checklistCount, { count: totalItems })}
        </span>
        <span className="inline-flex rounded-full bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-900 ring-1 ring-amber-200">
          {formatTemplate(text.openChecklistCount, { count: openItems })}
        </span>
        <span className="inline-flex rounded-full bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-900 ring-1 ring-emerald-200">
          {formatTemplate(text.completedChecklistCount, { count: completedItems })}
        </span>
      </div>
      <div className="space-y-2">
        {checklist.items.map((item, index) => (
          <div
            key={item.item_id || `checklist-item-${index}`}
            className={`flex items-start justify-between gap-3 rounded-2xl border px-4 py-3 ${
              item.status === 'completed'
                ? 'border-emerald-200 bg-emerald-50/60'
                : item.status === 'in_progress'
                  ? 'border-sky-200 bg-sky-50/70'
                  : 'border-slate-200 bg-white'
            }`}
          >
            <div className="min-w-0">
              <div className="text-xs uppercase tracking-[0.18em] text-slate-400">
                {text.checklistItemLabel} {index + 1}
              </div>
              <div className="mt-1 break-words text-sm font-medium text-slate-900">{item.active_form || item.content}</div>
              {item.active_form && item.active_form !== item.content ? (
                <div className="mt-1 text-xs text-slate-500">
                  {text.checklistOriginalContent}: {item.content}
                </div>
              ) : null}
            </div>
            <StatusPill value={item.status} />
          </div>
        ))}
      </div>
    </div>
  );
}

function CapabilityGapCard({
  blockedAgents,
  skillTitleById,
  toolTitleById,
  canFocusAgent,
  onFocusAgent,
  onOpenSkillPool,
}: {
  blockedAgents: Record<string, unknown>[];
  skillTitleById: Map<string, string>;
  toolTitleById: Map<string, string>;
  canFocusAgent?: (agentId: string) => boolean;
  onFocusAgent?: (agentId: string) => void;
  onOpenSkillPool?: () => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);

  if (blockedAgents.length === 0) {
    return <div className="mt-5 text-sm text-slate-500">{text.noCapabilityGaps}</div>;
  }

  return (
    <div className="mt-5 space-y-3">
      {blockedAgents.map((agent, index) => {
        const agentId = typeof agent.agent_id === 'string' ? agent.agent_id : `blocked-agent-${index}`;
        const agentName = typeof agent.agent_name === 'string' && agent.agent_name ? agent.agent_name : agentId;
        const missingSkills = coerceStringList(agent.missing_skills);
        const missingSkillDetails = coerceRecordList(agent.missing_skill_details);

        return (
          <div key={`${agentId}-${index}`} className="rounded-2xl border border-rose-200 bg-rose-50/70 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-rose-950">{agentName}</div>
                <div className="mt-1 text-xs text-rose-700">{text.missingCapabilityPacks}</div>
              </div>
              <StatusPill value="blocked" />
            </div>
            {missingSkillDetails.length > 0 ? (
              <div className="mt-4 space-y-3">
                {missingSkillDetails.map((detail, detailIndex) => {
                  const skillId = typeof detail.skill_id === 'string' ? detail.skill_id : '';
                  const title =
                    typeof detail.title === 'string' && detail.title
                      ? detail.title
                      : formatSkillTitle(skillId, skillTitleById);
                  const description = typeof detail.description === 'string' ? detail.description : null;
                  const promptHint = typeof detail.prompt_hint === 'string' ? detail.prompt_hint : null;
                  const suggestedTools = coerceStringList(detail.suggested_tool_ids);
                  const suggestedMcpServers = coerceStringList(detail.suggested_mcp_server_ids);

                  return (
                    <div key={`${agentId}-skill-${skillId || detailIndex}`} className="rounded-2xl border border-white/70 bg-white/90 p-4">
                      <div className="flex flex-wrap items-center gap-2">
                        <div className="text-sm font-semibold text-slate-900">{title}</div>
                        {skillId ? (
                          <span className="inline-flex rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
                            {skillId}
                          </span>
                        ) : null}
                      </div>
                      {description ? <div className="mt-2 text-sm text-slate-700">{description}</div> : null}
                      {promptHint ? (
                        <div className="mt-3 rounded-2xl border border-cyan-200 bg-cyan-50 px-3 py-3 text-sm text-cyan-950">
                          <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">{text.promptHintLabel}</div>
                          <div className="mt-2">{promptHint}</div>
                        </div>
                      ) : null}
                      {suggestedTools.length > 0 ? (
                        <div className="mt-3">
                          <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.suggestedToolsForGap}</div>
                          <div className="mt-2 flex flex-wrap gap-2">
                            {suggestedTools.map((toolId) => (
                              <span key={`${agentId}-${skillId}-tool-${toolId}`} className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
                                {formatSkillTitle(toolId, toolTitleById)}
                              </span>
                            ))}
                          </div>
                        </div>
                      ) : null}
                      {suggestedMcpServers.length > 0 ? (
                        <div className="mt-3">
                          <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.suggestedMcpForGap}</div>
                          <div className="mt-2 flex flex-wrap gap-2">
                            {suggestedMcpServers.map((serverId) => (
                              <span key={`${agentId}-${skillId}-mcp-${serverId}`} className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
                                {serverId}
                              </span>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            ) : missingSkills.length > 0 ? (
              <div className="mt-4 flex flex-wrap gap-2">
                {missingSkills.map((skillId) => (
                  <span key={`${agentId}-${skillId}`} className="rounded-full bg-white px-2.5 py-1 text-[10px] font-semibold text-rose-800 ring-1 ring-rose-200">
                    {formatSkillTitle(skillId, skillTitleById)}
                  </span>
                ))}
              </div>
            ) : (
              <div className="mt-4 text-sm text-slate-500">{text.noDetailCaptured}</div>
            )}
            {onOpenSkillPool || (canFocusAgent?.(agentId) && onFocusAgent) ? (
              <div className="mt-4 flex flex-wrap gap-2">
                {onOpenSkillPool ? (
                  <button
                    type="button"
                    onClick={onOpenSkillPool}
                    className="inline-flex items-center justify-center rounded-xl border border-rose-200 bg-white px-3 py-2 text-xs font-semibold text-rose-900 hover:bg-rose-100"
                  >
                    {text.openSkillPoolAction}
                  </button>
                ) : null}
                {canFocusAgent?.(agentId) && onFocusAgent ? (
                  <button
                    type="button"
                    onClick={() => onFocusAgent(agentId)}
                    className="inline-flex items-center justify-center rounded-xl border border-rose-200 bg-white px-3 py-2 text-xs font-semibold text-rose-900 hover:bg-rose-100"
                  >
                    {formatTemplate(text.focusNodeForRecovery, { name: agentName })}
                  </button>
                ) : null}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function StructuredArtifactCard({
  artifactId,
  artifact,
}: {
  artifactId: string;
  artifact: Record<string, unknown>;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const nodeKind = typeof artifact.node_kind === 'string' ? artifact.node_kind : 'agent';
  const title =
    typeof artifact.agent_name === 'string' && artifact.agent_name
      ? artifact.agent_name
      : typeof artifact.cluster_name === 'string' && artifact.cluster_name
        ? artifact.cluster_name
        : artifactId;

  if (nodeKind === 'cluster') {
    const winningStrategy = typeof artifact.winning_strategy === 'string' ? artifact.winning_strategy : null;
    const nextStep = typeof artifact.next_step === 'string' ? artifact.next_step : null;
    const dominantRisks = typeof artifact.dominant_risks === 'string' ? artifact.dominant_risks : null;
    const research = coerceRecord(artifact.research);
    const researchQueries = coerceStringList(research?.queries);
    const researchDigest = typeof research?.digest === 'string' ? research.digest : null;
    const researchReviewOutput = typeof research?.review_output === 'string' ? research.review_output : null;
    const paperCount = Array.isArray(research?.papers) ? research.papers.length : coerceNumber(research?.paper_count);
    const browserPreviewCount = Array.isArray(research?.browser_previews)
      ? research.browser_previews.length
      : coerceNumber(research?.browser_preview_count);
    const sourceCount = Array.isArray(research?.sources) ? research.sources.length : coerceNumber(research?.source_count);

    return (
      <div className="rounded-2xl border border-amber-200 bg-amber-50/70 p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold text-amber-950">{title}</div>
            <div className="mt-1 text-xs text-amber-700">{text.clusterSynthesis}</div>
          </div>
          <StatusPill value="cluster" />
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {winningStrategy ? (
            <div className="rounded-2xl border border-white/70 bg-white/90 p-3">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.clusterWinningStrategy}</div>
              <div className="mt-2 text-sm text-slate-900">{winningStrategy}</div>
            </div>
          ) : null}
          {nextStep ? (
            <div className="rounded-2xl border border-white/70 bg-white/90 p-3">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.clusterNextStep}</div>
              <div className="mt-2 text-sm text-slate-900">{nextStep}</div>
            </div>
          ) : null}
        </div>
        {dominantRisks ? (
          <div className="mt-3 rounded-2xl border border-rose-200 bg-white/90 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-rose-600">{text.clusterDominantRisks}</div>
            <div className="mt-2 text-sm text-slate-900">{dominantRisks}</div>
          </div>
        ) : null}
        {research ? (
          <div className="mt-3 rounded-2xl border border-sky-200 bg-white/90 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-700">{text.researchSync}</div>
            {(paperCount !== null && paperCount > 0) || (browserPreviewCount !== null && browserPreviewCount > 0) || (sourceCount !== null && sourceCount > 0) ? (
              <div className="mt-2 flex flex-wrap gap-2">
                {paperCount !== null && paperCount > 0 ? (
                  <span className="rounded-full bg-sky-50 px-2.5 py-1 text-[10px] font-semibold text-sky-800 ring-1 ring-sky-200">
                    {paperCount} {text.papers}
                  </span>
                ) : null}
                {browserPreviewCount !== null && browserPreviewCount > 0 ? (
                  <span className="rounded-full bg-sky-50 px-2.5 py-1 text-[10px] font-semibold text-sky-800 ring-1 ring-sky-200">
                    {browserPreviewCount} {text.browserPreviews}
                  </span>
                ) : null}
                {sourceCount !== null && sourceCount > 0 ? (
                  <span className="rounded-full bg-sky-50 px-2.5 py-1 text-[10px] font-semibold text-sky-800 ring-1 ring-sky-200">
                    {sourceCount} {text.sources}
                  </span>
                ) : null}
              </div>
            ) : null}
            {researchQueries.length > 0 ? (
              <div className="mt-3">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.researchQueriesLabel}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {researchQueries.map((query, index) => (
                    <span key={`${artifactId}-query-${index}`} className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
                      {query}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
            <div className="mt-3 rounded-2xl border border-white/70 bg-slate-50 p-3">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.researchDigestLabel}</div>
              <div className="mt-2 whitespace-pre-wrap break-words text-sm text-slate-900">{researchDigest || text.noResearchDigestCaptured}</div>
            </div>
            {researchReviewOutput ? (
              <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-900">
                {researchReviewOutput}
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    );
  }

  const handoffSummary = typeof artifact.handoff_summary === 'string' ? artifact.handoff_summary : null;
  const outputPreview = typeof artifact.output_preview === 'string' ? artifact.output_preview : null;
  const actionItems = coerceStringList(artifact.action_items);
  const openQuestions = coerceStringList(artifact.open_questions);
  const riskFlags = coerceStringList(artifact.risk_flags);
  const mcpServerIds = coerceStringList(artifact.mcp_server_ids);
  const missingMcpServerIds = coerceStringList(artifact.missing_mcp_server_ids);
  const consumedHandoffs = coerceRecordList(artifact.consumed_handoffs);
  const toolRuns = coerceRecordList(artifact.tool_runs);
  const downstreamHandoffs = coerceRecordList(artifact.downstream_handoffs);
  const finalOutput = artifact.final_output === true;

  return (
    <div className="rounded-2xl border border-cyan-200 bg-cyan-50/70 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-cyan-950">{title}</div>
          <div className="mt-1 text-xs text-cyan-700">{text.handoffArtifacts}</div>
        </div>
        {finalOutput ? <StatusPill value="completed" /> : <StatusPill value="handoff" />}
      </div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        {handoffSummary ? (
          <div className="rounded-2xl border border-white/70 bg-white/90 p-3 sm:col-span-2">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.handoffSummaryLabel}</div>
            <div className="mt-2 text-sm text-slate-900">{handoffSummary}</div>
          </div>
        ) : null}
        {outputPreview ? (
          <div className="rounded-2xl border border-white/70 bg-white/90 p-3 sm:col-span-2">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.outputPreviewLabel}</div>
            <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-words font-mono text-xs text-slate-900">{outputPreview}</pre>
          </div>
        ) : null}
        {toolRuns.length > 0 ? (
          <div className="rounded-2xl border border-white/70 bg-white/90 p-3 sm:col-span-2">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.toolRunsLabel}</div>
            <div className="mt-3 space-y-3">
              {toolRuns.map((toolRun, index) => {
                const toolId =
                  typeof toolRun.tool_id === 'string' && toolRun.tool_id ? toolRun.tool_id : `tool-${index + 1}`;
                const status = typeof toolRun.status === 'string' ? toolRun.status : 'success';
                const argsPreview = typeof toolRun.args_preview === 'string' ? toolRun.args_preview : null;
                const resultPreview = typeof toolRun.result_preview === 'string' ? toolRun.result_preview : null;
                return (
                  <div key={`${artifactId}-tool-${index}`} className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="text-sm font-semibold text-slate-900">{humanizeHarnessValue(toolId, text)}</div>
                      <StatusPill value={status} />
                    </div>
                    {argsPreview ? (
                      <div className="mt-2 text-sm text-slate-700">
                        <span className="font-medium text-slate-900">{text.toolArgumentsLabel}:</span> {argsPreview}
                      </div>
                    ) : null}
                    {resultPreview ? (
                      <div className="mt-2">
                        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.toolResultPreviewLabel}</div>
                        <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-xs text-slate-800">{resultPreview}</pre>
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}
        {actionItems.length > 0 ? (
          <div className="rounded-2xl border border-white/70 bg-white/90 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.nextActionsLabel}</div>
            <div className="mt-2 space-y-1 text-sm text-slate-900">
              {actionItems.map((item, index) => (
                <div key={`${artifactId}-action-${index}`} className="break-words">{item}</div>
              ))}
            </div>
          </div>
        ) : null}
        {openQuestions.length > 0 ? (
          <div className="rounded-2xl border border-white/70 bg-white/90 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.openQuestionsLabel}</div>
            <div className="mt-2 space-y-1 text-sm text-slate-900">
              {openQuestions.map((item, index) => (
                <div key={`${artifactId}-question-${index}`} className="break-words">{item}</div>
              ))}
            </div>
          </div>
        ) : null}
        {riskFlags.length > 0 ? (
          <div className="rounded-2xl border border-rose-200 bg-white/90 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-rose-600">{text.riskFlagsLabel}</div>
            <div className="mt-2 space-y-1 text-sm text-slate-900">
              {riskFlags.map((item, index) => (
                <div key={`${artifactId}-risk-${index}`} className="break-words">{item}</div>
              ))}
            </div>
          </div>
        ) : null}
        {mcpServerIds.length > 0 ? (
          <div className="rounded-2xl border border-violet-200 bg-white/90 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-violet-700">{text.mcpServersLabel}</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {mcpServerIds.map((serverId) => (
                <span key={`${artifactId}-mcp-${serverId}`} className="rounded-full bg-violet-50 px-2.5 py-1 text-[10px] font-semibold text-violet-800 ring-1 ring-violet-200">
                  {humanizeHarnessValue(serverId, text)}
                </span>
              ))}
            </div>
          </div>
        ) : null}
        {missingMcpServerIds.length > 0 ? (
          <div className="rounded-2xl border border-amber-200 bg-white/90 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-700">{text.missingMcpServersLabel}</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {missingMcpServerIds.map((serverId) => (
                <span key={`${artifactId}-missing-mcp-${serverId}`} className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
                  {humanizeHarnessValue(serverId, text)}
                </span>
              ))}
            </div>
          </div>
        ) : null}
        {consumedHandoffs.length > 0 ? (
          <div className="rounded-2xl border border-white/70 bg-white/90 p-3 sm:col-span-2">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.upstreamHandoffsLabel}</div>
            <div className="mt-3 space-y-3">
              {consumedHandoffs.map((handoff, index) => {
                const sourceName =
                  typeof handoff.source_agent_name === 'string' && handoff.source_agent_name
                    ? handoff.source_agent_name
                    : typeof handoff.source_agent_id === 'string'
                      ? handoff.source_agent_id
                      : text.unknownNode;
                const interaction =
                  typeof handoff.interaction === 'string' && handoff.interaction
                    ? handoff.interaction
                    : 'handoff';
                const artifactSummary = typeof handoff.artifact_summary === 'string' ? handoff.artifact_summary : null;
                const upstreamOutputPreview = typeof handoff.output_preview === 'string' ? handoff.output_preview : null;
                return (
                  <div key={`${artifactId}-upstream-${index}`} className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                    <div className="text-sm font-semibold text-slate-900">
                      {sourceName} · {humanizeHarnessValue(interaction, text)}
                    </div>
                    {artifactSummary ? (
                      <div className="mt-2 text-sm text-slate-700">
                        <span className="font-medium text-slate-900">{text.artifactSummaryLabel}:</span> {artifactSummary}
                      </div>
                    ) : null}
                    {upstreamOutputPreview ? (
                      <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-xs text-slate-800">{upstreamOutputPreview}</pre>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}
        {downstreamHandoffs.length > 0 ? (
          <div className="rounded-2xl border border-white/70 bg-white/90 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.downstreamHandoffsLabel}</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {downstreamHandoffs.map((handoff, index) => {
                const targetName =
                  typeof handoff.target_agent_name === 'string' && handoff.target_agent_name
                    ? handoff.target_agent_name
                    : typeof handoff.target_agent_id === 'string'
                      ? handoff.target_agent_id
                      : text.unknownNode;
                const interaction =
                  typeof handoff.interaction === 'string' && handoff.interaction
                    ? handoff.interaction
                    : 'handoff';
                return (
                  <span key={`${artifactId}-downstream-${index}`} className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
                    {targetName} · {humanizeHarnessValue(interaction, text)}
                  </span>
                );
              })}
            </div>
          </div>
        ) : null}
        {finalOutput ? (
          <div className="rounded-2xl border border-emerald-200 bg-white/90 p-3">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-600">{text.finalOutputReady}</div>
            <div className="mt-2 text-sm text-slate-900">{text.finalOutputReady}</div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function HandoffArtifactsCard({
  artifacts,
}: {
  artifacts: Array<{ artifactId: string; artifact: Record<string, unknown> }>;
}) {
  const text = useMessages(HARNESS_MESSAGES);

  if (artifacts.length === 0) {
    return <div className="mt-5 text-sm text-slate-500">{text.noHandoffArtifacts}</div>;
  }

  return (
    <div className="mt-5 space-y-3">
      {artifacts.map(({ artifactId, artifact }) => (
        <StructuredArtifactCard key={artifactId} artifactId={artifactId} artifact={artifact} />
      ))}
    </div>
  );
}

type EdgeDelegationDiagnostic = {
  edgeId: string;
  fit: 'strong' | 'good' | 'weak';
  score: number;
  rationale: string | null;
  sourceAgentName: string;
  targetAgentName: string;
  interaction: string;
  bestAlternative: HarnessDelegationTargetFitDTO | null;
};

function edgeStrokeColor(fit: string | null | undefined) {
  if (fit === 'strong') {
    return 'rgba(5,150,105,0.52)';
  }
  if (fit === 'good') {
    return 'rgba(8,145,178,0.46)';
  }
  return 'rgba(217,119,6,0.56)';
}

function edgeLabelColor(fit: string | null | undefined) {
  if (fit === 'strong') {
    return 'rgba(6,95,70,0.88)';
  }
  if (fit === 'good') {
    return 'rgba(15,118,110,0.88)';
  }
  return 'rgba(146,64,14,0.9)';
}

function edgeMarkerId(fit: string | null | undefined) {
  if (fit === 'strong') {
    return 'harness-canvas-arrow-strong';
  }
  if (fit === 'good') {
    return 'harness-canvas-arrow-good';
  }
  return 'harness-canvas-arrow-weak';
}

function buildEdgeDelegationDiagnostic(
  edge: HarnessCanvasEdgeDTO,
  summaryByAgentId: Map<string, HarnessAgentCapabilitySummaryDTO>,
  agentNameById: Map<string, string>
): EdgeDelegationDiagnostic | null {
  const sourceSummary = summaryByAgentId.get(edge.source_agent_id);
  const sourceAgentName = agentNameById.get(edge.source_agent_id) ?? edge.source_agent_id;
  const targetAgentName = agentNameById.get(edge.target_agent_id) ?? edge.target_agent_id;
  if (!sourceSummary) {
    return null;
  }

  const downstreamFit =
    (sourceSummary.downstream_handoff_scores ?? []).find((item) => item.agent_id === edge.target_agent_id) ?? null;
  const bestAlternative =
    (sourceSummary.recommended_collaborators ?? []).find(
      (item) =>
        item.agent_id !== edge.target_agent_id &&
        !item.edge_present &&
        (item.fit === 'strong' || item.fit === 'good')
    ) ?? null;

  if (!downstreamFit && !bestAlternative) {
    return null;
  }

  const fit = normalizeDelegationFit(downstreamFit?.fit);
  return {
    edgeId: edge.edge_id,
    fit,
    score: typeof downstreamFit?.score === 'number' ? downstreamFit.score : 0,
    rationale: typeof downstreamFit?.rationale === 'string' ? downstreamFit.rationale : null,
    sourceAgentName,
    targetAgentName,
    interaction: typeof downstreamFit?.interaction === 'string' && downstreamFit.interaction
      ? downstreamFit.interaction
      : typeof edge.interaction === 'string' && edge.interaction
        ? edge.interaction
        : 'handoff',
    bestAlternative,
  };
}

function rewireGraphEdge(
  graph: NonNullable<HarnessProjectDetailDTO['graph_json']>,
  {
    sourceAgentId,
    fromTargetAgentId,
    toTargetAgentId,
  }: {
    sourceAgentId: string;
    fromTargetAgentId: string;
    toTargetAgentId: string;
  }
) {
  if (!sourceAgentId || !fromTargetAgentId || !toTargetAgentId || fromTargetAgentId === toTargetAgentId) {
    return { changed: false, graph };
  }

  let changed = false;
  const alreadyExists = (graph.edges ?? []).some(
    (edge) => edge.source_agent_id === sourceAgentId && edge.target_agent_id === toTargetAgentId
  );

  const nextEdges = (graph.edges ?? []).flatMap((edge) => {
    if (edge.source_agent_id !== sourceAgentId || edge.target_agent_id !== fromTargetAgentId) {
      return [edge];
    }
    changed = true;
    if (alreadyExists) {
      return [];
    }
    return [{ ...edge, target_agent_id: toTargetAgentId }];
  });

  if (!changed) {
    return { changed: false, graph };
  }

  return {
    changed: true,
    graph: {
      ...graph,
      edges: nextEdges,
    },
  };
}

function insertGraphEdge(
  graph: NonNullable<HarnessProjectDetailDTO['graph_json']>,
  {
    sourceAgentId,
    targetAgentId,
    interaction = 'handoff',
  }: {
    sourceAgentId: string;
    targetAgentId: string;
    interaction?: string;
  }
) {
  if (!sourceAgentId || !targetAgentId || sourceAgentId === targetAgentId) {
    return { changed: false, graph };
  }

  const exists = (graph.edges ?? []).some(
    (edge) => edge.source_agent_id === sourceAgentId && edge.target_agent_id === targetAgentId
  );
  if (exists) {
    return { changed: false, graph };
  }

  return {
    changed: true,
    graph: {
      ...graph,
      edges: [
        ...(graph.edges ?? []),
        {
          edge_id: makeId('edge'),
          source_agent_id: sourceAgentId,
          target_agent_id: targetAgentId,
          interaction,
        },
      ],
    },
  };
}

function applySuggestedRewiresToGraph(
  graph: NonNullable<HarnessProjectDetailDTO['graph_json']>,
  diagnostics: Record<string, unknown> | null
) {
  let nextGraph = graph;
  let changedCount = 0;
  let actionableCount = 0;

  for (const edge of coerceRecordList(diagnostics?.weak_downstream_edges)) {
    const target = coerceRecord(edge.target);
    const suggestedReplacements = coerceRecordList(edge.suggested_replacements);
    const replacement = suggestedReplacements[0] ?? null;
    const sourceAgentId =
      typeof edge.source_agent_id === 'string' && edge.source_agent_id ? edge.source_agent_id : '';
    const fromTargetAgentId =
      typeof target?.agent_id === 'string' && target.agent_id ? target.agent_id : '';
    const toTargetAgentId =
      typeof replacement?.agent_id === 'string' && replacement.agent_id ? replacement.agent_id : '';
    if (!sourceAgentId || !fromTargetAgentId || !toTargetAgentId) {
      continue;
    }
    actionableCount += 1;
    const rewired = rewireGraphEdge(nextGraph, {
      sourceAgentId,
      fromTargetAgentId,
      toTargetAgentId,
    });
    if (!rewired.changed) {
      continue;
    }
    nextGraph = rewired.graph;
    changedCount += 1;
  }

  return {
    actionableCount,
    changedCount,
    graph: nextGraph,
  };
}

function applySuggestedHandoffsToGraph(
  graph: NonNullable<HarnessProjectDetailDTO['graph_json']>,
  diagnostics: Record<string, unknown> | null
) {
  let nextGraph = graph;
  let changedCount = 0;
  let actionableCount = 0;

  for (const item of coerceRecordList(diagnostics?.best_next_handoffs)) {
    const target = coerceRecord(item.target);
    const sourceAgentId =
      typeof item.source_agent_id === 'string' && item.source_agent_id ? item.source_agent_id : '';
    const targetAgentId =
      typeof target?.agent_id === 'string' && target.agent_id ? target.agent_id : '';
    if (!sourceAgentId || !targetAgentId) {
      continue;
    }
    actionableCount += 1;
    const inserted = insertGraphEdge(nextGraph, {
      sourceAgentId,
      targetAgentId,
    });
    if (!inserted.changed) {
      continue;
    }
    nextGraph = inserted.graph;
    changedCount += 1;
  }

  return {
    actionableCount,
    changedCount,
    graph: nextGraph,
  };
}

function applySuggestedCollaborationChangesToGraph(
  graph: NonNullable<HarnessProjectDetailDTO['graph_json']>,
  diagnostics: Record<string, unknown> | null
) {
  const rewired = applySuggestedRewiresToGraph(graph, diagnostics);
  const inserted = applySuggestedHandoffsToGraph(rewired.graph, diagnostics);
  return {
    actionableCount: rewired.actionableCount + inserted.actionableCount,
    changedCount: rewired.changedCount + inserted.changedCount,
    graph: inserted.graph,
  };
}

function filterDiagnosticsByAgentScope(
  diagnostics: Record<string, unknown> | null,
  selectedAgentIds: string[]
) {
  const selectedSet = new Set(
    selectedAgentIds.map((agentId) => String(agentId || '').trim()).filter(Boolean)
  );
  if (selectedSet.size === 0) {
    return {
      weak_downstream_edges: [],
      best_next_handoffs: [],
      weak_edge_count: 0,
      best_next_count: 0,
    };
  }

  const weakDownstreamEdges = coerceRecordList(diagnostics?.weak_downstream_edges)
    .filter((edge) => {
      const target = coerceRecord(edge.target);
      const sourceAgentId = typeof edge.source_agent_id === 'string' ? edge.source_agent_id : '';
      const targetAgentId = typeof target?.agent_id === 'string' ? target.agent_id : '';
      return selectedSet.has(sourceAgentId) && selectedSet.has(targetAgentId);
    })
    .map((edge) => ({
      ...edge,
      suggested_replacements: coerceRecordList(edge.suggested_replacements).filter((item) =>
        typeof item.agent_id === 'string' && selectedSet.has(item.agent_id)
      ),
    }));

  const bestNextHandoffs = coerceRecordList(diagnostics?.best_next_handoffs).filter((item) => {
    const target = coerceRecord(item.target);
    const sourceAgentId = typeof item.source_agent_id === 'string' ? item.source_agent_id : '';
    const targetAgentId = typeof target?.agent_id === 'string' ? target.agent_id : '';
    return selectedSet.has(sourceAgentId) && selectedSet.has(targetAgentId);
  });

  return {
    weak_downstream_edges: weakDownstreamEdges,
    best_next_handoffs: bestNextHandoffs,
    weak_edge_count: weakDownstreamEdges.length,
    best_next_count: bestNextHandoffs.length,
  };
}

function CapabilitySnapshotCard({
  snapshot,
  skillTitleById,
  toolTitleById,
  mcpServerTitleById,
}: {
  snapshot: Record<string, unknown> | null;
  skillTitleById: Map<string, string>;
  toolTitleById: Map<string, string>;
  mcpServerTitleById: Map<string, string>;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const agentCapabilities = coerceRecordList(snapshot?.agent_capabilities);
  const mcpInventory = coerceRecordList(snapshot?.mcp_server_catalog);

  if (agentCapabilities.length === 0 && mcpInventory.length === 0) {
    return <div className="mt-5 text-sm text-slate-500">{text.noCapabilitySnapshot}</div>;
  }

  return (
    <div className="mt-5 space-y-3">
      {agentCapabilities.map((agentCapability, index) => {
        const agentId =
          typeof agentCapability.agent_id === 'string' && agentCapability.agent_id
            ? agentCapability.agent_id
            : `agent-${index + 1}`;
        const agentName =
          typeof agentCapability.agent_name === 'string' && agentCapability.agent_name
            ? agentCapability.agent_name
            : agentId;
        const role =
          typeof agentCapability.role === 'string' && agentCapability.role
            ? agentCapability.role
            : 'specialist';
        const availabilityStatus = resolveCapabilityAvailabilityStatus(agentCapability);
        const availabilityBlockers = coerceStringList(agentCapability.availability_blockers);
        const availabilityWarnings = coerceStringList(agentCapability.availability_warnings);
        const requiredSkillIds = coerceStringList(agentCapability.required_skill_ids);
        const missingRequiredSkillIds = coerceStringList(agentCapability.missing_required_skill_ids);
        const requiresToolCalling = Boolean(agentCapability.requires_tool_calling);
        const requiredMcpServerIds = coerceStringList(agentCapability.required_mcp_server_ids);
        const missingRequiredMcpServerIds = coerceStringList(agentCapability.missing_required_mcp_server_ids);
        const readinessStatus = resolveCapabilityReadinessStatus(agentCapability);
        const readinessBlockers = coerceStringList(agentCapability.readiness_blockers);
        const readinessWarnings = coerceStringList(agentCapability.readiness_warnings);
        const delegationFocus =
          typeof agentCapability.delegation_focus === 'string' ? agentCapability.delegation_focus : null;
        const delegationLaneIds = coerceStringList(agentCapability.delegation_lane_ids);
        const loadedSkills = coerceStringList(agentCapability.loaded_skill_ids);
        const configuredAllowedTools = coerceStringList(agentCapability.configured_allowed_tool_ids);
        const configuredDeniedTools = coerceStringList(agentCapability.configured_denied_tool_ids);
        const enabledTools = coerceStringList(agentCapability.enabled_tool_ids);
        const configuredAllowedMcpServerIds = coerceStringList(agentCapability.configured_allowed_mcp_server_ids);
        const configuredDeniedMcpServerIds = coerceStringList(agentCapability.configured_denied_mcp_server_ids);
        const mcpServerIds = coerceStringList(agentCapability.mcp_server_ids);
        const missingMcpServerIds = coerceStringList(agentCapability.missing_mcp_server_ids);
        const recommendedCollaborators = coerceRecordList(agentCapability.recommended_collaborators);
        const downstreamHandoffScores = coerceRecordList(agentCapability.downstream_handoff_scores);

        return (
          <div key={`${agentId}-${index}`} className="rounded-2xl border border-slate-200 bg-white/90 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-slate-950">{agentName}</div>
                <div className="mt-1 text-xs text-slate-500">{humanizeHarnessValue(role, text)}</div>
              </div>
              <div className="flex flex-wrap gap-2">
                <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityAvailabilityBadgeClass(availabilityStatus)}`}>
                  {formatCapabilityAvailabilityLabel(availabilityStatus, text)}
                </span>
                <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityReadinessBadgeClass(readinessStatus)}`}>
                  {formatCapabilityReadinessLabel(readinessStatus, text)}
                </span>
              </div>
            </div>
            <div className="mt-3 rounded-2xl border border-slate-200 bg-slate-50 p-3">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.availabilityLabel}</div>
              {availabilityBlockers.length > 0 || availabilityWarnings.length > 0 || requiredSkillIds.length > 0 || requiredMcpServerIds.length > 0 || requiresToolCalling ? (
                <div className="mt-2 space-y-2 text-xs leading-5 text-slate-600">
                  {requiredSkillIds.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.requiredSkillsLabel}:</span>{' '}
                      {requiredSkillIds.map((skillId) => formatSkillTitle(skillId, skillTitleById)).join(' · ')}
                    </div>
                  ) : null}
                  {missingRequiredSkillIds.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.missingSkillsLabel}:</span>{' '}
                      {missingRequiredSkillIds.map((skillId) => formatSkillTitle(skillId, skillTitleById)).join(' · ')}
                    </div>
                  ) : null}
                  {requiredMcpServerIds.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.requiredMcpServersLabel}:</span>{' '}
                      {requiredMcpServerIds.map((serverId) => formatSkillTitle(serverId, mcpServerTitleById)).join(' · ')}
                    </div>
                  ) : null}
                  {missingRequiredMcpServerIds.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.missingMcpServersLabel}:</span>{' '}
                      {missingRequiredMcpServerIds.map((serverId) => formatSkillTitle(serverId, mcpServerTitleById)).join(' · ')}
                    </div>
                  ) : null}
                  {requiresToolCalling ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.requireToolCallingLabel}:</span>{' '}
                      {text.enabled}
                    </div>
                  ) : null}
                  {availabilityBlockers.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.availabilityBlockersLabel}:</span>{' '}
                      {availabilityBlockers.join(' · ')}
                    </div>
                  ) : null}
                  {availabilityWarnings.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.availabilityWarningsLabel}:</span>{' '}
                      {availabilityWarnings.join(' · ')}
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="mt-2 text-xs text-slate-500">{text.noAvailabilityIssues}</div>
              )}
            </div>
            <div className="mt-3 rounded-2xl border border-slate-200 bg-slate-50 p-3">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.readinessLabel}</div>
              {readinessBlockers.length > 0 || readinessWarnings.length > 0 ? (
                <div className="mt-2 space-y-2 text-xs leading-5 text-slate-600">
                  {readinessBlockers.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.readinessBlockersLabel}:</span>{' '}
                      {readinessBlockers.join(' · ')}
                    </div>
                  ) : null}
                  {readinessWarnings.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.readinessWarningsLabel}:</span>{' '}
                      {readinessWarnings.join(' · ')}
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="mt-2 text-xs text-slate-500">{text.noReadinessIssues}</div>
              )}
            </div>
            {delegationFocus ? (
              <div className="mt-3 rounded-2xl border border-cyan-200 bg-cyan-50/60 p-3">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">{text.delegationFocusLabel}</div>
                <div className="mt-2 text-sm text-slate-900">{delegationFocus}</div>
              </div>
            ) : null}
            <div className="mt-3 rounded-2xl border border-slate-200 bg-slate-50 p-3">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.delegationLanesLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {delegationLaneIds.length > 0 ? (
                  delegationLaneIds.map((laneId) => (
                    <span key={`${agentId}-lane-${laneId}`} className="rounded-full bg-sky-50 px-2.5 py-1 text-[10px] font-semibold text-sky-800 ring-1 ring-sky-200">
                      {formatSkillTitle(laneId, new Map<string, string>())}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-slate-500">{text.none}</span>
                )}
              </div>
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.approvedSkillsLabel}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {loadedSkills.length > 0 ? (
                    loadedSkills.map((skillId) => (
                      <span key={`${agentId}-skill-${skillId}`} className="rounded-full bg-cyan-50 px-2.5 py-1 text-[10px] font-semibold text-cyan-800 ring-1 ring-cyan-200">
                        {formatSkillTitle(skillId, skillTitleById)}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-slate-500">{text.none}</span>
                  )}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.enabledToolsLabel}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {enabledTools.length > 0 ? (
                    enabledTools.map((toolId) => (
                      <span key={`${agentId}-tool-${toolId}`} className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
                        {formatSkillTitle(toolId, toolTitleById)}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-slate-500">{text.none}</span>
                  )}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.toolPolicySummaryLabel}</div>
                <div className="mt-2 space-y-2 text-xs text-slate-600">
                  <div>
                    <span className="font-semibold text-slate-700">{text.toolAllowPolicyLabel}:</span>{' '}
                    {configuredAllowedTools.length > 0
                      ? configuredAllowedTools.map((toolId) => formatSkillTitle(toolId, toolTitleById)).join(', ')
                      : text.none}
                  </div>
                  <div>
                    <span className="font-semibold text-slate-700">{text.toolDenyPolicyLabel}:</span>{' '}
                    {configuredDeniedTools.length > 0
                      ? configuredDeniedTools.map((toolId) => formatSkillTitle(toolId, toolTitleById)).join(', ')
                      : text.none}
                  </div>
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.mcpServersLabel}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {mcpServerIds.length > 0 ? (
                    mcpServerIds.map((serverId) => (
                      <span key={`${agentId}-mcp-${serverId}`} className="rounded-full bg-violet-50 px-2.5 py-1 text-[10px] font-semibold text-violet-800 ring-1 ring-violet-200">
                        {formatSkillTitle(serverId, mcpServerTitleById)}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-slate-500">{text.noMcpServersConfigured}</span>
                  )}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.mcpPolicySummaryLabel}</div>
                <div className="mt-2 space-y-2 text-xs text-slate-600">
                  <div>
                    <span className="font-semibold text-slate-700">{text.mcpAllowPolicyLabel}:</span>{' '}
                    {configuredAllowedMcpServerIds.length > 0
                      ? configuredAllowedMcpServerIds.map((serverId) => formatSkillTitle(serverId, mcpServerTitleById)).join(', ')
                      : text.none}
                  </div>
                  <div>
                    <span className="font-semibold text-slate-700">{text.mcpDenyPolicyLabel}:</span>{' '}
                    {configuredDeniedMcpServerIds.length > 0
                      ? configuredDeniedMcpServerIds.map((serverId) => formatSkillTitle(serverId, mcpServerTitleById)).join(', ')
                      : text.none}
                  </div>
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.missingMcpServersLabel}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {missingMcpServerIds.length > 0 ? (
                    missingMcpServerIds.map((serverId) => (
                      <span key={`${agentId}-missing-mcp-${serverId}`} className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
                        {formatSkillTitle(serverId, mcpServerTitleById)}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-slate-500">{text.noMcpGaps}</span>
                  )}
                </div>
              </div>
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.recommendedCollaboratorsLabel}</div>
                <div className="mt-3 space-y-2">
                  {recommendedCollaborators.length > 0 ? (
                    recommendedCollaborators.map((item, recommendationIndex) => {
                      const recommendedAgentId =
                        typeof item.agent_id === 'string' && item.agent_id ? item.agent_id : `${agentId}-recommendation-${recommendationIndex}`;
                      const recommendedAgentName =
                        typeof item.agent_name === 'string' && item.agent_name ? item.agent_name : recommendedAgentId;
                      const fit = typeof item.fit === 'string' ? item.fit : 'weak';
                      const rationale = typeof item.rationale === 'string' ? item.rationale : null;
                      return (
                        <div key={`${agentId}-recommend-${recommendedAgentId}-${recommendationIndex}`} className="rounded-xl border border-slate-200 bg-white px-3 py-2">
                          <div className="flex items-center justify-between gap-2">
                            <div className="text-sm font-semibold text-slate-900">{recommendedAgentName}</div>
                            <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationFitBadgeClass(fit)}`}>
                              {formatDelegationFitLabel(fit, text)}
                            </span>
                          </div>
                          {rationale ? <div className="mt-1 text-xs leading-5 text-slate-600">{rationale}</div> : null}
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-xs text-slate-500">{text.noRecommendedCollaborators}</div>
                  )}
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.downstreamHandoffFitLabel}</div>
                <div className="mt-3 space-y-2">
                  {downstreamHandoffScores.length > 0 ? (
                    downstreamHandoffScores.map((item, recommendationIndex) => {
                      const downstreamAgentId =
                        typeof item.agent_id === 'string' && item.agent_id ? item.agent_id : `${agentId}-downstream-${recommendationIndex}`;
                      const downstreamAgentName =
                        typeof item.agent_name === 'string' && item.agent_name ? item.agent_name : downstreamAgentId;
                      const fit = typeof item.fit === 'string' ? item.fit : 'weak';
                      const rationale = typeof item.rationale === 'string' ? item.rationale : null;
                      return (
                        <div key={`${agentId}-downstream-fit-${downstreamAgentId}-${recommendationIndex}`} className="rounded-xl border border-slate-200 bg-white px-3 py-2">
                          <div className="flex items-center justify-between gap-2">
                            <div className="text-sm font-semibold text-slate-900">{downstreamAgentName}</div>
                            <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationFitBadgeClass(fit)}`}>
                              {formatDelegationFitLabel(fit, text)}
                            </span>
                          </div>
                          {rationale ? <div className="mt-1 text-xs leading-5 text-slate-600">{rationale}</div> : null}
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-xs text-slate-500">{text.noDownstreamHandoffFit}</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        );
      })}
      {mcpInventory.length > 0 ? (
        <div className="rounded-2xl border border-slate-200 bg-white/90 p-4">
          <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.projectMcpInventoryLabel}</div>
          <div className="mt-3 flex flex-wrap gap-2">
            {mcpInventory.map((server, index) => {
              const serverId =
                typeof server.server_id === 'string' && server.server_id ? server.server_id : `mcp-${index + 1}`;
              const status = typeof server.status === 'string' && server.status ? server.status : 'enabled';
              return (
                <span key={`${serverId}-${index}`} className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${
                  status === 'enabled'
                    ? 'bg-violet-50 text-violet-800 ring-violet-200'
                    : 'bg-slate-100 text-slate-700 ring-slate-200'
                }`}>
                  {formatSkillTitle(serverId, mcpServerTitleById)}
                </span>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function renderSegmentWindow(startChar: number | null, endChar: number | null) {
  if (startChar === null && endChar === null) {
    return null;
  }
  const startLabel = startChar === null ? '?' : `${startChar}`;
  const endLabel = endChar === null ? '?' : `${endChar}`;
  return `${startLabel} -> ${endLabel}`;
}

function getLatestReviewNotification(events?: HarnessEventDTO[] | null) {
  if (!events || events.length === 0) {
    return null;
  }

  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event.event_type !== 'run.notification_ready') {
      continue;
    }
    const details = event.details_json || null;
    return {
      title: typeof details?.title === 'string' ? details.title : null,
      message: typeof details?.message === 'string' ? details.message : null,
      deliveryStatus: typeof details?.delivery_status === 'string' ? details.delivery_status : null,
      reviewer: typeof details?.reviewer === 'string' ? details.reviewer : null,
      verdict: typeof details?.verdict === 'string' ? details.verdict : null,
      readyAt: event.created_at ?? null,
    };
  }

  return null;
}

function ApprovalSummary({
  approval,
  events,
}: {
  approval?: HarnessApprovalDTO | null;
  events?: HarnessEventDTO[] | null;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  if (!approval) {
    return <p className="text-sm text-slate-500">{text.noApprovalRecordYet}</p>;
  }

  const notification = getLatestReviewNotification(events);
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
  const artifactSnapshot = coerceRecord(payload?.artifact_snapshot);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-900">{humanizeHarnessValue(approval.action_type || 'approval', text)}</div>
          {approval.status && approval.status !== 'pending' && approval.resolved_by ? (
            <div className="mt-1 text-xs text-slate-500">{text.autoResolvedByReviewer}</div>
          ) : null}
        </div>
        <StatusPill value={approval.status} />
      </div>
      <dl className="grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
        <div>
          <dt className="text-slate-500">{text.requestedBy}</dt>
          <dd className="mt-1">{approval.requested_by || text.systemActor}</dd>
        </div>
        <div>
          <dt className="text-slate-500">{text.resolvedBy}</dt>
          <dd className="mt-1">{approval.resolved_by || text.notResolved}</dd>
        </div>
        <div>
          <dt className="text-slate-500">{text.createdAt}</dt>
          <dd className="mt-1">{formatTimestamp(approval.created_at, text.notRecorded)}</dd>
        </div>
        <div>
          <dt className="text-slate-500">{text.resolvedAt}</dt>
          <dd className="mt-1">{formatTimestamp(approval.resolved_at, text.notRecorded)}</dd>
        </div>
      </dl>
      {approval.status === 'pending' ? (
        <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950">
          {text.legacyApprovalCheckpoint}
        </div>
      ) : null}
      {approval.reason ? <div className="rounded-2xl bg-amber-50 p-4 text-sm text-amber-900">{approval.reason}</div> : null}
      {notification ? (
        <div className="rounded-2xl border border-sky-200 bg-sky-50 p-4 text-sm text-sky-950">
          <div className="font-semibold text-sky-950">{text.reviewNotification}</div>
          <dl className="mt-3 grid gap-3 sm:grid-cols-2">
            <div>
              <dt className="text-sky-700/80">{text.notificationTitleLabel}</dt>
              <dd className="mt-1 font-medium">{notification.title || text.noDetailCaptured}</dd>
            </div>
            <div>
              <dt className="text-sky-700/80">{text.notificationStatusLabel}</dt>
              <dd className="mt-1 font-medium">
                {notification.deliveryStatus === 'ready' ? text.notificationReadyState : notification.deliveryStatus || text.notRecorded}
              </dd>
            </div>
            <div>
              <dt className="text-sky-700/80">{text.reviewReviewerLabel}</dt>
              <dd className="mt-1">{notification.reviewer || approval.resolved_by || text.systemActor}</dd>
            </div>
            <div>
              <dt className="text-sky-700/80">{text.notificationReadyLabel}</dt>
              <dd className="mt-1">{formatTimestamp(notification.readyAt, text.notRecorded)}</dd>
            </div>
          </dl>
          <div className="mt-3 rounded-2xl bg-white/80 p-3 text-sm text-sky-950">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-sky-600">{text.notificationMessageLabel}</div>
            <div className="mt-2 whitespace-pre-wrap break-words">{notification.message || text.noReviewNotificationYet}</div>
          </div>
        </div>
      ) : null}
      {approval.action_type === 'orchestration_review' ? (
        <div className="rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-950">
          <div className="font-semibold">{text.blockedOrchestrationOutput}</div>
          <div className="mt-2 space-y-1">
            <div>{text.nodeLabel}: {blockedAgentName || blockedAgentId || text.unknownNode}</div>
            {reviewStage === 'cluster_research' ? <div>{text.stageLabel}: {text.clusterResearchEvidence}</div> : null}
            {reviewStage === 'agent_output_segment' ? <div>{text.stageLabel}: {text.pipelineOutputSegment}</div> : null}
            {reviewStage === 'agent_output_stream' ? <div>{text.stageLabel}: {text.liveStreamingOutputGuard}</div> : null}
            {loopNumber ? <div>{text.loopLabel}: {loopNumber}</div> : null}
            {segmentIndex !== null ? (
              <div>
                {text.segmentLabel}: {segmentIndex + 1}
                {segmentCount !== null ? ` / ${segmentCount}` : ''}
              </div>
            ) : null}
            {renderSegmentWindow(segmentStartChar, segmentEndChar) ? (
              <div>{text.characterWindow}: {renderSegmentWindow(segmentStartChar, segmentEndChar)}</div>
            ) : null}
            {typeof rollbackStepIndex === 'number' && Number.isFinite(rollbackStepIndex) ? (
              <div>{text.rollbackTarget}: {formatTemplate(text.previousSafeStateBeforeStep, { count: rollbackStepIndex + 1 })}</div>
            ) : null}
          </div>
          {reviewOutput ? <div className="mt-3 rounded-2xl bg-white/80 p-3 text-sm text-rose-900">{reviewOutput}</div> : null}
          {reviewStage === 'agent_output_stream' ? (
            <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-950">
              {text.streamApprovalReplayHint}
            </div>
          ) : null}
          {segmentPreview ? (
            <div className="mt-3 rounded-2xl border border-rose-200 bg-white/80 p-3 text-sm text-rose-950">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-rose-500">{text.blockedSegmentPreview}</div>
              <pre className="mt-2 whitespace-pre-wrap break-words font-mono text-xs">{segmentPreview}</pre>
            </div>
          ) : null}
          {partialOutput ? (
            <div className="mt-3 rounded-2xl border border-slate-200 bg-white/80 p-3 text-slate-950">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.acceptedPartialOutput}</div>
              <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words font-mono text-xs">{partialOutput}</pre>
            </div>
          ) : null}
          {artifactSnapshot ? (
            <div className="mt-3">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-rose-600">{text.approvalArtifactSnapshot}</div>
              <div className="mt-3">
                <StructuredArtifactCard artifactId={blockedAgentId || 'approval-artifact'} artifact={artifactSnapshot} />
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
      {approval.comment ? <div className="rounded-2xl bg-slate-100 p-4 text-sm text-slate-700">{approval.comment}</div> : null}
      {payload ? (
        <details className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <summary className="cursor-pointer text-sm font-medium text-slate-800">{text.approvalPayload}</summary>
          <div className="mt-3">
            <JsonBlock value={payload} />
          </div>
        </details>
      ) : null}
    </div>
  );
}

function EventRow({ event }: { event: HarnessEventDTO }) {
  const text = useMessages(HARNESS_MESSAGES);
  const details = event.details_json || null;
  const blockedSegmentIndex = coerceNumber(details?.blocked_segment_index);
  const segmentsReviewed = coerceNumber(details?.segments_reviewed);
  const segmentCount = coerceNumber(details?.segment_count);
  const checklistPreview = Array.isArray(details?.open_items_preview)
    ? details.open_items_preview.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
    : [];

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <StatusPill value={event.event_type} />
          <div className="text-xs uppercase tracking-[0.2em] text-slate-400">{humanizeHarnessValue(event.event_source || 'harness', text)}</div>
        </div>
        <div className="text-xs text-slate-500">{formatTimestamp(event.created_at, text.notRecorded)}</div>
      </div>
      <div className="mt-3 text-sm text-slate-700">
        {text.actorLabel}: <span className="font-medium text-slate-900">{event.actor || text.systemActor}</span>
      </div>
      {event.event_type === 'orchestration.review_segment_scan_completed' ? (
        <div className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-950">
          <div className="font-semibold">{text.pipelineReviewScan}</div>
          <div className="mt-2 space-y-1 text-sm">
            <div>{text.nodeLabel}: {typeof details?.agent_name === 'string' ? details.agent_name : typeof details?.agent_id === 'string' ? details.agent_id : text.unknownNode}</div>
            {segmentsReviewed !== null || segmentCount !== null ? (
              <div>
                {text.segmentsReviewed}: {segmentsReviewed ?? '?'}
                {segmentCount !== null ? ` / ${segmentCount}` : ''}
              </div>
            ) : null}
            {blockedSegmentIndex !== null ? <div>{text.blockedAtSegment}: {blockedSegmentIndex + 1}</div> : <div>{text.noRiskySegmentDetected}</div>}
          </div>
        </div>
      ) : null}
      {event.event_type === 'orchestration.review_stream_blocked' ? (
        <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-950">
          <div className="font-semibold">{text.liveStreamGuardBlockedOutput}</div>
          <div className="mt-2 space-y-1 text-sm">
            <div>{text.nodeLabel}: {typeof details?.agent_name === 'string' ? details.agent_name : typeof details?.agent_id === 'string' ? details.agent_id : text.unknownNode}</div>
            <div>
              {text.characterWindow}: {renderSegmentWindow(coerceNumber(details?.segment_start_char), coerceNumber(details?.segment_end_char)) || text.unknownLabel}
            </div>
            <div>{text.partialOutputLength}: {coerceNumber(details?.partial_length) ?? 0}</div>
          </div>
        </div>
      ) : null}
      {event.event_type === 'orchestration.checklist_loaded' ? (
        <div className="mt-3 rounded-2xl border border-cyan-200 bg-cyan-50 p-3 text-sm text-cyan-950">
          <div className="font-semibold">{text.checklistLoaded}</div>
          <div className="mt-2 space-y-1 text-sm">
            <div>{formatTemplate(text.checklistCount, { count: coerceNumber(details?.checklist_count) ?? 0 })}</div>
            <div>{formatTemplate(text.openChecklistCount, { count: coerceNumber(details?.open_item_count) ?? 0 })}</div>
          </div>
          {checklistPreview.length > 0 ? (
            <div className="mt-3 rounded-2xl border border-cyan-200 bg-white/80 p-3 text-sm text-cyan-950">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-600">{text.activeChecklistItems}</div>
              <div className="mt-2 space-y-1">
                {checklistPreview.map((item, index) => (
                  <div key={`${index}-${item}`} className="break-words">{item}</div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
      {event.event_type === 'orchestration.stream_continuation_resumed' ? (
        <div className="mt-3 rounded-2xl border border-sky-200 bg-sky-50 p-3 text-sm text-sky-950">
          <div className="font-semibold">{text.streamContinuationResumed}</div>
          <div className="mt-2 space-y-1 text-sm">
            <div>{text.nodeLabel}: {typeof details?.agent_name === 'string' ? details.agent_name : typeof details?.agent_id === 'string' ? details.agent_id : text.unknownNode}</div>
            <div>{text.partialPrefixLength}: {coerceNumber(details?.partial_length) ?? 0}</div>
            <div>{text.resumedAtStepIndex}: {coerceNumber(details?.next_step_index) ?? 0}</div>
          </div>
        </div>
      ) : null}
      {event.event_type === 'orchestration.stream_continuation_completed' ? (
        <div className="mt-3 rounded-2xl border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-950">
          <div className="font-semibold">{text.streamContinuationCompleted}</div>
          <div className="mt-2">
            {text.nodeLabel}: {typeof details?.agent_name === 'string' ? details.agent_name : typeof details?.agent_id === 'string' ? details.agent_id : text.unknownNode}
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
  const text = useMessages(HARNESS_MESSAGES);
  const continuationLabel = getRunContinuationLabel(run, text);
  const workflowSummary = getWorkflowProgressSummary(run.workflow_progress, text);
  const checklist = resolveRunChecklistSnapshot(run);
  const focusItem = getChecklistFocusItem(checklist);
  const totalChecklistItems = checklist?.total_items ?? checklist?.items?.length ?? 0;
  const completedChecklistItems =
    checklist?.completed_items ?? checklist?.items?.filter((item) => item.status === 'completed')?.length ?? 0;
  const openChecklistItems = checklist?.open_items ?? Math.max(totalChecklistItems - completedChecklistItems, 0);
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full rounded-[14px] border p-4 text-left transition ${
        selected
          ? 'border-cyan-300 bg-cyan-50 shadow-[0_18px_60px_-40px_rgba(8,145,178,0.65)]'
          : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-900">{humanizeHarnessValue(run.task_type || 'unknown_task', text)}</div>
          <div className="mt-1 text-xs text-slate-500">{run.run_id}</div>
        </div>
        <StatusPill value={run.status} />
      </div>
      <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-slate-500">{text.stepLabel}</dt>
          <dd className="mt-1 text-slate-900">{humanizeHarnessValue(run.current_step || 'idle', text)}</dd>
        </div>
        <div>
          <dt className="text-slate-500">{text.verificationLabel}</dt>
          <dd className="mt-1 text-slate-900">{humanizeHarnessValue(run.latest_verification?.status || run.verification_status || 'pending', text)}</dd>
        </div>
        <div>
          <dt className="text-slate-500">{text.approvalLabel}</dt>
          <dd className="mt-1 text-slate-900">{humanizeHarnessValue(run.latest_approval?.status || 'none', text)}</dd>
        </div>
        <div>
          <dt className="text-slate-500">{text.updatedLabel}</dt>
          <dd className="mt-1 text-slate-900">{formatTimestamp(run.updated_at, text.notRecorded)}</dd>
        </div>
      </dl>
      {continuationLabel ? (
        <div className="mt-3 inline-flex rounded-full bg-sky-50 px-3 py-1 text-xs font-semibold text-sky-900 ring-1 ring-inset ring-sky-200">
          {continuationLabel}
        </div>
      ) : null}
      {workflowSummary ? (
        <div className="mt-3 inline-flex rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-inset ring-slate-200">
          {workflowSummary}
        </div>
      ) : null}
      {checklist?.enabled ? (
        <div className="mt-3 rounded-2xl border border-cyan-200 bg-cyan-50/70 px-3 py-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">
              {text.checklistSnapshot}
            </div>
            <div className="inline-flex items-center gap-2">
              <span className="inline-flex rounded-full bg-white/80 px-2.5 py-1 text-[11px] font-semibold text-cyan-900 ring-1 ring-cyan-200">
                {formatTemplate(text.checklistProgressCount, {
                  completed: completedChecklistItems,
                  total: totalChecklistItems,
                })}
              </span>
              {openChecklistItems > 0 ? (
                <span className="inline-flex rounded-full bg-amber-50 px-2.5 py-1 text-[11px] font-semibold text-amber-900 ring-1 ring-amber-200">
                  {formatTemplate(text.openChecklistCount, { count: openChecklistItems })}
                </span>
              ) : null}
            </div>
          </div>
          {focusItem ? (
            <div className="mt-2 text-xs text-cyan-950">
              <span className="font-semibold">
                {focusItem.status === 'in_progress' ? text.checklistWorkingOn : text.checklistNextUp}
              </span>{' '}
              <span className="break-words">{focusItem.content}</span>
            </div>
          ) : (
            <div className="mt-2 text-xs text-cyan-900">{text.allChecklistItemsCompleted}</div>
          )}
        </div>
      ) : null}
    </button>
  );
}

export default function HarnessPage() {
  const text = useMessages(HARNESS_MESSAGES);
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [localDraftProject, setLocalDraftProject] = useState<HarnessProjectDetailDTO | null>(null);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [runFilter, setRunFilter] = useState<RunFilter>('all');
  const [runSectionJumpNonce, setRunSectionJumpNonce] = useState(0);
  const [connectionSourceId, setConnectionSourceId] = useState<string | null>(null);
  const [selectedAgentIdsForRun, setSelectedAgentIdsForRun] = useState<string[]>([]);
  const [loopCount, setLoopCount] = useState(1);
  const [taskText, setTaskText] = useState('');
  const [timeoutValue, setTimeoutValue] = useState<string>('');
  const [approvalCommentsByKey, setApprovalCommentsByKey] = useState<Record<string, string>>({});
  const [editorNotice, setEditorNotice] = useState<string | null>(null);
  const [activeCanvasPanel, setActiveCanvasPanel] = useState<CanvasPanel | null>(null);
  const [pendingCreationKind, setPendingCreationKind] = useState<CanvasCreationKind | null>(null);
  const [showCanvasHint, setShowCanvasHint] = useState(true);
  const [isNodeEditorOpen, setIsNodeEditorOpen] = useState(false);
  const [isCanvasPanning, setIsCanvasPanning] = useState(false);
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const pendingRunSectionJumpRef = useRef<RunQueueGroup | null>(null);
  const runSectionRefs = useRef<Partial<Record<RunQueueGroup, HTMLDivElement | null>>>({});
  const projectProvidersSectionRef = useRef<HTMLElement | null>(null);
  const projectMcpSectionRef = useRef<HTMLElement | null>(null);
  const dragRef = useRef<{ agentId: string; offsetX: number; offsetY: number } | null>(null);
  const panRef = useRef<{ startX: number; startY: number; originX: number; originY: number } | null>(null);
  const initializedViewportProjectIdRef = useRef<string | null>(null);

  const projectsQuery = useHarnessStudioProjectsQuery();
  const currentProjectQuery = useHarnessCurrentStudioProjectQuery();
  const createProjectMutation = useHarnessCreateStudioProjectMutation();
  const updateProjectMutation = useHarnessUpdateStudioProjectMutation();
  const approvalMutation = useHarnessApprovalMutation();
  const studioRunMutation = useHarnessStudioRunMutation();
  const skillRequestMutation = useHarnessSkillRequestMutation();
  const skillDecisionMutation = useHarnessSkillDecisionMutation();
  const runsQuery = useHarnessRunsQuery();
  const policiesQuery = useHarnessPoliciesQuery();
  const retryRunMutation = useHarnessRetryRunMutation();
  const providersQuery = useHarnessModelProvidersQuery();
  const knowledgeBasesQuery = useKnowledgeBasesQuery();

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
  const sortedRuns = useMemo(() => [...runs].sort(compareHarnessRuns), [runs]);
  const projectScopedRuns = useMemo(() => {
    if (!activeProjectId) {
      return sortedRuns;
    }
    return sortedRuns.filter((run) => {
      if (run.task_type !== 'agent_orchestration') {
        return true;
      }
      const runProjectId = typeof run.project_id === 'string' ? run.project_id : '';
      return !runProjectId || runProjectId === activeProjectId;
    });
  }, [activeProjectId, sortedRuns]);
  const filteredRuns = useMemo(
    () => projectScopedRuns.filter((run) => matchesRunFilter(run, runFilter)),
    [projectScopedRuns, runFilter]
  );
  const runFilterCounts = useMemo(
    () => ({
      all: projectScopedRuns.length,
      attention: projectScopedRuns.filter((run) => matchesRunFilter(run, 'attention')).length,
      active: projectScopedRuns.filter((run) => matchesRunFilter(run, 'active')).length,
      checklist: projectScopedRuns.filter((run) => matchesRunFilter(run, 'checklist')).length,
      completed: projectScopedRuns.filter((run) => matchesRunFilter(run, 'completed')).length,
    }),
    [projectScopedRuns]
  );
  const activeRunId = useMemo(() => {
    if (selectedRunId && filteredRuns.some((run) => run.run_id === selectedRunId)) {
      return selectedRunId;
    }
    return filteredRuns[0]?.run_id ?? null;
  }, [filteredRuns, selectedRunId]);
  const detailQuery = useHarnessRunDetailQuery(activeRunId);
  const selectedRun = detailQuery.data;
  const selectedRunChecklist = useMemo(() => resolveRunChecklistSnapshot(selectedRun), [selectedRun]);
  const pendingRunApproval =
    selectedRun?.latest_approval?.status === 'pending' ? selectedRun.latest_approval : null;
  const pendingRunApprovalKey = pendingRunApproval
    ? `${selectedRun?.run_id || ''}:${pendingRunApproval.approval_id || pendingRunApproval.action_type || 'pending'}`
    : '';
  const approvalComment = pendingRunApprovalKey ? approvalCommentsByKey[pendingRunApprovalKey] ?? '' : '';
  const runFilterOptions = useMemo(
    () => [
      { id: 'all' as const, label: text.runFilterAll, count: runFilterCounts.all },
      { id: 'attention' as const, label: text.runFilterAttention, count: runFilterCounts.attention },
      { id: 'active' as const, label: text.runFilterActive, count: runFilterCounts.active },
      { id: 'checklist' as const, label: text.runFilterChecklist, count: runFilterCounts.checklist },
      { id: 'completed' as const, label: text.runFilterCompleted, count: runFilterCounts.completed },
    ],
    [runFilterCounts, text]
  );
  const runQueueSections = useMemo(
    () =>
      [
        {
          id: 'approval' as const,
          label: text.runGroupApproval,
          description: text.runGroupApprovalDescription,
          tone: 'border-amber-200 bg-amber-50/80 text-amber-950',
        },
        {
          id: 'blocked' as const,
          label: text.runGroupBlocked,
          description: text.runGroupBlockedDescription,
          tone: 'border-rose-200 bg-rose-50/80 text-rose-950',
        },
        {
          id: 'checklist' as const,
          label: text.runGroupChecklist,
          description: text.runGroupChecklistDescription,
          tone: 'border-cyan-200 bg-cyan-50/80 text-cyan-950',
        },
        {
          id: 'active' as const,
          label: text.runGroupActive,
          description: text.runGroupActiveDescription,
          tone: 'border-sky-200 bg-sky-50/80 text-sky-950',
        },
        {
          id: 'completed' as const,
          label: text.runGroupCompleted,
          description: text.runGroupCompletedDescription,
          tone: 'border-emerald-200 bg-emerald-50/80 text-emerald-950',
        },
        {
          id: 'other' as const,
          label: text.runGroupOther,
          description: text.runGroupOtherDescription,
          tone: 'border-slate-200 bg-slate-50 text-slate-900',
        },
      ]
        .map((section) => ({
          ...section,
          runs: filteredRuns.filter((run) => getRunQueueGroup(run) === section.id),
        }))
        .filter((section) => section.runs.length > 0),
    [filteredRuns, text]
  );
  const runQueueSummary = useMemo(
    () =>
      [
        {
          id: 'approval' as const,
          label: text.runGroupApproval,
          caption: text.runSummaryApproval,
          count: sortedRuns.filter((run) => getRunQueueGroup(run) === 'approval').length,
          tone: 'border-amber-200 bg-amber-50/90 text-amber-950',
        },
        {
          id: 'blocked' as const,
          label: text.runGroupBlocked,
          caption: text.runSummaryBlocked,
          count: sortedRuns.filter((run) => getRunQueueGroup(run) === 'blocked').length,
          tone: 'border-rose-200 bg-rose-50/90 text-rose-950',
        },
        {
          id: 'checklist' as const,
          label: text.runGroupChecklist,
          caption: text.runSummaryChecklist,
          count: sortedRuns.filter((run) => getRunQueueGroup(run) === 'checklist').length,
          tone: 'border-cyan-200 bg-cyan-50/90 text-cyan-950',
        },
        {
          id: 'active' as const,
          label: text.runGroupActive,
          caption: text.runSummaryActive,
          count: sortedRuns.filter((run) => getRunQueueGroup(run) === 'active').length,
          tone: 'border-sky-200 bg-sky-50/90 text-sky-950',
        },
        {
          id: 'completed' as const,
          label: text.runGroupCompleted,
          caption: text.runSummaryCompleted,
          count: sortedRuns.filter((run) => getRunQueueGroup(run) === 'completed').length,
          tone: 'border-emerald-200 bg-emerald-50/90 text-emerald-950',
        },
      ].filter((item) => item.count > 0),
    [sortedRuns, text]
  );

  const handleSelectRun = useCallback((runId: string) => {
    setSelectedRunId(runId);
    setActiveCanvasPanel('runs');
  }, []);

  const handleJumpToRunSection = useCallback((group: RunQueueGroup) => {
    setActiveCanvasPanel('runs');
    setRunFilter('all');
    pendingRunSectionJumpRef.current = group;
    setRunSectionJumpNonce((value) => value + 1);
    const firstRun = sortedRuns.find((run) => getRunQueueGroup(run) === group);
    if (firstRun) {
      setSelectedRunId(firstRun.run_id);
    }
  }, [sortedRuns]);

  const studioProject = useMemo(() => {
    if (projectQuery.data) {
      return projectQuery.data;
    }
    if (activeProjectId && currentProjectQuery.data?.project_id === activeProjectId) {
      return currentProjectQuery.data;
    }
    return null;
  }, [activeProjectId, currentProjectQuery.data, projectQuery.data]);
  const normalizedStudioProject = useMemo(
    () => (studioProject ? normalizeProject(studioProject, text) : null),
    [studioProject, text]
  );

  useEffect(() => {
    if (!normalizedStudioProject?.project_id || !showCanvasHint) {
      return;
    }
    const timer = window.setTimeout(() => setShowCanvasHint(false), 3200);
    return () => window.clearTimeout(timer);
  }, [normalizedStudioProject?.project_id, showCanvasHint]);

  useEffect(() => {
    if (!editorNotice) {
      return;
    }
    const timer = window.setTimeout(() => setEditorNotice(null), 3600);
    return () => window.clearTimeout(timer);
  }, [editorNotice]);

  useEffect(() => {
    const pendingRunSectionJump = pendingRunSectionJumpRef.current;
    if (!pendingRunSectionJump || activeCanvasPanel !== 'runs') {
      return;
    }
    const targetSection = runQueueSections.find((section) => section.id === pendingRunSectionJump);
    if (!targetSection) {
      return;
    }
    const node = runSectionRefs.current[pendingRunSectionJump];
    if (!node) {
      return;
    }
    node.scrollIntoView({ behavior: 'smooth', block: 'start' });
    pendingRunSectionJumpRef.current = null;
  }, [activeCanvasPanel, runQueueSections, runSectionJumpNonce]);

  const draftProject = useMemo(() => {
    if (!normalizedStudioProject) {
      return null;
    }
    if (localDraftProject?.project_id === normalizedStudioProject.project_id) {
      return localDraftProject;
    }
    return normalizedStudioProject;
  }, [localDraftProject, normalizedStudioProject]);
  const hasUnsavedDraftChanges = useMemo(
    () => {
      if (!localDraftProject || !normalizedStudioProject) {
        return false;
      }
      if (localDraftProject.project_id !== normalizedStudioProject.project_id) {
        return false;
      }
      return projectDraftFingerprint(localDraftProject) !== projectDraftFingerprint(normalizedStudioProject);
    },
    [localDraftProject, normalizedStudioProject]
  );
  const updateDraftProject = useCallback((updater: (current: HarnessProjectDetailDTO) => HarnessProjectDetailDTO) => {
    setLocalDraftProject((current) => {
      const baseline = current?.project_id === draftProject?.project_id ? current : draftProject;
      return baseline ? updater(baseline) : baseline;
    });
  }, [draftProject]);
  const graph = draftProject?.graph_json;
  const agents = useMemo(() => graph?.agents ?? [], [graph?.agents]);
  const edges = useMemo(() => graph?.edges ?? [], [graph?.edges]);
  const studioGraphDiagnostics = useMemo(() => coerceRecord(graph?.graph_diagnostics), [graph?.graph_diagnostics]);
  const studioWeakDownstreamEdges = useMemo(
    () => coerceRecordList(studioGraphDiagnostics?.weak_downstream_edges),
    [studioGraphDiagnostics]
  );
  const studioBestNextHandoffs = useMemo(
    () => coerceRecordList(studioGraphDiagnostics?.best_next_handoffs),
    [studioGraphDiagnostics]
  );
  const studioWeakEdgeCount = useMemo(
    () => coerceNumber(studioGraphDiagnostics?.weak_edge_count) ?? studioWeakDownstreamEdges.length,
    [studioGraphDiagnostics, studioWeakDownstreamEdges.length]
  );
  const studioBestNextCount = useMemo(
    () => coerceNumber(studioGraphDiagnostics?.best_next_count) ?? studioBestNextHandoffs.length,
    [studioBestNextHandoffs.length, studioGraphDiagnostics]
  );
  const studioCollaborationSummary = useMemo(
    () => buildCollaborationScopeSummary(studioGraphDiagnostics),
    [studioGraphDiagnostics]
  );
  const studioCoordinationTopologySummary = useMemo(
    () =>
      buildCoordinationTopologySummary({
        agents,
        edges,
        capabilitySummaries: graph?.agent_capability_summaries ?? [],
        selectedAgentIds: null,
      }),
    [agents, edges, graph?.agent_capability_summaries]
  );
  const studioCapabilityCoverageSummary = useMemo(
    () =>
      buildCapabilityCoverageSummary({
        agents,
        capabilitySummaries: graph?.agent_capability_summaries ?? [],
        selectedAgentIds: null,
      }),
    [agents, graph?.agent_capability_summaries]
  );
  const mcpServerCatalog = useMemo(() => graph?.mcp_server_catalog ?? [], [graph?.mcp_server_catalog]);
  const referencedKnowledgeBaseIds = useMemo(() => graph?.knowledge_base_ids ?? [], [graph?.knowledge_base_ids]);
  const executionChecklist = useMemo(() => graph?.execution_checklist ?? [], [graph?.execution_checklist]);
  const openExecutionChecklistCount = useMemo(
    () =>
      executionChecklist.filter(
        (item) => (item.status ?? 'pending') !== 'completed' && Boolean(item.content?.trim())
      ).length,
    [executionChecklist]
  );
  const referencedKnowledgeBases = useMemo(() => {
    const lookup = new Map((knowledgeBasesQuery.data ?? []).map((item) => [item.knowledge_base_id, item]));
    return referencedKnowledgeBaseIds
      .map((knowledgeBaseId) => lookup.get(knowledgeBaseId))
      .filter((item): item is NonNullable<typeof item> => Boolean(item));
  }, [knowledgeBasesQuery.data, referencedKnowledgeBaseIds]);
  const canvasViewport = useMemo(() => normalizeCanvasViewport(graph?.canvas), [graph?.canvas]);
  const canvasZoom = canvasViewport.zoom;
  const effectiveSelectedAgentId = useMemo(() => {
    if (selectedAgentId && agents.some((agent) => agent.agent_id === selectedAgentId)) {
      return selectedAgentId;
    }
    return null;
  }, [agents, selectedAgentId]);
  const effectiveSelectedAgentIdsForRun = useMemo(
    () => selectedAgentIdsForRun.filter((agentId) => agents.some((agent) => agent.agent_id === agentId)),
    [agents, selectedAgentIdsForRun]
  );
  const selectedScopeGraphDiagnostics = useMemo(
    () => filterDiagnosticsByAgentScope(studioGraphDiagnostics, effectiveSelectedAgentIdsForRun),
    [effectiveSelectedAgentIdsForRun, studioGraphDiagnostics]
  );
  const selectedScopeWeakEdgeCount = useMemo(
    () => coerceNumber(selectedScopeGraphDiagnostics?.weak_edge_count) ?? 0,
    [selectedScopeGraphDiagnostics]
  );
  const selectedScopeBestNextCount = useMemo(
    () => coerceNumber(selectedScopeGraphDiagnostics?.best_next_count) ?? 0,
    [selectedScopeGraphDiagnostics]
  );
  const selectedScopeCollaborationSummary = useMemo(
    () => buildCollaborationScopeSummary(selectedScopeGraphDiagnostics),
    [selectedScopeGraphDiagnostics]
  );
  const selectedScopeCoordinationTopologySummary = useMemo(
    () =>
      buildCoordinationTopologySummary({
        agents,
        edges,
        capabilitySummaries: graph?.agent_capability_summaries ?? [],
        selectedAgentIds: effectiveSelectedAgentIdsForRun,
      }),
    [agents, edges, effectiveSelectedAgentIdsForRun, graph?.agent_capability_summaries]
  );
  const selectedScopeCapabilityCoverageSummary = useMemo(
    () =>
      buildCapabilityCoverageSummary({
        agents,
        capabilitySummaries: graph?.agent_capability_summaries ?? [],
        selectedAgentIds: effectiveSelectedAgentIdsForRun,
      }),
    [agents, effectiveSelectedAgentIdsForRun, graph?.agent_capability_summaries]
  );
  const selectedAgent = useMemo(
    () => agents.find((agent) => agent.agent_id === effectiveSelectedAgentId) ?? null,
    [agents, effectiveSelectedAgentId]
  );
  const applySkillProjectUpdate = useCallback((
    payload: HarnessProjectDetailDTO,
    notice: string
  ) => {
    const normalizedPayload = normalizeProject(payload, text);
    setLocalDraftProject((current) => {
      if (!current || current.project_id !== normalizedPayload.project_id) {
        return normalizedPayload;
      }
      return {
        ...current,
        loaded_skill_count: normalizedPayload.loaded_skill_count,
        pending_skill_request_count: normalizedPayload.pending_skill_request_count,
        graph_json: {
          ...current.graph_json,
          tool_catalog: normalizedPayload.graph_json.tool_catalog ?? [],
          mcp_server_catalog: normalizedPayload.graph_json.mcp_server_catalog ?? [],
          agent_capability_summaries: normalizedPayload.graph_json.agent_capability_summaries ?? [],
          graph_diagnostics: normalizedPayload.graph_json.graph_diagnostics ?? current.graph_json.graph_diagnostics,
          skill_catalog: normalizedPayload.graph_json.skill_catalog ?? [],
          skill_pool: mergeSkillPoolItems(
            current.graph_json.skill_pool ?? [],
            normalizedPayload.graph_json.skill_pool ?? []
          ),
          pending_skill_requests: mergeSkillRequests(
            current.graph_json.pending_skill_requests ?? [],
            normalizedPayload.graph_json.pending_skill_requests ?? []
          ),
        },
      };
    });
    setEditorNotice(notice);
  }, [text]);

  const handleAddChecklistItem = useCallback(() => {
    updateDraftProject((current) => ({
      ...current,
      graph_json: {
        ...current.graph_json,
        execution_checklist: [...(current.graph_json.execution_checklist ?? []), createExecutionChecklistItem()],
      },
    }));
  }, [updateDraftProject]);

  const updateChecklistItem = useCallback((
    itemId: string,
    changes: Partial<HarnessExecutionChecklistItemDTO>
  ) => {
    updateDraftProject((current) => ({
      ...current,
      graph_json: {
        ...current.graph_json,
        execution_checklist: (current.graph_json.execution_checklist ?? []).map((item) =>
          item.item_id === itemId
            ? {
                ...item,
                ...changes,
              }
            : item
        ),
      },
    }));
  }, [updateDraftProject]);

  const removeChecklistItem = useCallback((itemId: string) => {
    updateDraftProject((current) => ({
      ...current,
      graph_json: {
        ...current.graph_json,
        execution_checklist: (current.graph_json.execution_checklist ?? []).filter((item) => item.item_id !== itemId),
      },
    }));
  }, [updateDraftProject]);

  const clientToWorldPoint = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return { x: 0, y: 0 };
    }
    const rect = canvas.getBoundingClientRect();
    return {
      x: (clientX - rect.left - canvasViewport.x) / canvasZoom,
      y: (clientY - rect.top - canvasViewport.y) / canvasZoom,
    };
  }, [canvasViewport.x, canvasViewport.y, canvasZoom]);

  const updateCanvasViewport = useCallback((
    nextViewport:
      | { x: number; y: number; zoom: number }
      | ((current: { x: number; y: number; zoom: number }) => { x: number; y: number; zoom: number })
  ) => {
    updateDraftProject((current) => {
      const previous = normalizeCanvasViewport(current.graph_json.canvas);
      const resolved = typeof nextViewport === 'function' ? nextViewport(previous) : nextViewport;
      return {
        ...current,
        graph_json: {
          ...current.graph_json,
          canvas: {
            x: Math.round(resolved.x),
            y: Math.round(resolved.y),
            zoom: Number(clamp(resolved.zoom, MIN_CANVAS_ZOOM, MAX_CANVAS_ZOOM).toFixed(3)),
          },
        },
      };
    });
  }, [updateDraftProject]);

  const handleFitCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) {
      return;
    }

    if (agents.length === 0) {
      updateCanvasViewport({
        x: Math.round(width / 2),
        y: Math.round(height / 2),
        zoom: 0.8,
      });
      return;
    }

    const minX = Math.min(...agents.map((agent) => agent.position?.x ?? 0)) - CANVAS_PADDING;
    const minY = Math.min(...agents.map((agent) => agent.position?.y ?? 0)) - CANVAS_PADDING;
    const maxX = Math.max(...agents.map((agent) => (agent.position?.x ?? 0) + NODE_WIDTH)) + CANVAS_PADDING;
    const maxY = Math.max(...agents.map((agent) => (agent.position?.y ?? 0) + NODE_HEIGHT)) + CANVAS_PADDING;
    const boundsWidth = Math.max(maxX - minX, 1);
    const boundsHeight = Math.max(maxY - minY, 1);
    const nextZoom = clamp(
      Math.min((width - CANVAS_PADDING * 2) / boundsWidth, (height - CANVAS_PADDING * 2) / boundsHeight),
      MIN_CANVAS_ZOOM,
      MAX_CANVAS_ZOOM
    );

    updateCanvasViewport({
      x: (width - boundsWidth * nextZoom) / 2 - minX * nextZoom,
      y: (height - boundsHeight * nextZoom) / 2 - minY * nextZoom,
      zoom: nextZoom,
    });
  }, [agents, updateCanvasViewport]);

  const handleCanvasZoom = useCallback((nextZoom: number, anchorX?: number, anchorY?: number) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const cursorX = anchorX ?? width / 2;
    const cursorY = anchorY ?? height / 2;
    const worldX = (cursorX - canvasViewport.x) / canvasZoom;
    const worldY = (cursorY - canvasViewport.y) / canvasZoom;
    const clampedZoom = clamp(nextZoom, MIN_CANVAS_ZOOM, MAX_CANVAS_ZOOM);

    updateCanvasViewport({
      x: cursorX - worldX * clampedZoom,
      y: cursorY - worldY * clampedZoom,
      zoom: clampedZoom,
    });
  }, [canvasViewport.x, canvasViewport.y, canvasZoom, updateCanvasViewport]);

  useEffect(() => {
    const handleMove = (event: PointerEvent) => {
      const drag = dragRef.current;
      if (drag) {
        const point = clientToWorldPoint(event.clientX, event.clientY);
        updateDraftProject((current) => ({
          ...current,
          graph_json: {
            ...current.graph_json,
            agents: (current.graph_json.agents ?? []).map((agent) =>
              agent.agent_id === drag.agentId
                ? {
                    ...agent,
                    position: {
                      x: clamp(
                        point.x - drag.offsetX,
                        CANVAS_WORLD_MIN_X + NODE_SPAWN_MARGIN,
                        CANVAS_WORLD_MAX_X - NODE_WIDTH - NODE_SPAWN_MARGIN
                      ),
                      y: clamp(
                        point.y - drag.offsetY,
                        CANVAS_WORLD_MIN_Y + NODE_SPAWN_MARGIN,
                        CANVAS_WORLD_MAX_Y - NODE_HEIGHT - NODE_SPAWN_MARGIN
                      ),
                    },
                  }
                : agent
            ),
          },
        }));
        return;
      }

      const pan = panRef.current;
      if (!pan) {
        return;
      }

      updateCanvasViewport({
        x: pan.originX + (event.clientX - pan.startX),
        y: pan.originY + (event.clientY - pan.startY),
        zoom: canvasZoom,
      });
    };

    const handleUp = () => {
      dragRef.current = null;
      panRef.current = null;
      setIsCanvasPanning(false);
    };

    window.addEventListener('pointermove', handleMove);
    window.addEventListener('pointerup', handleUp);
    return () => {
      window.removeEventListener('pointermove', handleMove);
      window.removeEventListener('pointerup', handleUp);
    };
  }, [canvasZoom, clientToWorldPoint, updateCanvasViewport, updateDraftProject]);

  useEffect(() => {
    if (!draftProject?.project_id || !canvasRef.current) {
      return;
    }
    if (initializedViewportProjectIdRef.current === draftProject.project_id) {
      return;
    }

    initializedViewportProjectIdRef.current = draftProject.project_id;
    const savedViewport = normalizeCanvasViewport(draftProject.graph_json.canvas);
    const hasMeaningfulViewport =
      Math.abs(savedViewport.x) > 4 || Math.abs(savedViewport.y) > 4 || Math.abs(savedViewport.zoom - 1) > 0.01;

    if (!hasMeaningfulViewport) {
      requestAnimationFrame(() => handleFitCanvas());
    }
  }, [agents, draftProject?.graph_json.canvas, draftProject?.project_id, handleFitCanvas]);
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
  const selectedRunVerificationArtifacts = useMemo(
    () => coerceRecord(selectedRun?.latest_verification?.artifacts_json),
    [selectedRun?.latest_verification?.artifacts_json]
  );
  const selectedRunInput = useMemo(
    () => coerceRecord(selectedRun?.input_json),
    [selectedRun?.input_json]
  );
  const selectedRunMetadata = useMemo(
    () => coerceRecord(selectedRun?.metadata_json),
    [selectedRun?.metadata_json]
  );
  const selectedRunScopeAgentIds = useMemo(() => {
    const runScope = typeof selectedRunInput?.run_scope === 'string' ? selectedRunInput.run_scope : '';
    if (runScope !== 'selected') {
      return null;
    }
    const inputSelectedAgentIds = coerceStringList(selectedRunInput?.selected_agent_ids);
    if (inputSelectedAgentIds.length > 0) {
      return inputSelectedAgentIds;
    }
    const metadataSelectedAgentIds = coerceStringList(selectedRunMetadata?.selected_agent_ids);
    return metadataSelectedAgentIds.length > 0 ? metadataSelectedAgentIds : [];
  }, [selectedRunInput, selectedRunMetadata]);
  const selectedRunBlockedAgents = useMemo(
    () => coerceRecordList(selectedRunVerificationArtifacts?.blocked_agents),
    [selectedRunVerificationArtifacts]
  );
  const selectedRunCapabilitySnapshot = useMemo(
    () => coerceRecord(selectedRunVerificationArtifacts?.capability_snapshot),
    [selectedRunVerificationArtifacts]
  );
  const selectedRunCapabilityAvailability = useMemo(
    () => coerceRecord(selectedRunVerificationArtifacts?.capability_availability),
    [selectedRunVerificationArtifacts]
  );
  const selectedRunCapabilityReadiness = useMemo(
    () => coerceRecord(selectedRunVerificationArtifacts?.capability_readiness),
    [selectedRunVerificationArtifacts]
  );
  const selectedRunExecutionContracts = useMemo(
    () => coerceRecord(selectedRunVerificationArtifacts?.capability_execution_contracts),
    [selectedRunVerificationArtifacts]
  );
  const selectedRunCollaborationContracts = useMemo(
    () => coerceRecord(selectedRunVerificationArtifacts?.collaboration_contracts),
    [selectedRunVerificationArtifacts]
  );
  const selectedRunHandoffDiagnostics = useMemo(
    () => coerceRecord(selectedRunVerificationArtifacts?.handoff_diagnostics),
    [selectedRunVerificationArtifacts]
  );
  const selectedRunHandoffArtifacts = useMemo(() => {
    const outputArtifacts = coerceRecord(selectedRunVerificationArtifacts?.output_artifacts);
    if (!outputArtifacts) {
      return [] as Array<{ artifactId: string; artifact: Record<string, unknown> }>;
    }
    return Object.entries(outputArtifacts)
      .map(([artifactId, artifact]) => {
        const normalizedArtifact = coerceRecord(artifact);
        if (!normalizedArtifact) {
          return null;
        }
        return { artifactId, artifact: normalizedArtifact };
      })
      .filter((entry): entry is { artifactId: string; artifact: Record<string, unknown> } => entry !== null);
  }, [selectedRunVerificationArtifacts]);
  const orchestrationPolicy = useMemo(
    () => policiesQuery.data?.policies?.find((policy) => policy.task_type === 'agent_orchestration') ?? null,
    [policiesQuery.data?.policies]
  );
  const providerOptions = useMemo(() => providersQuery.data?.providers ?? [], [providersQuery.data?.providers]);
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
        return node ? getNodeExecutionLabels(node, text) : [agentId];
      });
      const withReview =
        graph?.review_agent?.enabled === false
          ? orderedNames
          : [...orderedNames, graph?.review_agent?.name || text.defaultReviewAgentName];
      const repeated = Array.from({ length: Math.max(loopCount, 1) }).flatMap((_, index) =>
        withReview.map((name) =>
          `${name}${loopCount > 1 ? ` (${formatTemplate(text.loopExecutionLabel, { count: index + 1 })})` : ''}`
        )
      );
      return repeated;
    };

    return {
      all: buildPreview([]),
      selected:
        effectiveSelectedAgentIdsForRun.length > 0 ? buildPreview(effectiveSelectedAgentIdsForRun) : [],
    };
  }, [agents, edges, effectiveSelectedAgentIdsForRun, graph?.review_agent?.enabled, graph?.review_agent?.name, loopCount, text]);
  const activeProjectLabel = draftProject?.name || text.loadingStudio;
  const reviewAgentEnabled = graph?.review_agent?.enabled ?? true;
  const runReady = !!draftProject?.project_id && agents.length > 0;
  const topCanvasMessage = editorNotice || (connectionSourceId ? text.linkModeHint : null);
  const floatingPanelWidthClass =
    activeCanvasPanel === 'runs' || activeCanvasPanel === 'control'
      ? 'w-[min(1100px,calc(100%-2rem))]'
      : 'w-[min(560px,calc(100%-2rem))]';
  const agentNameById = useMemo(() => new Map(agents.map((agent) => [agent.agent_id, agent.name])), [agents]);
  const agentById = useMemo(() => new Map(agents.map((agent) => [agent.agent_id, agent])), [agents]);
  const focusableAgentIds = useMemo(() => new Set(agents.map((agent) => agent.agent_id)), [agents]);
  const selectedRunPolicyRepairSummary = useMemo(
    () =>
      buildPolicyRepairScopeSummary(
        graph,
        graph?.agent_capability_summaries ?? [],
        agentNameById,
        selectedRunScopeAgentIds
      ),
    [agentNameById, graph, selectedRunScopeAgentIds]
  );
  const selectedRunRoleProfileSummary = useMemo(
    () =>
      buildRoleProfileScopeSummary(
        graph,
        graph?.agent_capability_summaries ?? [],
        agentNameById,
        selectedRunScopeAgentIds
      ),
    [agentNameById, graph, selectedRunScopeAgentIds]
  );
  const graphAvailabilitySummary = useMemo(
    () => buildAvailabilityScopeSummary(graph?.agent_capability_summaries ?? [], agentNameById, null),
    [agentNameById, graph?.agent_capability_summaries]
  );
  const selectedScopeAvailabilitySummary = useMemo(
    () =>
      buildAvailabilityScopeSummary(
        graph?.agent_capability_summaries ?? [],
        agentNameById,
        effectiveSelectedAgentIdsForRun
      ),
    [agentNameById, effectiveSelectedAgentIdsForRun, graph?.agent_capability_summaries]
  );
  const graphPolicyRepairSummary = useMemo(
    () => buildPolicyRepairScopeSummary(graph, graph?.agent_capability_summaries ?? [], agentNameById, null),
    [agentNameById, graph]
  );
  const graphRoleProfileSummary = useMemo(
    () => buildRoleProfileScopeSummary(graph, graph?.agent_capability_summaries ?? [], agentNameById, null),
    [agentNameById, graph]
  );
  const selectedScopePolicyRepairSummary = useMemo(
    () =>
      buildPolicyRepairScopeSummary(
        graph,
        graph?.agent_capability_summaries ?? [],
        agentNameById,
        effectiveSelectedAgentIdsForRun
      ),
    [agentNameById, effectiveSelectedAgentIdsForRun, graph]
  );
  const selectedScopeRoleProfileSummary = useMemo(
    () =>
      buildRoleProfileScopeSummary(
        graph,
        graph?.agent_capability_summaries ?? [],
        agentNameById,
        effectiveSelectedAgentIdsForRun
      ),
    [agentNameById, effectiveSelectedAgentIdsForRun, graph]
  );
  const persistedStudioOrchestrationBriefSummary = useMemo(
    () => coerceOrchestrationBriefSummary(graph?.orchestration_summary),
    [graph?.orchestration_summary]
  );
  const studioOrchestrationBriefSummary = useMemo(
    () =>
      persistedStudioOrchestrationBriefSummary ??
      buildOrchestrationBriefSummary({
        agents,
        edges,
        capabilitySummaries: graph?.agent_capability_summaries ?? [],
        selectedAgentIds: null,
        executionStepCount: executionPreview.all.length,
        reviewEnabled: reviewAgentEnabled,
        availabilitySummary: graphAvailabilitySummary,
        policyRepairSummary: graphPolicyRepairSummary,
        roleProfileSummary: graphRoleProfileSummary,
        collaborationSummary: studioCollaborationSummary,
        topologySummary: studioCoordinationTopologySummary,
        capabilityCoverageSummary: studioCapabilityCoverageSummary,
      }),
    [
      agents,
      edges,
      persistedStudioOrchestrationBriefSummary,
      graph?.agent_capability_summaries,
      executionPreview.all.length,
      graphAvailabilitySummary,
      graphPolicyRepairSummary,
      graphRoleProfileSummary,
      reviewAgentEnabled,
      studioCapabilityCoverageSummary,
      studioCollaborationSummary,
      studioCoordinationTopologySummary,
    ]
  );
  const selectedScopeOrchestrationBriefSummary = useMemo(
    () =>
      buildOrchestrationBriefSummary({
        agents,
        edges,
        capabilitySummaries: graph?.agent_capability_summaries ?? [],
        selectedAgentIds: effectiveSelectedAgentIdsForRun,
        executionStepCount: executionPreview.selected.length,
        reviewEnabled: reviewAgentEnabled,
        availabilitySummary: selectedScopeAvailabilitySummary,
        policyRepairSummary: selectedScopePolicyRepairSummary,
        roleProfileSummary: selectedScopeRoleProfileSummary,
        collaborationSummary: selectedScopeCollaborationSummary,
        topologySummary: selectedScopeCoordinationTopologySummary,
        capabilityCoverageSummary: selectedScopeCapabilityCoverageSummary,
      }),
    [
      agents,
      edges,
      effectiveSelectedAgentIdsForRun,
      graph?.agent_capability_summaries,
      executionPreview.selected.length,
      reviewAgentEnabled,
      selectedScopeAvailabilitySummary,
      selectedScopeCapabilityCoverageSummary,
      selectedScopeCollaborationSummary,
      selectedScopeCoordinationTopologySummary,
      selectedScopeRoleProfileSummary,
      selectedScopePolicyRepairSummary,
    ]
  );
  const runAllBlockedByAvailability = graphAvailabilitySummary.unavailableCount > 0;
  const runSelectedBlockedByAvailability =
    effectiveSelectedAgentIdsForRun.length > 0 && selectedScopeAvailabilitySummary.unavailableCount > 0;
  const selectedAssignableAgent = useMemo(
    () => (selectedAgent && selectedAgent.node_kind !== 'cluster' ? selectedAgent : null),
    [selectedAgent]
  );
  const selectedAgentSkillIds = useMemo(
    () => new Set((selectedAssignableAgent?.skill_ids ?? []).map((skillId) => normalizeSkillKey(skillId)).filter(Boolean)),
    [selectedAssignableAgent?.skill_ids]
  );
  const persistedAgentIds = useMemo(
    () => new Set((normalizedStudioProject?.graph_json?.agents ?? []).map((agent) => agent.agent_id)),
    [normalizedStudioProject?.graph_json?.agents]
  );
  const canRequestSourceSkills = Boolean(
    draftProject?.project_id &&
      selectedAssignableAgent &&
      persistedAgentIds.has(selectedAssignableAgent.agent_id)
  );
  const skillUsageById = useMemo(() => {
    const usage = new Map<string, { count: number; agentNames: string[] }>();
    for (const agent of agents) {
      for (const rawSkillId of agent.skill_ids ?? []) {
        const skillId = normalizeSkillKey(rawSkillId);
        if (!skillId) {
          continue;
        }
        const current = usage.get(skillId);
        if (current) {
          current.count += 1;
          current.agentNames.push(agent.name);
          continue;
        }
        usage.set(skillId, { count: 1, agentNames: [agent.name] });
      }
    }
    return usage;
  }, [agents]);
  const skillRequests = useMemo(() => graph?.pending_skill_requests ?? [], [graph?.pending_skill_requests]);
  const pendingSkillRequests = useMemo(
    () =>
      skillRequests
        .filter((request) => (request.status ?? 'pending') === 'pending')
        .sort((left, right) => getSkillRequestTimestamp(right) - getSkillRequestTimestamp(left)),
    [skillRequests]
  );
  const pendingSkillApprovalCount = pendingSkillRequests.length;
  const approvedSkillIds = useMemo(
    () => new Set((graph?.skill_pool ?? []).map((skill) => normalizeSkillKey(skill.skill_id)).filter(Boolean)),
    [graph?.skill_pool]
  );
  const pendingSkillRequestBySkillId = useMemo(() => {
    const requests = new Map<string, HarnessSkillRequestDTO>();
    for (const request of pendingSkillRequests) {
      const key = normalizeSkillKey(request.skill_id);
      if (!key || requests.has(key)) {
        continue;
      }
      requests.set(key, request);
    }
    return requests;
  }, [pendingSkillRequests]);
  const latestSkillRequestByAgentSkillId = useMemo(() => {
    const requests = new Map<string, HarnessSkillRequestDTO>();
    for (const request of skillRequests) {
      const key = buildAgentSkillRequestKey(request.agent_id, request.skill_id);
      if (!key) {
        continue;
      }
      const current = requests.get(key);
      if (!current || getSkillRequestTimestamp(request) >= getSkillRequestTimestamp(current)) {
        requests.set(key, request);
      }
    }
    return requests;
  }, [skillRequests]);
  const mergedSkillPool = useMemo(() => {
    const byId = new Map<
      string,
      {
        skill_id: string;
        title: string;
        description?: string | null;
        source: string;
        status?: string;
        approved: boolean;
        displayStatus: 'approved' | 'pending' | 'rejected' | 'available';
        used: boolean;
        assigned: boolean;
        usageCount: number;
        agentNames: string[];
        pendingRequest: HarnessSkillRequestDTO | null;
        selectedRequest: HarnessSkillRequestDTO | null;
      }
    >();

    const upsertSkill = (skill: {
      skill_id: string;
      title: string;
      description?: string | null;
      source: string;
      status?: string;
    }) => {
      const normalizedSkillId = normalizeSkillKey(skill.skill_id);
      if (!normalizedSkillId) {
        return;
      }
      const existing = byId.get(normalizedSkillId);
      const usage = skillUsageById.get(normalizedSkillId);
      const pendingRequest = pendingSkillRequestBySkillId.get(normalizedSkillId) ?? null;
      const selectedRequest = selectedAssignableAgent
        ? latestSkillRequestByAgentSkillId.get(
            buildAgentSkillRequestKey(selectedAssignableAgent.agent_id, normalizedSkillId)
          ) ?? null
        : null;
      const approved = approvedSkillIds.has(normalizedSkillId);
      const assigned = selectedAgentSkillIds.has(normalizedSkillId);
      const displayStatus = approved
        ? 'approved'
        : pendingRequest
          ? 'pending'
          : selectedRequest?.status === 'rejected'
            ? 'rejected'
            : 'available';
      byId.set(normalizedSkillId, {
        skill_id: skill.skill_id,
        title: skill.title,
        description: skill.description,
        source: skill.source,
        status: skill.status,
        approved,
        displayStatus,
        used: Boolean(usage),
        assigned,
        usageCount: usage?.count ?? existing?.usageCount ?? 0,
        agentNames: usage?.agentNames ?? existing?.agentNames ?? [],
        pendingRequest,
        selectedRequest,
      });
    };

    for (const skill of graph?.skill_catalog ?? []) {
      upsertSkill(skill);
    }

    for (const skill of graph?.skill_pool ?? []) {
      upsertSkill(skill);
    }

    for (const [skillId] of skillUsageById.entries()) {
      if (byId.has(skillId)) {
        continue;
      }
      upsertSkill({
        skill_id: skillId,
        title: skillId,
        description: null,
        source: text.skillPool,
        status: 'loaded',
      });
    }

    return Array.from(byId.values()).sort((left, right) => {
      const statusRank: Record<'approved' | 'pending' | 'rejected' | 'available', number> = {
        approved: 0,
        pending: 1,
        rejected: 2,
        available: 3,
      };
      if (statusRank[left.displayStatus] !== statusRank[right.displayStatus]) {
        return statusRank[left.displayStatus] - statusRank[right.displayStatus];
      }
      if (left.assigned !== right.assigned) {
        return left.assigned ? -1 : 1;
      }
      if (left.used !== right.used) {
        return left.used ? -1 : 1;
      }
      return left.title.localeCompare(right.title);
    });
  }, [
    approvedSkillIds,
    graph?.skill_catalog,
    graph?.skill_pool,
    latestSkillRequestByAgentSkillId,
    pendingSkillRequestBySkillId,
    selectedAgentSkillIds,
    selectedAssignableAgent,
    skillUsageById,
    text.skillPool,
  ]);
  const approvedSkillCount = approvedSkillIds.size;
  const approvedSkillPool = useMemo(
    () => mergedSkillPool.filter((skill) => skill.approved),
    [mergedSkillPool]
  );
  const sourceSkillPool = useMemo(
    () => mergedSkillPool.filter((skill) => !skill.approved),
    [mergedSkillPool]
  );
  const skillTitleById = useMemo(
    () => new Map(mergedSkillPool.map((skill) => [skill.skill_id, skill.title])),
    [mergedSkillPool]
  );
  const toolTitleById = useMemo(
    () => new Map((graph?.tool_catalog ?? []).map((tool) => [tool.tool_id, tool.title])),
    [graph?.tool_catalog]
  );
  const mcpServerTitleById = useMemo(
    () => new Map(mcpServerCatalog.map((server) => [server.server_id, server.title])),
    [mcpServerCatalog]
  );
  const agentCapabilitySummaryById = useMemo(
    () => new Map((graph?.agent_capability_summaries ?? []).map((summary) => [summary.agent_id, summary])),
    [graph?.agent_capability_summaries]
  );
  const edgeDelegationDiagnosticsById = useMemo(() => {
    const diagnostics = new Map<string, EdgeDelegationDiagnostic>();
    for (const edge of edges) {
      const diagnostic = buildEdgeDelegationDiagnostic(edge, agentCapabilitySummaryById, agentNameById);
      if (!diagnostic) {
        continue;
      }
      diagnostics.set(edge.edge_id, diagnostic);
    }
    return diagnostics;
  }, [agentCapabilitySummaryById, agentNameById, edges]);
  const selectedAgentCapabilitySummary = useMemo(
    () => (selectedAgent ? agentCapabilitySummaryById.get(selectedAgent.agent_id) ?? null : null),
    [agentCapabilitySummaryById, selectedAgent]
  );
  const selectedAgentRoleProfileSuggestion = selectedAgentCapabilitySummary?.role_profile_suggestion ?? null;
  const selectedAgentRoleProfileMissingSkillIds = selectedAgentRoleProfileSuggestion
    ? coerceStringList(selectedAgentRoleProfileSuggestion.missing_skill_ids)
    : [];
  const selectedAgentMissingSkillDetails = selectedAgentCapabilitySummary
    ? coerceRecordList(selectedAgentCapabilitySummary.missing_skill_details)
    : [];
  const selectedAgentMissingMcpServerDetails = selectedAgentCapabilitySummary
    ? coerceRecordList(selectedAgentCapabilitySummary.missing_mcp_server_details)
    : [];
  const selectedAgentAvailabilityStatus = resolveCapabilityAvailabilityStatus(selectedAgentCapabilitySummary);
  const selectedAgentAvailabilityBlockers = selectedAgentCapabilitySummary
    ? coerceStringList(selectedAgentCapabilitySummary.availability_blockers)
    : [];
  const selectedAgentAvailabilityWarnings = selectedAgentCapabilitySummary
    ? coerceStringList(selectedAgentCapabilitySummary.availability_warnings)
    : [];
  const selectedAgentReadinessStatus = resolveCapabilityReadinessStatus(selectedAgentCapabilitySummary);
  const selectedAgentReadinessBlockers = selectedAgentCapabilitySummary
    ? coerceStringList(selectedAgentCapabilitySummary.readiness_blockers)
    : [];
  const selectedAgentReadinessWarnings = selectedAgentCapabilitySummary
    ? coerceStringList(selectedAgentCapabilitySummary.readiness_warnings)
    : [];
  const selectedAgentMissingSkillIds = selectedAgentCapabilitySummary
    ? coerceStringList(selectedAgentCapabilitySummary.missing_skill_ids)
    : [];
  const selectedAgentMissingRequiredSkillIds = selectedAgentCapabilitySummary
    ? coerceStringList(selectedAgentCapabilitySummary.missing_required_skill_ids)
    : [];
  const selectedAgentPolicyBlockedToolIds = selectedAgentCapabilitySummary
    ? coerceStringList(selectedAgentCapabilitySummary.policy_blocked_tool_ids)
    : [];
  const selectedAgentPolicyBlockedMcpServerIds = selectedAgentCapabilitySummary
    ? coerceStringList(selectedAgentCapabilitySummary.policy_blocked_mcp_server_ids)
    : [];
  const selectedAgentActionableToolPolicyIds = computeActionableToolPolicySuggestionIds(
    selectedAssignableAgent,
    selectedAgentPolicyBlockedToolIds
  );
  const selectedAgentActionableMcpPolicyIds = computeActionableMcpPolicySuggestionIds(
    selectedAssignableAgent,
    selectedAgentPolicyBlockedMcpServerIds
  );
  const selectedAgentActionableToolRestrictionIds = computeCoordinatorToolPolicyRestrictionIds(
    selectedAssignableAgent,
    selectedAgentCapabilitySummary
  );
  const selectedAgentActionableMcpRestrictionIds = computeCoordinatorMcpPolicyRestrictionIds(
    selectedAssignableAgent,
    selectedAgentCapabilitySummary
  );
  const selectedAgentProviderLimitedToolIds = selectedAgentCapabilitySummary
    ? coerceStringList(selectedAgentCapabilitySummary.provider_limited_tool_ids)
    : [];
  const selectedAgentMissingMcpServerIds = selectedAgentCapabilitySummary
    ? coerceStringList(selectedAgentCapabilitySummary.missing_mcp_server_ids)
    : [];
  const selectedAgentShouldOpenSkillPool =
    selectedAgentMissingSkillIds.length > 0
    || selectedAgentMissingRequiredSkillIds.length > 0
    || selectedAgentMissingSkillDetails.length > 0;
  const selectedAgentCanRequestRoleProfileSkills =
    canRequestSourceSkills && selectedAgentRoleProfileMissingSkillIds.length > 0;
  const selectedAgentShouldOpenProjectProviders = shouldOpenProjectProvidersForDiagnostic(selectedAgentCapabilitySummary);
  const selectedAgentShouldOpenProjectMcp = shouldOpenProjectMcpForDiagnostic(selectedAgentCapabilitySummary);
  const selectedAgentDownstreamTargetIds = useMemo(() => {
    if (!selectedAgent) {
      return new Set<string>();
    }
    return new Set(
      edges
        .filter((edge) => edge.source_agent_id === selectedAgent.agent_id)
        .map((edge) => edge.target_agent_id)
    );
  }, [edges, selectedAgent]);
  const selectedAgentPrimarySuggestedCollaborator = useMemo(() => {
    if (!selectedAgentCapabilitySummary) {
      return null;
    }
    return (
      (selectedAgentCapabilitySummary.recommended_collaborators ?? []).find((item) => {
        if (!item.agent_id || selectedAgentDownstreamTargetIds.has(item.agent_id)) {
          return false;
        }
        return item.fit === 'strong' || item.fit === 'good';
      }) ?? null
    );
  }, [selectedAgentCapabilitySummary, selectedAgentDownstreamTargetIds]);
  const selectedAgentRoleProfilePeerDiagnostics = useMemo(
    () =>
      buildRoleProfilePeerOverlapDiagnostics(
        graph?.agent_capability_summaries ?? [],
        agentNameById,
        selectedAgent?.agent_id ?? null
      ),
    [agentNameById, graph?.agent_capability_summaries, selectedAgent]
  );
  const creationPreview = useMemo(() => {
    if (!pendingCreationKind) {
      return null;
    }
    if (pendingCreationKind === 'agent') {
      return {
        kind: pendingCreationKind,
        label: text.addAgent,
        description: text.newAgentDescription,
        toneClassName: 'border-slate-200 bg-white/96 text-slate-800',
        icon: GitBranchPlus,
      };
    }
    if (pendingCreationKind === 'brainstorm') {
      return {
        kind: pendingCreationKind,
        label: text.brainstormCluster,
        description: text.brainstormClusterDescription,
        toneClassName: 'border-amber-200 bg-amber-50/96 text-amber-950',
        icon: Sparkles,
      };
    }
    return {
      kind: pendingCreationKind,
      label: text.customCluster,
      description: text.customClusterDescription,
      toneClassName: 'border-emerald-200 bg-emerald-50/96 text-emerald-950',
      icon: Layers3,
    };
  }, [pendingCreationKind, text]);
  const openNodeEditor = (agentId: string) => {
    setSelectedAgentId(agentId);
    setIsNodeEditorOpen(true);
  };

  const focusControlPanelSection = useCallback((section: 'providers' | 'mcp') => {
    setPendingCreationKind(null);
    setActiveCanvasPanel('control');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const target = section === 'providers' ? projectProvidersSectionRef.current : projectMcpSectionRef.current;
        target?.scrollIntoView({ block: 'start', behavior: 'smooth' });
      });
    });
  }, []);

  const toggleCanvasPanel = (panel: CanvasPanel) => {
    setPendingCreationKind(null);
    setActiveCanvasPanel((current) => (current === panel ? null : panel));
  };
  const stopCanvasWheelPropagation = (event: ReactWheelEvent<HTMLElement>) => {
    event.stopPropagation();
  };

  const handleProjectCreate = () => {
    const nextIndex = (projectsQuery.data?.projects?.length ?? 0) + 1;
    createProjectMutation.mutate(
      {
        name: formatTemplate(text.agentStudioProjectName, { count: nextIndex }),
        description: text.agentStudioProjectDescription,
      },
      {
        onSuccess: (payload) => {
          setSelectedProjectId(payload.project_id);
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(text.newStudioProjectCreatedNotice);
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
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(text.studioSavedNotice);
        },
      }
    );
  };

  const handleApplySuggestedRewire = useCallback(
    ({
      sourceAgentId,
      fromTargetAgentId,
      toTargetAgentId,
    }: {
      sourceAgentId: string;
      fromTargetAgentId: string;
      toTargetAgentId: string;
    }) => {
      if (!draftProject) {
        return;
      }
      const { changed, graph: nextGraph } = rewireGraphEdge(draftProject.graph_json, {
        sourceAgentId,
        fromTargetAgentId,
        toTargetAgentId,
      });
      if (!changed) {
        setEditorNotice(text.noRewireChangeNotice);
        return;
      }
      updateProjectMutation.mutate(
        {
          projectId: draftProject.project_id,
          name: draftProject.name,
          description: draftProject.description,
          graphJson: nextGraph,
        },
        {
          onSuccess: (payload) => {
            setLocalDraftProject(normalizeProject(payload, text));
            setEditorNotice(text.rewireSavedNotice);
          },
        }
      );
    },
    [draftProject, text, updateProjectMutation]
  );

  const handleApplySuggestedHandoff = useCallback(
    ({
      sourceAgentId,
      targetAgentId,
    }: {
      sourceAgentId: string;
      targetAgentId: string;
    }) => {
      if (!draftProject) {
        return;
      }
      const inserted = insertGraphEdge(draftProject.graph_json, {
        sourceAgentId,
        targetAgentId,
      });
      if (!inserted.changed) {
        setEditorNotice(text.noSuggestedHandoffChangeNotice);
        return;
      }
      updateProjectMutation.mutate(
        {
          projectId: draftProject.project_id,
          name: draftProject.name,
          description: draftProject.description,
          graphJson: inserted.graph,
        },
        {
          onSuccess: (payload) => {
            setLocalDraftProject(normalizeProject(payload, text));
            setEditorNotice(text.suggestedHandoffSavedNotice);
          },
        }
      );
    },
    [draftProject, text, updateProjectMutation]
  );

  const handleApplySuggestedHandoffsForScope = useCallback((scope: 'all' | 'selected') => {
    if (!draftProject) {
      return;
    }
    const diagnostics = scope === 'selected' ? selectedScopeGraphDiagnostics : studioGraphDiagnostics;
    const inserted = applySuggestedHandoffsToGraph(draftProject.graph_json, diagnostics);
    if (inserted.actionableCount <= 0 || inserted.changedCount <= 0) {
      setEditorNotice(text.noActionableSuggestedHandoffsNotice);
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: inserted.graph,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(formatTemplate(text.suggestedHandoffsSavedNotice, { count: inserted.changedCount }));
        },
      }
    );
  }, [draftProject, selectedScopeGraphDiagnostics, studioGraphDiagnostics, text, updateProjectMutation]);

  const handleApplySuggestedCollaborationFixesForScope = useCallback((scope: 'all' | 'selected') => {
    if (!draftProject) {
      return;
    }
    const diagnostics = scope === 'selected' ? selectedScopeGraphDiagnostics : studioGraphDiagnostics;
    const applied = applySuggestedCollaborationChangesToGraph(draftProject.graph_json, diagnostics);
    if (applied.actionableCount <= 0 || applied.changedCount <= 0) {
      setEditorNotice(text.noCollaborationSuggestionsNotice);
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: applied.graph,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(formatTemplate(text.collaborationSuggestionsSavedNotice, { count: applied.changedCount }));
        },
      }
    );
  }, [draftProject, selectedScopeGraphDiagnostics, studioGraphDiagnostics, text, updateProjectMutation]);

  const handleApplyRunRecoveryCollaborationFixes = useCallback(() => {
    if (!draftProject) {
      return;
    }
    const applied = applySuggestedCollaborationChangesToGraph(draftProject.graph_json, selectedRunHandoffDiagnostics);
    if (applied.actionableCount <= 0 || applied.changedCount <= 0) {
      setEditorNotice(text.noCollaborationSuggestionsNotice);
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: applied.graph,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(formatTemplate(text.collaborationSuggestionsSavedNotice, { count: applied.changedCount }));
        },
      }
    );
  }, [draftProject, selectedRunHandoffDiagnostics, text, updateProjectMutation]);

  const handleApplySuggestedRewiresForScope = useCallback((scope: 'all' | 'selected') => {
    if (!draftProject) {
      return;
    }
    const diagnostics = scope === 'selected' ? selectedScopeGraphDiagnostics : studioGraphDiagnostics;
    const rewired = applySuggestedRewiresToGraph(draftProject.graph_json, diagnostics);
    if (rewired.actionableCount <= 0 || rewired.changedCount <= 0) {
      setEditorNotice(text.noActionableRewiresNotice);
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: rewired.graph,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(formatTemplate(text.rewiresSavedNotice, { count: rewired.changedCount }));
        },
      }
    );
  }, [draftProject, selectedScopeGraphDiagnostics, studioGraphDiagnostics, text, updateProjectMutation]);

  const handleApplyPolicyRepairsForScope = useCallback((scope: 'all' | 'selected') => {
    if (!draftProject) {
      return;
    }
    const applied = applyCapabilityPolicySuggestionsToGraph(
      draftProject.graph_json,
      draftProject.graph_json.agent_capability_summaries ?? [],
      scope === 'selected' ? effectiveSelectedAgentIdsForRun : null
    );
    if (applied.actionableAgentCount <= 0 || applied.changedAgentCount <= 0) {
      setEditorNotice(text.noActionablePolicyRepairsNotice);
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: applied.graph,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(
            formatTemplate(text.policyRepairsSavedNotice, {
              agents: applied.changedAgentCount,
              tools: applied.toolChangeCount,
              mcp: applied.mcpChangeCount,
            })
          );
        },
      }
    );
  }, [draftProject, effectiveSelectedAgentIdsForRun, text, updateProjectMutation]);

  const handleApplyRoleProfilesForScope = useCallback((scope: 'all' | 'selected') => {
    if (!draftProject) {
      return;
    }
    const applied = applyRoleProfilesToGraph(
      draftProject.graph_json,
      draftProject.graph_json.agent_capability_summaries ?? [],
      scope === 'selected' ? effectiveSelectedAgentIdsForRun : null
    );
    if (applied.actionableAgentCount <= 0 || applied.changedAgentCount <= 0) {
      setEditorNotice(text.noActionableRoleProfilesNotice);
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: applied.graph,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(
            formatTemplate(text.roleProfilesAppliedNotice, {
              agents: applied.changedAgentCount,
              skills: applied.skillChangeCount,
              tools: applied.toolChangeCount,
              mcp: applied.mcpChangeCount,
            })
          );
        },
      }
    );
  }, [draftProject, effectiveSelectedAgentIdsForRun, text, updateProjectMutation]);

  const handleApplyAgentCapabilityPolicySuggestions = useCallback(({
    agentId,
    skillIds = [],
    toolIds = [],
    mcpServerIds = [],
    denyToolIds = [],
    denyMcpServerIds = [],
    forceAllowToolIds = false,
    forceAllowMcpServerIds = false,
  }: {
    agentId: string;
    skillIds?: string[];
    toolIds?: string[];
    mcpServerIds?: string[];
    denyToolIds?: string[];
    denyMcpServerIds?: string[];
    forceAllowToolIds?: boolean;
    forceAllowMcpServerIds?: boolean;
  }) => {
    if (!draftProject || !agentId) {
      return;
    }
    const expectsSkillChanges = skillIds.length > 0;
    const expectsToolChanges = toolIds.length > 0 || denyToolIds.length > 0;
    const expectsMcpChanges = mcpServerIds.length > 0 || denyMcpServerIds.length > 0;
    const applied = applyAgentCapabilityPolicySuggestions(draftProject.graph_json, {
      agentId,
      skillIds,
      toolIds,
      mcpServerIds,
      denyToolIds,
      denyMcpServerIds,
      forceAllowToolIds,
      forceAllowMcpServerIds,
    });
    if (
      !applied.changed
      || (expectsSkillChanges && applied.skillChangeCount <= 0 && !expectsToolChanges && !expectsMcpChanges)
      || (expectsToolChanges && applied.toolChangeCount <= 0)
      || (expectsMcpChanges && applied.mcpChangeCount <= 0)
    ) {
      if (expectsSkillChanges && !expectsToolChanges && !expectsMcpChanges) {
        setEditorNotice(text.noActionableRoleProfilesNotice);
      } else if (expectsToolChanges && !expectsMcpChanges) {
        setEditorNotice(text.noToolPolicySuggestionsNotice);
      } else if (expectsMcpChanges && !expectsToolChanges) {
        setEditorNotice(text.noMcpPolicySuggestionsNotice);
      } else {
        setEditorNotice(text.noActionablePolicyRepairsNotice);
      }
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: applied.graph,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload, text));
          if (expectsSkillChanges && !expectsToolChanges && !expectsMcpChanges) {
            setEditorNotice(
              formatTemplate(text.roleProfilesAppliedNotice, {
                agents: 1,
                skills: applied.skillChangeCount,
                tools: 0,
                mcp: 0,
              })
            );
          } else if (expectsToolChanges && !expectsMcpChanges && !expectsSkillChanges) {
            setEditorNotice(formatTemplate(text.toolPolicySuggestionsAppliedNotice, { count: applied.toolChangeCount }));
          } else if (expectsMcpChanges && !expectsToolChanges && !expectsSkillChanges) {
            setEditorNotice(formatTemplate(text.mcpPolicySuggestionsAppliedNotice, { count: applied.mcpChangeCount }));
          } else {
            setEditorNotice(
              formatTemplate(expectsSkillChanges ? text.roleProfilesAppliedNotice : text.policyRepairsSavedNotice, {
                agents: 1,
                skills: applied.skillChangeCount,
                tools: applied.toolChangeCount,
                mcp: applied.mcpChangeCount,
              })
            );
          }
        },
      }
    );
  }, [draftProject, text, updateProjectMutation]);

  const handleApplySelectedAgentToolPolicySuggestions = () => {
    if (!selectedAssignableAgent) {
      return;
    }
    handleApplyAgentCapabilityPolicySuggestions({
      agentId: selectedAssignableAgent.agent_id,
      toolIds: selectedAgentActionableToolPolicyIds,
    });
  };

  const handleApplySelectedAgentMcpPolicySuggestions = () => {
    if (!selectedAssignableAgent) {
      return;
    }
    handleApplyAgentCapabilityPolicySuggestions({
      agentId: selectedAssignableAgent.agent_id,
      mcpServerIds: selectedAgentActionableMcpPolicyIds,
    });
  };

  const handleApplySelectedAgentToolPolicyRestrictions = () => {
    if (!selectedAssignableAgent) {
      return;
    }
    handleApplyAgentCapabilityPolicySuggestions({
      agentId: selectedAssignableAgent.agent_id,
      denyToolIds: selectedAgentActionableToolRestrictionIds,
    });
  };

  const handleApplySelectedAgentMcpPolicyRestrictions = () => {
    if (!selectedAssignableAgent) {
      return;
    }
    handleApplyAgentCapabilityPolicySuggestions({
      agentId: selectedAssignableAgent.agent_id,
      denyMcpServerIds: selectedAgentActionableMcpRestrictionIds,
    });
  };

  const handleApplySelectedAgentRoleProfile = () => {
    if (!draftProject || !selectedAssignableAgent || !selectedAgentRoleProfileSuggestion) {
      return;
    }
    const skillIds = coerceStringList(selectedAgentRoleProfileSuggestion.available_skill_ids);
    const toolIds = coerceStringList(selectedAgentRoleProfileSuggestion.suggested_tool_ids);
    const mcpServerIds = coerceStringList(selectedAgentRoleProfileSuggestion.suggested_mcp_server_ids);
    const denyToolIds = coerceStringList(selectedAgentRoleProfileSuggestion.restrictive_tool_ids);
    const denyMcpServerIds = coerceStringList(selectedAgentRoleProfileSuggestion.restrictive_mcp_server_ids);
    const applied = applyAgentCapabilityPolicySuggestions(draftProject.graph_json, {
      agentId: selectedAssignableAgent.agent_id,
      skillIds,
      toolIds,
      mcpServerIds,
      denyToolIds,
      denyMcpServerIds,
      forceAllowToolIds: toolIds.length > 0,
      forceAllowMcpServerIds: mcpServerIds.length > 0,
    });
    if (!applied.changed) {
      setEditorNotice(
        coerceStringList(selectedAgentRoleProfileSuggestion.missing_skill_ids).length > 0
          ? text.roleProfileNeedsSkillPoolNotice
          : text.noActionableRoleProfileNotice
      );
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: applied.graph,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(
            formatTemplate(text.roleProfileAppliedNotice, {
              skills: applied.skillChangeCount,
              tools: applied.toolChangeCount,
              mcp: applied.mcpChangeCount,
            })
          );
        },
      }
    );
  };

  const requestRoleProfileSkillsFromSummary = useCallback(async (
    summary: ReturnType<typeof buildRoleProfileScopeSummary>
  ) => {
    if (!draftProject) {
      return;
    }
    const requestableDiagnostics = summary.diagnostics.filter(
      (diagnostic) =>
        persistedAgentIds.has(diagnostic.agentId)
        && diagnostic.missingSkillIds.length > 0
    );
    if (requestableDiagnostics.length <= 0) {
      setEditorNotice(
        summary.missingSkillAgentCount > 0 ? text.saveStudioBeforeSkillRequests : text.noRoleProfileSkillRequestsNotice
      );
      return;
    }

    let latestPayload: HarnessProjectDetailDTO | null = null;
    let requestedAgentCount = 0;
    let requestedSkillCount = 0;
    let availableSkillCount = 0;

    for (const diagnostic of requestableDiagnostics) {
      const payload = await skillRequestMutation.mutateAsync({
        projectId: draftProject.project_id,
        agentId: diagnostic.agentId,
        requestedSkills: diagnostic.missingSkillIds,
      });
      latestPayload = payload;
      requestedAgentCount += 1;
      requestedSkillCount += payload.skill_request_result?.created_requests?.length ?? 0;
      availableSkillCount += payload.skill_request_result?.available_skill_ids?.length ?? 0;
    }

    if (!latestPayload || (requestedSkillCount <= 0 && availableSkillCount <= 0)) {
      setEditorNotice(text.noRoleProfileSkillRequestsNotice);
      return;
    }

    applySkillProjectUpdate(
      latestPayload,
      formatTemplate(text.roleProfileSkillRequestsSyncedNotice, {
        agents: requestedAgentCount,
        skills: requestedSkillCount,
        available: availableSkillCount,
      })
    );
  }, [
    applySkillProjectUpdate,
    draftProject,
    persistedAgentIds,
    skillRequestMutation,
    text,
  ]);

  const handleRequestRoleProfileSkillsForScope = useCallback((scope: 'all' | 'selected') => {
    const summary = scope === 'selected' ? selectedScopeRoleProfileSummary : graphRoleProfileSummary;
    void requestRoleProfileSkillsFromSummary(summary);
  }, [
    graphRoleProfileSummary,
    requestRoleProfileSkillsFromSummary,
    selectedScopeRoleProfileSummary,
  ]);

  const handleRequestSelectedAgentRoleProfileSkills = () => {
    if (!draftProject?.project_id || !selectedAssignableAgent || selectedAgentRoleProfileMissingSkillIds.length <= 0) {
      return;
    }
    if (!persistedAgentIds.has(selectedAssignableAgent.agent_id)) {
      setEditorNotice(text.saveStudioBeforeSkillRequests);
      return;
    }
    skillRequestMutation.mutate(
      {
        projectId: draftProject.project_id,
        agentId: selectedAssignableAgent.agent_id,
        requestedSkills: selectedAgentRoleProfileMissingSkillIds,
      },
      {
        onSuccess: (payload) =>
          applySkillProjectUpdate(
            payload,
            formatTemplate(text.roleProfileSkillRequestsSyncedNotice, {
              agents: 1,
              skills: payload.skill_request_result?.created_requests?.length ?? 0,
              available: payload.skill_request_result?.available_skill_ids?.length ?? 0,
            })
          ),
      }
    );
  };

  const handleApplyRunRoleProfiles = useCallback(() => {
    if (!draftProject || !selectedRun) {
      return;
    }
    const applied = applyRoleProfilesToGraph(
      draftProject.graph_json,
      draftProject.graph_json.agent_capability_summaries ?? [],
      selectedRunScopeAgentIds
    );
    if (applied.actionableAgentCount <= 0 || applied.changedAgentCount <= 0) {
      setEditorNotice(text.noActionableRoleProfilesNotice);
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: applied.graph,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(
            formatTemplate(text.roleProfilesAppliedNotice, {
              agents: applied.changedAgentCount,
              skills: applied.skillChangeCount,
              tools: applied.toolChangeCount,
              mcp: applied.mcpChangeCount,
            })
          );
        },
      }
    );
  }, [draftProject, selectedRun, selectedRunScopeAgentIds, text, updateProjectMutation]);

  const handleRequestRunRoleProfileSkills = useCallback(() => {
    void requestRoleProfileSkillsFromSummary(selectedRunRoleProfileSummary);
  }, [requestRoleProfileSkillsFromSummary, selectedRunRoleProfileSummary]);

  const handleApplySelectedRunPolicyRepairs = () => {
    if (!draftProject || !selectedRun) {
      return;
    }
    const applied = applyCapabilityPolicySuggestionsToGraph(
      draftProject.graph_json,
      draftProject.graph_json.agent_capability_summaries ?? [],
      selectedRunScopeAgentIds
    );
    if (applied.actionableAgentCount <= 0 || applied.changedAgentCount <= 0) {
      setEditorNotice(text.noActionablePolicyRepairsNotice);
      return;
    }
    updateProjectMutation.mutate(
      {
        projectId: draftProject.project_id,
        name: draftProject.name,
        description: draftProject.description,
        graphJson: applied.graph,
      },
      {
        onSuccess: (payload) => {
          setLocalDraftProject(normalizeProject(payload, text));
          setEditorNotice(
            formatTemplate(text.policyRepairsSavedNotice, {
              agents: applied.changedAgentCount,
              tools: applied.toolChangeCount,
              mcp: applied.mcpChangeCount,
            })
          );
        },
      }
    );
  };

  const createNodePosition = useCallback(
    (index: number) => getCanvasSpawnPosition(index, canvasViewport, canvasRef.current),
    [canvasViewport]
  );

  const appendCanvasNode = useCallback((nextNode: HarnessCanvasAgentDTO, notice: string) => {
    updateDraftProject((current) => ({
      ...current,
      graph_json: {
        ...current.graph_json,
        agents: [...(current.graph_json.agents ?? []), nextNode],
      },
    }));
    setSelectedAgentId(nextNode.agent_id);
    setPendingCreationKind(null);
    setShowCanvasHint(false);
    setEditorNotice(notice);
  }, [updateDraftProject]);

  const handleConfirmCreation = (kind: CanvasCreationKind) => {
    const position = createNodePosition(agents.length);
    const nextNode =
      kind === 'agent'
        ? createAgentSeed(agents.length, text, position)
        : kind === 'brainstorm'
          ? createBrainstormClusterSeed(agents.length, text, position)
          : createCustomClusterSeed(agents.length, text, position);

    appendCanvasNode(
      nextNode,
      kind === 'agent'
        ? text.addedCanvasAgentNotice
        : kind === 'brainstorm'
          ? text.addedBrainstormClusterNotice
          : text.addedCustomClusterNotice
    );
  };

  const openCreationPreview = (kind: CanvasCreationKind) => {
    setActiveCanvasPanel(null);
    setPendingCreationKind((current) => (current === kind ? null : kind));
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
        createClusterMemberSeed(
          formatTemplate(text.memberSeedName, { count: nextIndex }),
          text.specialistRoleSeed,
          'gpt-5.1-codex-mini',
          text.memberPerspectivePrompt
        ),
      ],
    });
  };

  const toggleSelectedAgentPolicyValue = (
    listKey:
      | 'required_skill_ids'
      | 'required_tool_ids'
      | 'allowed_tool_ids'
      | 'denied_tool_ids'
      | 'required_mcp_server_ids'
      | 'allowed_mcp_server_ids'
      | 'denied_mcp_server_ids',
    rawValue: string,
    oppositeListKey?:
      | 'allowed_tool_ids'
      | 'denied_tool_ids'
      | 'allowed_mcp_server_ids'
      | 'denied_mcp_server_ids'
  ) => {
    if (!selectedAssignableAgent) {
      return;
    }
    const normalizedValue = normalizeSkillKey(rawValue);
    if (!normalizedValue) {
      return;
    }
    const currentList = (selectedAssignableAgent[listKey] ?? []).filter(Boolean);
    const isSelected = currentList.some((value) => normalizeSkillKey(value) === normalizedValue);
    const nextList = isSelected
      ? currentList.filter((value) => normalizeSkillKey(value) !== normalizedValue)
      : [...currentList, normalizedValue];
    const updates: Partial<HarnessCanvasAgentDTO> = { [listKey]: nextList };
    if (oppositeListKey && !isSelected) {
      const oppositeList = (selectedAssignableAgent[oppositeListKey] ?? []).filter(Boolean);
      updates[oppositeListKey] = oppositeList.filter((value) => normalizeSkillKey(value) !== normalizedValue);
    }
    updateSelectedAgent(updates);
  };

  const toggleRunSelection = (agentId: string) => {
    setSelectedAgentIdsForRun((current) =>
      current.includes(agentId) ? current.filter((value) => value !== agentId) : [...current, agentId]
    );
  };

  const toggleSkillAssignment = (skillId: string) => {
    if (!selectedAssignableAgent) {
      return;
    }
    const normalizedSkillId = normalizeSkillKey(skillId);
    if (!normalizedSkillId) {
      return;
    }
    const currentSkillIds = selectedAssignableAgent.skill_ids ?? [];
    const isAssigned = currentSkillIds.some((value) => normalizeSkillKey(value) === normalizedSkillId);
    if (!isAssigned && !approvedSkillIds.has(normalizedSkillId)) {
      return;
    }
    updateSelectedAgent({
      skill_ids: isAssigned
        ? currentSkillIds.filter((value) => normalizeSkillKey(value) !== normalizedSkillId)
        : [...currentSkillIds, skillId],
    });
  };

  const handleRequestSkill = (skillId: string) => {
    if (!draftProject?.project_id || !selectedAssignableAgent) {
      return;
    }
    if (!persistedAgentIds.has(selectedAssignableAgent.agent_id)) {
      setEditorNotice(text.saveStudioBeforeSkillRequests);
      return;
    }
    skillRequestMutation.mutate(
      {
        projectId: draftProject.project_id,
        agentId: selectedAssignableAgent.agent_id,
        requestedSkills: [skillId],
      },
      {
        onSuccess: (payload) => applySkillProjectUpdate(payload, text.skillRequestsSyncedNotice),
      }
    );
  };

  const handleSkillDecision = (requestId: string, approved: boolean) => {
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
        onSuccess: (payload) =>
          applySkillProjectUpdate(
            payload,
            approved ? text.skillApprovedPoolNotice : text.skillRequestRejectedNotice
          ),
      }
    );
  };

  const handleResolveRunApproval = (approved: boolean) => {
    if (!selectedRun) {
      return;
    }
    approvalMutation.mutate(
      {
        runId: selectedRun.run_id,
        approved,
        comment: approvalComment,
      },
      {
        onSuccess: () => {
          if (!pendingRunApprovalKey) {
            return;
          }
          setApprovalCommentsByKey((current) => {
            if (!(pendingRunApprovalKey in current)) {
              return current;
            }
            const next = { ...current };
            delete next[pendingRunApprovalKey];
            return next;
          });
        },
      }
    );
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

  const startStudioRun = useCallback((
    projectId: string,
    scope: 'all' | 'selected',
    weakEdgeCountOverride?: number
  ) => {
    const weakEdgeCount = weakEdgeCountOverride ?? (scope === 'selected' ? selectedScopeWeakEdgeCount : studioWeakEdgeCount);
    studioRunMutation.mutate(
      {
        projectId,
        runScope: scope,
        agentIds: scope === 'selected' ? effectiveSelectedAgentIdsForRun : [],
        loopCount,
        task: taskText.trim() || undefined,
        timeoutSeconds: timeoutValue ? parseInt(timeoutValue, 10) : undefined,
      },
      {
        onSuccess: (payload) => {
          handleSelectRun(payload.run_id);
          setEditorNotice(
            weakEdgeCount > 0
              ? formatTemplate(text.startedRunWithWeakEdgesNotice, { count: weakEdgeCount })
              : scope === 'all'
                ? text.startedFullLoopNotice
                : text.startedPartialLoopNotice
          );
        },
      }
    );
  }, [
    effectiveSelectedAgentIdsForRun,
    handleSelectRun,
    loopCount,
    selectedScopeWeakEdgeCount,
    studioRunMutation,
    studioWeakEdgeCount,
    taskText,
    text,
    timeoutValue,
  ]);

  const handleRun = (scope: 'all' | 'selected') => {
    if (!draftProject?.project_id) {
      return;
    }
    const currentAvailabilitySummary =
      scope === 'selected' ? selectedScopeAvailabilitySummary : graphAvailabilitySummary;
    if (currentAvailabilitySummary.unavailableCount > 0) {
      setEditorNotice(
        formatTemplate(text.runBlockedByUnavailableAgentsNotice, {
          scope: scope === 'all' ? text.runAllScopeLabel : text.runSelectedScopeLabel,
          names: formatAvailabilityAgentPreview(currentAvailabilitySummary.unavailableAgentNames),
        })
      );
      return;
    }
    if (hasUnsavedDraftChanges) {
      updateProjectMutation.mutate(
        {
          projectId: draftProject.project_id,
          name: draftProject.name,
          description: draftProject.description,
          graphJson: draftProject.graph_json,
        },
        {
          onSuccess: (payload) => {
            const normalizedPayload = normalizeProject(payload, text);
            setLocalDraftProject(normalizedPayload);
            const normalizedAgentNameById = new Map(
              (normalizedPayload.graph_json?.agents ?? []).map((agent) => [agent.agent_id, agent.name])
            );
            const savedAvailabilitySummary = buildAvailabilityScopeSummary(
              normalizedPayload.graph_json?.agent_capability_summaries ?? [],
              normalizedAgentNameById,
              scope === 'selected' ? effectiveSelectedAgentIdsForRun : null
            );
            if (savedAvailabilitySummary.unavailableCount > 0) {
              setEditorNotice(
                formatTemplate(text.runBlockedByUnavailableAgentsNotice, {
                  scope: scope === 'all' ? text.runAllScopeLabel : text.runSelectedScopeLabel,
                  names: formatAvailabilityAgentPreview(savedAvailabilitySummary.unavailableAgentNames),
                })
              );
              return;
            }
            const scopedDiagnostics =
              scope === 'selected'
                ? filterDiagnosticsByAgentScope(
                    coerceRecord(normalizedPayload.graph_json?.graph_diagnostics),
                    effectiveSelectedAgentIdsForRun
                  )
                : coerceRecord(normalizedPayload.graph_json?.graph_diagnostics) ?? null;
            const weakEdgeCount =
              coerceNumber(scopedDiagnostics?.weak_edge_count)
              ?? (scope === 'selected' ? selectedScopeWeakEdgeCount : studioWeakEdgeCount);
            startStudioRun(normalizedPayload.project_id, scope, weakEdgeCount);
          },
        }
      );
      return;
    }
    startStudioRun(
      draftProject.project_id,
      scope,
      scope === 'selected' ? selectedScopeWeakEdgeCount : studioWeakEdgeCount
    );
  };

  const handleRetry = () => {
    if (!selectedRun) {
      return;
    }
    retryRunMutation.mutate(
      { runId: selectedRun.run_id },
      {
        onSuccess: (payload) => handleSelectRun(payload.run_id),
      }
    );
  };

  const handleCanvasPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) {
      return;
    }
    const target = event.target as HTMLElement;
    if (target.closest('[data-canvas-ui="true"]') || target.closest('[data-node-card="true"]')) {
      return;
    }
    setActiveCanvasPanel(null);
    setPendingCreationKind(null);
    setShowCanvasHint(false);
    panRef.current = {
      startX: event.clientX,
      startY: event.clientY,
      originX: canvasViewport.x,
      originY: canvasViewport.y,
    };
    setIsCanvasPanning(true);
  };

  const handleCanvasWheel = (event: ReactWheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const nextZoom = canvasZoom * (event.deltaY > 0 ? 0.92 : 1.08);
    handleCanvasZoom(nextZoom, event.clientX - rect.left, event.clientY - rect.top);
  };

  const startDragging = (event: ReactPointerEvent<HTMLButtonElement>, agent: HarnessCanvasAgentDTO) => {
    const position = agent.position ?? { x: 0, y: 0 };
    const point = clientToWorldPoint(event.clientX, event.clientY);
    dragRef.current = {
      agentId: agent.agent_id,
      offsetX: point.x - position.x,
      offsetY: point.y - position.y,
    };
    setSelectedAgentId(agent.agent_id);
    setIsCanvasPanning(false);
    panRef.current = null;
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.16),_transparent_26%),radial-gradient(circle_at_bottom_right,_rgba(250,204,21,0.18),_transparent_24%),linear-gradient(180deg,#f8fafc_0%,#ecfeff_42%,#fff7ed_100%)]">
      <div className="mx-auto max-w-[1920px] px-4 py-4">
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex items-center gap-2 rounded-full bg-slate-950 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em] text-cyan-200">
              <Workflow className="h-3.5 w-3.5" />
              {text.studioBadge}
            </div>
            <div className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-700 shadow-sm">
              {activeProjectLabel}
            </div>
            <div className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-700 shadow-sm">
              {formatTemplate(text.nodesCount, { count: agents.length })}
            </div>
            <div className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-700 shadow-sm">
              {formatTemplate(text.edgeCount, { count: edges.length })}
            </div>
            <div className="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-700 shadow-sm">
              {reviewAgentEnabled ? text.reviewEnabledShort : text.reviewDisabledShort}
            </div>
          </div>

          <div className="relative">
            <section className="relative min-w-0">
              <div className="rounded-[14px] border border-slate-200 bg-white/90 p-3 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)] xl:flex xl:h-[calc(100dvh-140px)] xl:flex-col xl:overflow-hidden">
                <div className="min-h-0 flex-1">
                  <div
                    ref={canvasRef}
                    onPointerDown={handleCanvasPointerDown}
                    onWheel={handleCanvasWheel}
                    className={`relative h-full w-full overflow-hidden rounded-[18px] border border-slate-200 shadow-[inset_0_1px_0_rgba(255,255,255,0.6)] ${
                      isCanvasPanning ? 'cursor-grabbing' : 'cursor-grab'
                    }`}
                    style={{
                      backgroundImage:
                        'linear-gradient(90deg,rgba(148,163,184,0.08)_1px,transparent_1px),linear-gradient(180deg,rgba(148,163,184,0.08)_1px,transparent_1px),radial-gradient(circle_at_top,rgba(255,255,255,0.98),rgba(248,250,252,0.98)_46%,rgba(241,245,249,0.98)_100%)',
                      backgroundSize: `${28 * canvasZoom}px ${28 * canvasZoom}px, ${28 * canvasZoom}px ${28 * canvasZoom}px, 100% 100%`,
                      backgroundPosition: `${canvasViewport.x}px ${canvasViewport.y}px, ${canvasViewport.x}px ${canvasViewport.y}px, center`,
                    }}
                  >
                    <div data-canvas-ui="true" className="pointer-events-none absolute left-4 top-4 z-20 flex max-w-[calc(100%-2rem)] flex-wrap gap-2 lg:max-w-[calc(100%-480px)]">
                      <button
                        type="button"
                        onClick={() => openCreationPreview('agent')}
                        disabled={!draftProject}
                        className="pointer-events-auto inline-flex h-10 items-center gap-2 rounded-[10px] border border-slate-200 bg-white/95 px-4 text-xs font-semibold text-slate-800 shadow-[0_18px_40px_-34px_rgba(15,23,42,0.5)] transition hover:bg-slate-50 disabled:opacity-50"
                      >
                        <GitBranchPlus className="h-4 w-4" />
                        {text.addAgent}
                      </button>
                      <button
                        type="button"
                        onClick={() => openCreationPreview('brainstorm')}
                        disabled={!draftProject}
                        className="pointer-events-auto inline-flex h-10 items-center gap-2 rounded-[10px] border border-amber-200 bg-amber-50/95 px-4 text-xs font-semibold text-amber-900 shadow-[0_18px_40px_-34px_rgba(15,23,42,0.45)] transition hover:bg-amber-100 disabled:opacity-50"
                      >
                        <Sparkles className="h-4 w-4" />
                        {text.brainstormCluster}
                      </button>
                      <button
                        type="button"
                        onClick={() => openCreationPreview('custom')}
                        disabled={!draftProject}
                        className="pointer-events-auto inline-flex h-10 items-center gap-2 rounded-[10px] border border-emerald-200 bg-emerald-50/95 px-4 text-xs font-semibold text-emerald-900 shadow-[0_18px_40px_-34px_rgba(15,23,42,0.45)] transition hover:bg-emerald-100 disabled:opacity-50"
                      >
                        <Layers3 className="h-4 w-4" />
                        {text.customCluster}
                      </button>
                    </div>

                    {creationPreview ? (
                      <div
                        data-canvas-ui="true"
                        onWheelCapture={stopCanvasWheelPropagation}
                        className="absolute left-4 top-16 z-30 w-[min(360px,calc(100%-2rem))]"
                      >
                        <div className={`rounded-[18px] border px-4 py-4 shadow-[0_28px_80px_-45px_rgba(15,23,42,0.45)] backdrop-blur ${creationPreview.toneClassName}`}>
                          <div className="flex items-start gap-3">
                            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-white/80 ring-1 ring-black/5">
                              <creationPreview.icon className="h-4 w-4" />
                            </div>
                            <div className="min-w-0 flex-1">
                              <div className="text-sm font-semibold">{creationPreview.label}</div>
                              <div className="mt-1 text-sm leading-6 text-slate-600">{creationPreview.description}</div>
                            </div>
                          </div>
                          <div className="mt-4 rounded-[12px] bg-white/80 px-3 py-3 text-xs leading-6 text-slate-600 ring-1 ring-black/5">
                            {text.spawnAtCanvasCenter}
                          </div>
                          <div className="mt-4 flex justify-end gap-2">
                            <button
                              type="button"
                              onClick={() => setPendingCreationKind(null)}
                              className="rounded-[10px] border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                            >
                              {text.cancel}
                            </button>
                            <button
                              type="button"
                              onClick={() => handleConfirmCreation(creationPreview.kind)}
                              className="rounded-[10px] bg-slate-950 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-800"
                            >
                              {text.confirmCreateNode}
                            </button>
                          </div>
                        </div>
                      </div>
                    ) : null}

                    {topCanvasMessage ? (
                      <div data-canvas-ui="true" className="pointer-events-none absolute inset-x-0 top-16 z-20 flex justify-center px-4 lg:top-4 lg:px-24">
                        <div className="pointer-events-auto max-w-2xl rounded-[14px] border border-slate-200 bg-white/96 px-4 py-3 text-sm text-slate-700 shadow-[0_18px_40px_-30px_rgba(15,23,42,0.45)] backdrop-blur">
                          {topCanvasMessage}
                        </div>
                      </div>
                    ) : null}

                    <div data-canvas-ui="true" className="absolute right-4 top-4 z-20 flex max-w-[calc(100%-1rem)] flex-wrap justify-end gap-2 lg:max-w-[520px]">
                      <button
                        type="button"
                        onClick={() => toggleCanvasPanel('control')}
                        className={`inline-flex h-10 items-center gap-2 rounded-[10px] border px-4 text-xs font-semibold shadow-[0_18px_40px_-34px_rgba(15,23,42,0.45)] transition ${
                          activeCanvasPanel === 'control'
                            ? 'border-slate-900 bg-slate-950 text-white'
                            : 'border-slate-200 bg-white/95 text-slate-800 hover:bg-slate-50'
                        }`}
                      >
                        <Workflow className="h-4 w-4" />
                        {text.controlCenter}
                      </button>
                      <button
                        type="button"
                        onClick={() => toggleCanvasPanel('skills')}
                        className={`inline-flex h-10 items-center gap-2 rounded-[10px] border px-4 text-xs font-semibold shadow-[0_18px_40px_-34px_rgba(15,23,42,0.45)] transition ${
                          activeCanvasPanel === 'skills'
                            ? 'border-slate-900 bg-slate-950 text-white'
                            : 'border-slate-200 bg-white/95 text-slate-800 hover:bg-slate-50'
                        }`}
                      >
                        <Sparkles className="h-4 w-4" />
                        {text.skillPool}
                        <span className={`rounded-full px-2 py-0.5 text-[10px] ${activeCanvasPanel === 'skills' ? 'bg-white/15 text-white' : 'bg-slate-100 text-slate-600'}`}>
                          {approvedSkillCount}
                        </span>
                      </button>
                      <button
                        type="button"
                        onClick={() => toggleCanvasPanel('runs')}
                        className={`inline-flex h-10 items-center gap-2 rounded-[10px] border px-4 text-xs font-semibold shadow-[0_18px_40px_-34px_rgba(15,23,42,0.45)] transition ${
                          activeCanvasPanel === 'runs'
                            ? 'border-slate-900 bg-slate-950 text-white'
                            : 'border-slate-200 bg-white/95 text-slate-800 hover:bg-slate-50'
                        }`}
                      >
                        <Clock3 className="h-4 w-4" />
                        {text.runsAndEvidence}
                      </button>
                      <button
                        type="button"
                        onClick={() => toggleCanvasPanel('review')}
                        className={`inline-flex h-10 items-center gap-2 rounded-[10px] border px-4 text-xs font-semibold shadow-[0_18px_40px_-34px_rgba(15,23,42,0.45)] transition ${
                          activeCanvasPanel === 'review'
                            ? 'border-slate-900 bg-slate-950 text-white'
                            : 'border-slate-200 bg-white/95 text-slate-800 hover:bg-slate-50'
                        }`}
                      >
                        <ShieldCheck className="h-4 w-4" />
                        {text.reviewPath}
                      </button>
                    </div>

                    {activeCanvasPanel ? (
                      <div
                        data-canvas-ui="true"
                        onWheelCapture={stopCanvasWheelPropagation}
                        onWheel={stopCanvasWheelPropagation}
                        className={`absolute bottom-4 right-4 top-16 z-30 flex overflow-hidden rounded-[18px] border border-slate-200 bg-white/96 shadow-[0_28px_80px_-45px_rgba(15,23,42,0.5)] backdrop-blur ${floatingPanelWidthClass}`}
                      >
                        <div className="flex min-h-0 flex-1 flex-col">
                          <div className="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-3">
                            <div className="text-sm font-semibold text-slate-900">
                              {activeCanvasPanel === 'control'
                                ? text.controlCenter
                                : activeCanvasPanel === 'skills'
                                  ? text.skillPool
                                  : activeCanvasPanel === 'runs'
                                    ? text.runsAndEvidence
                                    : text.reviewPath}
                            </div>
                            <button
                              type="button"
                              onClick={() => setActiveCanvasPanel(null)}
                              className="rounded-[8px] border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50"
                            >
                              {text.closePanel}
                            </button>
                          </div>
                          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-4" onWheel={stopCanvasWheelPropagation}>
                          {activeCanvasPanel === 'control' ? (
                            <div className="grid gap-4 xl:grid-cols-2">
                              <section className="rounded-[14px] border border-slate-200 bg-white p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.2)]">
                                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                  <Workflow className="h-4 w-4 text-cyan-600" />
                                  {text.project}
                                </div>
                                <div className="mt-4 space-y-3">
                                  <label className="block text-sm font-medium text-slate-800">
                                    {text.project}
                                    <select
                                      value={activeProjectId ?? ''}
                                      onChange={(event) => setSelectedProjectId(event.target.value || null)}
                                      className="mt-2 w-full rounded-[8px] border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                    >
                                      {(projectsQuery.data?.projects ?? []).map((project) => (
                                        <option key={project.project_id} value={project.project_id}>
                                          {localizeStudioProjectName(project.name, text)}
                                        </option>
                                      ))}
                                    </select>
                                  </label>
                                  <div className="grid grid-cols-2 gap-3">
                                    <button
                                      type="button"
                                      onClick={handleProjectCreate}
                                      disabled={createProjectMutation.isPending}
                                      className="inline-flex items-center justify-center gap-2 rounded-[8px] bg-cyan-600 px-4 py-3 text-sm font-semibold text-white hover:bg-cyan-500 disabled:opacity-50"
                                    >
                                      <PlusCircle className="h-4 w-4" />
                                      {createProjectMutation.isPending ? text.creating : text.newProject}
                                    </button>
                                    <button
                                      type="button"
                                      onClick={handleSaveProject}
                                      disabled={!draftProject || updateProjectMutation.isPending}
                                      className="inline-flex items-center justify-center gap-2 rounded-[8px] bg-slate-950 px-4 py-3 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
                                    >
                                      <Save className="h-4 w-4" />
                                      {updateProjectMutation.isPending ? text.saving : text.saveStudio}
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => {
                                        projectsQuery.refetch();
                                        currentProjectQuery.refetch();
                                        projectQuery.refetch();
                                        knowledgeBasesQuery.refetch();
                                        runsQuery.refetch();
                                        detailQuery.refetch();
                                      }}
                                      className="inline-flex items-center justify-center gap-2 rounded-[8px] border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-800 hover:bg-slate-50"
                                    >
                                      <RefreshCw className="h-4 w-4" />
                                      {text.refresh}
                                    </button>
                                    <Link
                                      href="/chat"
                                      className="inline-flex items-center justify-center rounded-[8px] border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50"
                                    >
                                      {text.backToChat}
                                    </Link>
                                  </div>
                                  <div className="grid gap-3 sm:grid-cols-2">
                                    <div className="rounded-[12px] bg-slate-50 p-4">
                                      <div className="text-xs uppercase tracking-[0.16em] text-slate-400">{text.policy}</div>
                                      <div className="mt-2 text-sm font-semibold text-slate-900">
                                        {orchestrationPolicy?.policy_id || text.orchestrationPolicyFallback}
                                      </div>
                                    </div>
                                    <div className="rounded-[12px] bg-slate-50 p-4">
                                      <div className="text-xs uppercase tracking-[0.16em] text-slate-400">{text.retryBudget}</div>
                                      <div className="mt-2 text-sm font-semibold text-slate-900">{orchestrationPolicy?.retry_budget ?? 1}</div>
                                    </div>
                                  </div>
                                </div>
                              </section>

                              <section className="rounded-[14px] border border-slate-200 bg-white p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.2)]">
                                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                  <Layers3 className="h-4 w-4 text-emerald-600" />
                                  {text.referencedKnowledgeBases}
                                </div>
                                <div className="mt-1 text-sm text-slate-500">{text.referencedKnowledgeBasesDescription}</div>
                                <div className="mt-4 space-y-3">
                                  <div className="rounded-[12px] bg-slate-50 p-4">
                                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">{text.referencedKnowledgeBases}</div>
                                    <div className="mt-2 text-sm font-semibold text-slate-900">
                                      {formatTemplate(text.knowledgeBasesCountShort, { count: referencedKnowledgeBases.length })}
                                    </div>
                                  </div>
                                  {(knowledgeBasesQuery.data ?? []).length > 0 ? (
                                    <div className="grid gap-3">
                                      {(knowledgeBasesQuery.data ?? []).map((knowledgeBase) => {
                                        const checked = referencedKnowledgeBaseIds.includes(knowledgeBase.knowledge_base_id);
                                        return (
                                          <label
                                            key={knowledgeBase.knowledge_base_id}
                                            className={`flex cursor-pointer items-start gap-3 rounded-[12px] border px-4 py-3 ${
                                              checked ? 'border-cyan-200 bg-cyan-50/70' : 'border-slate-200 bg-slate-50'
                                            }`}
                                          >
                                            <input
                                              type="checkbox"
                                              checked={checked}
                                              onChange={() =>
                                                updateDraftProject((current) => {
                                                  const currentIds = current.graph_json.knowledge_base_ids ?? [];
                                                  const nextIds = checked
                                                    ? currentIds.filter((item) => item !== knowledgeBase.knowledge_base_id)
                                                    : [...currentIds, knowledgeBase.knowledge_base_id];
                                                  return {
                                                    ...current,
                                                    graph_json: {
                                                      ...current.graph_json,
                                                      knowledge_base_ids: nextIds,
                                                    },
                                                  };
                                                })
                                              }
                                              className="mt-1 h-4 w-4 rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                                            />
                                            <span className="min-w-0 flex-1">
                                              <span className="block text-sm font-semibold text-slate-900">{knowledgeBase.name}</span>
                                              <span className="mt-1 block text-xs leading-5 text-slate-500">
                                                {knowledgeBase.description || formatTemplate(text.libraryDocumentCount, { count: knowledgeBase.document_count ?? 0 })}
                                              </span>
                                            </span>
                                          </label>
                                        );
                                      })}
                                    </div>
                                  ) : (
                                    <div className="rounded-[12px] border border-dashed border-slate-300 bg-slate-50 px-4 py-4 text-sm text-slate-500">
                                      {text.noKnowledgeBasesAvailable}
                                    </div>
                                  )}
                                </div>
                              </section>

                              <section className="rounded-[14px] border border-slate-200 bg-white p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.2)]">
                                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                  <Play className="h-4 w-4 text-sky-600" />
                                  {text.canvasOrchestration}
                                </div>
                                <div className="mt-4 space-y-3">
                                  <label className="block text-sm font-medium text-slate-800">
                                    {text.taskLabel}
                                    <input
                                      type="text"
                                      value={taskText}
                                      onChange={(event) => setTaskText(event.target.value)}
                                      placeholder={text.inheritProjectName}
                                      className="mt-2 w-full rounded-[8px] border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                    />
                                  </label>
                                  <div className="grid gap-3 sm:grid-cols-2">
                                    <label className="block text-sm font-medium text-slate-800">
                                      {text.timeoutSeconds}
                                      <input
                                        type="number"
                                        min={5}
                                        max={600}
                                        value={timeoutValue}
                                        onChange={(event) => setTimeoutValue(event.target.value)}
                                        placeholder={text.timeoutPlaceholder}
                                        className="mt-2 w-full rounded-[8px] border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                      />
                                    </label>
                                    <label className="block text-sm font-medium text-slate-800">
                                      {text.loopCount}
                                      <input
                                        type="number"
                                        min={1}
                                        max={10}
                                        value={loopCount}
                                        onChange={(event) => setLoopCount(Number(event.target.value) || 1)}
                                        className="mt-2 w-full rounded-[8px] border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                      />
                                    </label>
                                  </div>
                                  <div className="grid grid-cols-2 gap-3">
                                    <button
                                      type="button"
                                      onClick={() => handleRun('selected')}
                                      disabled={
                                        !runReady ||
                                        effectiveSelectedAgentIdsForRun.length === 0 ||
                                        runSelectedBlockedByAvailability ||
                                        studioRunMutation.isPending ||
                                        updateProjectMutation.isPending
                                      }
                                      className="inline-flex items-center justify-center gap-2 rounded-[8px] border border-sky-200 bg-sky-50 px-4 py-3 text-sm font-semibold text-sky-900 hover:bg-sky-100 disabled:opacity-50"
                                    >
                                      <Play className="h-4 w-4" />
                                      {text.runSelected}
                                    </button>
                                    <button
                                      type="button"
                                      onClick={() => handleRun('all')}
                                      disabled={
                                        !runReady ||
                                        runAllBlockedByAvailability ||
                                        studioRunMutation.isPending ||
                                        updateProjectMutation.isPending
                                      }
                                      className="inline-flex items-center justify-center gap-2 rounded-[8px] bg-slate-950 px-4 py-3 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
                                    >
                                      <Workflow className="h-4 w-4" />
                                      {text.runAll}
                                    </button>
                                  </div>
                                  {hasUnsavedDraftChanges ? (
                                    <div className="rounded-[12px] border border-sky-200 bg-sky-50/70 px-4 py-3 text-sm text-sky-900">
                                      {text.unsavedDraftRunHint}
                                    </div>
                                  ) : null}
                                  <div
                                    className={`rounded-[12px] border p-4 ${
                                      graphAvailabilitySummary.unavailableCount > 0
                                        ? 'border-rose-200 bg-rose-50/70'
                                        : graphAvailabilitySummary.limitedCount > 0
                                          ? 'border-amber-200 bg-amber-50/70'
                                          : 'border-emerald-200 bg-emerald-50/60'
                                    }`}
                                  >
                                    <div>
                                      <div className="text-sm font-semibold text-slate-900">{text.preflightAvailabilityCheckLabel}</div>
                                      <div className="mt-1 text-sm text-slate-600">
                                        {graphAvailabilitySummary.unavailableCount > 0
                                          ? text.preflightAvailabilityBlockedHint
                                          : graphAvailabilitySummary.limitedCount > 0
                                            ? text.preflightAvailabilityLimitedHint
                                            : text.preflightAvailabilityReadyHint}
                                      </div>
                                      <div className="mt-2 text-xs leading-5 text-slate-500">{text.preflightAvailabilitySavedGraphHint}</div>
                                    </div>
                                    <div className="mt-4 grid gap-4 xl:grid-cols-2">
                                      <div className={`rounded-[12px] border p-4 ${availabilityScopeToneClasses(graphAvailabilitySummary).panel}`}>
                                        <div className="flex flex-wrap items-start justify-between gap-3">
                                          <div>
                                            <div className={`text-xs font-semibold uppercase tracking-[0.18em] ${availabilityScopeToneClasses(graphAvailabilitySummary).accent}`}>{text.runAllScopeLabel}</div>
                                            <div className="mt-1 text-sm text-slate-700">
                                              {graphAvailabilitySummary.unavailableCount > 0
                                                ? text.preflightAvailabilityBlockedHint
                                                : graphAvailabilitySummary.limitedCount > 0
                                                  ? text.preflightAvailabilityLimitedHint
                                                  : text.preflightAvailabilityReadyHint}
                                            </div>
                                          </div>
                                        </div>
                                        <AvailabilityPreflightCard
                                          summary={graphAvailabilitySummary}
                                          emptyLabel={text.noAvailabilityPreflightIssues}
                                        />
                                      </div>
                                      <div
                                        className={`rounded-[12px] border p-4 ${
                                          effectiveSelectedAgentIdsForRun.length === 0
                                            ? 'border-slate-200 bg-slate-50/60'
                                            : availabilityScopeToneClasses(selectedScopeAvailabilitySummary).panel
                                        }`}
                                      >
                                        <div className="flex flex-wrap items-start justify-between gap-3">
                                          <div>
                                            <div
                                              className={`text-xs font-semibold uppercase tracking-[0.18em] ${
                                                effectiveSelectedAgentIdsForRun.length === 0
                                                  ? 'text-slate-500'
                                                  : availabilityScopeToneClasses(selectedScopeAvailabilitySummary).accent
                                              }`}
                                            >
                                              {text.runSelectedScopeLabel}
                                            </div>
                                            <div className="mt-1 text-sm text-slate-700">
                                              {effectiveSelectedAgentIdsForRun.length === 0
                                                ? text.noSelectedRunScopeAvailabilityYet
                                                : selectedScopeAvailabilitySummary.unavailableCount > 0
                                                  ? text.preflightAvailabilityBlockedHint
                                                  : selectedScopeAvailabilitySummary.limitedCount > 0
                                                    ? text.preflightAvailabilityLimitedHint
                                                    : text.preflightAvailabilityReadyHint}
                                            </div>
                                          </div>
                                        </div>
                                        <div className="mt-2 text-xs leading-5 text-slate-500">
                                          {effectiveSelectedAgentIdsForRun.length > 0
                                            ? text.preflightAvailabilitySelectedScopeHint
                                            : text.noSelectedRunScopeAvailabilityYet}
                                        </div>
                                        <AvailabilityPreflightCard
                                          summary={selectedScopeAvailabilitySummary}
                                          emptyLabel={text.noSelectedRunScopeAvailabilityYet}
                                        />
                                      </div>
                                    </div>
                                  </div>
                                  <div
                                    className={`rounded-[12px] border p-4 ${
                                      graphPolicyRepairSummary.agentCount > 0
                                        ? 'border-cyan-200 bg-cyan-50/60'
                                        : 'border-slate-200 bg-slate-50/60'
                                    }`}
                                  >
                                    <div>
                                      <div className="text-sm font-semibold text-slate-900">{text.preflightPolicyRepairLabel}</div>
                                      <div className="mt-1 text-sm text-slate-600">
                                        {graphPolicyRepairSummary.agentCount > 0
                                          ? text.preflightPolicyRepairHint
                                          : text.preflightPolicyRepairReadyHint}
                                      </div>
                                      <div className="mt-2 text-xs leading-5 text-slate-500">{text.preflightSavedGraphHint}</div>
                                    </div>
                                    <div className="mt-4 grid gap-4 xl:grid-cols-2">
                                      <div className="rounded-[12px] border border-white/80 bg-white/70 p-4">
                                        <div className="flex flex-wrap items-start justify-between gap-3">
                                          <div>
                                            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.runAllScopeLabel}</div>
                                            <div className="mt-1 text-sm text-slate-700">
                                              {graphPolicyRepairSummary.agentCount > 0
                                                ? text.preflightPolicyRepairHint
                                                : text.preflightPolicyRepairReadyHint}
                                            </div>
                                          </div>
                                          {graphPolicyRepairSummary.agentCount > 0 ? (
                                            <button
                                              type="button"
                                              onClick={() => handleApplyPolicyRepairsForScope('all')}
                                              disabled={updateProjectMutation.isPending}
                                              className="inline-flex items-center justify-center rounded-[10px] border border-cyan-300 bg-white px-3 py-1.5 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                                            >
                                              {updateProjectMutation.isPending ? text.applyingPolicyRepairs : text.applyAllSuggestedPolicyRepairs}
                                            </button>
                                          ) : null}
                                        </div>
                                        <div className="mt-3 flex flex-wrap gap-2">
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.policyRepairAgentsCountLabel, { count: graphPolicyRepairSummary.agentCount })}
                                          </span>
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.policyRepairToolsCountLabel, { count: graphPolicyRepairSummary.toolSuggestionCount })}
                                          </span>
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.policyRepairMcpCountLabel, { count: graphPolicyRepairSummary.mcpSuggestionCount })}
                                          </span>
                                        </div>
                                        <div className="mt-3 text-xs leading-5 text-slate-500">
                                          {graphPolicyRepairSummary.agentCount > 0
                                            ? graphPolicyRepairSummary.diagnostics
                                                .slice(0, 3)
                                                .map((item) => item.agentName)
                                                .join(' · ')
                                            : text.noActionablePolicyRepairsNotice}
                                        </div>
                                      </div>
                                      <div className="rounded-[12px] border border-white/80 bg-white/70 p-4">
                                        <div className="flex flex-wrap items-start justify-between gap-3">
                                          <div>
                                            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.runSelectedScopeLabel}</div>
                                            <div className="mt-1 text-sm text-slate-700">
                                              {effectiveSelectedAgentIdsForRun.length === 0
                                                ? text.noSelectedRunScopeYet
                                                : selectedScopePolicyRepairSummary.agentCount > 0
                                                  ? text.preflightPolicyRepairHint
                                                  : text.preflightPolicyRepairReadyHint}
                                            </div>
                                          </div>
                                          {effectiveSelectedAgentIdsForRun.length > 0 && selectedScopePolicyRepairSummary.agentCount > 0 ? (
                                            <button
                                              type="button"
                                              onClick={() => handleApplyPolicyRepairsForScope('selected')}
                                              disabled={updateProjectMutation.isPending}
                                              className="inline-flex items-center justify-center rounded-[10px] border border-cyan-300 bg-white px-3 py-1.5 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                                            >
                                              {updateProjectMutation.isPending ? text.applyingPolicyRepairs : text.applySelectedSuggestedPolicyRepairs}
                                            </button>
                                          ) : null}
                                        </div>
                                        <div className="mt-3 flex flex-wrap gap-2">
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.policyRepairAgentsCountLabel, { count: selectedScopePolicyRepairSummary.agentCount })}
                                          </span>
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.policyRepairToolsCountLabel, { count: selectedScopePolicyRepairSummary.toolSuggestionCount })}
                                          </span>
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.policyRepairMcpCountLabel, { count: selectedScopePolicyRepairSummary.mcpSuggestionCount })}
                                          </span>
                                        </div>
                                        <div className="mt-3 text-xs leading-5 text-slate-500">
                                          {effectiveSelectedAgentIdsForRun.length === 0
                                            ? text.noSelectedRunScopeYet
                                            : selectedScopePolicyRepairSummary.agentCount > 0
                                              ? selectedScopePolicyRepairSummary.diagnostics
                                                  .slice(0, 3)
                                                  .map((item) => item.agentName)
                                                  .join(' · ')
                                              : text.noActionablePolicyRepairsNotice}
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                  <div
                                    className={`rounded-[12px] border p-4 ${
                                      studioWeakEdgeCount > 0
                                        ? 'border-amber-200 bg-amber-50/70'
                                        : 'border-emerald-200 bg-emerald-50/60'
                                    }`}
                                  >
                                    <div>
                                      <div className="text-sm font-semibold text-slate-900">{text.preflightHandoffCheckLabel}</div>
                                      <div className="mt-1 text-sm text-slate-600">
                                        {studioWeakEdgeCount > 0 ? text.preflightWeakEdgeHint : text.preflightReadyHint}
                                      </div>
                                      <div className="mt-2 text-xs leading-5 text-slate-500">{text.preflightSavedGraphHint}</div>
                                    </div>
                                    <div className="mt-4 grid gap-4 xl:grid-cols-2">
                                      <div className="rounded-[12px] border border-white/80 bg-white/70 p-4">
                                        <div className="flex flex-wrap items-start justify-between gap-3">
                                          <div>
                                            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.runAllScopeLabel}</div>
                                            <div className="mt-1 text-sm text-slate-700">
                                              {studioWeakEdgeCount > 0 ? text.preflightWeakEdgeHint : text.preflightReadyHint}
                                            </div>
                                          </div>
                                          <div className="flex flex-wrap gap-2">
                                            {studioWeakEdgeCount > 0 || studioBestNextCount > 0 ? (
                                              <button
                                                type="button"
                                                onClick={() => handleApplySuggestedCollaborationFixesForScope('all')}
                                                disabled={updateProjectMutation.isPending}
                                                className="inline-flex items-center justify-center rounded-[10px] border border-emerald-300 bg-white px-3 py-1.5 text-xs font-semibold text-emerald-900 hover:bg-emerald-100 disabled:opacity-50"
                                              >
                                                {updateProjectMutation.isPending
                                                  ? text.applyingSuggestedCollaborationFixes
                                                  : text.applyAllSuggestedCollaborationFixes}
                                              </button>
                                            ) : null}
                                            {studioWeakEdgeCount > 0 ? (
                                              <button
                                                type="button"
                                                onClick={() => handleApplySuggestedRewiresForScope('all')}
                                                disabled={updateProjectMutation.isPending}
                                                className="inline-flex items-center justify-center rounded-[10px] border border-amber-300 bg-white px-3 py-1.5 text-xs font-semibold text-amber-900 hover:bg-amber-100 disabled:opacity-50"
                                              >
                                                {updateProjectMutation.isPending ? text.applyingAllSuggestedRewires : text.applyAllSuggestedRewires}
                                              </button>
                                            ) : null}
                                            {studioBestNextCount > 0 ? (
                                              <button
                                                type="button"
                                                onClick={() => handleApplySuggestedHandoffsForScope('all')}
                                                disabled={updateProjectMutation.isPending}
                                                className="inline-flex items-center justify-center rounded-[10px] border border-cyan-300 bg-white px-3 py-1.5 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                                              >
                                                {updateProjectMutation.isPending ? text.applyingAllSuggestedHandoffs : text.applyAllSuggestedHandoffs}
                                              </button>
                                            ) : null}
                                          </div>
                                        </div>
                                        <div className="mt-3 flex flex-wrap gap-2">
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.weakEdgesCountLabel, { count: studioWeakEdgeCount })}
                                          </span>
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.bestNextCountLabel, { count: studioBestNextCount })}
                                          </span>
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.collaborationSourceAgentsCountLabel, { count: studioCollaborationSummary.actionableSourceAgentCount })}
                                          </span>
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.collaborationLaneCountLabel, { count: studioCollaborationSummary.laneCount })}
                                          </span>
                                          {studioCollaborationSummary.focusCount > 0 ? (
                                            <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                              {formatTemplate(text.collaborationFocusCountLabel, { count: studioCollaborationSummary.focusCount })}
                                            </span>
                                          ) : null}
                                        </div>
                                        <div className="mt-3 text-xs leading-5 text-slate-500">
                                          <span className="font-semibold text-slate-700">{text.collaborationFocusPreviewLabel}:</span>{' '}
                                          {studioCollaborationSummary.focusPreview.length > 0
                                            ? studioCollaborationSummary.focusPreview.join(' · ')
                                            : text.noCollaborationFocusPreview}
                                        </div>
                                        <GraphCollaborationSummary
                                          collaborationSummary={studioCollaborationSummary}
                                          topologySummary={studioCoordinationTopologySummary}
                                          focusableAgentIds={focusableAgentIds}
                                          onFocusAgent={openNodeEditor}
                                        />
                                        <GraphCapabilityCoverageSummary
                                          summary={studioCapabilityCoverageSummary}
                                          skillTitleById={skillTitleById}
                                          toolTitleById={toolTitleById}
                                          mcpServerTitleById={mcpServerTitleById}
                                          focusableAgentIds={focusableAgentIds}
                                          onFocusAgent={openNodeEditor}
                                        />
                                        <HandoffDiagnosticsCard
                                          diagnostics={studioGraphDiagnostics}
                                          emptyLabel={text.noStudioGraphDiagnostics}
                                          onApplyRewire={handleApplySuggestedRewire}
                                          onApplySuggestedHandoff={handleApplySuggestedHandoff}
                                          isApplying={updateProjectMutation.isPending}
                                        />
                                      </div>
                                      <div className="rounded-[12px] border border-white/80 bg-white/70 p-4">
                                        <div className="flex flex-wrap items-start justify-between gap-3">
                                          <div>
                                            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">{text.runSelectedScopeLabel}</div>
                                            <div className="mt-1 text-sm text-slate-700">
                                              {effectiveSelectedAgentIdsForRun.length > 0 ? text.preflightSelectedScopeHint : text.noSelectedRunScopeYet}
                                            </div>
                                          </div>
                                          <div className="flex flex-wrap gap-2">
                                            {effectiveSelectedAgentIdsForRun.length > 0 && (selectedScopeWeakEdgeCount > 0 || selectedScopeBestNextCount > 0) ? (
                                              <button
                                                type="button"
                                                onClick={() => handleApplySuggestedCollaborationFixesForScope('selected')}
                                                disabled={updateProjectMutation.isPending}
                                                className="inline-flex items-center justify-center rounded-[10px] border border-emerald-300 bg-white px-3 py-1.5 text-xs font-semibold text-emerald-900 hover:bg-emerald-100 disabled:opacity-50"
                                              >
                                                {updateProjectMutation.isPending
                                                  ? text.applyingSuggestedCollaborationFixes
                                                  : text.applySelectedSuggestedCollaborationFixes}
                                              </button>
                                            ) : null}
                                            {effectiveSelectedAgentIdsForRun.length > 0 && selectedScopeWeakEdgeCount > 0 ? (
                                              <button
                                                type="button"
                                                onClick={() => handleApplySuggestedRewiresForScope('selected')}
                                                disabled={updateProjectMutation.isPending}
                                                className="inline-flex items-center justify-center rounded-[10px] border border-cyan-300 bg-white px-3 py-1.5 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                                              >
                                                {updateProjectMutation.isPending ? text.applyingAllSuggestedRewires : text.applySelectedSuggestedRewires}
                                              </button>
                                            ) : null}
                                            {effectiveSelectedAgentIdsForRun.length > 0 && selectedScopeBestNextCount > 0 ? (
                                              <button
                                                type="button"
                                                onClick={() => handleApplySuggestedHandoffsForScope('selected')}
                                                disabled={updateProjectMutation.isPending}
                                                className="inline-flex items-center justify-center rounded-[10px] border border-cyan-300 bg-white px-3 py-1.5 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                                              >
                                                {updateProjectMutation.isPending ? text.applyingAllSuggestedHandoffs : text.applySelectedSuggestedHandoffs}
                                              </button>
                                            ) : null}
                                          </div>
                                        </div>
                                        <div className="mt-3 flex flex-wrap gap-2">
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.weakEdgesCountLabel, { count: selectedScopeWeakEdgeCount })}
                                          </span>
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.bestNextCountLabel, { count: selectedScopeBestNextCount })}
                                          </span>
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.collaborationSourceAgentsCountLabel, { count: selectedScopeCollaborationSummary.actionableSourceAgentCount })}
                                          </span>
                                          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                            {formatTemplate(text.collaborationLaneCountLabel, { count: selectedScopeCollaborationSummary.laneCount })}
                                          </span>
                                          {selectedScopeCollaborationSummary.focusCount > 0 ? (
                                            <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                              {formatTemplate(text.collaborationFocusCountLabel, { count: selectedScopeCollaborationSummary.focusCount })}
                                            </span>
                                          ) : null}
                                        </div>
                                        <div className="mt-3 text-xs leading-5 text-slate-500">
                                          <span className="font-semibold text-slate-700">{text.collaborationFocusPreviewLabel}:</span>{' '}
                                          {effectiveSelectedAgentIdsForRun.length === 0
                                            ? text.noSelectedRunScopeYet
                                            : selectedScopeCollaborationSummary.focusPreview.length > 0
                                              ? selectedScopeCollaborationSummary.focusPreview.join(' · ')
                                              : text.noCollaborationFocusPreview}
                                        </div>
                                        <GraphCollaborationSummary
                                          collaborationSummary={selectedScopeCollaborationSummary}
                                          topologySummary={selectedScopeCoordinationTopologySummary}
                                          focusableAgentIds={focusableAgentIds}
                                          onFocusAgent={openNodeEditor}
                                        />
                                        <GraphCapabilityCoverageSummary
                                          summary={selectedScopeCapabilityCoverageSummary}
                                          skillTitleById={skillTitleById}
                                          toolTitleById={toolTitleById}
                                          mcpServerTitleById={mcpServerTitleById}
                                          focusableAgentIds={focusableAgentIds}
                                          onFocusAgent={openNodeEditor}
                                        />
                                        <HandoffDiagnosticsCard
                                          diagnostics={selectedScopeGraphDiagnostics}
                                          emptyLabel={text.noSelectedRunScopeYet}
                                          onApplyRewire={handleApplySuggestedRewire}
                                          onApplySuggestedHandoff={handleApplySuggestedHandoff}
                                          isApplying={updateProjectMutation.isPending}
                                        />
                                      </div>
                                    </div>
                                  </div>
                                  <div className="rounded-[12px] border border-slate-200 bg-slate-50 p-4">
                                    <div className="flex flex-wrap items-center justify-between gap-3">
                                      <div>
                                        <div className="text-sm font-semibold text-slate-900">{text.executionChecklist}</div>
                                        <div className="mt-1 text-sm text-slate-500">{text.executionChecklistDescription}</div>
                                      </div>
                                      <button
                                        type="button"
                                        onClick={handleAddChecklistItem}
                                        className="inline-flex items-center gap-2 rounded-[8px] border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-100"
                                      >
                                        <PlusCircle className="h-3.5 w-3.5" />
                                        {text.addChecklistItem}
                                      </button>
                                    </div>
                                    <div className="mt-3 flex flex-wrap gap-2">
                                      <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                        {formatTemplate(text.checklistCount, { count: executionChecklist.length })}
                                      </span>
                                      <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                                        {formatTemplate(text.openChecklistCount, { count: openExecutionChecklistCount })}
                                      </span>
                                    </div>
                                    <div className="mt-4 space-y-3">
                                      {executionChecklist.length === 0 ? (
                                        <div className="rounded-[10px] border border-dashed border-slate-300 bg-white px-4 py-4 text-sm text-slate-500">
                                          {text.noChecklistItems}
                                        </div>
                                      ) : (
                                        executionChecklist.map((item) => (
                                          <div key={item.item_id} className="grid gap-3 rounded-[10px] border border-slate-200 bg-white p-3 sm:grid-cols-[132px_minmax(0,1fr)_auto]">
                                            <select
                                              value={item.status ?? 'pending'}
                                              onChange={(event) =>
                                                updateChecklistItem(item.item_id, {
                                                  status: (event.target.value as 'pending' | 'in_progress' | 'completed') || 'pending',
                                                })
                                              }
                                              className="rounded-[8px] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                            >
                                              <option value="pending">{text.pendingState}</option>
                                              <option value="in_progress">{text.inProgressState}</option>
                                              <option value="completed">{text.completedState}</option>
                                            </select>
                                            <input
                                              type="text"
                                              value={item.content}
                                              onChange={(event) =>
                                                updateChecklistItem(item.item_id, {
                                                  content: event.target.value,
                                                  active_form: event.target.value,
                                                })
                                              }
                                              placeholder={text.checklistItemPlaceholder}
                                              className="rounded-[8px] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                            />
                                            <button
                                              type="button"
                                              onClick={() => removeChecklistItem(item.item_id)}
                                              className="inline-flex items-center justify-center rounded-[8px] px-3 py-2 text-xs font-semibold text-rose-600 hover:bg-rose-50"
                                            >
                                              {text.remove}
                                            </button>
                                          </div>
                                        ))
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </section>

                              <section ref={projectProvidersSectionRef} className="rounded-[14px] border border-slate-200 bg-white p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.2)]">
                                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                  <ShieldCheck className="h-4 w-4 text-amber-600" />
                                  {text.projectProviders}
                                </div>
                                <div className="mt-1 text-sm text-slate-500">{text.projectProvidersDescription}</div>
                                <div className="mt-4 space-y-4">
                                  <ProviderSelect
                                    label={text.preferredProvider}
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
                                    emptyLabel={text.defaultSettingsModel}
                                  />
                                  <ProviderSelect
                                    label={text.fallbackProvider}
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
                                    emptyLabel={text.noneOption}
                                  />
                                  <div className="rounded-[12px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                                    {selectedAgent
                                      ? formatTemplate(text.selectedNodeHint, { name: selectedAgent.name })
                                      : text.selectNodeToEdit}
                                  </div>
                                </div>
                              </section>

                              <section ref={projectMcpSectionRef} className="rounded-[14px] border border-slate-200 bg-white p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.2)]">
                                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                  <Server className="h-4 w-4 text-violet-600" />
                                  {text.projectMcpServers}
                                </div>
                                <div className="mt-1 text-sm text-slate-500">{text.projectMcpServersDescription}</div>
                                <div className="mt-4 space-y-3">
                                  <div className="rounded-[12px] bg-slate-50 p-4">
                                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">{text.projectMcpServers}</div>
                                    <div className="mt-2 text-sm font-semibold text-slate-900">
                                      {formatTemplate(text.mcpServerCountShort, { count: mcpServerCatalog.length })}
                                    </div>
                                  </div>
                                  {mcpServerCatalog.length > 0 ? (
                                    <div className="grid gap-3">
                                      {mcpServerCatalog.map((server) => (
                                        <div key={server.server_id} className="rounded-[12px] border border-slate-200 bg-slate-50 px-4 py-3">
                                          <div className="flex flex-wrap items-start justify-between gap-3">
                                            <div className="min-w-0 flex-1">
                                              <div className="flex flex-wrap items-center gap-2">
                                                <div className="text-sm font-semibold text-slate-900">{server.title}</div>
                                                <span className="rounded-full bg-white px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
                                                  {server.server_id}
                                                </span>
                                                <StatusPill value={server.status || 'enabled'} />
                                              </div>
                                              {server.description ? (
                                                <div className="mt-2 text-sm text-slate-600">{server.description}</div>
                                              ) : null}
                                              {server.command_preview ? (
                                                <div className="mt-3 rounded-[10px] border border-slate-200 bg-white px-3 py-3 text-xs text-slate-600">
                                                  <div className="font-semibold uppercase tracking-[0.16em] text-slate-400">{text.commandPreviewLabel}</div>
                                                  <div className="mt-2 break-all font-mono text-[11px] text-slate-800">{server.command_preview}</div>
                                                </div>
                                              ) : null}
                                            </div>
                                          </div>
                                        </div>
                                      ))}
                                    </div>
                                  ) : (
                                    <div className="rounded-[12px] border border-dashed border-slate-300 bg-slate-50 px-4 py-4 text-sm text-slate-500">
                                      {text.noProjectMcpServers}
                                    </div>
                                  )}
                                </div>
                              </section>

                              <section className="rounded-[14px] border border-slate-200 bg-white p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.2)]">
                                <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                  <Waypoints className="h-4 w-4 text-cyan-600" />
                                  {text.graphTab}
                                </div>
                                <div className="mt-4 space-y-4">
                                  <div className="grid gap-4 xl:grid-cols-2">
                                    <GraphOrchestrationBrief
                                      scopeLabel={text.runAllScopeLabel}
                                      previewLabel={text.runAllPreview}
                                      previewSteps={executionPreview.all}
                                      emptyLabel={text.addAgentsToPreview}
                                      summary={studioOrchestrationBriefSummary}
                                      reviewAgentName={graph?.review_agent?.name || text.defaultReviewAgentName}
                                      applyRoleProfilesLabel={
                                        graphRoleProfileSummary.actionableAgentCount > 0 ? text.applyAllRoleProfiles : null
                                      }
                                      requestRoleProfileSkillsLabel={
                                        graphRoleProfileSummary.missingSkillAgentCount > 0 ? text.requestAllRoleProfileSkills : null
                                      }
                                      applyPolicyRepairsLabel={
                                        graphPolicyRepairSummary.agentCount > 0 ? text.applyAllSuggestedPolicyRepairs : null
                                      }
                                      applyCollaborationFixesLabel={
                                        studioWeakEdgeCount > 0 || studioBestNextCount > 0
                                          ? text.applyAllSuggestedCollaborationFixes
                                          : null
                                      }
                                      isApplying={updateProjectMutation.isPending}
                                      isRequestingRoleProfileSkills={skillRequestMutation.isPending}
                                      onApplyRoleProfiles={() => handleApplyRoleProfilesForScope('all')}
                                      onRequestRoleProfileSkills={() => void handleRequestRoleProfileSkillsForScope('all')}
                                      onApplyPolicyRepairs={() => handleApplyPolicyRepairsForScope('all')}
                                      onApplyCollaborationFixes={() => handleApplySuggestedCollaborationFixesForScope('all')}
                                      skillTitleById={skillTitleById}
                                      toolTitleById={toolTitleById}
                                      mcpServerTitleById={mcpServerTitleById}
                                      focusableAgentIds={focusableAgentIds}
                                      onFocusAgent={openNodeEditor}
                                    />
                                    <GraphOrchestrationBrief
                                      scopeLabel={text.runSelectedScopeLabel}
                                      previewLabel={text.runSelectedPreview}
                                      previewSteps={executionPreview.selected}
                                      emptyLabel={text.noSelectedRunScopeYet}
                                      summary={selectedScopeOrchestrationBriefSummary}
                                      reviewAgentName={graph?.review_agent?.name || text.defaultReviewAgentName}
                                      applyRoleProfilesLabel={
                                        effectiveSelectedAgentIdsForRun.length > 0 && selectedScopeRoleProfileSummary.actionableAgentCount > 0
                                          ? text.applySelectedRoleProfiles
                                          : null
                                      }
                                      requestRoleProfileSkillsLabel={
                                        effectiveSelectedAgentIdsForRun.length > 0 && selectedScopeRoleProfileSummary.missingSkillAgentCount > 0
                                          ? text.requestSelectedRoleProfileSkills
                                          : null
                                      }
                                      applyPolicyRepairsLabel={
                                        effectiveSelectedAgentIdsForRun.length > 0 && selectedScopePolicyRepairSummary.agentCount > 0
                                          ? text.applySelectedSuggestedPolicyRepairs
                                          : null
                                      }
                                      applyCollaborationFixesLabel={
                                        effectiveSelectedAgentIdsForRun.length > 0 && (selectedScopeWeakEdgeCount > 0 || selectedScopeBestNextCount > 0)
                                          ? text.applySelectedSuggestedCollaborationFixes
                                          : null
                                      }
                                      isApplying={updateProjectMutation.isPending}
                                      isRequestingRoleProfileSkills={skillRequestMutation.isPending}
                                      onApplyRoleProfiles={() => handleApplyRoleProfilesForScope('selected')}
                                      onRequestRoleProfileSkills={() => void handleRequestRoleProfileSkillsForScope('selected')}
                                      onApplyPolicyRepairs={() => handleApplyPolicyRepairsForScope('selected')}
                                      onApplyCollaborationFixes={() => handleApplySuggestedCollaborationFixesForScope('selected')}
                                      skillTitleById={skillTitleById}
                                      toolTitleById={toolTitleById}
                                      mcpServerTitleById={mcpServerTitleById}
                                      focusableAgentIds={focusableAgentIds}
                                      onFocusAgent={openNodeEditor}
                                    />
                                  </div>
                                  <div className="rounded-[12px] bg-slate-50 p-4">
                                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">{text.edges}</div>
                                    <div className="mt-3 space-y-2">
                                      {edges.length === 0 ? (
                                        <div className="text-sm text-slate-500">{text.noEdgesYet}</div>
                                      ) : (
                                        edges.map((edge) => {
                                          const diagnostic = edgeDelegationDiagnosticsById.get(edge.edge_id) ?? null;
                                          return (
                                            <div
                                              key={edge.edge_id}
                                              className={`flex items-center justify-between gap-3 rounded-[10px] px-3 py-3 text-sm ring-1 ${
                                                diagnostic?.fit === 'weak'
                                                  ? 'bg-amber-50/80 text-amber-950 ring-amber-200'
                                                  : diagnostic?.fit === 'strong'
                                                    ? 'bg-emerald-50/70 text-emerald-950 ring-emerald-200'
                                                    : 'bg-white text-slate-700 ring-slate-200'
                                              }`}
                                            >
                                              <div className="min-w-0">
                                                <div className="truncate">
                                                  <span className="font-medium text-slate-900">{agentNameById.get(edge.source_agent_id) ?? edge.source_agent_id}</span>
                                                  {' -> '}
                                                  <span className="font-medium text-slate-900">{agentNameById.get(edge.target_agent_id) ?? edge.target_agent_id}</span>
                                                </div>
                                                <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                                                  <span>{humanizeEdgeInteraction(edge.interaction, text)}</span>
                                                  {diagnostic ? (
                                                    <span className={`rounded-full px-2 py-0.5 font-semibold ring-1 ${delegationFitBadgeClass(diagnostic.fit)}`}>
                                                      {formatDelegationFitLabel(diagnostic.fit, text)}
                                                    </span>
                                                  ) : null}
                                                </div>
                                                {diagnostic?.fit === 'weak' && diagnostic.bestAlternative?.agent_name ? (
                                                  <div className="mt-2 text-xs text-amber-900">
                                                    {formatTemplate(text.tryCollaboratorHint, { name: diagnostic.bestAlternative.agent_name })}
                                                  </div>
                                                ) : null}
                                                {diagnostic?.rationale ? (
                                                  <div className="mt-2 text-xs leading-5 text-slate-600">{diagnostic.rationale}</div>
                                                ) : null}
                                              </div>
                                              <div className="flex shrink-0 flex-col items-end gap-2">
                                                {diagnostic?.fit === 'weak' && diagnostic.bestAlternative?.agent_id ? (
                                                  <button
                                                    type="button"
                                                    onClick={() =>
                                                      handleApplySuggestedRewire({
                                                        sourceAgentId: edge.source_agent_id,
                                                        fromTargetAgentId: edge.target_agent_id,
                                                        toTargetAgentId: diagnostic.bestAlternative?.agent_id || '',
                                                      })
                                                    }
                                                    disabled={updateProjectMutation.isPending}
                                                    className="rounded-[8px] border border-amber-300 bg-white px-3 py-2 text-xs font-semibold text-amber-900 hover:bg-amber-100 disabled:opacity-50"
                                                  >
                                                    {updateProjectMutation.isPending
                                                      ? text.applyingSuggestedRewire
                                                      : formatTemplate(text.applySuggestedRewire, {
                                                          name: diagnostic.bestAlternative.agent_name || text.unknownNode,
                                                        })}
                                                  </button>
                                                ) : null}
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
                                                  className="text-xs font-semibold text-rose-600 hover:text-rose-500"
                                                >
                                                  {text.remove}
                                                </button>
                                              </div>
                                            </div>
                                          );
                                        })
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </section>
                            </div>
                          ) : null}

                          {activeCanvasPanel === 'skills' ? (
                            <div className="space-y-4">
                              <div className="flex flex-wrap items-center justify-between gap-3 rounded-[16px] border border-slate-200 bg-slate-50 px-4 py-3">
                                <div>
                                  <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">{text.skillPool}</div>
                                  <div className="mt-1 text-sm text-slate-600">{text.skillPoolDialogDescription}</div>
                                </div>
                                <div className="flex flex-wrap items-center gap-2">
                                  <div className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-600 ring-1 ring-slate-200">
                                    {formatTemplate(text.approvedSkillsCount, { count: approvedSkillCount })}
                                  </div>
                                  <div className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-600 ring-1 ring-slate-200">
                                    {formatTemplate(text.pendingApprovalsCount, { count: pendingSkillApprovalCount })}
                                  </div>
                                  <div className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-600 ring-1 ring-slate-200">
                                    {formatTemplate(text.sourceSkillsCount, { count: sourceSkillPool.length })}
                                  </div>
                                </div>
                              </div>
                              <div className="rounded-[14px] border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600">
                                {selectedAssignableAgent
                                  ? formatTemplate(text.selectedNodeHint, { name: selectedAssignableAgent.name })
                                  : text.selectNonClusterAgentForSkills}
                                {selectedAssignableAgent && !canRequestSourceSkills ? (
                                  <div className="mt-2 rounded-[10px] border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                                    {text.saveStudioBeforeSkillRequests}
                                  </div>
                                ) : null}
                              </div>

                              <section className="rounded-[16px] border border-slate-200 bg-white p-4">
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                  <div>
                                    <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">{text.skillPool}</div>
                                    <div className="mt-1 text-sm font-semibold text-slate-900">
                                      {formatTemplate(text.approvedSkillsCount, { count: approvedSkillCount })}
                                    </div>
                                  </div>
                                  <div className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                                    {formatTemplate(text.skillsCountShort, { count: approvedSkillCount })}
                                  </div>
                                </div>
                                <div className="mt-4 space-y-2">
                                  {approvedSkillPool.length > 0 ? (
                                    approvedSkillPool.map((skill) => {
                                      const content = (
                                        <>
                                          <div className="min-w-0 flex-1">
                                            <div className="flex flex-wrap items-center gap-2">
                                              <span className={`inline-flex h-2.5 w-2.5 rounded-full ${skill.used ? 'bg-cyan-500' : 'bg-slate-300'}`} />
                                              <span className={`line-clamp-1 text-sm font-semibold ${skill.used ? 'text-slate-900' : 'text-slate-700'}`}>
                                                {skill.title}
                                              </span>
                                              <StatusPill value="approved" />
                                            </div>
                                            <div className="mt-1 text-xs text-slate-500">{skill.description || skill.source}</div>
                                            {skill.agentNames.length > 0 ? (
                                              <div className="mt-2 line-clamp-1 text-[11px] text-cyan-700">{skill.agentNames.join(', ')}</div>
                                            ) : null}
                                          </div>
                                          <div className={`rounded-full px-2 py-1 text-[10px] font-semibold ${
                                            skill.used ? 'bg-cyan-50 text-cyan-700' : 'bg-slate-100 text-slate-500'
                                          }`}>
                                            {skill.usageCount > 0 ? skill.usageCount : text.none}
                                          </div>
                                        </>
                                      );

                                      if (!selectedAssignableAgent) {
                                        return (
                                          <div
                                            key={skill.skill_id}
                                            className={`flex items-start gap-3 rounded-[12px] border px-3 py-3 ${
                                              skill.used ? 'border-cyan-200 bg-cyan-50/70' : 'border-slate-200 bg-slate-50'
                                            }`}
                                          >
                                            {content}
                                          </div>
                                        );
                                      }

                                      return (
                                        <label
                                          key={skill.skill_id}
                                          className={`flex cursor-pointer items-start gap-3 rounded-[12px] border px-3 py-3 ${
                                            skill.used || skill.assigned ? 'border-cyan-200 bg-cyan-50/70' : 'border-slate-200 bg-slate-50'
                                          }`}
                                        >
                                          <input
                                            type="checkbox"
                                            checked={skill.assigned}
                                            onChange={() => toggleSkillAssignment(skill.skill_id)}
                                            className="mt-1 h-4 w-4 rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                                          />
                                          {content}
                                        </label>
                                      );
                                    })
                                  ) : (
                                    <div className="rounded-[12px] border border-dashed border-slate-300 bg-slate-50 px-3 py-4 text-sm text-slate-500">
                                      {text.noApprovedSkills}
                                    </div>
                                  )}
                                </div>
                              </section>

                              <section className="rounded-[16px] border border-slate-200 bg-white p-4">
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                  <div>
                                    <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">{text.pendingSkillApprovals}</div>
                                    <div className="mt-1 text-sm font-semibold text-slate-900">
                                      {formatTemplate(text.pendingApprovalsCount, { count: pendingSkillApprovalCount })}
                                    </div>
                                  </div>
                                </div>
                                <div className="mt-4 space-y-3">
                                  {pendingSkillRequests.length > 0 ? (
                                    pendingSkillRequests.map((request) => (
                                      <div key={request.request_id} className="rounded-[12px] border border-amber-200 bg-amber-50/70 px-4 py-3">
                                        <div className="flex flex-wrap items-start justify-between gap-3">
                                          <div className="min-w-0 flex-1">
                                            <div className="flex flex-wrap items-center gap-2">
                                              <div className="text-sm font-semibold text-slate-900">{request.title}</div>
                                              <StatusPill value={request.status || 'pending'} />
                                            </div>
                                            <div className="mt-1 text-xs text-slate-500">{request.source}</div>
                                            <div className="mt-2 text-[11px] text-amber-900">
                                              {formatTemplate(text.pendingForAgent, {
                                                name: agentNameById.get(request.agent_id) ?? request.agent_id,
                                              })}
                                            </div>
                                            {request.reason ? (
                                              <div className="mt-2 rounded-[10px] border border-amber-200 bg-white/80 px-3 py-2 text-xs text-amber-900">
                                                {request.reason}
                                              </div>
                                            ) : null}
                                          </div>
                                          <div className="flex items-center gap-2">
                                            <button
                                              type="button"
                                              onClick={() => handleSkillDecision(request.request_id, true)}
                                              disabled={skillDecisionMutation.isPending}
                                              className="rounded-[10px] bg-emerald-600 px-3 py-2 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
                                            >
                                              {text.approve}
                                            </button>
                                            <button
                                              type="button"
                                              onClick={() => handleSkillDecision(request.request_id, false)}
                                              disabled={skillDecisionMutation.isPending}
                                              className="rounded-[10px] border border-rose-200 bg-white px-3 py-2 text-xs font-semibold text-rose-700 hover:bg-rose-50 disabled:opacity-50"
                                            >
                                              {text.reject}
                                            </button>
                                          </div>
                                        </div>
                                      </div>
                                    ))
                                  ) : (
                                    <div className="rounded-[12px] border border-dashed border-slate-300 bg-slate-50 px-3 py-4 text-sm text-slate-500">
                                      {text.noPendingSkillRequests}
                                    </div>
                                  )}
                                </div>
                              </section>

                              <section className="rounded-[16px] border border-slate-200 bg-white p-4">
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                  <div>
                                    <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">{text.availableSourceSkills}</div>
                                    <div className="mt-1 text-sm font-semibold text-slate-900">
                                      {formatTemplate(text.sourceSkillsCount, { count: sourceSkillPool.length })}
                                    </div>
                                  </div>
                                  <div className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                                    {formatTemplate(text.sourceSkillsCount, { count: sourceSkillPool.length })}
                                  </div>
                                </div>
                                <div className="mt-4 space-y-3">
                                  {sourceSkillPool.length > 0 ? (
                                    sourceSkillPool.map((skill) => {
                                      const pendingAgentName = skill.pendingRequest
                                        ? agentNameById.get(skill.pendingRequest.agent_id) ?? skill.pendingRequest.agent_id
                                        : null;
                                      const requestable =
                                        Boolean(selectedAssignableAgent) &&
                                        canRequestSourceSkills &&
                                        skill.displayStatus !== 'pending';
                                      return (
                                        <div
                                          key={skill.skill_id}
                                          className={`rounded-[12px] border px-4 py-3 ${
                                            skill.assigned
                                              ? 'border-cyan-200 bg-cyan-50/70'
                                              : skill.displayStatus === 'pending'
                                                ? 'border-amber-200 bg-amber-50/70'
                                                : skill.displayStatus === 'rejected'
                                                  ? 'border-rose-200 bg-rose-50/70'
                                                  : 'border-slate-200 bg-slate-50'
                                          }`}
                                        >
                                          <div className="flex flex-wrap items-start justify-between gap-3">
                                            <div className="min-w-0 flex-1">
                                              <div className="flex flex-wrap items-center gap-2">
                                                <div className="text-sm font-semibold text-slate-900">{skill.title}</div>
                                                <StatusPill value={skill.displayStatus} />
                                                {skill.assigned ? (
                                                  <span className="rounded-full bg-cyan-100 px-2 py-1 text-[10px] font-semibold text-cyan-800">
                                                    {text.currentAssignmentState}
                                                  </span>
                                                ) : null}
                                              </div>
                                              <div className="mt-1 text-xs text-slate-500">{skill.description || skill.source}</div>
                                              {pendingAgentName ? (
                                                <div className="mt-2 text-[11px] text-amber-900">
                                                  {formatTemplate(text.pendingForAgent, { name: pendingAgentName })}
                                                </div>
                                              ) : null}
                                              {skill.selectedRequest?.reason ? (
                                                <div className="mt-2 rounded-[10px] border border-slate-200 bg-white/80 px-3 py-2 text-xs text-slate-600">
                                                  {skill.selectedRequest.reason}
                                                </div>
                                              ) : null}
                                            </div>
                                            <div className="flex items-center gap-2">
                                              {skill.assigned ? (
                                                <button
                                                  type="button"
                                                  onClick={() => toggleSkillAssignment(skill.skill_id)}
                                                  className="rounded-[10px] border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                                                >
                                                  {text.remove}
                                                </button>
                                              ) : null}
                                              {requestable ? (
                                                <button
                                                  type="button"
                                                  onClick={() => handleRequestSkill(skill.skill_id)}
                                                  disabled={skillRequestMutation.isPending}
                                                  className="rounded-[10px] bg-slate-950 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
                                                >
                                                  {skillRequestMutation.isPending ? text.requesting : text.requestThisSkill}
                                                </button>
                                              ) : null}
                                            </div>
                                          </div>
                                        </div>
                                      );
                                    })
                                  ) : (
                                    <div className="rounded-[12px] border border-dashed border-slate-300 bg-slate-50 px-3 py-4 text-sm text-slate-500">
                                      {text.noSkillIntentsAvailable}
                                    </div>
                                  )}
                                </div>
                              </section>
                            </div>
                          ) : null}

                          {activeCanvasPanel === 'runs' ? (
                            <div className="space-y-4">
                              <div className="flex flex-wrap items-center justify-between gap-3">
                                <div className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                                  {selectedRun ? selectedRun.run_id : text.drawerCollapsedHint}
                                </div>
                                <div className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-slate-600">
                                  {formatTemplate(text.showingRunsCount, { visible: filteredRuns.length, total: sortedRuns.length })}
                                </div>
                              </div>
                              {runQueueSummary.length > 0 ? (
                                <div className="rounded-[20px] border border-slate-200 bg-white p-4">
                                  <div className="flex flex-wrap items-start justify-between gap-3">
                                    <div>
                                      <div className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                                        {text.runQueueSummary}
                                      </div>
                                      <div className="mt-1 text-sm text-slate-600">{text.runQueueSummaryDescription}</div>
                                    </div>
                                    <div className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                                      {formatTemplate(text.runsCount, { count: sortedRuns.length })}
                                    </div>
                                  </div>
                                  <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                                    {runQueueSummary.map((item) => (
                                      <button
                                        key={item.id}
                                        type="button"
                                        onClick={() => handleJumpToRunSection(item.id)}
                                        className={`rounded-[16px] border px-4 py-4 text-left transition hover:shadow-[0_18px_40px_-36px_rgba(15,23,42,0.45)] ${item.tone}`}
                                      >
                                        <div className="flex items-center justify-between gap-3">
                                          <div className="text-[11px] font-semibold uppercase tracking-[0.18em]">
                                            {item.label}
                                          </div>
                                          <div className="rounded-full bg-white/85 px-3 py-1 text-sm font-semibold ring-1 ring-black/5">
                                            {item.count}
                                          </div>
                                        </div>
                                        <div className="mt-2 text-sm opacity-85">{item.caption}</div>
                                        <div className="mt-3 text-xs font-semibold opacity-80">{text.jumpToRunGroup}</div>
                                      </button>
                                    ))}
                                  </div>
                                </div>
                              ) : null}
                              <div className="flex flex-wrap gap-2">
                                {runFilterOptions.map((option) => (
                                  <button
                                    key={option.id}
                                    type="button"
                                    onClick={() => setRunFilter(option.id)}
                                    className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold ring-1 ring-inset transition ${
                                      runFilter === option.id
                                        ? 'bg-cyan-50 text-cyan-900 ring-cyan-200'
                                        : 'bg-white text-slate-600 ring-slate-200 hover:bg-slate-50'
                                    }`}
                                  >
                                    <span>{option.label}</span>
                                    <span className={`rounded-full px-2 py-0.5 text-[10px] ${
                                      runFilter === option.id ? 'bg-white/80 text-cyan-900' : 'bg-slate-100 text-slate-600'
                                    }`}>
                                      {option.count}
                                    </span>
                                  </button>
                                ))}
                              </div>
                              <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
                                <div className="space-y-3 xl:overflow-y-auto">
                                  {sortedRuns.length === 0 ? (
                                    <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-8 text-sm text-slate-500">
                                      {text.noHarnessRunsYet}
                                    </div>
                                  ) : filteredRuns.length === 0 ? (
                                    <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-8 text-sm text-slate-500">
                                      {text.noRunsMatchFilter}
                                    </div>
                                  ) : (
                                    runQueueSections.map((section) => (
                                      <div
                                        key={section.id}
                                        ref={(node) => {
                                          runSectionRefs.current[section.id] = node;
                                        }}
                                        className="space-y-3"
                                      >
                                        <div className={`rounded-[16px] border px-4 py-3 ${section.tone}`}>
                                          <div className="flex items-center justify-between gap-3">
                                            <div className="min-w-0">
                                              <div className="text-[11px] font-semibold uppercase tracking-[0.2em]">
                                                {section.label}
                                              </div>
                                              <div className="mt-1 text-xs opacity-80">{section.description}</div>
                                            </div>
                                            <div className="rounded-full bg-white/80 px-3 py-1 text-[11px] font-semibold ring-1 ring-black/5">
                                              {section.runs.length}
                                            </div>
                                          </div>
                                        </div>
                                        {section.runs.map((run) => (
                                          <RunRow
                                            key={run.run_id}
                                            run={run}
                                            selected={run.run_id === activeRunId}
                                            onSelect={() => handleSelectRun(run.run_id)}
                                          />
                                        ))}
                                      </div>
                                    ))
                                  )}
                                </div>
                                <div className="space-y-6">
                                  {!selectedRun ? (
                                    <div className="rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-8 text-sm text-slate-500">
                                      {sortedRuns.length > 0 && filteredRuns.length === 0 ? text.noRunsMatchFilter : text.selectRunHint}
                                    </div>
                                  ) : (
                                    <>
                                      <div className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
                                        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                                          <div>
                                            <div className="text-xs uppercase tracking-[0.24em] text-slate-400">{text.runDetail}</div>
                                            <h3 className="mt-2 text-xl font-semibold text-slate-950">
                                              {humanizeHarnessValue(selectedRun.task_type || 'unknown_task', text)}
                                            </h3>
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
                                            <dt className="text-slate-500">{text.currentStep}</dt>
                                            <dd className="mt-1 font-medium text-slate-900">{humanizeHarnessValue(selectedRun.current_step || 'idle', text)}</dd>
                                          </div>
                                          <div className="rounded-2xl bg-white p-4">
                                            <dt className="text-slate-500">{text.retryBudget}</dt>
                                            <dd className="mt-1 font-medium text-slate-900">{selectedRun.policy?.retry_budget ?? 0}</dd>
                                          </div>
                                          <div className="rounded-2xl bg-white p-4">
                                            <dt className="text-slate-500">{text.recoveryModeLabel}</dt>
                                            <dd className="mt-1 font-medium text-slate-900">{humanizeRecoveryMode(recoveryMode, text)}</dd>
                                          </div>
                                        </dl>
                                        {pendingRunApproval ? (
                                          <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 p-4">
                                            <div className="flex flex-wrap items-start justify-between gap-3">
                                              <div className="min-w-0 flex-1">
                                                <div className="flex flex-wrap items-center gap-2">
                                                  <div className="text-sm font-semibold text-amber-950">
                                                    {humanizeHarnessValue(pendingRunApproval.action_type || 'approval', text)}
                                                  </div>
                                                  <StatusPill value={pendingRunApproval.status} />
                                                </div>
                                                <div className="mt-2 text-sm leading-6 text-amber-900">
                                                  {pendingRunApproval.reason || text.legacyApprovalCheckpoint}
                                                </div>
                                              </div>
                                            </div>
                                            <label className="mt-4 block text-sm font-medium text-amber-950">
                                              {text.optionalDecisionContext}
                                              <textarea
                                                rows={3}
                                                value={approvalComment}
                                                onChange={(event) => {
                                                  if (!pendingRunApprovalKey) {
                                                    return;
                                                  }
                                                  const nextValue = event.target.value;
                                                  setApprovalCommentsByKey((current) => ({
                                                    ...current,
                                                    [pendingRunApprovalKey]: nextValue,
                                                  }));
                                                }}
                                                placeholder={text.approvalCommentPlaceholder}
                                                className="mt-2 w-full rounded-2xl border border-amber-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                                              />
                                            </label>
                                            <div className="mt-4 flex flex-wrap gap-3">
                                              <button
                                                type="button"
                                                onClick={() => handleResolveRunApproval(true)}
                                                disabled={approvalMutation.isPending}
                                                className="inline-flex items-center justify-center rounded-2xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-50"
                                              >
                                                {approvalMutation.isPending ? text.resolvingApproval : text.approve}
                                              </button>
                                              <button
                                                type="button"
                                                onClick={() => handleResolveRunApproval(false)}
                                                disabled={approvalMutation.isPending}
                                                className="inline-flex items-center justify-center rounded-2xl border border-rose-200 bg-white px-4 py-3 text-sm font-semibold text-rose-700 hover:bg-rose-50 disabled:opacity-50"
                                              >
                                                {approvalMutation.isPending ? text.resolvingApproval : text.reject}
                                              </button>
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
                                                ? text.starting
                                                : rejectedReviewApproval
                                                  ? rejectedReviewStage === 'cluster_research'
                                                    ? text.continueWithoutResearch
                                                    : text.continueFromRollback
                                                  : text.retryRun}
                                            </button>
                                          </div>
                                        ) : null}
                                      </div>

                                      <div className="rounded-3xl border border-slate-200 bg-white p-5">
                                        <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                          <Sparkles className="h-4 w-4 text-sky-600" />
                                          {text.recoveryGuide}
                                        </div>
                                        <RunRecoveryGuideCard
                                          run={selectedRun}
                                          blockedAgents={selectedRunBlockedAgents}
                                          availabilityDiagnostics={selectedRunCapabilityAvailability}
                                          readinessDiagnostics={selectedRunCapabilityReadiness}
                                          collaborationContractDiagnostics={selectedRunCollaborationContracts}
                                          handoffDiagnostics={selectedRunHandoffDiagnostics}
                                          policyRepairSummary={selectedRunPolicyRepairSummary}
                                          roleProfileSummary={selectedRunRoleProfileSummary}
                                          recoveryMode={recoveryMode}
                                          graphAgentsById={agentById}
                                          canFocusAgent={(agentId) => focusableAgentIds.has(agentId)}
                                          onFocusAgent={openNodeEditor}
                                          onOpenSkillPool={() => {
                                            setPendingCreationKind(null);
                                            setActiveCanvasPanel('skills');
                                          }}
                                          onOpenProjectProviders={() => focusControlPanelSection('providers')}
                                          onOpenProjectMcpInventory={() => focusControlPanelSection('mcp')}
                                          onApplyRewire={handleApplySuggestedRewire}
                                          onApplySuggestedHandoff={handleApplySuggestedHandoff}
                                          onApplyAllCollaborationFixes={handleApplyRunRecoveryCollaborationFixes}
                                          onApplyRoleProfiles={handleApplyRunRoleProfiles}
                                          onApplyPolicyRepairs={handleApplySelectedRunPolicyRepairs}
                                          onApplyAgentPolicyRepair={handleApplyAgentCapabilityPolicySuggestions}
                                          onRequestRoleProfileSkills={handleRequestRunRoleProfileSkills}
                                          onRetry={handleRetry}
                                          isRetrying={retryRunMutation.isPending}
                                          isApplyingGraphChange={updateProjectMutation.isPending}
                                          isRequestingSkills={skillRequestMutation.isPending}
                                          StatusPillComponent={StatusPill}
                                        />
                                      </div>

                                      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
                                        <div className="rounded-3xl border border-slate-200 bg-white p-5">
                                          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                            <Waypoints className="h-4 w-4 text-cyan-600" />
                                            {text.executionProgress}
                                          </div>
                                          <WorkflowProgressCard workflow={selectedRun.workflow_progress} />
                                        </div>
                                        <div className="rounded-3xl border border-slate-200 bg-white p-5">
                                          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                                            {text.checklistSnapshot}
                                          </div>
                                          <ChecklistSnapshotCard checklist={selectedRunChecklist} />
                                        </div>
                                      </div>

                                      <div className="grid gap-6 lg:grid-cols-2">
                                        <div className="rounded-3xl border border-slate-200 bg-white p-5">
                                          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                            <ShieldCheck className="h-4 w-4 text-amber-600" />
                                            {text.approvalState}
                                          </div>
                                          <div className="mt-5">
                                            <ApprovalSummary approval={selectedRun.latest_approval} events={selectedRun.events} />
                                          </div>
                                        </div>

                                        <div className="rounded-3xl border border-slate-200 bg-white p-5">
                                          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                                            {text.verificationSummary}
                                          </div>
                                          {!selectedRun.latest_verification ? (
                                            <div className="mt-5 text-sm text-slate-500">{text.noVerificationYet}</div>
                                          ) : (
                                            <div className="mt-5 space-y-4">
                                              <div className="flex items-center justify-between gap-3">
                                                <div className="text-sm font-semibold text-slate-900">
                                                  {selectedRun.latest_verification.summary || text.verificationRecorded}
                                                </div>
                                                <StatusPill value={selectedRun.latest_verification.status} />
                                              </div>
                                              <dl className="grid gap-3 text-sm text-slate-700 sm:grid-cols-2">
                                                <div>
                                                  <dt className="text-slate-500">{text.recordedLabel}</dt>
                                                  <dd className="mt-1">{formatTimestamp(selectedRun.latest_verification.created_at, text.notRecorded)}</dd>
                                                </div>
                                                <div>
                                                  <dt className="text-slate-500">{text.checksLabel}</dt>
                                                  <dd className="mt-1">{selectedRun.latest_verification.checks_json?.checks_run?.join(', ') || text.none}</dd>
                                                </div>
                                              </dl>
                                              <div className="grid gap-4 xl:grid-cols-3">
                                                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                                                    {text.capabilityGaps}
                                                  </div>
                                                  <CapabilityGapCard
                                                    blockedAgents={selectedRunBlockedAgents}
                                                    skillTitleById={skillTitleById}
                                                    toolTitleById={toolTitleById}
                                                    canFocusAgent={(agentId) => focusableAgentIds.has(agentId)}
                                                    onFocusAgent={openNodeEditor}
                                                    onOpenSkillPool={() => {
                                                      setPendingCreationKind(null);
                                                      setActiveCanvasPanel('skills');
                                                    }}
                                                  />
                                                </div>
                                                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                                                    {text.handoffArtifacts}
                                                  </div>
                                                  <HandoffArtifactsCard artifacts={selectedRunHandoffArtifacts} />
                                                </div>
                                                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                                                    {text.handoffDiagnostics}
                                                  </div>
                                                  <HandoffDiagnosticsCard
                                                    diagnostics={selectedRunHandoffDiagnostics}
                                                    onApplyRewire={handleApplySuggestedRewire}
                                                    onApplySuggestedHandoff={handleApplySuggestedHandoff}
                                                    isApplying={updateProjectMutation.isPending}
                                                  />
                                                </div>
                                              </div>
                                              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                                                  {text.capabilitySnapshot}
                                                </div>
                                                <CapabilitySnapshotCard
                                                  snapshot={selectedRunCapabilitySnapshot}
                                                  skillTitleById={skillTitleById}
                                                  toolTitleById={toolTitleById}
                                                  mcpServerTitleById={mcpServerTitleById}
                                                />
                                              </div>
                                              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                                                  {text.capabilityExecutionContractsEvidence}
                                                </div>
                                                <CapabilityExecutionContractEvidenceCard
                                                  diagnostics={selectedRunExecutionContracts}
                                                  canFocusAgent={(agentId) => focusableAgentIds.has(agentId)}
                                                  onFocusAgent={openNodeEditor}
                                                  onOpenProjectProviders={() => focusControlPanelSection('providers')}
                                                  onOpenProjectMcpInventory={() => focusControlPanelSection('mcp')}
                                                />
                                              </div>
                                              <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                                                  {text.collaborationContractsEvidence}
                                                </div>
                                                <CollaborationContractEvidenceCard
                                                  diagnostics={selectedRunCollaborationContracts}
                                                  policyRepairSummary={selectedRunPolicyRepairSummary}
                                                  roleProfileSummary={selectedRunRoleProfileSummary}
                                                  canFocusAgent={(agentId) => focusableAgentIds.has(agentId)}
                                                  onFocusAgent={openNodeEditor}
                                                  onOpenSkillPool={() => {
                                                    setPendingCreationKind(null);
                                                    setActiveCanvasPanel('skills');
                                                  }}
                                                  onApplyAgentPolicyRepair={handleApplyAgentCapabilityPolicySuggestions}
                                                  onApplySuggestedHandoff={handleApplySuggestedHandoff}
                                                  isApplyingGraphChange={updateProjectMutation.isPending}
                                                />
                                              </div>
                                              <div className="grid gap-4 xl:grid-cols-2">
                                                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                                                    {text.capabilityAvailabilityEvidence}
                                                  </div>
                                                  <CapabilityAvailabilityEvidenceCard
                                                    diagnostics={selectedRunCapabilityAvailability}
                                                    graphAgentsById={agentById}
                                                    canFocusAgent={(agentId) => focusableAgentIds.has(agentId)}
                                                    onFocusAgent={openNodeEditor}
                                                    onOpenSkillPool={() => {
                                                      setPendingCreationKind(null);
                                                      setActiveCanvasPanel('skills');
                                                    }}
                                                    onOpenProjectProviders={() => focusControlPanelSection('providers')}
                                                    onOpenProjectMcpInventory={() => focusControlPanelSection('mcp')}
                                                    onApplyAgentPolicyRepair={handleApplyAgentCapabilityPolicySuggestions}
                                                    isApplyingPolicyRepair={updateProjectMutation.isPending}
                                                  />
                                                </div>
                                                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                                                    {text.capabilityReadinessEvidence}
                                                  </div>
                                                  <CapabilityReadinessEvidenceCard
                                                    diagnostics={selectedRunCapabilityReadiness}
                                                    graphAgentsById={agentById}
                                                    canFocusAgent={(agentId) => focusableAgentIds.has(agentId)}
                                                    onFocusAgent={openNodeEditor}
                                                    onOpenSkillPool={() => {
                                                      setPendingCreationKind(null);
                                                      setActiveCanvasPanel('skills');
                                                    }}
                                                    onOpenProjectProviders={() => focusControlPanelSection('providers')}
                                                    onOpenProjectMcpInventory={() => focusControlPanelSection('mcp')}
                                                    onApplyAgentPolicyRepair={handleApplyAgentCapabilityPolicySuggestions}
                                                    isApplyingPolicyRepair={updateProjectMutation.isPending}
                                                  />
                                                </div>
                                              </div>
                                              <details className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                                <summary className="cursor-pointer text-sm font-medium text-slate-800">{text.rawVerificationArtifacts}</summary>
                                                <div className="mt-3">
                                                  <JsonBlock value={selectedRun.latest_verification.artifacts_json} />
                                                </div>
                                              </details>
                                            </div>
                                          )}
                                        </div>
                                      </div>

                                      <div className="grid gap-6 lg:grid-cols-2">
                                        <div className="rounded-3xl border border-slate-200 bg-white p-5">
                                          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                            <Clock3 className="h-4 w-4 text-sky-600" />
                                            {text.inputPayload}
                                          </div>
                                          <div className="mt-5">
                                            <JsonBlock value={selectedRun.input_json} />
                                          </div>
                                        </div>
                                        <div className="rounded-3xl border border-slate-200 bg-white p-5">
                                          <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                            <Sparkles className="h-4 w-4 text-fuchsia-600" />
                                            {text.metadataPanel}
                                          </div>
                                          <div className="mt-5 space-y-4">
                                            <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                                              <RunPreflightSummaryCard metadata={selectedRunMetadata} />
                                            </div>
                                            <JsonBlock value={selectedRun.metadata_json || null} />
                                          </div>
                                        </div>
                                      </div>

                                      <div className="rounded-3xl border border-slate-200 bg-white p-5">
                                        <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                                          <Workflow className="h-4 w-4 text-cyan-600" />
                                          {text.eventTimeline}
                                        </div>
                                        {!selectedRun.events || selectedRun.events.length === 0 ? (
                                          <div className="mt-5 text-sm text-slate-500">{text.noEventEvidence}</div>
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
                          ) : null}

                          {activeCanvasPanel === 'review' ? (
                            <div className="space-y-4">
                              <label className="flex items-center gap-3 rounded-[12px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-800">
                                <input
                                  type="checkbox"
                                  checked={graph?.review_agent?.enabled ?? true}
                                  onChange={(event) =>
                                    updateDraftProject((current) => ({
                                      ...current,
                                      graph_json: {
                                        ...current.graph_json,
                                        review_agent: {
                                          enabled: event.target.checked,
                                          hidden: current.graph_json.review_agent?.hidden ?? true,
                                          name: current.graph_json.review_agent?.name ?? text.defaultReviewAgentName,
                                          model: current.graph_json.review_agent?.model ?? 'gpt-5.1-codex-mini',
                                          preferred_provider_id: current.graph_json.review_agent?.preferred_provider_id ?? null,
                                          fallback_provider_id: current.graph_json.review_agent?.fallback_provider_id ?? null,
                                          system_prompt: current.graph_json.review_agent?.system_prompt ?? '',
                                        },
                                      },
                                    }))
                                  }
                                  className="h-4 w-4 rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                                />
                                {text.enabled}
                              </label>
                              <div className="rounded-[12px] border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
                                {text.hiddenReviewerHint}
                              </div>
                              <label className="block text-sm font-medium text-slate-800">
                                {text.nameLabel}
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
                                {text.modelLabel}
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
                                          name: current.graph_json.review_agent?.name ?? text.defaultReviewAgentName,
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
                                  label={text.preferredProvider}
                                  value={graph?.review_agent?.preferred_provider_id}
                                  onChange={(nextValue) =>
                                    updateDraftProject((current) => ({
                                      ...current,
                                      graph_json: {
                                        ...current.graph_json,
                                        review_agent: {
                                          enabled: current.graph_json.review_agent?.enabled ?? true,
                                          hidden: current.graph_json.review_agent?.hidden ?? true,
                                          name: current.graph_json.review_agent?.name ?? text.defaultReviewAgentName,
                                          model: current.graph_json.review_agent?.model ?? 'gpt-5.1-codex-mini',
                                          preferred_provider_id: nextValue,
                                          fallback_provider_id: current.graph_json.review_agent?.fallback_provider_id ?? null,
                                          system_prompt: current.graph_json.review_agent?.system_prompt ?? '',
                                        },
                                      },
                                    }))
                                  }
                                  providers={providerOptions}
                                  emptyLabel={text.projectPreferenceLabel}
                                />
                                <ProviderSelect
                                  label={text.fallbackProvider}
                                  value={graph?.review_agent?.fallback_provider_id}
                                  onChange={(nextValue) =>
                                    updateDraftProject((current) => ({
                                      ...current,
                                      graph_json: {
                                        ...current.graph_json,
                                        review_agent: {
                                          enabled: current.graph_json.review_agent?.enabled ?? true,
                                          hidden: current.graph_json.review_agent?.hidden ?? true,
                                          name: current.graph_json.review_agent?.name ?? text.defaultReviewAgentName,
                                          model: current.graph_json.review_agent?.model ?? 'gpt-5.1-codex-mini',
                                          preferred_provider_id: current.graph_json.review_agent?.preferred_provider_id ?? null,
                                          fallback_provider_id: nextValue,
                                          system_prompt: current.graph_json.review_agent?.system_prompt ?? '',
                                        },
                                      },
                                    }))
                                  }
                                  providers={providerOptions}
                                  emptyLabel={text.noneOption}
                                />
                              </div>
                              <label className="block text-sm font-medium text-slate-800">
                                {text.systemPrompt}
                                <textarea
                                  rows={6}
                                  value={graph?.review_agent?.system_prompt ?? ''}
                                  onChange={(event) =>
                                    updateDraftProject((current) => ({
                                      ...current,
                                      graph_json: {
                                        ...current.graph_json,
                                        review_agent: {
                                          enabled: current.graph_json.review_agent?.enabled ?? true,
                                          hidden: current.graph_json.review_agent?.hidden ?? true,
                                          name: current.graph_json.review_agent?.name ?? text.defaultReviewAgentName,
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
                          ) : null}
                        </div>
                      </div>
                      </div>
                    ) : null}

                    <div data-canvas-ui="true" className="absolute bottom-4 right-4 z-20 flex items-center gap-2 rounded-[14px] border border-slate-200 bg-white/96 px-3 py-2 shadow-[0_18px_40px_-34px_rgba(15,23,42,0.45)] backdrop-blur">
                      <button
                        type="button"
                        onClick={() => handleCanvasZoom(canvasZoom / 1.12)}
                        className="inline-flex h-8 w-8 items-center justify-center rounded-[8px] border border-slate-200 bg-slate-50 text-sm font-semibold text-slate-700 hover:bg-white"
                      >
                        -
                      </button>
                      <div className="min-w-[58px] text-center text-xs font-semibold text-slate-600">
                        {Math.round(canvasZoom * 100)}%
                      </div>
                      <button
                        type="button"
                        onClick={() => handleCanvasZoom(canvasZoom * 1.12)}
                        className="inline-flex h-8 w-8 items-center justify-center rounded-[8px] border border-slate-200 bg-slate-50 text-sm font-semibold text-slate-700 hover:bg-white"
                      >
                        +
                      </button>
                      <button
                        type="button"
                        onClick={handleFitCanvas}
                        className="inline-flex h-8 items-center justify-center rounded-[8px] border border-slate-200 bg-slate-50 px-3 text-xs font-semibold text-slate-700 hover:bg-white"
                      >
                        {text.fitView}
                      </button>
                    </div>

                    {showCanvasHint ? (
                      <div data-canvas-ui="true" className="pointer-events-none absolute inset-x-0 bottom-16 z-20 flex justify-center px-4">
                        <div className="pointer-events-auto max-w-3xl rounded-[16px] border border-slate-200 bg-white/96 px-4 py-3 text-xs leading-6 text-slate-600 shadow-[0_24px_80px_-46px_rgba(15,23,42,0.45)] backdrop-blur">
                          <div>{text.canvasNavigationHint}</div>
                          <div className="mt-1">
                            <span className="font-semibold">{text.immutableRuleLabel}</span> {text.immutableRuleBody}
                          </div>
                        </div>
                      </div>
                    ) : null}

                    <div
                      className="absolute inset-0"
                      style={{
                        transform: `translate(${canvasViewport.x}px, ${canvasViewport.y}px) scale(${canvasZoom})`,
                        transformOrigin: '0 0',
                      }}
                    >
                      <div
                        className="absolute"
                        style={{
                          left: CANVAS_WORLD_MIN_X,
                          top: CANVAS_WORLD_MIN_Y,
                          height: CANVAS_WORLD_HEIGHT,
                          width: CANVAS_WORLD_WIDTH,
                        }}
                      >
                        <svg className="pointer-events-none absolute inset-0 h-full w-full">
                          <defs>
                            <marker id="harness-canvas-arrow-strong" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(5,150,105,0.54)" />
                            </marker>
                            <marker id="harness-canvas-arrow-good" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(8,145,178,0.48)" />
                            </marker>
                            <marker id="harness-canvas-arrow-weak" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(217,119,6,0.58)" />
                            </marker>
                          </defs>
                          {edges.map((edge) => {
                            const source = agents.find((agent) => agent.agent_id === edge.source_agent_id);
                            const target = agents.find((agent) => agent.agent_id === edge.target_agent_id);
                            if (!source?.position || !target?.position) {
                              return null;
                            }
                            const x1 = source.position.x - CANVAS_WORLD_MIN_X + NODE_CENTER_X;
                            const y1 = source.position.y - CANVAS_WORLD_MIN_Y + NODE_CENTER_Y;
                            const x2 = target.position.x - CANVAS_WORLD_MIN_X + NODE_CENTER_X;
                            const y2 = target.position.y - CANVAS_WORLD_MIN_Y + NODE_CENTER_Y;
                            const bend = Math.max(48, Math.abs(x2 - x1) / 2);
                            const path = `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
                            const diagnostic = edgeDelegationDiagnosticsById.get(edge.edge_id) ?? null;
                            const fit = diagnostic?.fit ?? 'good';
                            const edgeLabel = diagnostic
                              ? `${humanizeEdgeInteraction(diagnostic.interaction, text)} · ${formatDelegationFitLabel(diagnostic.fit, text)}`
                              : humanizeEdgeInteraction(edge.interaction, text);
                            const alternateHint = diagnostic?.fit === 'weak' && diagnostic.bestAlternative?.agent_name
                              ? formatTemplate(text.tryCollaboratorHint, { name: diagnostic.bestAlternative.agent_name })
                              : null;
                            return (
                              <g key={edge.edge_id}>
                                <path
                                  d={path}
                                  fill="none"
                                  stroke={edgeStrokeColor(fit)}
                                  strokeWidth="3"
                                  strokeLinecap="round"
                                  markerEnd={`url(#${edgeMarkerId(fit)})`}
                                />
                                <text
                                  x={(x1 + x2) / 2}
                                  y={(y1 + y2) / 2 - 10}
                                  fill={edgeLabelColor(fit)}
                                  className="text-[10px] font-semibold uppercase tracking-[0.16em]"
                                  textAnchor="middle"
                                >
                                  <tspan x={(x1 + x2) / 2} dy="0">
                                    {edgeLabel}
                                  </tspan>
                                  {alternateHint ? (
                                    <tspan x={(x1 + x2) / 2} dy="12">
                                      {alternateHint}
                                    </tspan>
                                  ) : null}
                                </text>
                              </g>
                            );
                          })}
                        </svg>

                        {agents.map((agent) => (
                          <div
                            key={agent.agent_id}
                            data-node-card="true"
                            className={`absolute w-60 rounded-[18px] border bg-white/96 shadow-[0_22px_70px_-42px_rgba(15,23,42,0.45)] transition ${
                              effectiveSelectedAgentId === agent.agent_id
                                ? 'border-cyan-300 ring-4 ring-cyan-100'
                                : 'border-slate-200 hover:border-slate-300'
                            }`}
                            style={{
                              left: (agent.position?.x ?? 0) - CANVAS_WORLD_MIN_X,
                              top: (agent.position?.y ?? 0) - CANVAS_WORLD_MIN_Y,
                            }}
                          >
                        {effectiveSelectedAgentId === agent.agent_id ? (
                          <div className="pointer-events-none absolute inset-[-8px] rounded-[24px] border-2 border-dashed border-cyan-500 shadow-[0_0_0_8px_rgba(34,211,238,0.12)]" />
                        ) : null}
                        <button
                          type="button"
                          onPointerDown={(event) => {
                            event.stopPropagation();
                            startDragging(event, agent);
                          }}
                          onClick={() => setSelectedAgentId(agent.agent_id)}
                          className="w-full cursor-grab rounded-[18px] px-4 pb-3 pt-4 text-left active:cursor-grabbing"
                        >
                          {(() => {
                            const metaBadges =
                              agent.node_kind === 'cluster'
                                ? [
                                    `${agent.cluster_strategy === 'brainstorm' ? text.brainstorm : text.custom} ${text.clusterNodeLabel}`,
                                    agent.role || text.specialistRoleSeed,
                                    formatTemplate(text.membersCountShort, { count: agent.cluster_members?.length ?? 0 }),
                                  ]
                                : [
                                    text.agentNodeLabel,
                                    agent.role || text.specialistRoleSeed,
                                    (agent.skill_ids ?? []).length > 0
                                      ? formatTemplate(text.skillsCountShort, { count: agent.skill_ids?.length ?? 0 })
                                      : humanizeHarnessValue('no_skills', text),
                                  ];

                            return (
                          <div className="min-w-0">
                            <div className="flex items-start justify-between gap-2">
                              <div className="min-w-0">
                                <div className="line-clamp-1 text-sm font-semibold text-slate-950">{agent.name}</div>
                                <div className="mt-1 line-clamp-1 text-xs text-slate-500">{agent.model || 'gpt-5.2'}</div>
                              </div>
                            </div>
                            <div className="mt-3 grid grid-cols-3 gap-2">
                              {metaBadges.map((badge, index) => (
                                <span
                                  key={`${agent.agent_id}-meta-${index}`}
                                  className={`inline-flex h-8 min-w-0 items-center justify-center rounded-[12px] px-2 text-center text-[10px] font-semibold ${
                                    index === 2 && agent.node_kind === 'cluster'
                                      ? 'bg-amber-50 text-amber-900'
                                      : index === 2
                                        ? 'bg-sky-50 text-sky-900'
                                        : 'bg-slate-100 text-slate-600'
                                  }`}
                                >
                                  <span className="truncate">{badge}</span>
                                </span>
                              ))}
                            </div>
                          </div>
                            );
                          })()}
                        </button>
                        <div className="border-t border-slate-100 px-4 pb-4 pt-3">
                          <p className="min-h-[60px] text-xs leading-5 text-slate-600">{agent.description || text.noDescriptionYet}</p>
                          <div className="mt-3 min-h-[34px]">
                            {(() => {
                              const capabilitySummary = agentCapabilitySummaryById.get(agent.agent_id);
                              const availabilityStatus = resolveCapabilityAvailabilityStatus(capabilitySummary);
                              return agent.node_kind === 'cluster' ? (
                              <div className="flex flex-wrap gap-2">
                                <span className="rounded-full bg-white px-2.5 py-1 text-[10px] font-semibold text-slate-600 ring-1 ring-slate-200">
                                  {humanizeHarnessValue(agent.cluster_auto_research ? 'cluster_research' : 'cluster', text)}
                                </span>
                              </div>
                              ) : (agent.skill_ids ?? []).length === 0 && !(capabilitySummary?.enabled_tool_ids?.length) ? (
                              <div className="flex flex-wrap gap-2">
                                <span className="rounded-full bg-white px-2.5 py-1 text-[10px] font-semibold text-slate-600 ring-1 ring-slate-200">
                                  {humanizeHarnessValue('no_skills', text)}
                                </span>
                              </div>
                            ) : (
                              <div className="group relative flex flex-wrap gap-2">
                                {(agent.skill_ids ?? []).slice(0, 2).map((skillId) => (
                                  <span key={skillId} className="rounded-full bg-white px-2.5 py-1 text-[10px] font-semibold text-slate-600 ring-1 ring-slate-200">
                                    {formatSkillTitle(skillId, skillTitleById)}
                                  </span>
                                ))}
                                {(agent.skill_ids?.length ?? 0) > 2 ? (
                                  <>
                                    <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-600 ring-1 ring-slate-200">
                                      +{(agent.skill_ids?.length ?? 0) - 2}
                                    </span>
                                    <div className="pointer-events-none absolute bottom-full left-0 z-30 mb-2 w-full rounded-[14px] border border-white/15 bg-slate-950/72 px-3 py-3 text-xs text-white opacity-0 shadow-[0_18px_50px_-24px_rgba(15,23,42,0.7)] backdrop-blur-sm transition duration-150 group-hover:opacity-100">
                                      <div className="flex flex-wrap gap-2">
                                        {(agent.skill_ids ?? []).map((skillId) => (
                                          <span key={`${agent.agent_id}-tooltip-${skillId}`} className="rounded-full bg-white/12 px-2.5 py-1 text-[10px] font-semibold text-white ring-1 ring-white/15">
                                            {formatSkillTitle(skillId, skillTitleById)}
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  </>
                                ) : null}
                                {capabilitySummary?.enabled_tool_ids?.length ? (
                                  <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
                                    {formatTemplate(text.toolsCountShort, { count: capabilitySummary.enabled_tool_ids.length })}
                                  </span>
                                ) : null}
                                {capabilitySummary?.provider_limited_tool_ids?.length ? (
                                  <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
                                    {formatTemplate(text.providerLimitedToolsCountShort, { count: capabilitySummary.provider_limited_tool_ids.length })}
                                  </span>
                                ) : null}
                                {capabilitySummary && availabilityStatus !== 'available' ? (
                                  <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityAvailabilityBadgeClass(availabilityStatus)}`}>
                                    {formatCapabilityAvailabilityLabel(availabilityStatus, text)}
                                  </span>
                                ) : null}
                              </div>
                              );
                            })()}
                          </div>
                          <div className="mt-4 grid grid-cols-2 gap-2">
                            <button
                              type="button"
                              onClick={() => toggleRunSelection(agent.agent_id)}
                              className={`inline-flex items-center justify-center gap-1 rounded-[8px] px-3 py-2 text-xs font-semibold transition ${
                                effectiveSelectedAgentIdsForRun.includes(agent.agent_id)
                                  ? 'bg-sky-600 text-white'
                                  : 'border border-slate-200 bg-slate-50 text-slate-700 hover:bg-white'
                              }`}
                            >
                              <Play className="h-3.5 w-3.5" />
                              {effectiveSelectedAgentIdsForRun.includes(agent.agent_id)
                                ? text.selectedForRun
                                : text.selectForRun}
                            </button>
                            <button
                              type="button"
                              onClick={() => openNodeEditor(agent.agent_id)}
                              className="inline-flex items-center justify-center gap-1 rounded-[8px] border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-white"
                            >
                              <Sparkles className="h-3.5 w-3.5" />
                              {text.configure}
                            </button>
                            <button
                              type="button"
                              onClick={() => setConnectionSourceId(agent.agent_id)}
                              className={`inline-flex items-center justify-center gap-1 rounded-[8px] px-3 py-2 text-xs font-semibold transition ${
                                connectionSourceId === agent.agent_id
                                  ? 'bg-cyan-600 text-white'
                                  : 'border border-slate-200 bg-slate-50 text-slate-700 hover:bg-white'
                              }`}
                            >
                              <Link2 className="h-3.5 w-3.5" />
                              {text.linkOut}
                            </button>
                            <button
                              type="button"
                              onClick={() => handleConnectAgents(agent.agent_id)}
                              disabled={!connectionSourceId || connectionSourceId === agent.agent_id}
                              className="inline-flex items-center justify-center gap-1 rounded-[8px] border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-700 transition hover:bg-white disabled:opacity-50"
                            >
                              <Layers3 className="h-3.5 w-3.5" />
                              {text.linkIn}
                            </button>
                          </div>
                          <div className="mt-3 flex items-center justify-between gap-3">
                            <div className="text-[11px] text-slate-400">
                              {connectionSourceId === agent.agent_id ? text.linkModeHint : `${agent.agent_id.slice(0, 8)}`}
                            </div>
                            <button
                              type="button"
                              onClick={() => handleRemoveAgent(agent.agent_id)}
                              className="text-xs font-semibold text-rose-600 hover:text-rose-500"
                            >
                              {text.remove}
                            </button>
                          </div>
                        </div>
                          </div>
                        ))}

                        {agents.length === 0 ? (
                          <div
                            className="flex items-center justify-center px-6"
                            style={{
                              height: CANVAS_WORLD_HEIGHT,
                              width: CANVAS_WORLD_WIDTH,
                            }}
                          >
                            <div className="max-w-md rounded-[30px] border border-dashed border-slate-300 bg-white/86 px-8 py-8 text-center shadow-[0_18px_50px_-42px_rgba(15,23,42,0.35)]">
                              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-sky-100 text-sky-700">
                                <Workflow className="h-6 w-6" />
                              </div>
                              <div className="mt-5 text-lg font-semibold text-slate-950">{text.emptyCanvas}</div>
                              <div className="mt-3 text-sm leading-6 text-slate-500">{text.canvasFocusHint}</div>
                            </div>
                          </div>
                        ) : null}
                        </div>
                    </div>
                  </div>
                </div>
              </div>

            </section>


          <OverlayDialog
            open={isNodeEditorOpen}
            onClose={() => setIsNodeEditorOpen(false)}
            title={selectedAgent ? formatTemplate(text.nodeSettingsTitle, { name: selectedAgent.name }) : text.nodeSettings}
            description={text.nodeSettingsDescription}
          >
            {!selectedAgent ? (
              <div className="text-sm text-slate-500">{text.selectNodeToEdit}</div>
            ) : (
              <div className="space-y-4">
                <label className="block text-sm font-medium text-slate-800">
                  {text.nameLabel}
                  <input
                    value={selectedAgent.name}
                    onChange={(event) => updateSelectedAgent({ name: event.target.value })}
                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                  />
                </label>
                <label className="block text-sm font-medium text-slate-800">
                  {text.roleLabel}
                  <input
                    value={selectedAgent.role || ''}
                    onChange={(event) => updateSelectedAgent({ role: event.target.value })}
                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                  />
                </label>
                <label className="block text-sm font-medium text-slate-800">
                  {text.descriptionLabel}
                  <textarea
                    rows={3}
                    value={selectedAgent.description || ''}
                    onChange={(event) => updateSelectedAgent({ description: event.target.value })}
                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                  />
                </label>
                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="block text-sm font-medium text-slate-800">
                    {text.modelLabel}
                    <input
                      value={selectedAgent.model || ''}
                      onChange={(event) => updateSelectedAgent({ model: event.target.value })}
                      className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                    />
                  </label>
                  <ProviderSelect
                    label={text.preferredProvider}
                    value={selectedAgent.preferred_provider_id}
                    onChange={(nextValue) => updateSelectedAgent({ preferred_provider_id: nextValue })}
                    providers={providerOptions}
                    emptyLabel={text.projectPreferenceLabel}
                  />
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  <ProviderSelect
                    label={text.fallbackProvider}
                    value={selectedAgent.fallback_provider_id}
                    onChange={(nextValue) => updateSelectedAgent({ fallback_provider_id: nextValue })}
                    providers={providerOptions}
                    emptyLabel={text.noneOption}
                  />
                  <label className="block text-sm font-medium text-slate-800">
                    {text.maxIterations}
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
                        <div className="text-sm font-semibold text-slate-900">{text.clusterConfiguration}</div>
                        <div className="mt-1 text-xs text-slate-600">{text.clusterMembersHint}</div>
                      </div>
                      <button
                        type="button"
                        onClick={handleAddClusterMember}
                        className="inline-flex items-center gap-2 rounded-2xl border border-amber-200 bg-white px-3 py-2 text-xs font-semibold text-amber-900 hover:bg-amber-100"
                      >
                        <PlusCircle className="h-3.5 w-3.5" />
                        {text.addMember}
                      </button>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <label className="block text-sm font-medium text-slate-800">
                        {text.clusterStrategy}
                        <select
                          value={selectedAgent.cluster_strategy || 'custom'}
                          onChange={(event) =>
                            updateSelectedAgent({
                              cluster_strategy: (event.target.value as 'brainstorm' | 'custom') || 'custom',
                            })
                          }
                          className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                        >
                          <option value="brainstorm">{text.brainstorm}</option>
                          <option value="custom">{text.custom}</option>
                        </select>
                      </label>
                      <label className="block text-sm font-medium text-slate-800">
                        {text.debateRounds}
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
                        {text.autoResearchPass}
                      </label>
                      <label className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800">
                        <input
                          type="checkbox"
                          checked={selectedAgent.cluster_auto_review ?? true}
                          onChange={(event) => updateSelectedAgent({ cluster_auto_review: event.target.checked })}
                          className="h-4 w-4 rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                        />
                        {text.autoAttachReviewAgent}
                      </label>
                    </div>
                    <div className="space-y-3">
                      {(selectedAgent.cluster_members ?? []).map((member) => (
                        <div key={member.member_id} className="rounded-2xl border border-slate-200 bg-white p-4">
                          <div className="grid gap-3 sm:grid-cols-2">
                            <label className="block text-sm font-medium text-slate-800">
                              {text.memberName}
                              <input
                                value={member.name}
                                onChange={(event) => updateClusterMember(member.member_id, { name: event.target.value })}
                                className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                              />
                            </label>
                            <label className="block text-sm font-medium text-slate-800">
                              {text.roleLabel}
                              <input
                                value={member.role || ''}
                                onChange={(event) => updateClusterMember(member.member_id, { role: event.target.value })}
                                className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                              />
                            </label>
                          </div>
                          <div className="mt-3 grid gap-3 sm:grid-cols-2">
                            <label className="block text-sm font-medium text-slate-800">
                              {text.modelLabel}
                              <input
                                value={member.model || ''}
                                onChange={(event) => updateClusterMember(member.member_id, { model: event.target.value })}
                                className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                              />
                            </label>
                            <ProviderSelect
                              label={text.preferredProvider}
                              value={member.preferred_provider_id}
                              onChange={(nextValue) => updateClusterMember(member.member_id, { preferred_provider_id: nextValue })}
                              providers={providerOptions}
                              emptyLabel={text.clusterOrProjectPreference}
                            />
                          </div>
                          <div className="mt-3">
                            <label className="block text-sm font-medium text-slate-800">
                              {text.systemPrompt}
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
                ) : (
                  <>
                    <div className="space-y-4">
                      <div className="text-sm font-medium text-slate-800">{text.skillPool}</div>
                      {approvedSkillPool.length > 0 ? (
                        <div className="mt-3 grid gap-3 sm:grid-cols-2">
                          {approvedSkillPool.map((skill) => {
                            return (
                              <label
                                key={skill.skill_id}
                                className={`flex cursor-pointer items-start gap-3 rounded-2xl border px-4 py-3 transition ${
                                  skill.assigned
                                    ? 'border-cyan-200 bg-cyan-50/70'
                                    : 'border-slate-200 bg-white hover:bg-slate-50'
                                }`}
                              >
                                <input
                                  type="checkbox"
                                  checked={skill.assigned}
                                  onChange={() => toggleSkillAssignment(skill.skill_id)}
                                  className="mt-1 h-4 w-4 rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                                />
                                <span className="min-w-0 flex-1">
                                  <span className="flex flex-wrap items-center gap-2">
                                    <span className="block text-sm font-semibold text-slate-900">{skill.title}</span>
                                    <StatusPill value="approved" />
                                  </span>
                                  <span className="mt-1 block text-xs leading-5 text-slate-500">
                                    {skill.description || skill.source}
                                  </span>
                                </span>
                              </label>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="mt-3 rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-4 py-4 text-sm text-slate-500">
                          {text.noApprovedSkills}
                        </div>
                      )}
                    </div>
                    <div className="space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <div className="text-sm font-medium text-slate-800">{text.availableSourceSkills}</div>
                        <div className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-600">
                          {formatTemplate(text.sourceSkillsCount, { count: sourceSkillPool.length })}
                        </div>
                      </div>
                      {!canRequestSourceSkills ? (
                        <div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
                          {text.saveStudioBeforeSkillRequests}
                        </div>
                      ) : null}
                    {sourceSkillPool.length > 0 ? (
                        <div className="grid gap-3 sm:grid-cols-2">
                          {sourceSkillPool.map((skill) => {
                            const pendingAgentName = skill.pendingRequest
                              ? agentNameById.get(skill.pendingRequest.agent_id) ?? skill.pendingRequest.agent_id
                              : null;
                            const requestable = canRequestSourceSkills && skill.displayStatus !== 'pending';
                            return (
                              <div
                                key={skill.skill_id}
                                className={`rounded-2xl border px-4 py-3 ${
                                  skill.assigned
                                    ? 'border-cyan-200 bg-cyan-50/70'
                                    : skill.displayStatus === 'pending'
                                      ? 'border-amber-200 bg-amber-50/70'
                                      : skill.displayStatus === 'rejected'
                                        ? 'border-rose-200 bg-rose-50/70'
                                        : 'border-slate-200 bg-white'
                                }`}
                              >
                                <div className="flex flex-wrap items-start justify-between gap-3">
                                  <span className="min-w-0 flex-1">
                                    <span className="flex flex-wrap items-center gap-2">
                                      <span className="block text-sm font-semibold text-slate-900">{skill.title}</span>
                                      <StatusPill value={skill.displayStatus} />
                                      {skill.assigned ? (
                                        <span className="rounded-full bg-cyan-100 px-2 py-1 text-[10px] font-semibold text-cyan-800">
                                          {text.currentAssignmentState}
                                        </span>
                                      ) : null}
                                    </span>
                                    <span className="mt-1 block text-xs leading-5 text-slate-500">
                                      {skill.description || skill.source}
                                    </span>
                                    {pendingAgentName ? (
                                      <span className="mt-2 block text-[11px] text-amber-900">
                                        {formatTemplate(text.pendingForAgent, { name: pendingAgentName })}
                                      </span>
                                    ) : null}
                                    {skill.selectedRequest?.reason ? (
                                      <span className="mt-2 block rounded-[10px] border border-slate-200 bg-white/80 px-3 py-2 text-xs text-slate-600">
                                        {skill.selectedRequest.reason}
                                      </span>
                                    ) : null}
                                  </span>
                                  <div className="flex items-center gap-2">
                                    {skill.assigned ? (
                                      <button
                                        type="button"
                                        onClick={() => toggleSkillAssignment(skill.skill_id)}
                                        className="rounded-[10px] border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                                      >
                                        {text.remove}
                                      </button>
                                    ) : null}
                                    {requestable ? (
                                      <button
                                        type="button"
                                        onClick={() => handleRequestSkill(skill.skill_id)}
                                        disabled={skillRequestMutation.isPending}
                                        className="rounded-[10px] bg-slate-950 px-3 py-2 text-xs font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
                                      >
                                        {skillRequestMutation.isPending ? text.requesting : text.requestThisSkill}
                                      </button>
                                    ) : null}
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                    ) : (
                        <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-4 py-4 text-sm text-slate-500">
                          {text.noSkillIntentsAvailable}
                        </div>
                      )}
                    </div>
                    <div className="space-y-4 rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
                      <div>
                        <div className="text-sm font-semibold text-slate-900">{text.agentAvailabilityPrerequisites}</div>
                        <div className="mt-1 text-xs leading-5 text-slate-500">{text.availabilityPrerequisitesHint}</div>
                      </div>
                      <label className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800">
                        <input
                          type="checkbox"
                          checked={selectedAssignableAgent?.requires_tool_calling ?? false}
                          onChange={(event) => updateSelectedAgent({ requires_tool_calling: event.target.checked })}
                          className="h-4 w-4 rounded border-slate-300 text-cyan-600 focus:ring-cyan-500"
                        />
                        {text.requireToolCallingLabel}
                      </label>
                      <div className="grid gap-3 lg:grid-cols-3">
                        <div className="rounded-2xl bg-white p-3">
                          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{text.requiredSkillsLabel}</div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {(graph?.skill_catalog ?? []).length > 0 ? (
                              (graph?.skill_catalog ?? []).map((skill) => {
                                const skillId = normalizeSkillKey(skill.skill_id ?? '');
                                const isSelected = (selectedAssignableAgent?.required_skill_ids ?? []).some(
                                  (value) => normalizeSkillKey(value) === skillId
                                );
                                return (
                                  <button
                                    key={`required-skill-${skillId}`}
                                    type="button"
                                    onClick={() => toggleSelectedAgentPolicyValue('required_skill_ids', skillId)}
                                    className={`rounded-full px-3 py-1.5 text-[11px] font-semibold ring-1 transition ${
                                      isSelected
                                        ? 'bg-cyan-50 text-cyan-800 ring-cyan-200'
                                        : 'bg-white text-slate-700 ring-slate-200 hover:bg-slate-50'
                                    }`}
                                  >
                                    {formatSkillTitle(skillId, skillTitleById)}
                                  </button>
                                );
                              })
                            ) : (
                              <span className="text-xs text-slate-500">{text.none}</span>
                            )}
                          </div>
                        </div>
                        <div className="rounded-2xl bg-white p-3">
                          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{text.requiredToolsLabel}</div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {(graph?.tool_catalog ?? []).length > 0 ? (
                              (graph?.tool_catalog ?? []).map((tool) => {
                                const toolId = normalizeSkillKey(tool.tool_id ?? '');
                                const isSelected = (selectedAssignableAgent?.required_tool_ids ?? []).some(
                                  (value) => normalizeSkillKey(value) === toolId
                                );
                                return (
                                  <button
                                    key={`required-tool-${toolId}`}
                                    type="button"
                                    onClick={() => toggleSelectedAgentPolicyValue('required_tool_ids', toolId)}
                                    className={`rounded-full px-3 py-1.5 text-[11px] font-semibold ring-1 transition ${
                                      isSelected
                                        ? 'bg-emerald-50 text-emerald-800 ring-emerald-200'
                                        : tool.status === 'disabled'
                                          ? 'bg-slate-100 text-slate-500 ring-slate-200'
                                          : 'bg-white text-slate-700 ring-slate-200 hover:bg-slate-50'
                                    }`}
                                  >
                                    {formatSkillTitle(toolId, toolTitleById)}
                                  </button>
                                );
                              })
                            ) : (
                              <span className="text-xs text-slate-500">{text.none}</span>
                            )}
                          </div>
                        </div>
                        <div className="rounded-2xl bg-white p-3">
                          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{text.requiredMcpServersLabel}</div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {(graph?.mcp_server_catalog ?? []).length > 0 ? (
                              (graph?.mcp_server_catalog ?? []).map((server) => {
                                const serverId = normalizeSkillKey(server.server_id ?? '');
                                const isSelected = (selectedAssignableAgent?.required_mcp_server_ids ?? []).some(
                                  (value) => normalizeSkillKey(value) === serverId
                                );
                                return (
                                  <button
                                    key={`required-mcp-${serverId}`}
                                    type="button"
                                    onClick={() => toggleSelectedAgentPolicyValue('required_mcp_server_ids', serverId)}
                                    className={`rounded-full px-3 py-1.5 text-[11px] font-semibold ring-1 transition ${
                                      isSelected
                                        ? 'bg-violet-50 text-violet-800 ring-violet-200'
                                        : server.status === 'disabled'
                                          ? 'bg-slate-100 text-slate-500 ring-slate-200'
                                          : 'bg-white text-slate-700 ring-slate-200 hover:bg-slate-50'
                                    }`}
                                  >
                                    {formatSkillTitle(serverId, mcpServerTitleById)}
                                  </button>
                                );
                              })
                            ) : (
                              <span className="text-xs text-slate-500">{text.noProjectMcpServers}</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4 rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
                      <div>
                        <div className="text-sm font-semibold text-slate-900">{text.agentToolPolicy}</div>
                        <div className="mt-1 text-xs leading-5 text-slate-500">{text.toolPolicyHint}</div>
                      </div>
                      <div className="grid gap-3 sm:grid-cols-2">
                        <div className="rounded-2xl bg-white p-3">
                          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{text.toolAllowPolicyLabel}</div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {(graph?.tool_catalog ?? []).length > 0 ? (
                              (graph?.tool_catalog ?? []).map((tool) => {
                                const toolId = normalizeSkillKey(tool.tool_id ?? '');
                                const isSelected = (selectedAssignableAgent?.allowed_tool_ids ?? []).some(
                                  (value) => normalizeSkillKey(value) === toolId
                                );
                                return (
                                  <button
                                    key={`allow-tool-${toolId}`}
                                    type="button"
                                    onClick={() => toggleSelectedAgentPolicyValue('allowed_tool_ids', toolId, 'denied_tool_ids')}
                                    className={`rounded-full px-3 py-1.5 text-[11px] font-semibold ring-1 transition ${
                                      isSelected
                                        ? 'bg-emerald-50 text-emerald-800 ring-emerald-200'
                                        : tool.status === 'disabled'
                                          ? 'bg-slate-100 text-slate-500 ring-slate-200'
                                          : 'bg-white text-slate-700 ring-slate-200 hover:bg-slate-50'
                                    }`}
                                  >
                                    {formatSkillTitle(toolId, toolTitleById)}
                                  </button>
                                );
                              })
                            ) : (
                              <span className="text-xs text-slate-500">{text.none}</span>
                            )}
                          </div>
                        </div>
                        <div className="rounded-2xl bg-white p-3">
                          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{text.toolDenyPolicyLabel}</div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {(graph?.tool_catalog ?? []).length > 0 ? (
                              (graph?.tool_catalog ?? []).map((tool) => {
                                const toolId = normalizeSkillKey(tool.tool_id ?? '');
                                const isSelected = (selectedAssignableAgent?.denied_tool_ids ?? []).some(
                                  (value) => normalizeSkillKey(value) === toolId
                                );
                                return (
                                  <button
                                    key={`deny-tool-${toolId}`}
                                    type="button"
                                    onClick={() => toggleSelectedAgentPolicyValue('denied_tool_ids', toolId, 'allowed_tool_ids')}
                                    className={`rounded-full px-3 py-1.5 text-[11px] font-semibold ring-1 transition ${
                                      isSelected
                                        ? 'bg-rose-50 text-rose-800 ring-rose-200'
                                        : 'bg-white text-slate-700 ring-slate-200 hover:bg-slate-50'
                                    }`}
                                  >
                                    {formatSkillTitle(toolId, toolTitleById)}
                                  </button>
                                );
                              })
                            ) : (
                              <span className="text-xs text-slate-500">{text.none}</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4 rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
                      <div>
                        <div className="text-sm font-semibold text-slate-900">{text.agentMcpPolicy}</div>
                        <div className="mt-1 text-xs leading-5 text-slate-500">{text.mcpPolicyHint}</div>
                      </div>
                      <div className="grid gap-3 sm:grid-cols-2">
                        <div className="rounded-2xl bg-white p-3">
                          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{text.mcpAllowPolicyLabel}</div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {(graph?.mcp_server_catalog ?? []).length > 0 ? (
                              (graph?.mcp_server_catalog ?? []).map((server) => {
                                const serverId = normalizeSkillKey(server.server_id ?? '');
                                const isSelected = (selectedAssignableAgent?.allowed_mcp_server_ids ?? []).some(
                                  (value) => normalizeSkillKey(value) === serverId
                                );
                                return (
                                  <button
                                    key={`allow-mcp-${serverId}`}
                                    type="button"
                                    onClick={() =>
                                      toggleSelectedAgentPolicyValue(
                                        'allowed_mcp_server_ids',
                                        serverId,
                                        'denied_mcp_server_ids'
                                      )
                                    }
                                    className={`rounded-full px-3 py-1.5 text-[11px] font-semibold ring-1 transition ${
                                      isSelected
                                        ? 'bg-violet-50 text-violet-800 ring-violet-200'
                                        : server.status === 'disabled'
                                          ? 'bg-slate-100 text-slate-500 ring-slate-200'
                                          : 'bg-white text-slate-700 ring-slate-200 hover:bg-slate-50'
                                    }`}
                                  >
                                    {formatSkillTitle(serverId, mcpServerTitleById)}
                                  </button>
                                );
                              })
                            ) : (
                              <span className="text-xs text-slate-500">{text.noProjectMcpServers}</span>
                            )}
                          </div>
                        </div>
                        <div className="rounded-2xl bg-white p-3">
                          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{text.mcpDenyPolicyLabel}</div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {(graph?.mcp_server_catalog ?? []).length > 0 ? (
                              (graph?.mcp_server_catalog ?? []).map((server) => {
                                const serverId = normalizeSkillKey(server.server_id ?? '');
                                const isSelected = (selectedAssignableAgent?.denied_mcp_server_ids ?? []).some(
                                  (value) => normalizeSkillKey(value) === serverId
                                );
                                return (
                                  <button
                                    key={`deny-mcp-${serverId}`}
                                    type="button"
                                    onClick={() =>
                                      toggleSelectedAgentPolicyValue(
                                        'denied_mcp_server_ids',
                                        serverId,
                                        'allowed_mcp_server_ids'
                                      )
                                    }
                                    className={`rounded-full px-3 py-1.5 text-[11px] font-semibold ring-1 transition ${
                                      isSelected
                                        ? 'bg-amber-50 text-amber-800 ring-amber-200'
                                        : 'bg-white text-slate-700 ring-slate-200 hover:bg-slate-50'
                                    }`}
                                  >
                                    {formatSkillTitle(serverId, mcpServerTitleById)}
                                  </button>
                                );
                              })
                            ) : (
                              <span className="text-xs text-slate-500">{text.noProjectMcpServers}</span>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                    {selectedAgentCapabilitySummary ? (
                      <SelectedAgentCapabilityPanel
                        selectedAgent={selectedAgent}
                        selectedAgentCapabilitySummary={selectedAgentCapabilitySummary}
                        skillTitleById={skillTitleById}
                        toolTitleById={toolTitleById}
                        mcpServerTitleById={mcpServerTitleById}
                        agentNameById={agentNameById}
                        selectedAgentAvailabilityStatus={selectedAgentAvailabilityStatus}
                        selectedAgentAvailabilityBlockers={selectedAgentAvailabilityBlockers}
                        selectedAgentAvailabilityWarnings={selectedAgentAvailabilityWarnings}
                        selectedAgentReadinessStatus={selectedAgentReadinessStatus}
                        selectedAgentReadinessBlockers={selectedAgentReadinessBlockers}
                        selectedAgentReadinessWarnings={selectedAgentReadinessWarnings}
                        selectedAgentMissingSkillIds={selectedAgentMissingSkillIds}
                        selectedAgentMissingRequiredSkillIds={selectedAgentMissingRequiredSkillIds}
                        selectedAgentMissingSkillDetails={selectedAgentMissingSkillDetails}
                        selectedAgentPolicyBlockedToolIds={selectedAgentPolicyBlockedToolIds}
                        selectedAgentActionableToolPolicyIds={selectedAgentActionableToolPolicyIds}
                        selectedAgentActionableMcpPolicyIds={selectedAgentActionableMcpPolicyIds}
                        selectedAgentActionableToolRestrictionIds={selectedAgentActionableToolRestrictionIds}
                        selectedAgentActionableMcpRestrictionIds={selectedAgentActionableMcpRestrictionIds}
                        selectedAgentProviderLimitedToolIds={selectedAgentProviderLimitedToolIds}
                        selectedAgentMissingMcpServerIds={selectedAgentMissingMcpServerIds}
                        selectedAgentMissingMcpServerDetails={selectedAgentMissingMcpServerDetails}
                        selectedAgentPolicyBlockedMcpServerIds={selectedAgentPolicyBlockedMcpServerIds}
                        selectedAgentShouldOpenSkillPool={selectedAgentShouldOpenSkillPool}
                        selectedAgentShouldOpenProjectProviders={selectedAgentShouldOpenProjectProviders}
                        selectedAgentShouldOpenProjectMcp={selectedAgentShouldOpenProjectMcp}
                        selectedAgentDownstreamTargetIds={selectedAgentDownstreamTargetIds}
                        selectedAgentPrimarySuggestedCollaborator={selectedAgentPrimarySuggestedCollaborator}
                        selectedAgentRoleProfilePeerDiagnostics={selectedAgentRoleProfilePeerDiagnostics}
                        focusableAgentIds={focusableAgentIds}
                        isApplying={updateProjectMutation.isPending}
                        isRequestingSkills={skillRequestMutation.isPending}
                        canRequestSelectedAgentRoleProfileSkills={selectedAgentCanRequestRoleProfileSkills}
                        onOpenSkillPool={() => {
                          setPendingCreationKind(null);
                          setActiveCanvasPanel('skills');
                        }}
                        onApplySelectedAgentRoleProfile={handleApplySelectedAgentRoleProfile}
                        onRequestSelectedAgentRoleProfileSkills={handleRequestSelectedAgentRoleProfileSkills}
                        onApplySelectedAgentToolPolicySuggestions={handleApplySelectedAgentToolPolicySuggestions}
                        onApplySelectedAgentMcpPolicySuggestions={handleApplySelectedAgentMcpPolicySuggestions}
                        onApplySelectedAgentToolPolicyRestrictions={handleApplySelectedAgentToolPolicyRestrictions}
                        onApplySelectedAgentMcpPolicyRestrictions={handleApplySelectedAgentMcpPolicyRestrictions}
                        onOpenProjectProviders={() => focusControlPanelSection('providers')}
                        onOpenProjectMcp={() => focusControlPanelSection('mcp')}
                        onApplySuggestedHandoff={handleApplySuggestedHandoff}
                        onApplySuggestedRewire={handleApplySuggestedRewire}
                        onOpenNodeEditor={openNodeEditor}
                      />
                    ) : null}
                  </>
                )}
                <label className="block text-sm font-medium text-slate-800">
                  {text.systemPrompt}
                  <textarea
                    rows={7}
                    value={selectedAgent.system_prompt || ''}
                    onChange={(event) => updateSelectedAgent({ system_prompt: event.target.value })}
                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                  />
                </label>
              </div>
            )}
          </OverlayDialog>
        </div>
      </div>
    </div>
  </div>
  );
}
