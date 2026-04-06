import type {
  HarnessAgentCapabilitySummaryDTO,
  HarnessCanvasAgentDTO,
  HarnessCanvasEdgeDTO,
  HarnessOrchestrationSummaryDTO,
} from '@/domains/harness/hooks';
import type { PolicyRepairScopeSummary, RoleProfileScopeSummary } from './policy-repair';
import { coerceNumber, coerceRecord, coerceRecordList, coerceStringList } from './utils';

type LocalizedText = Record<string, string>;

export type DelegationFit = 'strong' | 'good' | 'weak';

export function normalizeDelegationFit(fit: string | null | undefined): DelegationFit {
  if (fit === 'strong' || fit === 'good') {
    return fit;
  }
  return 'weak';
}

export function formatDelegationFitLabel(fit: string | null | undefined, text: LocalizedText) {
  if (fit === 'strong') {
    return text.handoffFitStrong;
  }
  if (fit === 'good') {
    return text.handoffFitGood;
  }
  return text.handoffFitWeak;
}

export function delegationFitBadgeClass(fit: string | null | undefined) {
  if (fit === 'strong') {
    return 'bg-emerald-50 text-emerald-800 ring-emerald-200';
  }
  if (fit === 'good') {
    return 'bg-cyan-50 text-cyan-800 ring-cyan-200';
  }
  return 'bg-amber-50 text-amber-800 ring-amber-200';
}

export function resolveCapabilityReadinessStatus(agentCapability: Record<string, unknown> | null | undefined) {
  const status =
    typeof agentCapability?.readiness_status === 'string'
      ? agentCapability.readiness_status
      : typeof agentCapability?.status === 'string'
        ? agentCapability.status
        : '';
  if (status === 'ready' || status === 'limited' || status === 'blocked') {
    return status;
  }
  const missingSkills = coerceStringList(agentCapability?.missing_skill_ids);
  const missingMcp = coerceStringList(agentCapability?.missing_mcp_server_ids);
  const providerLimitedTools = coerceStringList(agentCapability?.provider_limited_tool_ids);
  if (missingSkills.length > 0) {
    return 'blocked';
  }
  if (missingMcp.length > 0 || providerLimitedTools.length > 0) {
    return 'limited';
  }
  return 'ready';
}

export function resolveCapabilityAvailabilityStatus(agentCapability: Record<string, unknown> | null | undefined) {
  const status = typeof agentCapability?.availability_status === 'string' ? agentCapability.availability_status : '';
  if (status === 'available' || status === 'limited' || status === 'unavailable') {
    return status;
  }
  const missingRequiredSkills = coerceStringList(agentCapability?.missing_required_skill_ids);
  const missingRequiredTools = coerceStringList(agentCapability?.missing_required_tool_ids);
  const missingRequiredMcp = coerceStringList(agentCapability?.missing_required_mcp_server_ids);
  const requiresToolCalling = Boolean(agentCapability?.requires_tool_calling);
  const toolExecutionSupport =
    typeof agentCapability?.tool_execution_support === 'string' ? agentCapability.tool_execution_support : '';
  if (missingRequiredSkills.length > 0 || missingRequiredTools.length > 0 || missingRequiredMcp.length > 0) {
    return 'unavailable';
  }
  if (requiresToolCalling && toolExecutionSupport === 'unsupported') {
    return 'unavailable';
  }
  if (requiresToolCalling && toolExecutionSupport !== 'supported') {
    return 'limited';
  }
  return 'available';
}

function hasRecoveryAction(
  agentCapability: Record<string, unknown> | null | undefined,
  action: 'focus_agent' | 'open_skill_pool' | 'open_project_mcp_inventory' | 'open_project_providers'
) {
  const recoveryActions = coerceRecord(agentCapability?.recovery_actions);
  return recoveryActions?.[action] === true;
}

export function shouldOpenSkillPoolForDiagnostic(agentCapability: Record<string, unknown> | null | undefined) {
  if (hasRecoveryAction(agentCapability, 'open_skill_pool')) {
    return true;
  }
  const missingSkillIds = coerceStringList(agentCapability?.missing_skill_ids);
  const missingSkillDetails = coerceRecordList(agentCapability?.missing_skill_details);
  const missingRequiredSkillIds = coerceStringList(agentCapability?.missing_required_skill_ids);
  return missingSkillIds.length > 0 || missingSkillDetails.length > 0 || missingRequiredSkillIds.length > 0;
}

export function shouldOpenProjectMcpForDiagnostic(agentCapability: Record<string, unknown> | null | undefined) {
  if (hasRecoveryAction(agentCapability, 'open_project_mcp_inventory')) {
    return true;
  }
  const missingMcpServerIds = coerceStringList(agentCapability?.missing_mcp_server_ids);
  const missingRequiredMcpServerIds = coerceStringList(agentCapability?.missing_required_mcp_server_ids);
  return missingMcpServerIds.length > 0 || missingRequiredMcpServerIds.length > 0;
}

export function shouldOpenProjectProvidersForDiagnostic(agentCapability: Record<string, unknown> | null | undefined) {
  if (hasRecoveryAction(agentCapability, 'open_project_providers')) {
    return true;
  }
  const providerLimitedToolIds = coerceStringList(agentCapability?.provider_limited_tool_ids);
  const requiresToolCalling = Boolean(agentCapability?.requires_tool_calling);
  const toolExecutionSupport =
    typeof agentCapability?.tool_execution_support === 'string' ? agentCapability.tool_execution_support : '';
  return providerLimitedToolIds.length > 0 || (requiresToolCalling && toolExecutionSupport !== 'supported');
}

export function formatCapabilityAvailabilityLabel(status: string | null | undefined, text: LocalizedText) {
  if (status === 'unavailable') {
    return text.availabilityUnavailable;
  }
  if (status === 'limited') {
    return text.availabilityLimited;
  }
  return text.availabilityAvailable;
}

export function capabilityAvailabilityBadgeClass(status: string | null | undefined) {
  if (status === 'unavailable') {
    return 'bg-rose-50 text-rose-800 ring-rose-200';
  }
  if (status === 'limited') {
    return 'bg-amber-50 text-amber-800 ring-amber-200';
  }
  return 'bg-emerald-50 text-emerald-800 ring-emerald-200';
}

export function formatCapabilityReadinessLabel(status: string | null | undefined, text: LocalizedText) {
  if (status === 'blocked') {
    return text.readinessBlocked;
  }
  if (status === 'limited') {
    return text.readinessLimited;
  }
  return text.readinessReady;
}

export function capabilityReadinessBadgeClass(status: string | null | undefined) {
  if (status === 'blocked') {
    return 'bg-rose-50 text-rose-800 ring-rose-200';
  }
  if (status === 'limited') {
    return 'bg-amber-50 text-amber-800 ring-amber-200';
  }
  return 'bg-emerald-50 text-emerald-800 ring-emerald-200';
}

export type AgentAvailabilityDiagnostic = {
  agentId: string;
  agentName: string;
  status: 'available' | 'limited' | 'unavailable';
  blockers: string[];
  warnings: string[];
};

export type AvailabilityScopeSummary = {
  totalCount: number;
  availableCount: number;
  limitedCount: number;
  unavailableCount: number;
  flaggedAgents: AgentAvailabilityDiagnostic[];
  unavailableAgentNames: string[];
};

export type RunPreflightScopeCounts = {
  totalCount: number;
  weakEdgeCount: number;
  bestNextCount: number;
  readyCount: number;
  limitedReadinessCount: number;
  blockedCount: number;
  availableCount: number;
  limitedAvailabilityCount: number;
  unavailableCount: number;
  directExecutionAgentCount: number;
  planningOnlyToolAgentCount: number;
  planningOnlyMcpAgentCount: number;
  coordinatorAgentCount: number;
  parallelCoordinatorAgentCount: number;
  finalOutputAgentCount: number;
  verificationAgentCount: number;
};

export type CollaborationScopeSummary = {
  weakEdgeCount: number;
  bestNextCount: number;
  actionableSourceAgentCount: number;
  laneCount: number;
  focusCount: number;
  sourceAgentNames: string[];
  focusPreview: string[];
};

export type CoordinationAgentPreview = {
  agentId: string;
  agentName: string;
};

export type CapabilityOwnerEntry = {
  capabilityId: string;
  ownerAgents: CoordinationAgentPreview[];
};

export type CoordinationTopologySummary = {
  totalAgentCount: number;
  totalLaneCount: number;
  sharedLaneCount: number;
  singleOwnerLaneCount: number;
  isolatedAgentCount: number;
  underconnectedAgentCount: number;
  isolatedAgents: CoordinationAgentPreview[];
  underconnectedAgents: CoordinationAgentPreview[];
  sharedLaneIds: string[];
  singleOwnerLaneIds: string[];
};

export type CapabilityCoverageSummary = {
  totalSkillCount: number;
  totalToolCount: number;
  totalMcpCount: number;
  sharedSkillIds: string[];
  singleOwnerSkills: CapabilityOwnerEntry[];
  sharedToolIds: string[];
  singleOwnerTools: CapabilityOwnerEntry[];
  sharedMcpServerIds: string[];
  singleOwnerMcpServers: CapabilityOwnerEntry[];
  missingSkillIds: string[];
  blockedToolIds: string[];
  missingMcpServerIds: string[];
};

export type OrchestrationBriefCapabilityRisk = {
  kind: 'skill' | 'tool' | 'mcp';
  capabilityId: string;
  ownerAgents: CoordinationAgentPreview[];
};

export type OrchestrationPhaseId = 'research' | 'synthesis' | 'implementation' | 'verification';

export type OrchestrationPhaseSummary = {
  phaseId: OrchestrationPhaseId;
  agentCount: number;
  agents: CoordinationAgentPreview[];
};

export type OrchestrationRepairPriorityId =
  | 'availability'
  | 'capability_gaps'
  | 'policy_repair'
  | 'role_profile_alignment'
  | 'weak_handoffs'
  | 'best_next_handoffs'
  | 'connectivity'
  | 'single_owner_capabilities'
  | 'review_path';

export type OrchestrationRepairPriority = {
  priorityId: OrchestrationRepairPriorityId;
  severity: 'high' | 'medium' | 'low';
  count: number;
};

export type OrchestrationAgentRoutingSummary = {
  coordinatorAnchors: CoordinationAgentPreview[];
  researchAnchors: CoordinationAgentPreview[];
  implementationAnchors: CoordinationAgentPreview[];
  verificationAnchors: CoordinationAgentPreview[];
  skillCapableAnchors: CoordinationAgentPreview[];
  toolCapableAnchors: CoordinationAgentPreview[];
  mcpCapableAnchors: CoordinationAgentPreview[];
};

export type OrchestrationBriefSummary = {
  totalAgentCount: number;
  executionStepCount: number;
  reviewEnabled: boolean;
  readiness: 'blocked' | 'repair' | 'watch' | 'ready';
  startAgents: CoordinationAgentPreview[];
  terminalAgents: CoordinationAgentPreview[];
  sharedLaneCount: number;
  singleOwnerCapabilityCount: number;
  singleOwnerCapabilityRisks: OrchestrationBriefCapabilityRisk[];
  unavailableCount: number;
  limitedAvailabilityCount: number;
  policyRepairAgentCount: number;
  roleProfileDriftAgentCount: number;
  roleProfileOverlapRiskCount: number;
  weakEdgeCount: number;
  bestNextCount: number;
  capabilityGapCount: number;
  isolatedAgentCount: number;
  underconnectedAgentCount: number;
  phases: OrchestrationPhaseSummary[];
  repairPriorities: OrchestrationRepairPriority[];
  agentRouting: OrchestrationAgentRoutingSummary;
};

const EMPTY_COORDINATION_TOPOLOGY_SUMMARY: CoordinationTopologySummary = {
  totalAgentCount: 0,
  totalLaneCount: 0,
  sharedLaneCount: 0,
  singleOwnerLaneCount: 0,
  isolatedAgentCount: 0,
  underconnectedAgentCount: 0,
  isolatedAgents: [],
  underconnectedAgents: [],
  sharedLaneIds: [],
  singleOwnerLaneIds: [],
};

const EMPTY_CAPABILITY_COVERAGE_SUMMARY: CapabilityCoverageSummary = {
  totalSkillCount: 0,
  totalToolCount: 0,
  totalMcpCount: 0,
  sharedSkillIds: [],
  singleOwnerSkills: [],
  sharedToolIds: [],
  singleOwnerTools: [],
  sharedMcpServerIds: [],
  singleOwnerMcpServers: [],
  missingSkillIds: [],
  blockedToolIds: [],
  missingMcpServerIds: [],
};

const EMPTY_ORCHESTRATION_BRIEF_SUMMARY: OrchestrationBriefSummary = {
  totalAgentCount: 0,
  executionStepCount: 0,
  reviewEnabled: false,
  readiness: 'ready',
  startAgents: [],
  terminalAgents: [],
  sharedLaneCount: 0,
  singleOwnerCapabilityCount: 0,
  singleOwnerCapabilityRisks: [],
  unavailableCount: 0,
  limitedAvailabilityCount: 0,
  policyRepairAgentCount: 0,
  roleProfileDriftAgentCount: 0,
  roleProfileOverlapRiskCount: 0,
  weakEdgeCount: 0,
  bestNextCount: 0,
  capabilityGapCount: 0,
  isolatedAgentCount: 0,
  underconnectedAgentCount: 0,
  phases: [],
  repairPriorities: [],
  agentRouting: {
    coordinatorAnchors: [],
    researchAnchors: [],
    implementationAnchors: [],
    verificationAnchors: [],
    skillCapableAnchors: [],
    toolCapableAnchors: [],
    mcpCapableAnchors: [],
  },
};

function coerceCoordinationAgentPreviewList(value: unknown): CoordinationAgentPreview[] {
  return coerceRecordList(value)
    .map((item) => {
      const agentId = typeof item.agent_id === 'string' ? item.agent_id.trim() : '';
      const agentName =
        typeof item.agent_name === 'string' && item.agent_name.trim()
          ? item.agent_name.trim()
          : agentId;
      if (!agentId || !agentName) {
        return null;
      }
      return {
        agentId,
        agentName,
      };
    })
    .filter((item): item is CoordinationAgentPreview => item !== null);
}

export function coerceOrchestrationBriefSummary(
  summary: HarnessOrchestrationSummaryDTO | null | undefined
): OrchestrationBriefSummary | null {
  if (!summary) {
    return null;
  }
  const readiness =
    summary.readiness === 'blocked' ||
    summary.readiness === 'repair' ||
    summary.readiness === 'watch' ||
    summary.readiness === 'ready'
      ? summary.readiness
      : 'ready';
  const phases = coerceRecordList(summary.phases)
    .map((item) => {
      const phaseId =
        item.phase_id === 'research' ||
        item.phase_id === 'synthesis' ||
        item.phase_id === 'implementation' ||
        item.phase_id === 'verification'
          ? item.phase_id
          : null;
      if (!phaseId) {
        return null;
      }
      return {
        phaseId,
        agentCount: coerceNumber(item.agent_count) ?? coerceCoordinationAgentPreviewList(item.agents).length,
        agents: coerceCoordinationAgentPreviewList(item.agents),
      };
    })
    .filter((item): item is OrchestrationPhaseSummary => item !== null);
  const repairPriorities = coerceRecordList(summary.repair_priorities)
    .map((item) => {
      const priorityId =
        item.priority_id === 'availability' ||
        item.priority_id === 'capability_gaps' ||
        item.priority_id === 'policy_repair' ||
        item.priority_id === 'role_profile_alignment' ||
        item.priority_id === 'weak_handoffs' ||
        item.priority_id === 'best_next_handoffs' ||
        item.priority_id === 'connectivity' ||
        item.priority_id === 'single_owner_capabilities' ||
        item.priority_id === 'review_path'
          ? item.priority_id
          : null;
      const severity =
        item.severity === 'high' || item.severity === 'medium' || item.severity === 'low' ? item.severity : null;
      if (!priorityId || !severity) {
        return null;
      }
      return {
        priorityId,
        severity,
        count: coerceNumber(item.count) ?? 0,
      };
    })
    .filter((item): item is OrchestrationRepairPriority => item !== null);
  const singleOwnerCapabilityRisks = coerceRecordList(summary.single_owner_capability_risks)
    .map((item) => {
      const kind = item.kind === 'skill' || item.kind === 'tool' || item.kind === 'mcp' ? item.kind : null;
      const capabilityId = typeof item.capability_id === 'string' ? item.capability_id.trim() : '';
      if (!kind || !capabilityId) {
        return null;
      }
      return {
        kind,
        capabilityId,
        ownerAgents: coerceCoordinationAgentPreviewList(item.owner_agents),
      };
    })
    .filter((item): item is OrchestrationBriefSummary['singleOwnerCapabilityRisks'][number] => item !== null);
  const routingRecord = coerceRecord(summary.agent_routing);
  return {
    ...EMPTY_ORCHESTRATION_BRIEF_SUMMARY,
    totalAgentCount: coerceNumber(summary.total_agent_count) ?? 0,
    executionStepCount: coerceNumber(summary.execution_step_count) ?? 0,
    reviewEnabled: summary.review_enabled ?? false,
    readiness,
    startAgents: coerceCoordinationAgentPreviewList(summary.start_agents),
    terminalAgents: coerceCoordinationAgentPreviewList(summary.terminal_agents),
    sharedLaneCount: coerceNumber(summary.shared_lane_count) ?? 0,
    singleOwnerCapabilityCount:
      coerceNumber(summary.single_owner_capability_count) ?? singleOwnerCapabilityRisks.length,
    singleOwnerCapabilityRisks,
    unavailableCount: coerceNumber(summary.unavailable_count) ?? 0,
    limitedAvailabilityCount: coerceNumber(summary.limited_availability_count) ?? 0,
    policyRepairAgentCount: coerceNumber(summary.policy_repair_agent_count) ?? 0,
    roleProfileDriftAgentCount: coerceNumber(summary.role_profile_drift_agent_count) ?? 0,
    roleProfileOverlapRiskCount: coerceNumber(summary.role_profile_overlap_risk_count) ?? 0,
    weakEdgeCount: coerceNumber(summary.weak_edge_count) ?? 0,
    bestNextCount: coerceNumber(summary.best_next_count) ?? 0,
    capabilityGapCount: coerceNumber(summary.capability_gap_count) ?? 0,
    isolatedAgentCount: coerceNumber(summary.isolated_agent_count) ?? 0,
    underconnectedAgentCount: coerceNumber(summary.underconnected_agent_count) ?? 0,
    phases,
    repairPriorities,
    agentRouting: {
      coordinatorAnchors: coerceCoordinationAgentPreviewList(routingRecord?.coordinator_anchors),
      researchAnchors: coerceCoordinationAgentPreviewList(routingRecord?.research_anchors),
      implementationAnchors: coerceCoordinationAgentPreviewList(routingRecord?.implementation_anchors),
      verificationAnchors: coerceCoordinationAgentPreviewList(routingRecord?.verification_anchors),
      skillCapableAnchors: coerceCoordinationAgentPreviewList(routingRecord?.skill_capable_anchors),
      toolCapableAnchors: coerceCoordinationAgentPreviewList(routingRecord?.tool_capable_anchors),
      mcpCapableAnchors: coerceCoordinationAgentPreviewList(routingRecord?.mcp_capable_anchors),
    },
  };
}

const RESEARCH_PHASE_KEYWORDS = [
  'research',
  'researcher',
  'rag',
  'retrieve',
  'retrieval',
  'search',
  'evidence',
  'context',
  'docs',
  'document',
  'study',
  'investig',
  '研究',
  '检索',
  '证据',
  '文档',
  '背景',
];

const SYNTHESIS_PHASE_KEYWORDS = [
  'plan',
  'planner',
  'coordinator',
  'coordination',
  'orchestrat',
  'delegate',
  'delegation',
  'synthes',
  'synthesis',
  'chair',
  'strateg',
  'cluster',
  '编排',
  '协调',
  '规划',
  '综合',
  '主持',
  '策略',
];

const IMPLEMENTATION_PHASE_KEYWORDS = [
  'build',
  'builder',
  'implement',
  'implementation',
  'execute',
  'execution',
  'engineer',
  'code',
  'develop',
  'delivery',
  '执行',
  '实现',
  '开发',
  '编码',
];

const VERIFICATION_PHASE_KEYWORDS = [
  'review',
  'reviewer',
  'verify',
  'verification',
  'qa',
  'test',
  'critic',
  'audit',
  'compliance',
  '审查',
  '验证',
  '测试',
  '质疑',
  '合规',
];

function normalizeDiagnosticText(value: string | null | undefined) {
  return String(value || '')
    .trim()
    .toLowerCase();
}

function matchesDiagnosticKeywords(values: Array<string | null | undefined>, keywords: string[]) {
  const haystack = values.map((value) => normalizeDiagnosticText(value)).filter(Boolean).join(' ');
  if (!haystack) {
    return false;
  }
  return keywords.some((keyword) => haystack.includes(keyword));
}

function pickTopCoordinationAnchors(
  entries: Array<CoordinationAgentPreview & { score: number }>,
  maxCount = 3
) {
  return entries
    .filter((entry) => entry.score > 0)
    .slice()
    .sort((left, right) => {
      if (left.score !== right.score) {
        return right.score - left.score;
      }
      return left.agentName.localeCompare(right.agentName);
    })
    .slice(0, maxCount)
    .map(({ agentId, agentName }) => ({ agentId, agentName }));
}

export function buildRunPreflightScopeCounts(
  metadata: Record<string, unknown> | null | undefined,
  prefix: 'graph' | 'scope'
): RunPreflightScopeCounts | null {
  const totalCount = coerceNumber(metadata?.[`${prefix}_total_agent_count`]);
  const weakEdgeCount = coerceNumber(metadata?.[`${prefix === 'graph' ? 'graph' : 'handoff_scope'}_weak_edge_count`]);
  const bestNextCount = coerceNumber(metadata?.[`${prefix === 'graph' ? 'graph' : 'handoff_scope'}_best_next_count`]);
  const readyCount = coerceNumber(metadata?.[`${prefix}_ready_agent_count`]);
  const limitedReadinessCount = coerceNumber(metadata?.[`${prefix}_limited_agent_count`]);
  const blockedCount = coerceNumber(metadata?.[`${prefix}_blocked_agent_count`]);
  const availableCount = coerceNumber(metadata?.[`${prefix}_available_agent_count`]);
  const limitedAvailabilityCount = coerceNumber(metadata?.[`${prefix}_availability_limited_agent_count`]);
  const unavailableCount = coerceNumber(metadata?.[`${prefix}_unavailable_agent_count`]);
  const directExecutionAgentCount = coerceNumber(metadata?.[`${prefix}_direct_execution_agent_count`]);
  const planningOnlyToolAgentCount = coerceNumber(metadata?.[`${prefix}_planning_only_tool_agent_count`]);
  const planningOnlyMcpAgentCount = coerceNumber(metadata?.[`${prefix}_planning_only_mcp_agent_count`]);
  const coordinatorAgentCount = coerceNumber(metadata?.[`${prefix}_coordinator_agent_count`]);
  const parallelCoordinatorAgentCount = coerceNumber(
    metadata?.[`${prefix}_parallel_coordinator_agent_count`]
  );
  const finalOutputAgentCount = coerceNumber(metadata?.[`${prefix}_final_output_agent_count`]);
  const verificationAgentCount = coerceNumber(metadata?.[`${prefix}_verification_agent_count`]);
  const hasStructuredSummary = [
    totalCount,
    weakEdgeCount,
    bestNextCount,
    readyCount,
    limitedReadinessCount,
    blockedCount,
    availableCount,
    limitedAvailabilityCount,
    unavailableCount,
    directExecutionAgentCount,
    planningOnlyToolAgentCount,
    planningOnlyMcpAgentCount,
    coordinatorAgentCount,
    parallelCoordinatorAgentCount,
    finalOutputAgentCount,
    verificationAgentCount,
  ].some((value) => value !== null);
  if (!hasStructuredSummary) {
    return null;
  }
  const derivedTotalCount = Math.max(
    (readyCount ?? 0) + (limitedReadinessCount ?? 0) + (blockedCount ?? 0),
    (availableCount ?? 0) + (limitedAvailabilityCount ?? 0) + (unavailableCount ?? 0)
  );
  return {
    totalCount: totalCount ?? derivedTotalCount,
    weakEdgeCount: weakEdgeCount ?? 0,
    bestNextCount: bestNextCount ?? 0,
    readyCount: readyCount ?? 0,
    limitedReadinessCount: limitedReadinessCount ?? 0,
    blockedCount: blockedCount ?? 0,
    availableCount: availableCount ?? 0,
    limitedAvailabilityCount: limitedAvailabilityCount ?? 0,
    unavailableCount: unavailableCount ?? 0,
    directExecutionAgentCount: directExecutionAgentCount ?? 0,
    planningOnlyToolAgentCount: planningOnlyToolAgentCount ?? 0,
    planningOnlyMcpAgentCount: planningOnlyMcpAgentCount ?? 0,
    coordinatorAgentCount: coordinatorAgentCount ?? 0,
    parallelCoordinatorAgentCount: parallelCoordinatorAgentCount ?? 0,
    finalOutputAgentCount: finalOutputAgentCount ?? 0,
    verificationAgentCount: verificationAgentCount ?? 0,
  };
}

export function buildCollaborationScopeSummary(
  diagnostics: Record<string, unknown> | null | undefined
): CollaborationScopeSummary {
  const weakEdges = coerceRecordList(diagnostics?.weak_downstream_edges);
  const bestNextHandoffs = coerceRecordList(diagnostics?.best_next_handoffs);
  const sourceAgentNames = new Set<string>();
  const laneIds = new Set<string>();
  const focusPreview = new Set<string>();

  for (const item of [...weakEdges, ...bestNextHandoffs]) {
    const sourceAgentId = typeof item.source_agent_id === 'string' ? item.source_agent_id.trim() : '';
    const sourceAgentName =
      typeof item.source_agent_name === 'string' && item.source_agent_name.trim()
        ? item.source_agent_name.trim()
        : sourceAgentId;
    if (sourceAgentName) {
      sourceAgentNames.add(sourceAgentName);
    }
    for (const laneId of coerceStringList(item.source_lane_ids)) {
      laneIds.add(laneId);
    }
    const delegationFocus = typeof item.delegation_focus === 'string' ? item.delegation_focus.trim() : '';
    if (delegationFocus) {
      focusPreview.add(delegationFocus);
    }
  }

  return {
    weakEdgeCount: coerceNumber(diagnostics?.weak_edge_count) ?? weakEdges.length,
    bestNextCount: coerceNumber(diagnostics?.best_next_count) ?? bestNextHandoffs.length,
    actionableSourceAgentCount: sourceAgentNames.size,
    laneCount: laneIds.size,
    focusCount: focusPreview.size,
    sourceAgentNames: Array.from(sourceAgentNames).sort((left, right) => left.localeCompare(right)),
    focusPreview: Array.from(focusPreview).slice(0, 3),
  };
}

export function buildCoordinationTopologySummary({
  agents,
  edges,
  capabilitySummaries,
  selectedAgentIds = null,
}: {
  agents: HarnessCanvasAgentDTO[];
  edges: HarnessCanvasEdgeDTO[];
  capabilitySummaries: HarnessAgentCapabilitySummaryDTO[];
  selectedAgentIds?: string[] | null;
}): CoordinationTopologySummary {
  const selectedSet =
    selectedAgentIds === null ? null : new Set(selectedAgentIds.map((agentId) => String(agentId || '').trim()).filter(Boolean));
  const filteredAgents = agents.filter((agent) => {
    const agentId = String(agent.agent_id || '').trim();
    return agentId && (!selectedSet || selectedSet.has(agentId));
  });
  if (filteredAgents.length === 0) {
    return EMPTY_COORDINATION_TOPOLOGY_SUMMARY;
  }
  const includedAgentIds = new Set(filteredAgents.map((agent) => agent.agent_id));
  const summaryByAgentId = new Map(
    capabilitySummaries
      .filter((summary) => {
        const agentId = String(summary.agent_id || '').trim();
        return agentId && includedAgentIds.has(agentId);
      })
      .map((summary) => [summary.agent_id, summary])
  );

  const inboundCountByAgentId = new Map(filteredAgents.map((agent) => [agent.agent_id, 0]));
  const outboundCountByAgentId = new Map(filteredAgents.map((agent) => [agent.agent_id, 0]));
  for (const edge of edges) {
    if (!includedAgentIds.has(edge.source_agent_id) || !includedAgentIds.has(edge.target_agent_id)) {
      continue;
    }
    outboundCountByAgentId.set(edge.source_agent_id, (outboundCountByAgentId.get(edge.source_agent_id) ?? 0) + 1);
    inboundCountByAgentId.set(edge.target_agent_id, (inboundCountByAgentId.get(edge.target_agent_id) ?? 0) + 1);
  }

  const laneOwners = new Map<string, CoordinationAgentPreview[]>();
  const isolatedAgents: CoordinationAgentPreview[] = [];
  const underconnectedAgents: CoordinationAgentPreview[] = [];

  for (const agent of filteredAgents) {
    const agentId = agent.agent_id;
    const agentName = agent.name || agentId;
    const summary = summaryByAgentId.get(agentId);
    const laneIds = coerceStringList(summary?.delegation_lane_ids);
    for (const laneId of laneIds) {
      const owners = laneOwners.get(laneId) ?? [];
      owners.push({ agentId, agentName });
      laneOwners.set(laneId, owners);
    }

    const inboundCount = inboundCountByAgentId.get(agentId) ?? 0;
    const outboundCount = outboundCountByAgentId.get(agentId) ?? 0;
    if (inboundCount === 0 && outboundCount === 0) {
      isolatedAgents.push({ agentId, agentName });
      continue;
    }

    const hasBridgeOpportunity = (summary?.recommended_collaborators ?? []).some((item) => {
      if (!item.agent_id || item.edge_present) {
        return false;
      }
      return item.fit === 'strong' || item.fit === 'good';
    });
    const hasWeakDownstream = (summary?.downstream_handoff_scores ?? []).some((item) => {
      if (!item.edge_present) {
        return false;
      }
      return item.fit === 'weak' || !item.fit;
    });
    if (hasBridgeOpportunity || hasWeakDownstream) {
      underconnectedAgents.push({ agentId, agentName });
    }
  }

  const sharedLaneIds: string[] = [];
  const singleOwnerLaneIds: string[] = [];
  for (const [laneId, owners] of laneOwners.entries()) {
    if (owners.length > 1) {
      sharedLaneIds.push(laneId);
    } else {
      singleOwnerLaneIds.push(laneId);
    }
  }

  isolatedAgents.sort((left, right) => left.agentName.localeCompare(right.agentName));
  underconnectedAgents.sort((left, right) => left.agentName.localeCompare(right.agentName));
  sharedLaneIds.sort((left, right) => left.localeCompare(right));
  singleOwnerLaneIds.sort((left, right) => left.localeCompare(right));

  return {
    totalAgentCount: filteredAgents.length,
    totalLaneCount: laneOwners.size,
    sharedLaneCount: sharedLaneIds.length,
    singleOwnerLaneCount: singleOwnerLaneIds.length,
    isolatedAgentCount: isolatedAgents.length,
    underconnectedAgentCount: underconnectedAgents.length,
    isolatedAgents,
    underconnectedAgents,
    sharedLaneIds,
    singleOwnerLaneIds,
  };
}

export function buildCapabilityCoverageSummary({
  agents,
  capabilitySummaries,
  selectedAgentIds = null,
}: {
  agents: HarnessCanvasAgentDTO[];
  capabilitySummaries: HarnessAgentCapabilitySummaryDTO[];
  selectedAgentIds?: string[] | null;
}): CapabilityCoverageSummary {
  const selectedSet =
    selectedAgentIds === null ? null : new Set(selectedAgentIds.map((agentId) => String(agentId || '').trim()).filter(Boolean));
  const filteredAgents = agents.filter((agent) => {
    const agentId = String(agent.agent_id || '').trim();
    return agentId && (!selectedSet || selectedSet.has(agentId));
  });
  if (filteredAgents.length === 0) {
    return EMPTY_CAPABILITY_COVERAGE_SUMMARY;
  }

  const agentNameById = new Map(filteredAgents.map((agent) => [agent.agent_id, agent.name || agent.agent_id]));
  const includedAgentIds = new Set(filteredAgents.map((agent) => agent.agent_id));
  const capabilityOwnerMap = {
    skills: new Map<string, CoordinationAgentPreview[]>(),
    tools: new Map<string, CoordinationAgentPreview[]>(),
    mcp: new Map<string, CoordinationAgentPreview[]>(),
  };
  const missingSkillIds = new Set<string>();
  const blockedToolIds = new Set<string>();
  const missingMcpServerIds = new Set<string>();

  const appendOwner = (
    map: Map<string, CoordinationAgentPreview[]>,
    capabilityId: string,
    owner: CoordinationAgentPreview
  ) => {
    const existingOwners = map.get(capabilityId) ?? [];
    if (!existingOwners.some((item) => item.agentId === owner.agentId)) {
      existingOwners.push(owner);
      existingOwners.sort((left, right) => left.agentName.localeCompare(right.agentName));
      map.set(capabilityId, existingOwners);
    }
  };

  for (const summary of capabilitySummaries) {
    const agentId = String(summary.agent_id || '').trim();
    if (!agentId || !includedAgentIds.has(agentId)) {
      continue;
    }
    const owner = {
      agentId,
      agentName: agentNameById.get(agentId) ?? agentId,
    };
    for (const skillId of coerceStringList(summary.loaded_skill_ids)) {
      appendOwner(capabilityOwnerMap.skills, skillId, owner);
    }
    for (const toolId of coerceStringList(summary.enabled_tool_ids)) {
      appendOwner(capabilityOwnerMap.tools, toolId, owner);
    }
    for (const serverId of coerceStringList(summary.mcp_server_ids)) {
      appendOwner(capabilityOwnerMap.mcp, serverId, owner);
    }
    for (const skillId of [
      ...coerceStringList(summary.missing_skill_ids),
      ...coerceStringList(summary.missing_required_skill_ids),
    ]) {
      missingSkillIds.add(skillId);
    }
    for (const toolId of [
      ...coerceStringList(summary.policy_blocked_tool_ids),
      ...coerceStringList(summary.provider_limited_tool_ids),
      ...coerceStringList(summary.missing_required_tool_ids),
    ]) {
      blockedToolIds.add(toolId);
    }
    for (const serverId of [
      ...coerceStringList(summary.missing_mcp_server_ids),
      ...coerceStringList(summary.missing_required_mcp_server_ids),
      ...coerceStringList(summary.policy_blocked_mcp_server_ids),
    ]) {
      missingMcpServerIds.add(serverId);
    }
  }

  const splitCoverage = (map: Map<string, CoordinationAgentPreview[]>) => {
    const sharedIds: string[] = [];
    const singleOwnerEntries: CapabilityOwnerEntry[] = [];
    for (const [capabilityId, owners] of map.entries()) {
      if (owners.length > 1) {
        sharedIds.push(capabilityId);
      } else {
        singleOwnerEntries.push({ capabilityId, ownerAgents: owners });
      }
    }
    sharedIds.sort((left, right) => left.localeCompare(right));
    singleOwnerEntries.sort((left, right) => left.capabilityId.localeCompare(right.capabilityId));
    return { sharedIds, singleOwnerEntries };
  };

  const skillCoverage = splitCoverage(capabilityOwnerMap.skills);
  const toolCoverage = splitCoverage(capabilityOwnerMap.tools);
  const mcpCoverage = splitCoverage(capabilityOwnerMap.mcp);

  return {
    totalSkillCount: capabilityOwnerMap.skills.size,
    totalToolCount: capabilityOwnerMap.tools.size,
    totalMcpCount: capabilityOwnerMap.mcp.size,
    sharedSkillIds: skillCoverage.sharedIds,
    singleOwnerSkills: skillCoverage.singleOwnerEntries,
    sharedToolIds: toolCoverage.sharedIds,
    singleOwnerTools: toolCoverage.singleOwnerEntries,
    sharedMcpServerIds: mcpCoverage.sharedIds,
    singleOwnerMcpServers: mcpCoverage.singleOwnerEntries,
    missingSkillIds: Array.from(missingSkillIds).sort((left, right) => left.localeCompare(right)),
    blockedToolIds: Array.from(blockedToolIds).sort((left, right) => left.localeCompare(right)),
    missingMcpServerIds: Array.from(missingMcpServerIds).sort((left, right) => left.localeCompare(right)),
  };
}

export function buildOrchestrationBriefSummary({
  agents,
  edges,
  capabilitySummaries,
  selectedAgentIds = null,
  executionStepCount,
  reviewEnabled,
  availabilitySummary,
  policyRepairSummary,
  roleProfileSummary,
  collaborationSummary,
  topologySummary,
  capabilityCoverageSummary,
}: {
  agents: HarnessCanvasAgentDTO[];
  edges: HarnessCanvasEdgeDTO[];
  capabilitySummaries: HarnessAgentCapabilitySummaryDTO[];
  selectedAgentIds?: string[] | null;
  executionStepCount: number;
  reviewEnabled: boolean;
  availabilitySummary: AvailabilityScopeSummary;
  policyRepairSummary: PolicyRepairScopeSummary;
  roleProfileSummary: RoleProfileScopeSummary;
  collaborationSummary: CollaborationScopeSummary;
  topologySummary: CoordinationTopologySummary;
  capabilityCoverageSummary: CapabilityCoverageSummary;
}): OrchestrationBriefSummary {
  const selectedSet =
    selectedAgentIds === null ? null : new Set(selectedAgentIds.map((agentId) => String(agentId || '').trim()).filter(Boolean));
  const filteredAgents = agents.filter((agent) => {
    const agentId = String(agent.agent_id || '').trim();
    return agentId && (!selectedSet || selectedSet.has(agentId));
  });
  if (filteredAgents.length === 0) {
    return {
      ...EMPTY_ORCHESTRATION_BRIEF_SUMMARY,
      reviewEnabled,
    };
  }

  const includedAgentIds = new Set(filteredAgents.map((agent) => agent.agent_id));
  const summaryByAgentId = new Map(
    capabilitySummaries
      .filter((summary) => {
        const agentId = String(summary.agent_id || '').trim();
        return agentId && includedAgentIds.has(agentId);
      })
      .map((summary) => [summary.agent_id, summary])
  );
  const inboundCountByAgentId = new Map(filteredAgents.map((agent) => [agent.agent_id, 0]));
  const outboundCountByAgentId = new Map(filteredAgents.map((agent) => [agent.agent_id, 0]));

  for (const edge of edges) {
    if (!includedAgentIds.has(edge.source_agent_id) || !includedAgentIds.has(edge.target_agent_id)) {
      continue;
    }
    outboundCountByAgentId.set(edge.source_agent_id, (outboundCountByAgentId.get(edge.source_agent_id) ?? 0) + 1);
    inboundCountByAgentId.set(edge.target_agent_id, (inboundCountByAgentId.get(edge.target_agent_id) ?? 0) + 1);
  }

  const startAgents = filteredAgents
    .filter((agent) => (inboundCountByAgentId.get(agent.agent_id) ?? 0) === 0)
    .map((agent) => ({
      agentId: agent.agent_id,
      agentName: agent.name || agent.agent_id,
    }))
    .sort((left, right) => left.agentName.localeCompare(right.agentName));
  const terminalAgents = filteredAgents
    .filter((agent) => (outboundCountByAgentId.get(agent.agent_id) ?? 0) === 0)
    .map((agent) => ({
      agentId: agent.agent_id,
      agentName: agent.name || agent.agent_id,
    }))
    .sort((left, right) => left.agentName.localeCompare(right.agentName));

  const singleOwnerCapabilityRisks: OrchestrationBriefCapabilityRisk[] = [
    ...capabilityCoverageSummary.singleOwnerSkills.map((entry) => ({
      kind: 'skill' as const,
      capabilityId: entry.capabilityId,
      ownerAgents: entry.ownerAgents,
    })),
    ...capabilityCoverageSummary.singleOwnerTools.map((entry) => ({
      kind: 'tool' as const,
      capabilityId: entry.capabilityId,
      ownerAgents: entry.ownerAgents,
    })),
    ...capabilityCoverageSummary.singleOwnerMcpServers.map((entry) => ({
      kind: 'mcp' as const,
      capabilityId: entry.capabilityId,
      ownerAgents: entry.ownerAgents,
    })),
  ].sort((left, right) => {
    const order = { skill: 0, tool: 1, mcp: 2 };
    if (order[left.kind] !== order[right.kind]) {
      return order[left.kind] - order[right.kind];
    }
    return left.capabilityId.localeCompare(right.capabilityId);
  });

  const capabilityGapCount =
    capabilityCoverageSummary.missingSkillIds.length +
    capabilityCoverageSummary.blockedToolIds.length +
    capabilityCoverageSummary.missingMcpServerIds.length;
  const phaseEntries = {
    research: [] as Array<CoordinationAgentPreview & { score: number }>,
    synthesis: [] as Array<CoordinationAgentPreview & { score: number }>,
    implementation: [] as Array<CoordinationAgentPreview & { score: number }>,
    verification: [] as Array<CoordinationAgentPreview & { score: number }>,
  };
  const routingEntries = {
    coordinator: [] as Array<CoordinationAgentPreview & { score: number }>,
    research: [] as Array<CoordinationAgentPreview & { score: number }>,
    implementation: [] as Array<CoordinationAgentPreview & { score: number }>,
    verification: [] as Array<CoordinationAgentPreview & { score: number }>,
    skill: [] as Array<CoordinationAgentPreview & { score: number }>,
    tool: [] as Array<CoordinationAgentPreview & { score: number }>,
    mcp: [] as Array<CoordinationAgentPreview & { score: number }>,
  };

  for (const agent of filteredAgents) {
    const agentId = agent.agent_id;
    const inboundCount = inboundCountByAgentId.get(agentId) ?? 0;
    const outboundCount = outboundCountByAgentId.get(agentId) ?? 0;
    const summary = summaryByAgentId.get(agentId);
    const laneCount = coerceStringList(summary?.delegation_lane_ids).length;
    const strongCollaboratorCount = (summary?.recommended_collaborators ?? []).filter(
      (item) => item.fit === 'strong' || item.fit === 'good'
    ).length;
    const enabledToolCount = coerceStringList(summary?.enabled_tool_ids).length;
    const mcpServerCount = coerceStringList(summary?.mcp_server_ids).length;
    const loadedSkillCount = coerceStringList(summary?.loaded_skill_ids).length;
    const signals = [
      agent.name,
      agent.role,
      agent.description,
      agent.system_prompt,
      typeof summary?.delegation_focus === 'string' ? summary.delegation_focus : '',
      typeof summary?.review_mode === 'string' ? summary.review_mode : '',
      ...coerceStringList(agent.skill_intents),
      ...coerceStringList(summary?.delegation_lane_ids),
    ];

    let researchScore = matchesDiagnosticKeywords(signals, RESEARCH_PHASE_KEYWORDS) ? 2 : 0;
    let synthesisScore = matchesDiagnosticKeywords(signals, SYNTHESIS_PHASE_KEYWORDS) ? 2 : 0;
    let implementationScore = matchesDiagnosticKeywords(signals, IMPLEMENTATION_PHASE_KEYWORDS) ? 2 : 0;
    let verificationScore = matchesDiagnosticKeywords(signals, VERIFICATION_PHASE_KEYWORDS) ? 2 : 0;

    if (agent.cluster_auto_research) {
      researchScore += 2;
    }
    if (agent.node_kind === 'cluster') {
      synthesisScore += 2;
    }
    if (outboundCount > 1) {
      synthesisScore += 1;
    }
    if (inboundCount === 0 && outboundCount > 0) {
      synthesisScore += 1;
    }
    if (coerceStringList(summary?.enabled_tool_ids).length > 0 || agent.requires_tool_calling) {
      implementationScore += 1;
    }
    if (outboundCount === 0 && inboundCount > 0) {
      implementationScore += 1;
    }
    if (agent.cluster_auto_review) {
      verificationScore += 1;
    }
    if (summary?.review_mode) {
      verificationScore += 1;
    }

    let phaseId: OrchestrationPhaseId;
    if (verificationScore >= Math.max(3, researchScore, synthesisScore, implementationScore + 1)) {
      phaseId = 'verification';
    } else if (synthesisScore >= Math.max(3, researchScore, implementationScore, verificationScore)) {
      phaseId = 'synthesis';
    } else if (implementationScore >= Math.max(2, researchScore, verificationScore)) {
      phaseId = 'implementation';
    } else if (researchScore >= 2) {
      phaseId = 'research';
    } else if (agent.cluster_auto_research) {
      phaseId = 'research';
    } else if (agent.node_kind === 'cluster' || outboundCount > 1 || (inboundCount === 0 && outboundCount > 0)) {
      phaseId = 'synthesis';
    } else if (verificationScore > 0 || agent.cluster_auto_review) {
      phaseId = 'verification';
    } else if (inboundCount === 0) {
      phaseId = 'research';
    } else {
      phaseId = 'implementation';
    }

    const baseScoreByPhase = {
      research: researchScore,
      synthesis: synthesisScore,
      implementation: implementationScore,
      verification: verificationScore,
    };
    phaseEntries[phaseId].push({
      agentId,
      agentName: agent.name || agentId,
      score: baseScoreByPhase[phaseId],
    });

    const preview = {
      agentId,
      agentName: agent.name || agentId,
    };
    const coordinatorScore =
      synthesisScore * 10
      + outboundCount * 3
      + laneCount * 2
      + strongCollaboratorCount
      + (agent.node_kind === 'cluster' ? 2 : 0);
    const researchRoutingScore =
      researchScore * 10
      + (agent.cluster_auto_research ? 3 : 0)
      + (inboundCount === 0 ? 1 : 0);
    const implementationRoutingScore =
      implementationScore * 10
      + enabledToolCount * 2
      + mcpServerCount
      + (outboundCount === 0 && inboundCount > 0 ? 1 : 0);
    const verificationRoutingScore =
      verificationScore * 10
      + (agent.cluster_auto_review ? 2 : 0)
      + (summary?.review_mode ? 1 : 0);
    const skillRoutingScore = loadedSkillCount * 10 + researchScore + synthesisScore;
    const toolRoutingScore = enabledToolCount * 10 + (agent.requires_tool_calling ? 3 : 0) + implementationScore;
    const mcpRoutingScore = mcpServerCount * 10 + implementationScore + synthesisScore;

    routingEntries.coordinator.push({ ...preview, score: coordinatorScore });
    routingEntries.research.push({ ...preview, score: researchRoutingScore });
    routingEntries.implementation.push({ ...preview, score: implementationRoutingScore });
    routingEntries.verification.push({ ...preview, score: verificationRoutingScore });
    routingEntries.skill.push({ ...preview, score: skillRoutingScore });
    routingEntries.tool.push({ ...preview, score: toolRoutingScore });
    routingEntries.mcp.push({ ...preview, score: mcpRoutingScore });
  }

  const toPhaseSummary = (phaseId: OrchestrationPhaseId, fallbackAgents: CoordinationAgentPreview[] = []) => {
    const explicitEntries = phaseEntries[phaseId]
      .slice()
      .sort((left, right) => {
        if (left.score !== right.score) {
          return right.score - left.score;
        }
        return left.agentName.localeCompare(right.agentName);
      })
      .map(({ agentId, agentName }) => ({ agentId, agentName }));
    const phaseAgents = explicitEntries.length > 0 ? explicitEntries : fallbackAgents;
    return {
      phaseId,
      agentCount: phaseAgents.length,
      agents: phaseAgents,
    };
  };

  const implementationFallbackAgents = terminalAgents.length > 0 ? terminalAgents : filteredAgents.slice(0, 2).map((agent) => ({
    agentId: agent.agent_id,
    agentName: agent.name || agent.agent_id,
  }));
  const verificationFallbackAgents =
    reviewEnabled || terminalAgents.length === 0
      ? terminalAgents
      : terminalAgents;
  const phases: OrchestrationPhaseSummary[] = [
    toPhaseSummary('research', startAgents),
    toPhaseSummary(
      'synthesis',
      filteredAgents
        .filter((agent) => (outboundCountByAgentId.get(agent.agent_id) ?? 0) > 0)
        .slice(0, 3)
        .map((agent) => ({
          agentId: agent.agent_id,
          agentName: agent.name || agent.agent_id,
        }))
    ),
    toPhaseSummary('implementation', implementationFallbackAgents),
    toPhaseSummary('verification', verificationFallbackAgents),
  ];
  const agentRouting = {
    coordinatorAnchors: pickTopCoordinationAnchors(routingEntries.coordinator),
    researchAnchors: pickTopCoordinationAnchors(routingEntries.research),
    implementationAnchors: pickTopCoordinationAnchors(routingEntries.implementation),
    verificationAnchors: pickTopCoordinationAnchors(routingEntries.verification),
    skillCapableAnchors: pickTopCoordinationAnchors(routingEntries.skill),
    toolCapableAnchors: pickTopCoordinationAnchors(routingEntries.tool),
    mcpCapableAnchors: pickTopCoordinationAnchors(routingEntries.mcp),
  };

  const repairPriorities: OrchestrationRepairPriority[] = [
    availabilitySummary.unavailableCount > 0
      ? { priorityId: 'availability', severity: 'high' as const, count: availabilitySummary.unavailableCount }
      : availabilitySummary.limitedCount > 0
        ? { priorityId: 'availability', severity: 'medium' as const, count: availabilitySummary.limitedCount }
        : null,
    capabilityGapCount > 0
      ? { priorityId: 'capability_gaps', severity: 'high' as const, count: capabilityGapCount }
      : null,
    policyRepairSummary.agentCount > 0
      ? { priorityId: 'policy_repair', severity: 'medium' as const, count: policyRepairSummary.agentCount }
      : null,
    roleProfileSummary.actionableAgentCount > 0 || roleProfileSummary.missingSkillAgentCount > 0
      ? {
          priorityId: 'role_profile_alignment',
          severity: roleProfileSummary.actionableAgentCount > 0 ? 'medium' as const : 'low' as const,
          count: roleProfileSummary.actionableAgentCount + roleProfileSummary.missingSkillAgentCount,
        }
      : null,
    collaborationSummary.weakEdgeCount > 0
      ? { priorityId: 'weak_handoffs', severity: 'medium' as const, count: collaborationSummary.weakEdgeCount }
      : null,
    collaborationSummary.bestNextCount > 0
      ? { priorityId: 'best_next_handoffs', severity: 'low' as const, count: collaborationSummary.bestNextCount }
      : null,
    topologySummary.isolatedAgentCount + topologySummary.underconnectedAgentCount > 0
      ? {
          priorityId: 'connectivity',
          severity: 'low' as const,
          count: topologySummary.isolatedAgentCount + topologySummary.underconnectedAgentCount,
        }
      : null,
    singleOwnerCapabilityRisks.length > 0
      ? {
          priorityId: 'single_owner_capabilities',
          severity: 'low' as const,
          count: singleOwnerCapabilityRisks.length,
        }
      : null,
    !reviewEnabled ? { priorityId: 'review_path', severity: 'low' as const, count: 1 } : null,
  ]
    .filter((item): item is OrchestrationRepairPriority => item !== null)
    .sort((left, right) => {
      const order = { high: 0, medium: 1, low: 2 };
      if (order[left.severity] !== order[right.severity]) {
        return order[left.severity] - order[right.severity];
      }
      return right.count - left.count;
    });
  const readiness =
    availabilitySummary.unavailableCount > 0
      ? 'blocked'
      : collaborationSummary.weakEdgeCount > 0 ||
          policyRepairSummary.agentCount > 0 ||
          roleProfileSummary.actionableAgentCount > 0 ||
          capabilityGapCount > 0
        ? 'repair'
        : roleProfileSummary.missingSkillAgentCount > 0 ||
            topologySummary.isolatedAgentCount > 0 ||
            topologySummary.underconnectedAgentCount > 0 ||
            singleOwnerCapabilityRisks.length > 0 ||
            !reviewEnabled ||
            startAgents.length === 0 ||
            terminalAgents.length === 0
          ? 'watch'
          : 'ready';

  return {
    totalAgentCount: filteredAgents.length,
    executionStepCount,
    reviewEnabled,
    readiness,
    startAgents,
    terminalAgents,
    sharedLaneCount: topologySummary.sharedLaneCount,
    singleOwnerCapabilityCount: singleOwnerCapabilityRisks.length,
    singleOwnerCapabilityRisks,
    unavailableCount: availabilitySummary.unavailableCount,
    limitedAvailabilityCount: availabilitySummary.limitedCount,
    policyRepairAgentCount: policyRepairSummary.agentCount,
    roleProfileDriftAgentCount: roleProfileSummary.actionableAgentCount,
    roleProfileOverlapRiskCount: 0,
    weakEdgeCount: collaborationSummary.weakEdgeCount,
    bestNextCount: collaborationSummary.bestNextCount,
    capabilityGapCount,
    isolatedAgentCount: topologySummary.isolatedAgentCount,
    underconnectedAgentCount: topologySummary.underconnectedAgentCount,
    phases,
    repairPriorities,
    agentRouting,
  };
}

export function buildAvailabilityScopeSummary(
  summaries: HarnessAgentCapabilitySummaryDTO[],
  agentNameById: Map<string, string>,
  selectedAgentIds: string[] | null
): AvailabilityScopeSummary {
  const selectedSet =
    selectedAgentIds === null ? null : new Set(selectedAgentIds.map((agentId) => String(agentId || '').trim()).filter(Boolean));
  if (selectedSet && selectedSet.size === 0) {
    return {
      totalCount: 0,
      availableCount: 0,
      limitedCount: 0,
      unavailableCount: 0,
      flaggedAgents: [],
      unavailableAgentNames: [],
    };
  }

  const flaggedAgents: AgentAvailabilityDiagnostic[] = [];
  let totalCount = 0;
  let availableCount = 0;
  let limitedCount = 0;
  let unavailableCount = 0;

  for (const summary of summaries) {
    const agentId = typeof summary.agent_id === 'string' ? summary.agent_id : '';
    if (!agentId || (selectedSet && !selectedSet.has(agentId))) {
      continue;
    }
    totalCount += 1;
    const status = resolveCapabilityAvailabilityStatus(summary);
    if (status === 'unavailable') {
      unavailableCount += 1;
    } else if (status === 'limited') {
      limitedCount += 1;
    } else {
      availableCount += 1;
    }
    const blockers = coerceStringList(summary.availability_blockers);
    const warnings = coerceStringList(summary.availability_warnings);
    if (status !== 'available' || blockers.length > 0 || warnings.length > 0) {
      flaggedAgents.push({
        agentId,
        agentName: agentNameById.get(agentId) ?? agentId,
        status,
        blockers,
        warnings,
      });
    }
  }

  flaggedAgents.sort((left, right) => {
    const order = { unavailable: 0, limited: 1, available: 2 };
    if (order[left.status] !== order[right.status]) {
      return order[left.status] - order[right.status];
    }
    return left.agentName.localeCompare(right.agentName);
  });

  return {
    totalCount,
    availableCount,
    limitedCount,
    unavailableCount,
    flaggedAgents,
    unavailableAgentNames: flaggedAgents
      .filter((agent) => agent.status === 'unavailable')
      .map((agent) => agent.agentName),
  };
}

export function formatAvailabilityAgentPreview(agentNames: string[], limit = 4) {
  if (agentNames.length <= limit) {
    return agentNames.join(', ');
  }
  return `${agentNames.slice(0, limit).join(', ')} (+${agentNames.length - limit})`;
}

export function availabilityScopeToneClasses(summary: { limitedCount: number; unavailableCount: number }) {
  if (summary.unavailableCount > 0) {
    return {
      panel: 'border-rose-200 bg-rose-50/60',
      accent: 'text-rose-700',
    };
  }
  if (summary.limitedCount > 0) {
    return {
      panel: 'border-amber-200 bg-amber-50/60',
      accent: 'text-amber-700',
    };
  }
  return {
    panel: 'border-emerald-200 bg-emerald-50/60',
    accent: 'text-emerald-700',
  };
}
