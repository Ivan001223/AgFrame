import type {
  HarnessAgentCapabilitySummaryDTO,
  HarnessCanvasAgentDTO,
  HarnessProjectDetailDTO,
} from '@/domains/harness/hooks';

const COORDINATOR_ALLOWED_DIRECT_TOOL_IDS = new Set(['get_current_time']);
const ROLE_PROFILE_SPLIT_LANE_HINTS: Record<string, string[]> = {
  coordinator: ['coordination', 'memory', 'grounding'],
  research: ['research', 'grounding', 'repository'],
  implementation: ['implementation', 'repository', 'grounding'],
  verification: ['verification', 'grounding', 'repository'],
  generalist: ['grounding', 'implementation', 'verification'],
};

function normalizeHarnessIdentifier(value: string | null | undefined) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

function coerceStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter(Boolean);
}

function uniqueNormalizedValues(values: string[] | undefined) {
  return Array.from(new Set((values ?? []).map((value) => normalizeHarnessIdentifier(value)).filter(Boolean)));
}

function isCoordinatorRestrictionCandidate(
  summary: HarnessAgentCapabilitySummaryDTO | Record<string, unknown> | null | undefined
) {
  if (!summary) {
    return false;
  }
  const summaryRecord = summary as Record<string, unknown>;
  const delegationContract =
    summaryRecord.delegation_contract && typeof summaryRecord.delegation_contract === 'object'
      ? (summaryRecord.delegation_contract as Record<string, unknown>)
      : null;
  const primaryRoleMode = normalizeHarnessIdentifier(
    typeof delegationContract?.primary_role_mode === 'string'
      ? delegationContract.primary_role_mode
      : typeof summaryRecord.primary_role_mode === 'string'
        ? summaryRecord.primary_role_mode
        : ''
  );
  const shouldCoordinateParallelWork = Boolean(
    delegationContract?.should_coordinate_parallel_work ?? summaryRecord.should_coordinate_parallel_work
  );
  const shouldProduceFinalOutput = Boolean(
    delegationContract?.should_produce_final_output ?? summaryRecord.should_produce_final_output
  );
  return !shouldProduceFinalOutput && (primaryRoleMode === 'coordinator' || shouldCoordinateParallelWork);
}

function mergeNormalizedPolicyValues(currentValues: string[] | undefined, additions: string[]) {
  const nextValues: string[] = [];
  const seen = new Set<string>();
  for (const value of currentValues ?? []) {
    const normalizedValue = normalizeHarnessIdentifier(value);
    if (!normalizedValue || seen.has(normalizedValue)) {
      continue;
    }
    seen.add(normalizedValue);
    nextValues.push(normalizedValue);
  }
  let addedCount = 0;
  for (const value of additions) {
    const normalizedValue = normalizeHarnessIdentifier(value);
    if (!normalizedValue || seen.has(normalizedValue)) {
      continue;
    }
    seen.add(normalizedValue);
    nextValues.push(normalizedValue);
    addedCount += 1;
  }
  return {
    values: nextValues,
    addedCount,
  };
}

function removeNormalizedPolicyValues(currentValues: string[] | undefined, removals: string[]) {
  const removalSet = new Set(removals.map((value) => normalizeHarnessIdentifier(value)).filter(Boolean));
  if (removalSet.size === 0) {
    return {
      values: [...(currentValues ?? [])],
      removedCount: 0,
    };
  }
  const nextValues: string[] = [];
  let removedCount = 0;
  for (const value of currentValues ?? []) {
    const normalizedValue = normalizeHarnessIdentifier(value);
    if (!normalizedValue) {
      continue;
    }
    if (removalSet.has(normalizedValue)) {
      removedCount += 1;
      continue;
    }
    nextValues.push(normalizedValue);
  }
  return {
    values: nextValues,
    removedCount,
  };
}

export type AgentPolicyRepairDiagnostic = {
  agentId: string;
  agentName: string;
  allowToolIds: string[];
  allowMcpServerIds: string[];
  denyToolIds: string[];
  denyMcpServerIds: string[];
};

export type PolicyRepairScopeSummary = {
  totalCount: number;
  agentCount: number;
  toolSuggestionCount: number;
  mcpSuggestionCount: number;
  diagnostics: AgentPolicyRepairDiagnostic[];
};

export type AgentRoleProfileDiagnostic = {
  agentId: string;
  agentName: string;
  availableSkillIds: string[];
  missingSkillIds: string[];
  toolIds: string[];
  mcpServerIds: string[];
  denyToolIds: string[];
  denyMcpServerIds: string[];
};

export type RoleProfileScopeSummary = {
  totalCount: number;
  actionableAgentCount: number;
  missingSkillAgentCount: number;
  availableSkillCount: number;
  missingSkillCount: number;
  toolSuggestionCount: number;
  mcpSuggestionCount: number;
  diagnostics: AgentRoleProfileDiagnostic[];
};

export type AgentRoleProfilePeerDiagnostic = {
  peerAgentId: string;
  peerAgentName: string;
  profileId: 'coordinator' | 'research' | 'implementation' | 'verification' | 'generalist';
  sharedLaneIds: string[];
  selectedFocusLaneIds: string[];
  peerFocusLaneIds: string[];
  selectedUniqueSkillIds: string[];
  peerUniqueSkillIds: string[];
  selectedUniqueToolIds: string[];
  peerUniqueToolIds: string[];
  selectedUniqueMcpServerIds: string[];
  peerUniqueMcpServerIds: string[];
};

export function computeActionableToolPolicySuggestionIds(
  agent: HarnessCanvasAgentDTO | null | undefined,
  blockedToolIds: string[]
) {
  if (!agent) {
    return [];
  }
  const normalizedBlockedToolIds = Array.from(
    new Set(blockedToolIds.map((value) => normalizeHarnessIdentifier(value)).filter(Boolean))
  );
  if (normalizedBlockedToolIds.length === 0) {
    return [];
  }
  const allowedToolIds = new Set(
    (agent.allowed_tool_ids ?? []).map((value) => normalizeHarnessIdentifier(value)).filter(Boolean)
  );
  const deniedToolIds = new Set(
    (agent.denied_tool_ids ?? []).map((value) => normalizeHarnessIdentifier(value)).filter(Boolean)
  );
  const hasExplicitAllowedTools = allowedToolIds.size > 0;
  return normalizedBlockedToolIds.filter(
    (toolId) => (hasExplicitAllowedTools && !allowedToolIds.has(toolId)) || deniedToolIds.has(toolId)
  );
}

export function computeActionableMcpPolicySuggestionIds(
  agent: HarnessCanvasAgentDTO | null | undefined,
  blockedMcpServerIds: string[]
) {
  if (!agent) {
    return [];
  }
  const normalizedBlockedMcpServerIds = Array.from(
    new Set(blockedMcpServerIds.map((value) => normalizeHarnessIdentifier(value)).filter(Boolean))
  );
  if (normalizedBlockedMcpServerIds.length === 0) {
    return [];
  }
  const allowedMcpServerIds = new Set(
    (agent.allowed_mcp_server_ids ?? []).map((value) => normalizeHarnessIdentifier(value)).filter(Boolean)
  );
  const deniedMcpServerIds = new Set(
    (agent.denied_mcp_server_ids ?? []).map((value) => normalizeHarnessIdentifier(value)).filter(Boolean)
  );
  const hasExplicitAllowedMcpServers = allowedMcpServerIds.size > 0;
  return normalizedBlockedMcpServerIds.filter(
    (serverId) => (hasExplicitAllowedMcpServers && !allowedMcpServerIds.has(serverId)) || deniedMcpServerIds.has(serverId)
  );
}

export function computeCoordinatorToolPolicyRestrictionIds(
  agent: HarnessCanvasAgentDTO | null | undefined,
  summary: HarnessAgentCapabilitySummaryDTO | Record<string, unknown> | null | undefined
) {
  if (!agent || !summary || !isCoordinatorRestrictionCandidate(summary)) {
    return [];
  }
  const summaryRecord = summary as Record<string, unknown>;
  const executionContract =
    summaryRecord.execution_contract && typeof summaryRecord.execution_contract === 'object'
      ? (summaryRecord.execution_contract as Record<string, unknown>)
      : null;
  const toolAccessMode = normalizeHarnessIdentifier(
    typeof executionContract?.tool_access_mode === 'string'
      ? executionContract.tool_access_mode
      : typeof summaryRecord.tool_access_mode === 'string'
        ? summaryRecord.tool_access_mode
        : ''
  );
  let executableToolIds = uniqueNormalizedValues(
    coerceStringList(executionContract?.executable_tool_ids)
  );
  if (executableToolIds.length === 0 && (toolAccessMode === 'direct_execution' || toolAccessMode === 'mixed')) {
    executableToolIds = uniqueNormalizedValues(coerceStringList(summaryRecord.enabled_tool_ids));
  }
  if (executableToolIds.length === 0) {
    return [];
  }
  const deniedToolIds = new Set(
    (agent.denied_tool_ids ?? []).map((value) => normalizeHarnessIdentifier(value)).filter(Boolean)
  );
  return executableToolIds.filter(
    (toolId) => !COORDINATOR_ALLOWED_DIRECT_TOOL_IDS.has(toolId) && !deniedToolIds.has(toolId)
  );
}

export function computeCoordinatorMcpPolicyRestrictionIds(
  agent: HarnessCanvasAgentDTO | null | undefined,
  summary: HarnessAgentCapabilitySummaryDTO | Record<string, unknown> | null | undefined
) {
  if (!agent || !summary || !isCoordinatorRestrictionCandidate(summary)) {
    return [];
  }
  const summaryRecord = summary as Record<string, unknown>;
  const executionContract =
    summaryRecord.execution_contract && typeof summaryRecord.execution_contract === 'object'
      ? (summaryRecord.execution_contract as Record<string, unknown>)
      : null;
  const mcpAccessMode = normalizeHarnessIdentifier(
    typeof executionContract?.mcp_access_mode === 'string'
      ? executionContract.mcp_access_mode
      : typeof summaryRecord.mcp_access_mode === 'string'
        ? summaryRecord.mcp_access_mode
        : ''
  );
  let planningOnlyMcpServerIds = uniqueNormalizedValues(
    coerceStringList(executionContract?.planning_only_mcp_server_ids)
  );
  if (planningOnlyMcpServerIds.length === 0 && mcpAccessMode === 'planning_only') {
    planningOnlyMcpServerIds = uniqueNormalizedValues(coerceStringList(summaryRecord.mcp_server_ids));
  }
  if (planningOnlyMcpServerIds.length === 0) {
    return [];
  }
  const deniedMcpServerIds = new Set(
    (agent.denied_mcp_server_ids ?? []).map((value) => normalizeHarnessIdentifier(value)).filter(Boolean)
  );
  return planningOnlyMcpServerIds.filter((serverId) => !deniedMcpServerIds.has(serverId));
}

function computeMissingRoleProfileAvailableSkillIds(
  summary: HarnessAgentCapabilitySummaryDTO,
  availableSkillIds: string[]
) {
  const loadedSkillIds = new Set(uniqueNormalizedValues(coerceStringList(summary.loaded_skill_ids)));
  return uniqueNormalizedValues(availableSkillIds).filter((skillId) => !loadedSkillIds.has(skillId));
}

function computeMissingRoleProfileToolIds(
  summary: HarnessAgentCapabilitySummaryDTO,
  suggestedToolIds: string[]
) {
  const currentToolIds = new Set(
    uniqueNormalizedValues([
      ...coerceStringList(summary.enabled_tool_ids),
      ...coerceStringList(summary.provider_limited_tool_ids),
    ])
  );
  return uniqueNormalizedValues(suggestedToolIds).filter((toolId) => !currentToolIds.has(toolId));
}

function computeMissingRoleProfileMcpServerIds(
  summary: HarnessAgentCapabilitySummaryDTO,
  suggestedMcpServerIds: string[]
) {
  const currentMcpServerIds = new Set(uniqueNormalizedValues(coerceStringList(summary.mcp_server_ids)));
  return uniqueNormalizedValues(suggestedMcpServerIds).filter((serverId) => !currentMcpServerIds.has(serverId));
}

function pickRoleProfileFocusLaneIds(
  profileId: AgentRoleProfilePeerDiagnostic['profileId'],
  sharedLaneIds: string[],
  preferredLaneIds: string[],
  takenLaneIds: string[] = []
) {
  const taken = new Set(takenLaneIds.map((value) => normalizeHarnessIdentifier(value)).filter(Boolean));
  const values: string[] = [];
  for (const laneId of [...preferredLaneIds, ...sharedLaneIds, ...(ROLE_PROFILE_SPLIT_LANE_HINTS[profileId] ?? [])]) {
    const normalizedLaneId = normalizeHarnessIdentifier(laneId);
    if (!normalizedLaneId || taken.has(normalizedLaneId) || values.includes(normalizedLaneId)) {
      continue;
    }
    values.push(normalizedLaneId);
  }
  return values.slice(0, 1);
}

function normalizeRoleProfileId(
  value: unknown
): AgentRoleProfilePeerDiagnostic['profileId'] {
  const normalized = normalizeHarnessIdentifier(typeof value === 'string' ? value : '');
  if (
    normalized === 'coordinator'
    || normalized === 'research'
    || normalized === 'implementation'
    || normalized === 'verification'
    || normalized === 'generalist'
  ) {
    return normalized;
  }
  return 'generalist';
}

export function buildRoleProfilePeerOverlapDiagnostics(
  summaries: HarnessAgentCapabilitySummaryDTO[],
  agentNameById: Map<string, string>,
  selectedAgentId: string | null | undefined
): AgentRoleProfilePeerDiagnostic[] {
  const normalizedSelectedAgentId = String(selectedAgentId || '').trim();
  if (!normalizedSelectedAgentId) {
    return [];
  }
  const summaryByAgentId = new Map(
    summaries
      .map((summary) => [typeof summary.agent_id === 'string' ? summary.agent_id : '', summary] as const)
      .filter(([agentId]) => Boolean(agentId))
  );
  const selectedSummary = summaryByAgentId.get(normalizedSelectedAgentId);
  const selectedRoleProfile =
    selectedSummary?.role_profile_suggestion && typeof selectedSummary.role_profile_suggestion === 'object'
      ? selectedSummary.role_profile_suggestion
      : null;
  if (!selectedSummary || !selectedRoleProfile) {
    return [];
  }
  const profileId = normalizeRoleProfileId(selectedRoleProfile.profile_id);
  const selectedLaneIds = uniqueNormalizedValues(coerceStringList(selectedSummary.delegation_lane_ids)).filter(
    (laneId) => laneId !== 'generalist' && laneId !== 'reasoning_only'
  );
  const selectedSkillIds = new Set(uniqueNormalizedValues(coerceStringList(selectedSummary.loaded_skill_ids)));
  const selectedToolIds = new Set(
    uniqueNormalizedValues([
      ...coerceStringList(selectedSummary.enabled_tool_ids),
      ...coerceStringList(selectedSummary.provider_limited_tool_ids),
    ])
  );
  const selectedMcpServerIds = new Set(uniqueNormalizedValues(coerceStringList(selectedSummary.mcp_server_ids)));
  const diagnostics: AgentRoleProfilePeerDiagnostic[] = [];

  for (const summary of summaries) {
    const peerAgentId = typeof summary.agent_id === 'string' ? summary.agent_id : '';
    if (!peerAgentId || peerAgentId === normalizedSelectedAgentId) {
      continue;
    }
    const peerRoleProfile =
      summary.role_profile_suggestion && typeof summary.role_profile_suggestion === 'object'
        ? summary.role_profile_suggestion
        : null;
    if (!peerRoleProfile || normalizeRoleProfileId(peerRoleProfile.profile_id) !== profileId) {
      continue;
    }

    const peerLaneIds = uniqueNormalizedValues(coerceStringList(summary.delegation_lane_ids)).filter(
      (laneId) => laneId !== 'generalist' && laneId !== 'reasoning_only'
    );
    const sharedLaneIds = selectedLaneIds.filter((laneId) => peerLaneIds.includes(laneId));
    if (sharedLaneIds.length === 0) {
      continue;
    }

    const peerSkillIds = new Set(uniqueNormalizedValues(coerceStringList(summary.loaded_skill_ids)));
    const peerToolIds = new Set(
      uniqueNormalizedValues([
        ...coerceStringList(summary.enabled_tool_ids),
        ...coerceStringList(summary.provider_limited_tool_ids),
      ])
    );
    const peerMcpServerIds = new Set(uniqueNormalizedValues(coerceStringList(summary.mcp_server_ids)));

    const selectedUniqueLaneIds = selectedLaneIds.filter((laneId) => !peerLaneIds.includes(laneId));
    const peerUniqueLaneIds = peerLaneIds.filter((laneId) => !selectedLaneIds.includes(laneId));
    const selectedUniqueSkillIds = Array.from(selectedSkillIds).filter((skillId) => !peerSkillIds.has(skillId));
    const peerUniqueSkillIds = Array.from(peerSkillIds).filter((skillId) => !selectedSkillIds.has(skillId));
    const selectedUniqueToolIds = Array.from(selectedToolIds).filter((toolId) => !peerToolIds.has(toolId));
    const peerUniqueToolIds = Array.from(peerToolIds).filter((toolId) => !selectedToolIds.has(toolId));
    const selectedUniqueMcpServerIds = Array.from(selectedMcpServerIds).filter(
      (serverId) => !peerMcpServerIds.has(serverId)
    );
    const peerUniqueMcpServerIds = Array.from(peerMcpServerIds).filter(
      (serverId) => !selectedMcpServerIds.has(serverId)
    );
    const capabilityDeltaCount =
      selectedUniqueSkillIds.length
      + peerUniqueSkillIds.length
      + selectedUniqueToolIds.length
      + peerUniqueToolIds.length
      + selectedUniqueMcpServerIds.length
      + peerUniqueMcpServerIds.length;

    const overlapRisk =
      (sharedLaneIds.length >= 2 && selectedUniqueLaneIds.length + peerUniqueLaneIds.length <= 1 && capabilityDeltaCount <= 2)
      || (selectedUniqueLaneIds.length + peerUniqueLaneIds.length === 0 && capabilityDeltaCount === 0)
      || (profileId === 'coordinator'
        && sharedLaneIds.includes('coordination')
        && selectedUniqueLaneIds.length + peerUniqueLaneIds.length === 0
        && capabilityDeltaCount <= 2);

    if (!overlapRisk) {
      continue;
    }

    const selectedFocusLaneIds = pickRoleProfileFocusLaneIds(profileId, sharedLaneIds, selectedUniqueLaneIds);
    const peerFocusLaneIds = pickRoleProfileFocusLaneIds(
      profileId,
      sharedLaneIds,
      peerUniqueLaneIds,
      selectedFocusLaneIds
    );
    diagnostics.push({
      peerAgentId,
      peerAgentName: agentNameById.get(peerAgentId) ?? peerAgentId,
      profileId,
      sharedLaneIds,
      selectedFocusLaneIds,
      peerFocusLaneIds,
      selectedUniqueSkillIds,
      peerUniqueSkillIds,
      selectedUniqueToolIds,
      peerUniqueToolIds,
      selectedUniqueMcpServerIds,
      peerUniqueMcpServerIds,
    });
  }

  diagnostics.sort((left, right) => {
    if (left.sharedLaneIds.length !== right.sharedLaneIds.length) {
      return right.sharedLaneIds.length - left.sharedLaneIds.length;
    }
    return left.peerAgentName.localeCompare(right.peerAgentName);
  });
  return diagnostics;
}

export function buildPolicyRepairScopeSummary(
  graph: NonNullable<HarnessProjectDetailDTO['graph_json']> | undefined,
  summaries: HarnessAgentCapabilitySummaryDTO[],
  agentNameById: Map<string, string>,
  selectedAgentIds: string[] | null
): PolicyRepairScopeSummary {
  const selectedSet =
    selectedAgentIds === null
      ? null
      : new Set(selectedAgentIds.map((agentId) => String(agentId || '').trim()).filter(Boolean));
  if (selectedSet && selectedSet.size === 0) {
    return {
      totalCount: 0,
      agentCount: 0,
      toolSuggestionCount: 0,
      mcpSuggestionCount: 0,
      diagnostics: [],
    };
  }

  const agentById = new Map((graph?.agents ?? []).map((agent) => [agent.agent_id, agent]));
  const diagnostics: AgentPolicyRepairDiagnostic[] = [];
  let totalCount = 0;
  let toolSuggestionCount = 0;
  let mcpSuggestionCount = 0;

  for (const summary of summaries) {
    const agentId = typeof summary.agent_id === 'string' ? summary.agent_id : '';
    if (!agentId || (selectedSet && !selectedSet.has(agentId))) {
      continue;
    }
    totalCount += 1;
    const agent = agentById.get(agentId) ?? null;
    const allowToolIds = computeActionableToolPolicySuggestionIds(agent, coerceStringList(summary.policy_blocked_tool_ids));
    const allowMcpServerIds = computeActionableMcpPolicySuggestionIds(
      agent,
      coerceStringList(summary.policy_blocked_mcp_server_ids)
    );
    const denyToolIds = computeCoordinatorToolPolicyRestrictionIds(agent, summary);
    const denyMcpServerIds = computeCoordinatorMcpPolicyRestrictionIds(agent, summary);
    if (
      allowToolIds.length === 0
      && allowMcpServerIds.length === 0
      && denyToolIds.length === 0
      && denyMcpServerIds.length === 0
    ) {
      continue;
    }
    toolSuggestionCount += allowToolIds.length + denyToolIds.length;
    mcpSuggestionCount += allowMcpServerIds.length + denyMcpServerIds.length;
    diagnostics.push({
      agentId,
      agentName: agentNameById.get(agentId) ?? agentId,
      allowToolIds,
      allowMcpServerIds,
      denyToolIds,
      denyMcpServerIds,
    });
  }

  diagnostics.sort((left, right) => {
    const leftCount =
      left.allowToolIds.length
      + left.allowMcpServerIds.length
      + left.denyToolIds.length
      + left.denyMcpServerIds.length;
    const rightCount =
      right.allowToolIds.length
      + right.allowMcpServerIds.length
      + right.denyToolIds.length
      + right.denyMcpServerIds.length;
    if (leftCount !== rightCount) {
      return rightCount - leftCount;
    }
    return left.agentName.localeCompare(right.agentName);
  });

  return {
    totalCount,
    agentCount: diagnostics.length,
    toolSuggestionCount,
    mcpSuggestionCount,
    diagnostics,
  };
}

export function applyAgentCapabilityPolicySuggestions(
  graph: NonNullable<HarnessProjectDetailDTO['graph_json']>,
  {
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
  }
) {
  const normalizedSkillIds = Array.from(
    new Set(skillIds.map((value) => normalizeHarnessIdentifier(value)).filter(Boolean))
  );
  const normalizedToolIds = Array.from(
    new Set(toolIds.map((value) => normalizeHarnessIdentifier(value)).filter(Boolean))
  );
  const normalizedMcpServerIds = Array.from(
    new Set(mcpServerIds.map((value) => normalizeHarnessIdentifier(value)).filter(Boolean))
  );
  const normalizedDeniedToolIds = Array.from(
    new Set(denyToolIds.map((value) => normalizeHarnessIdentifier(value)).filter(Boolean))
  );
  const normalizedDeniedMcpServerIds = Array.from(
    new Set(denyMcpServerIds.map((value) => normalizeHarnessIdentifier(value)).filter(Boolean))
  );
  if (
    !agentId
    || (
      normalizedSkillIds.length === 0
      && normalizedToolIds.length === 0
      && normalizedMcpServerIds.length === 0
      && normalizedDeniedToolIds.length === 0
      && normalizedDeniedMcpServerIds.length === 0
    )
  ) {
    return {
      changed: false,
      graph,
      skillChangeCount: 0,
      toolChangeCount: 0,
      mcpChangeCount: 0,
    };
  }

  let changed = false;
  let skillChangeCount = 0;
  let toolChangeCount = 0;
  let mcpChangeCount = 0;
  let matchedAgent = false;

  const nextAgents = (graph.agents ?? []).map((agent) => {
    if (agent.agent_id !== agentId) {
      return agent;
    }
    matchedAgent = true;
    let nextAgent = { ...agent };

    if (normalizedSkillIds.length > 0) {
      const skillValues = mergeNormalizedPolicyValues(nextAgent.skill_ids, normalizedSkillIds);
      if (skillValues.addedCount > 0) {
        nextAgent = {
          ...nextAgent,
          skill_ids: skillValues.values,
        };
        skillChangeCount += skillValues.addedCount;
        changed = true;
      }
    }

    if (normalizedToolIds.length > 0) {
      const hadExplicitAllowedTools = (nextAgent.allowed_tool_ids ?? []).some((value) =>
        normalizeHarnessIdentifier(value)
      );
      const allowedTools = (forceAllowToolIds || hadExplicitAllowedTools)
        ? mergeNormalizedPolicyValues(nextAgent.allowed_tool_ids, normalizedToolIds)
        : { values: [...(nextAgent.allowed_tool_ids ?? [])], addedCount: 0 };
      const deniedTools = removeNormalizedPolicyValues(nextAgent.denied_tool_ids, normalizedToolIds);
      const localToolChanges = allowedTools.addedCount + deniedTools.removedCount;
      if (localToolChanges > 0) {
        nextAgent = {
          ...nextAgent,
          allowed_tool_ids: allowedTools.values,
          denied_tool_ids: deniedTools.values,
        };
        toolChangeCount += localToolChanges;
        changed = true;
      }
    }

    if (normalizedDeniedToolIds.length > 0) {
      const deniedTools = mergeNormalizedPolicyValues(nextAgent.denied_tool_ids, normalizedDeniedToolIds);
      const allowedTools = removeNormalizedPolicyValues(nextAgent.allowed_tool_ids, normalizedDeniedToolIds);
      const localToolChanges = deniedTools.addedCount + allowedTools.removedCount;
      if (localToolChanges > 0) {
        nextAgent = {
          ...nextAgent,
          allowed_tool_ids: allowedTools.values,
          denied_tool_ids: deniedTools.values,
        };
        toolChangeCount += localToolChanges;
        changed = true;
      }
    }

    if (normalizedMcpServerIds.length > 0) {
      const hadExplicitAllowedMcpServers = (nextAgent.allowed_mcp_server_ids ?? []).some((value) =>
        normalizeHarnessIdentifier(value)
      );
      const allowedMcpServers = (forceAllowMcpServerIds || hadExplicitAllowedMcpServers)
        ? mergeNormalizedPolicyValues(nextAgent.allowed_mcp_server_ids, normalizedMcpServerIds)
        : { values: [...(nextAgent.allowed_mcp_server_ids ?? [])], addedCount: 0 };
      const deniedMcpServers = removeNormalizedPolicyValues(nextAgent.denied_mcp_server_ids, normalizedMcpServerIds);
      const localMcpChanges = allowedMcpServers.addedCount + deniedMcpServers.removedCount;
      if (localMcpChanges > 0) {
        nextAgent = {
          ...nextAgent,
          allowed_mcp_server_ids: allowedMcpServers.values,
          denied_mcp_server_ids: deniedMcpServers.values,
        };
        mcpChangeCount += localMcpChanges;
        changed = true;
      }
    }

    if (normalizedDeniedMcpServerIds.length > 0) {
      const deniedMcpServers = mergeNormalizedPolicyValues(nextAgent.denied_mcp_server_ids, normalizedDeniedMcpServerIds);
      const allowedMcpServers = removeNormalizedPolicyValues(nextAgent.allowed_mcp_server_ids, normalizedDeniedMcpServerIds);
      const localMcpChanges = deniedMcpServers.addedCount + allowedMcpServers.removedCount;
      if (localMcpChanges > 0) {
        nextAgent = {
          ...nextAgent,
          allowed_mcp_server_ids: allowedMcpServers.values,
          denied_mcp_server_ids: deniedMcpServers.values,
        };
        mcpChangeCount += localMcpChanges;
        changed = true;
      }
    }

    return nextAgent;
  });

  if (!matchedAgent || !changed) {
    return {
      changed: false,
      graph,
      skillChangeCount: 0,
      toolChangeCount: 0,
      mcpChangeCount: 0,
    };
  }

  return {
    changed: true,
    graph: {
      ...graph,
      agents: nextAgents,
    },
    skillChangeCount,
    toolChangeCount,
    mcpChangeCount,
  };
}

export function buildRoleProfileScopeSummary(
  graph: NonNullable<HarnessProjectDetailDTO['graph_json']> | undefined,
  summaries: HarnessAgentCapabilitySummaryDTO[],
  agentNameById: Map<string, string>,
  selectedAgentIds: string[] | null
): RoleProfileScopeSummary {
  const selectedSet =
    selectedAgentIds === null
      ? null
      : new Set(selectedAgentIds.map((agentId) => String(agentId || '').trim()).filter(Boolean));
  if (selectedSet && selectedSet.size === 0) {
    return {
      totalCount: 0,
      actionableAgentCount: 0,
      missingSkillAgentCount: 0,
      availableSkillCount: 0,
      missingSkillCount: 0,
      toolSuggestionCount: 0,
      mcpSuggestionCount: 0,
      diagnostics: [],
    };
  }

  const agentById = new Map((graph?.agents ?? []).map((agent) => [agent.agent_id, agent]));
  const diagnostics: AgentRoleProfileDiagnostic[] = [];
  let totalCount = 0;
  let actionableAgentCount = 0;
  let missingSkillAgentCount = 0;
  let availableSkillCount = 0;
  let missingSkillCount = 0;
  let toolSuggestionCount = 0;
  let mcpSuggestionCount = 0;

  for (const summary of summaries) {
    const agentId = typeof summary.agent_id === 'string' ? summary.agent_id : '';
    if (!agentId || (selectedSet && !selectedSet.has(agentId))) {
      continue;
    }
    totalCount += 1;
    const agent = agentById.get(agentId) ?? null;
    const roleProfile =
      summary.role_profile_suggestion && typeof summary.role_profile_suggestion === 'object'
        ? summary.role_profile_suggestion
        : null;
    if (!roleProfile) {
      continue;
    }
    const availableSkillIds = computeMissingRoleProfileAvailableSkillIds(
      summary,
      coerceStringList(roleProfile.available_skill_ids)
    );
    const missingSkillIds = coerceStringList(roleProfile.missing_skill_ids);
    const toolIds = computeMissingRoleProfileToolIds(summary, coerceStringList(roleProfile.suggested_tool_ids));
    const mcpServerIds = computeMissingRoleProfileMcpServerIds(
      summary,
      coerceStringList(roleProfile.suggested_mcp_server_ids)
    );
    const denyToolIds = computeCoordinatorToolPolicyRestrictionIds(agent, summary);
    const denyMcpServerIds = computeCoordinatorMcpPolicyRestrictionIds(agent, summary);
    const actionableApplyCount =
      availableSkillIds.length + toolIds.length + mcpServerIds.length + denyToolIds.length + denyMcpServerIds.length;
    if (actionableApplyCount <= 0 && missingSkillIds.length <= 0) {
      continue;
    }
    if (actionableApplyCount > 0) {
      actionableAgentCount += 1;
    }
    if (missingSkillIds.length > 0) {
      missingSkillAgentCount += 1;
    }
    availableSkillCount += availableSkillIds.length;
    missingSkillCount += missingSkillIds.length;
    toolSuggestionCount += toolIds.length + denyToolIds.length;
    mcpSuggestionCount += mcpServerIds.length + denyMcpServerIds.length;
    diagnostics.push({
      agentId,
      agentName: agentNameById.get(agentId) ?? agent?.name ?? agentId,
      availableSkillIds,
      missingSkillIds,
      toolIds,
      mcpServerIds,
      denyToolIds,
      denyMcpServerIds,
    });
  }

  diagnostics.sort((left, right) => {
    const leftCount =
      left.availableSkillIds.length
      + left.missingSkillIds.length
      + left.toolIds.length
      + left.mcpServerIds.length
      + left.denyToolIds.length
      + left.denyMcpServerIds.length;
    const rightCount =
      right.availableSkillIds.length
      + right.missingSkillIds.length
      + right.toolIds.length
      + right.mcpServerIds.length
      + right.denyToolIds.length
      + right.denyMcpServerIds.length;
    if (leftCount !== rightCount) {
      return rightCount - leftCount;
    }
    return left.agentName.localeCompare(right.agentName);
  });

  return {
    totalCount,
    actionableAgentCount,
    missingSkillAgentCount,
    availableSkillCount,
    missingSkillCount,
    toolSuggestionCount,
    mcpSuggestionCount,
    diagnostics,
  };
}

export function applyRoleProfilesToGraph(
  graph: NonNullable<HarnessProjectDetailDTO['graph_json']>,
  summaries: HarnessAgentCapabilitySummaryDTO[],
  selectedAgentIds: string[] | null
) {
  const selectedSet =
    selectedAgentIds === null
      ? null
      : new Set(selectedAgentIds.map((agentId) => String(agentId || '').trim()).filter(Boolean));
  let nextGraph = graph;
  let actionableAgentCount = 0;
  let changedAgentCount = 0;
  let skillChangeCount = 0;
  let toolChangeCount = 0;
  let mcpChangeCount = 0;

  for (const summary of summaries) {
    const agentId = typeof summary.agent_id === 'string' ? summary.agent_id : '';
    if (!agentId || (selectedSet && !selectedSet.has(agentId))) {
      continue;
    }
    const roleProfile =
      summary.role_profile_suggestion && typeof summary.role_profile_suggestion === 'object'
        ? summary.role_profile_suggestion
        : null;
    if (!roleProfile) {
      continue;
    }
    const skillIds = coerceStringList(roleProfile.available_skill_ids);
    const toolIds = coerceStringList(roleProfile.suggested_tool_ids);
    const mcpServerIds = coerceStringList(roleProfile.suggested_mcp_server_ids);
    const denyToolIds = coerceStringList(roleProfile.restrictive_tool_ids);
    const denyMcpServerIds = coerceStringList(roleProfile.restrictive_mcp_server_ids);
    if (
      skillIds.length === 0
      && toolIds.length === 0
      && mcpServerIds.length === 0
      && denyToolIds.length === 0
      && denyMcpServerIds.length === 0
    ) {
      continue;
    }
    actionableAgentCount += 1;
    const applied = applyAgentCapabilityPolicySuggestions(nextGraph, {
      agentId,
      skillIds,
      toolIds,
      mcpServerIds,
      denyToolIds,
      denyMcpServerIds,
      forceAllowToolIds: toolIds.length > 0,
      forceAllowMcpServerIds: mcpServerIds.length > 0,
    });
    if (!applied.changed) {
      continue;
    }
    nextGraph = applied.graph;
    changedAgentCount += 1;
    skillChangeCount += applied.skillChangeCount;
    toolChangeCount += applied.toolChangeCount;
    mcpChangeCount += applied.mcpChangeCount;
  }

  return {
    actionableAgentCount,
    changedAgentCount,
    skillChangeCount,
    toolChangeCount,
    mcpChangeCount,
    graph: nextGraph,
  };
}

export function applyCapabilityPolicySuggestionsToGraph(
  graph: NonNullable<HarnessProjectDetailDTO['graph_json']>,
  summaries: HarnessAgentCapabilitySummaryDTO[],
  selectedAgentIds: string[] | null,
  {
    includeTools = true,
    includeMcp = true,
  }: {
    includeTools?: boolean;
    includeMcp?: boolean;
  } = {}
) {
  const selectedSet =
    selectedAgentIds === null
      ? null
      : new Set(selectedAgentIds.map((agentId) => String(agentId || '').trim()).filter(Boolean));
  let nextGraph = graph;
  let changedAgentCount = 0;
  let actionableAgentCount = 0;
  let toolChangeCount = 0;
  let mcpChangeCount = 0;

  for (const summary of summaries) {
    const agentId = typeof summary.agent_id === 'string' ? summary.agent_id : '';
    if (!agentId || (selectedSet && !selectedSet.has(agentId))) {
      continue;
    }
    const currentAgent = (nextGraph.agents ?? []).find((agent) => agent.agent_id === agentId) ?? null;
    const actionableToolIds = includeTools
      ? computeActionableToolPolicySuggestionIds(currentAgent, coerceStringList(summary.policy_blocked_tool_ids))
      : [];
    const actionableMcpServerIds = includeMcp
      ? computeActionableMcpPolicySuggestionIds(currentAgent, coerceStringList(summary.policy_blocked_mcp_server_ids))
      : [];
    const restrictiveToolIds = includeTools
      ? computeCoordinatorToolPolicyRestrictionIds(currentAgent, summary)
      : [];
    const restrictiveMcpServerIds = includeMcp
      ? computeCoordinatorMcpPolicyRestrictionIds(currentAgent, summary)
      : [];
    if (
      actionableToolIds.length === 0
      && actionableMcpServerIds.length === 0
      && restrictiveToolIds.length === 0
      && restrictiveMcpServerIds.length === 0
    ) {
      continue;
    }
    actionableAgentCount += 1;
    const applied = applyAgentCapabilityPolicySuggestions(nextGraph, {
      agentId,
      toolIds: actionableToolIds,
      mcpServerIds: actionableMcpServerIds,
      denyToolIds: restrictiveToolIds,
      denyMcpServerIds: restrictiveMcpServerIds,
    });
    if (!applied.changed) {
      continue;
    }
    nextGraph = applied.graph;
    changedAgentCount += 1;
    toolChangeCount += applied.toolChangeCount;
    mcpChangeCount += applied.mcpChangeCount;
  }

  return {
    actionableAgentCount,
    changedAgentCount,
    toolChangeCount,
    mcpChangeCount,
    graph: nextGraph,
  };
}
