import type { ReactNode } from 'react';
import type {
  HarnessAgentCapabilitySummaryDTO,
  HarnessAgentDelegationContractDTO,
  HarnessAgentExecutionContractDTO,
  HarnessCanvasAgentDTO,
  HarnessCoordinationAgentPreviewDTO,
  HarnessDelegationTargetFitDTO,
} from '@/domains/harness/hooks';
import { useMessages } from '@/lib/i18n';
import {
  capabilityAvailabilityBadgeClass,
  capabilityReadinessBadgeClass,
  delegationFitBadgeClass,
  formatCapabilityAvailabilityLabel,
  formatCapabilityReadinessLabel,
  formatDelegationFitLabel,
} from './diagnostics';
import { HARNESS_MESSAGES } from './messages';
import type { AgentRoleProfilePeerDiagnostic } from './policy-repair';
import { coerceStringList, formatSkillTitle, formatTemplate } from './utils';

const EMPTY_LOOKUP = new Map<string, string>();
const PREVIEW_LIMIT = 6;

type CapabilityTone = 'slate' | 'cyan' | 'emerald' | 'violet' | 'rose' | 'amber' | 'sky';

function uniqueIds(...lists: Array<readonly string[] | undefined>) {
  const values: string[] = [];
  for (const list of lists) {
    if (!list) {
      continue;
    }
    for (const item of list) {
      const normalized = typeof item === 'string' ? item.trim() : '';
      if (normalized) {
        values.push(normalized);
      }
    }
  }
  return Array.from(new Set(values));
}

function chipToneClasses(tone: CapabilityTone) {
  if (tone === 'cyan') {
    return 'bg-cyan-50 text-cyan-800 ring-cyan-200';
  }
  if (tone === 'emerald') {
    return 'bg-emerald-50 text-emerald-800 ring-emerald-200';
  }
  if (tone === 'violet') {
    return 'bg-violet-50 text-violet-800 ring-violet-200';
  }
  if (tone === 'rose') {
    return 'bg-rose-50 text-rose-800 ring-rose-200';
  }
  if (tone === 'amber') {
    return 'bg-amber-50 text-amber-800 ring-amber-200';
  }
  if (tone === 'sky') {
    return 'bg-sky-50 text-sky-800 ring-sky-200';
  }
  return 'bg-slate-100 text-slate-700 ring-slate-200';
}

function countToneClasses(tone: CapabilityTone) {
  if (tone === 'cyan') {
    return 'bg-cyan-100 text-cyan-900 ring-cyan-200';
  }
  if (tone === 'emerald') {
    return 'bg-emerald-100 text-emerald-900 ring-emerald-200';
  }
  if (tone === 'violet') {
    return 'bg-violet-100 text-violet-900 ring-violet-200';
  }
  if (tone === 'rose') {
    return 'bg-rose-100 text-rose-900 ring-rose-200';
  }
  if (tone === 'amber') {
    return 'bg-amber-100 text-amber-900 ring-amber-200';
  }
  if (tone === 'sky') {
    return 'bg-sky-100 text-sky-900 ring-sky-200';
  }
  return 'bg-slate-100 text-slate-700 ring-slate-200';
}

function contractModeBadgeClass(mode: string | null | undefined) {
  if (mode === 'direct_execution') {
    return 'bg-emerald-50 text-emerald-800 ring-emerald-200';
  }
  if (mode === 'mixed') {
    return 'bg-cyan-50 text-cyan-800 ring-cyan-200';
  }
  if (mode === 'planning_only') {
    return 'bg-amber-50 text-amber-800 ring-amber-200';
  }
  if (mode === 'guidance_only') {
    return 'bg-sky-50 text-sky-800 ring-sky-200';
  }
  return 'bg-slate-100 text-slate-700 ring-slate-200';
}

function resolveExecutionContractToolAccessMode(
  contract: HarnessAgentExecutionContractDTO | null | undefined,
  executableToolIds: string[],
  planningOnlyToolIds: string[]
) {
  const mode = contract?.tool_access_mode;
  if (mode === 'direct_execution' || mode === 'planning_only' || mode === 'mixed' || mode === 'none') {
    return mode;
  }
  if (executableToolIds.length > 0 && planningOnlyToolIds.length > 0) {
    return 'mixed';
  }
  if (executableToolIds.length > 0) {
    return 'direct_execution';
  }
  if (planningOnlyToolIds.length > 0) {
    return 'planning_only';
  }
  return 'none';
}

function resolveExecutionContractMcpAccessMode(
  contract: HarnessAgentExecutionContractDTO | null | undefined,
  planningOnlyMcpServerIds: string[]
) {
  const mode = contract?.mcp_access_mode;
  if (mode === 'planning_only' || mode === 'none') {
    return mode;
  }
  return planningOnlyMcpServerIds.length > 0 ? 'planning_only' : 'none';
}

function resolveDelegationPrimaryRoleMode(
  contract: HarnessAgentDelegationContractDTO | null | undefined
) {
  const mode = contract?.primary_role_mode;
  if (
    mode === 'coordinator'
    || mode === 'research'
    || mode === 'implementation'
    || mode === 'verification'
    || mode === 'generalist'
  ) {
    return mode;
  }
  return 'generalist';
}

function resolveDelegationWorkStrategy(
  contract: HarnessAgentDelegationContractDTO | null | undefined
) {
  const strategy = contract?.work_strategy;
  if (
    strategy === 'synthesize_and_route'
    || strategy === 'gather_then_handoff'
    || strategy === 'implement_then_handoff'
    || strategy === 'verify_and_close'
    || strategy === 'self_contained_delivery'
    || strategy === 'flexible'
  ) {
    return strategy;
  }
  return 'flexible';
}

function delegationContractBadgeClass(mode: string | boolean | null | undefined) {
  if (mode === 'coordinator' || mode === 'synthesize_and_route' || mode === true) {
    return 'bg-cyan-50 text-cyan-800 ring-cyan-200';
  }
  if (mode === 'research' || mode === 'verification' || mode === 'gather_then_handoff' || mode === 'verify_and_close') {
    return 'bg-sky-50 text-sky-800 ring-sky-200';
  }
  if (mode === 'implementation' || mode === 'implement_then_handoff' || mode === 'self_contained_delivery') {
    return 'bg-emerald-50 text-emerald-800 ring-emerald-200';
  }
  return 'bg-slate-100 text-slate-700 ring-slate-200';
}

function collaboratorSignalBadgeClass(kind: 'same_profile' | 'overlap_risk') {
  if (kind === 'same_profile') {
    return 'bg-violet-50 text-violet-800 ring-violet-200';
  }
  return 'bg-amber-50 text-amber-800 ring-amber-200';
}

type SelectedAgentCapabilityPanelProps = {
  selectedAgent: HarnessCanvasAgentDTO;
  selectedAgentCapabilitySummary: HarnessAgentCapabilitySummaryDTO;
  skillTitleById: Map<string, string>;
  toolTitleById: Map<string, string>;
  mcpServerTitleById: Map<string, string>;
  agentNameById: Map<string, string>;
  selectedAgentAvailabilityStatus: 'available' | 'limited' | 'unavailable';
  selectedAgentAvailabilityBlockers: string[];
  selectedAgentAvailabilityWarnings: string[];
  selectedAgentReadinessStatus: 'ready' | 'limited' | 'blocked';
  selectedAgentReadinessBlockers: string[];
  selectedAgentReadinessWarnings: string[];
  selectedAgentMissingSkillIds: string[];
  selectedAgentMissingRequiredSkillIds: string[];
  selectedAgentMissingSkillDetails: Record<string, unknown>[];
  selectedAgentPolicyBlockedToolIds: string[];
  selectedAgentActionableToolPolicyIds: string[];
  selectedAgentActionableMcpPolicyIds: string[];
  selectedAgentActionableToolRestrictionIds: string[];
  selectedAgentActionableMcpRestrictionIds: string[];
  selectedAgentProviderLimitedToolIds: string[];
  selectedAgentMissingMcpServerIds: string[];
  selectedAgentMissingMcpServerDetails: Record<string, unknown>[];
  selectedAgentPolicyBlockedMcpServerIds: string[];
  selectedAgentShouldOpenSkillPool: boolean;
  selectedAgentShouldOpenProjectProviders: boolean;
  selectedAgentShouldOpenProjectMcp: boolean;
  selectedAgentDownstreamTargetIds: Set<string>;
  selectedAgentPrimarySuggestedCollaborator: HarnessDelegationTargetFitDTO | null;
  selectedAgentRoleProfilePeerDiagnostics: AgentRoleProfilePeerDiagnostic[];
  focusableAgentIds: Set<string>;
  isApplying: boolean;
  isRequestingSkills: boolean;
  canRequestSelectedAgentRoleProfileSkills: boolean;
  onOpenSkillPool: () => void;
  onApplySelectedAgentRoleProfile: () => void;
  onRequestSelectedAgentRoleProfileSkills: () => void;
  onApplySelectedAgentToolPolicySuggestions: () => void;
  onApplySelectedAgentMcpPolicySuggestions: () => void;
  onApplySelectedAgentToolPolicyRestrictions: () => void;
  onApplySelectedAgentMcpPolicyRestrictions: () => void;
  onOpenProjectProviders: () => void;
  onOpenProjectMcp: () => void;
  onApplySuggestedHandoff: (params: { sourceAgentId: string; targetAgentId: string }) => void;
  onApplySuggestedRewire: (params: {
    sourceAgentId: string;
    fromTargetAgentId: string;
    toTargetAgentId: string;
  }) => void;
  onOpenNodeEditor: (agentId: string) => void;
};

export function SelectedAgentCapabilityPanel({
  selectedAgent,
  selectedAgentCapabilitySummary,
  skillTitleById,
  toolTitleById,
  mcpServerTitleById,
  agentNameById,
  selectedAgentAvailabilityStatus,
  selectedAgentAvailabilityBlockers,
  selectedAgentAvailabilityWarnings,
  selectedAgentReadinessStatus,
  selectedAgentReadinessBlockers,
  selectedAgentReadinessWarnings,
  selectedAgentMissingSkillIds,
  selectedAgentMissingRequiredSkillIds,
  selectedAgentMissingSkillDetails,
  selectedAgentPolicyBlockedToolIds,
  selectedAgentActionableToolPolicyIds,
  selectedAgentActionableMcpPolicyIds,
  selectedAgentActionableToolRestrictionIds,
  selectedAgentActionableMcpRestrictionIds,
  selectedAgentProviderLimitedToolIds,
  selectedAgentMissingMcpServerIds,
  selectedAgentMissingMcpServerDetails,
  selectedAgentPolicyBlockedMcpServerIds,
  selectedAgentShouldOpenSkillPool,
  selectedAgentShouldOpenProjectProviders,
  selectedAgentShouldOpenProjectMcp,
  selectedAgentDownstreamTargetIds,
  selectedAgentPrimarySuggestedCollaborator,
  selectedAgentRoleProfilePeerDiagnostics,
  focusableAgentIds,
  isApplying,
  isRequestingSkills,
  canRequestSelectedAgentRoleProfileSkills,
  onOpenSkillPool,
  onApplySelectedAgentRoleProfile,
  onRequestSelectedAgentRoleProfileSkills,
  onApplySelectedAgentToolPolicySuggestions,
  onApplySelectedAgentMcpPolicySuggestions,
  onApplySelectedAgentToolPolicyRestrictions,
  onApplySelectedAgentMcpPolicyRestrictions,
  onOpenProjectProviders,
  onOpenProjectMcp,
  onApplySuggestedHandoff,
  onApplySuggestedRewire,
  onOpenNodeEditor,
}: SelectedAgentCapabilityPanelProps) {
  const text = useMessages(HARNESS_MESSAGES);

  const requiredSkillIds = uniqueIds(selectedAgentCapabilitySummary.required_skill_ids);
  const requiredToolIds = uniqueIds(selectedAgentCapabilitySummary.required_tool_ids);
  const missingRequiredToolIds = uniqueIds(selectedAgentCapabilitySummary.missing_required_tool_ids);
  const requiredMcpServerIds = uniqueIds(selectedAgentCapabilitySummary.required_mcp_server_ids);
  const missingRequiredMcpServerIds = uniqueIds(selectedAgentCapabilitySummary.missing_required_mcp_server_ids);
  const loadedSkillIds = uniqueIds(selectedAgentCapabilitySummary.loaded_skill_ids);
  const suggestedSkillIds = uniqueIds(selectedAgentCapabilitySummary.suggested_skill_ids);
  const enabledToolIds = uniqueIds(selectedAgentCapabilitySummary.enabled_tool_ids);
  const disabledToolIds = uniqueIds(selectedAgentCapabilitySummary.disabled_tool_ids);
  const configuredAllowedToolIds = uniqueIds(selectedAgentCapabilitySummary.configured_allowed_tool_ids);
  const configuredDeniedToolIds = uniqueIds(selectedAgentCapabilitySummary.configured_denied_tool_ids);
  const policyAddedToolIds = uniqueIds(selectedAgentCapabilitySummary.policy_added_tool_ids);
  const unknownAllowedToolIds = uniqueIds(selectedAgentCapabilitySummary.unknown_allowed_tool_ids);
  const configuredAllowedMcpServerIds = uniqueIds(selectedAgentCapabilitySummary.configured_allowed_mcp_server_ids);
  const configuredDeniedMcpServerIds = uniqueIds(selectedAgentCapabilitySummary.configured_denied_mcp_server_ids);
  const mcpServerIds = uniqueIds(selectedAgentCapabilitySummary.mcp_server_ids);
  const policyAddedMcpServerIds = uniqueIds(selectedAgentCapabilitySummary.policy_added_mcp_server_ids);
  const unknownAllowedMcpServerIds = uniqueIds(selectedAgentCapabilitySummary.unknown_allowed_mcp_server_ids);
  const delegationLaneIds = uniqueIds(selectedAgentCapabilitySummary.delegation_lane_ids);
  const missingSkillGapIds = uniqueIds(selectedAgentMissingRequiredSkillIds, selectedAgentMissingSkillIds);
  const toolGapIds = uniqueIds(
    selectedAgentPolicyBlockedToolIds,
    selectedAgentProviderLimitedToolIds,
    disabledToolIds
  );
  const mcpGapIds = uniqueIds(selectedAgentMissingMcpServerIds, selectedAgentPolicyBlockedMcpServerIds);
  const capabilityGapCount = uniqueIds(missingSkillGapIds, toolGapIds, mcpGapIds).length;
  const recommendedCollaborators = selectedAgentCapabilitySummary.recommended_collaborators ?? [];
  const downstreamHandoffScores = selectedAgentCapabilitySummary.downstream_handoff_scores ?? [];
  const hasRepairActions =
    selectedAgentShouldOpenSkillPool
    || selectedAgentActionableToolPolicyIds.length > 0
    || selectedAgentActionableMcpPolicyIds.length > 0
    || selectedAgentActionableToolRestrictionIds.length > 0
    || selectedAgentActionableMcpRestrictionIds.length > 0
    || selectedAgentShouldOpenProjectProviders
    || selectedAgentShouldOpenProjectMcp;
  const executionContract = selectedAgentCapabilitySummary.execution_contract;
  const executionContractApprovedSkillIds = uniqueIds(executionContract?.approved_skill_ids, loadedSkillIds);
  const executionContractSuggestedSkillIds = uniqueIds(executionContract?.suggested_skill_ids, suggestedSkillIds);
  const executionContractExecutableToolIds = uniqueIds(executionContract?.executable_tool_ids, enabledToolIds);
  const executionContractPlanningOnlyToolIds = uniqueIds(
    executionContract?.planning_only_tool_ids,
    selectedAgentProviderLimitedToolIds
  );
  const executionContractDisabledToolIds = uniqueIds(executionContract?.disabled_tool_ids, disabledToolIds);
  const executionContractPlanningOnlyMcpServerIds = uniqueIds(
    executionContract?.planning_only_mcp_server_ids,
    mcpServerIds
  );
  const executionContractMissingMcpServerIds = uniqueIds(
    executionContract?.missing_mcp_server_ids,
    selectedAgentMissingMcpServerIds
  );
  const executionContractToolAccessMode = resolveExecutionContractToolAccessMode(
    executionContract,
    executionContractExecutableToolIds,
    executionContractPlanningOnlyToolIds
  );
  const executionContractMcpAccessMode = resolveExecutionContractMcpAccessMode(
    executionContract,
    executionContractPlanningOnlyMcpServerIds
  );
  const executionContractSkillExecutionMode =
    executionContract?.skill_execution_mode === 'guidance_only' ? 'guidance_only' : 'guidance_only';
  const delegationContract = selectedAgentCapabilitySummary.delegation_contract;
  const delegationContractPrimaryRoleMode = resolveDelegationPrimaryRoleMode(delegationContract);
  const delegationContractSupportingRoleModes = uniqueIds(delegationContract?.supporting_role_modes);
  const delegationContractWorkStrategy = resolveDelegationWorkStrategy(delegationContract);
  const delegationContractUpstreamAgents = (delegationContract?.upstream_agents ?? []).filter(
    (item): item is HarnessCoordinationAgentPreviewDTO => Boolean(item && typeof item.agent_id === 'string')
  );
  const delegationContractDownstreamAgents = (delegationContract?.downstream_agents ?? []).filter(
    (item): item is HarnessCoordinationAgentPreviewDTO => Boolean(item && typeof item.agent_id === 'string')
  );
  const delegationContractPreferredCollaborators = (delegationContract?.preferred_collaborators ?? []).filter(
    (item): item is HarnessCoordinationAgentPreviewDTO => Boolean(item && typeof item.agent_id === 'string')
  );
  const delegationContractWeakHandoffTargets = (delegationContract?.weak_handoff_targets ?? []).filter(
    (item): item is HarnessCoordinationAgentPreviewDTO => Boolean(item && typeof item.agent_id === 'string')
  );
  const delegationContractWatchouts = coerceStringList(delegationContract?.watchouts);
  const roleProfileSuggestion = selectedAgentCapabilitySummary.role_profile_suggestion;
  const roleProfileId = (
    roleProfileSuggestion?.profile_id === 'coordinator'
    || roleProfileSuggestion?.profile_id === 'research'
    || roleProfileSuggestion?.profile_id === 'implementation'
    || roleProfileSuggestion?.profile_id === 'verification'
    || roleProfileSuggestion?.profile_id === 'generalist'
  )
    ? roleProfileSuggestion.profile_id
    : delegationContractPrimaryRoleMode;
  const roleProfileAvailableSkillIds = uniqueIds(roleProfileSuggestion?.available_skill_ids);
  const roleProfileMissingSkillIds = uniqueIds(roleProfileSuggestion?.missing_skill_ids);
  const roleProfileSuggestedToolIds = uniqueIds(roleProfileSuggestion?.suggested_tool_ids);
  const roleProfileSuggestedMcpServerIds = uniqueIds(roleProfileSuggestion?.suggested_mcp_server_ids);
  const roleProfileRestrictiveToolIds = uniqueIds(roleProfileSuggestion?.restrictive_tool_ids);
  const roleProfileRestrictiveMcpServerIds = uniqueIds(roleProfileSuggestion?.restrictive_mcp_server_ids);
  const hasRoleProfileApplyAction =
    roleProfileAvailableSkillIds.length > 0
    || roleProfileSuggestedToolIds.length > 0
    || roleProfileSuggestedMcpServerIds.length > 0
    || roleProfileRestrictiveToolIds.length > 0
    || roleProfileRestrictiveMcpServerIds.length > 0;
  const shouldOpenSkillPoolForRoleProfile = roleProfileMissingSkillIds.length > 0;

  const renderPills = (
    ids: string[],
    lookup: Map<string, string>,
    tone: CapabilityTone,
    emptyLabel: string,
    keyPrefix: string,
    limit?: number
  ) => {
    if (ids.length === 0) {
      return <span className="text-xs text-slate-500">{emptyLabel}</span>;
    }
    const visibleIds = typeof limit === 'number' ? ids.slice(0, limit) : ids;
    const remainingCount = ids.length - visibleIds.length;
    return (
      <>
        {visibleIds.map((id) => (
          <span
            key={`${selectedAgent.agent_id}-${keyPrefix}-${id}`}
            className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${chipToneClasses(tone)}`}
          >
            {formatSkillTitle(id, lookup)}
          </span>
        ))}
        {remainingCount > 0 ? (
          <span
            key={`${selectedAgent.agent_id}-${keyPrefix}-remaining`}
            className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${countToneClasses('slate')}`}
          >
            {formatTemplate(text.additionalItemsCount, { count: remainingCount })}
          </span>
        ) : null}
      </>
    );
  };

  const renderAgentPreviewPills = (
    entries: HarnessCoordinationAgentPreviewDTO[],
    tone: CapabilityTone,
    emptyLabel: string,
    keyPrefix: string,
    limit?: number
  ) => {
    if (entries.length === 0) {
      return <span className="text-xs text-slate-500">{emptyLabel}</span>;
    }
    const visibleEntries = typeof limit === 'number' ? entries.slice(0, limit) : entries;
    const remainingCount = entries.length - visibleEntries.length;
    return (
      <>
        {visibleEntries.map((item, index) => {
          const agentId = typeof item.agent_id === 'string' ? item.agent_id : '';
          const agentName =
            item.agent_name
            || (agentId ? agentNameById.get(agentId) ?? agentId : '')
            || text.none;
          const canFocus = agentId ? focusableAgentIds.has(agentId) : false;
          const className = `rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${chipToneClasses(tone)}`;
          if (canFocus) {
            return (
              <button
                key={`${selectedAgent.agent_id}-${keyPrefix}-${agentId || index}`}
                type="button"
                onClick={() => onOpenNodeEditor(agentId)}
                className={className}
              >
                {agentName}
              </button>
            );
          }
          return (
            <span key={`${selectedAgent.agent_id}-${keyPrefix}-${agentId || index}`} className={className}>
              {agentName}
            </span>
          );
        })}
        {remainingCount > 0 ? (
          <span
            key={`${selectedAgent.agent_id}-${keyPrefix}-remaining`}
            className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${countToneClasses('slate')}`}
          >
            {formatTemplate(text.additionalItemsCount, { count: remainingCount })}
          </span>
        ) : null}
      </>
    );
  };

  const renderSummaryBucket = ({
    title,
    countLabel,
    ids,
    lookup,
    tone,
    emptyLabel,
    keyPrefix,
    footer,
  }: {
    title: string;
    countLabel: string;
    ids: string[];
    lookup: Map<string, string>;
    tone: CapabilityTone;
    emptyLabel: string;
    keyPrefix: string;
    footer?: ReactNode;
  }) => (
    <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{title}</div>
        <span
          className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${countToneClasses(tone)}`}
        >
          {countLabel}
        </span>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {renderPills(ids, lookup, tone, emptyLabel, `${keyPrefix}-summary`, PREVIEW_LIMIT)}
      </div>
      {footer}
    </div>
  );

  const renderGapGroup = (
    title: string,
    ids: string[],
    lookup: Map<string, string>,
    tone: CapabilityTone,
    keyPrefix: string
  ) => {
    if (ids.length === 0) {
      return null;
    }
    return (
      <div>
        <div className="text-[11px] font-semibold text-slate-500">{title}</div>
        <div className="mt-2 flex flex-wrap gap-2">
          {renderPills(ids, lookup, tone, text.none, `${keyPrefix}-gap`, PREVIEW_LIMIT)}
        </div>
      </div>
    );
  };

  const formatDelegationRoleModeLabel = (mode: string) => {
    if (mode === 'coordinator') {
      return text.roleCoordinator;
    }
    if (mode === 'research') {
      return text.roleResearch;
    }
    if (mode === 'implementation') {
      return text.roleImplementation;
    }
    if (mode === 'verification') {
      return text.roleVerification;
    }
    return text.roleGeneralist;
  };

  const formatDelegationWorkStrategyLabel = (strategy: string) => {
    if (strategy === 'synthesize_and_route') {
      return text.workStrategySynthesizeAndRoute;
    }
    if (strategy === 'gather_then_handoff') {
      return text.workStrategyGatherThenHandoff;
    }
    if (strategy === 'implement_then_handoff') {
      return text.workStrategyImplementThenHandoff;
    }
    if (strategy === 'verify_and_close') {
      return text.workStrategyVerifyAndClose;
    }
    if (strategy === 'self_contained_delivery') {
      return text.workStrategySelfContainedDelivery;
    }
    return text.workStrategyFlexible;
  };

  const formatRoleProfileTitle = (profileId: string) => {
    if (profileId === 'coordinator') {
      return text.roleProfileCoordinatorTitle;
    }
    if (profileId === 'research') {
      return text.roleProfileResearchTitle;
    }
    if (profileId === 'implementation') {
      return text.roleProfileImplementationTitle;
    }
    if (profileId === 'verification') {
      return text.roleProfileVerificationTitle;
    }
    return text.roleProfileGeneralistTitle;
  };

  const formatRoleProfileDescription = (profileId: string) => {
    if (profileId === 'coordinator') {
      return text.roleProfileCoordinatorDescription;
    }
    if (profileId === 'research') {
      return text.roleProfileResearchDescription;
    }
    if (profileId === 'implementation') {
      return text.roleProfileImplementationDescription;
    }
    if (profileId === 'verification') {
      return text.roleProfileVerificationDescription;
    }
    return text.roleProfileGeneralistDescription;
  };

  const renderCollaboratorSignalBadges = (
    item: HarnessDelegationTargetFitDTO,
    keyPrefix: string
  ) => (
    <>
      {item.same_role_profile ? (
        <span
          key={`${selectedAgent.agent_id}-${keyPrefix}-same-profile`}
          className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${collaboratorSignalBadgeClass(
            'same_profile'
          )}`}
        >
          {text.sameRoleProfileBadgeLabel}
        </span>
      ) : null}
      {item.same_role_profile_overlap_risk ? (
        <span
          key={`${selectedAgent.agent_id}-${keyPrefix}-overlap-risk`}
          className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${collaboratorSignalBadgeClass(
            'overlap_risk'
          )}`}
        >
          {text.sameRoleProfileOverlapRiskBadgeLabel}
        </span>
      ) : null}
    </>
  );

  return (
    <div className="space-y-3 rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-semibold text-slate-900">{text.capabilitySummary}</div>
        <div className="flex flex-wrap gap-2">
          <span
            className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityAvailabilityBadgeClass(selectedAgentAvailabilityStatus)}`}
          >
            {formatCapabilityAvailabilityLabel(selectedAgentAvailabilityStatus, text)}
          </span>
          <span
            className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityReadinessBadgeClass(selectedAgentReadinessStatus)}`}
          >
            {formatCapabilityReadinessLabel(selectedAgentReadinessStatus, text)}
          </span>
        </div>
      </div>
      {hasRepairActions ? (
        <div className="rounded-2xl border border-slate-200 bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.repairActionsLabel}
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {selectedAgentShouldOpenSkillPool ? (
              <button
                type="button"
                onClick={onOpenSkillPool}
                className="inline-flex items-center justify-center rounded-xl border border-rose-200 bg-white px-3 py-2 text-xs font-semibold text-rose-900 hover:bg-rose-100"
              >
                {text.openSkillPoolAction}
              </button>
            ) : null}
            {selectedAgentActionableToolPolicyIds.length > 0 ? (
              <button
                type="button"
                onClick={onApplySelectedAgentToolPolicySuggestions}
                disabled={isApplying}
                className="inline-flex items-center justify-center rounded-xl border border-emerald-200 bg-white px-3 py-2 text-xs font-semibold text-emerald-900 hover:bg-emerald-100 disabled:opacity-50"
              >
                {text.allowSuggestedToolsAction}
              </button>
            ) : null}
            {selectedAgentActionableMcpPolicyIds.length > 0 ? (
              <button
                type="button"
                onClick={onApplySelectedAgentMcpPolicySuggestions}
                disabled={isApplying}
                className="inline-flex items-center justify-center rounded-xl border border-violet-200 bg-white px-3 py-2 text-xs font-semibold text-violet-900 hover:bg-violet-100 disabled:opacity-50"
              >
                {text.allowSuggestedMcpAction}
              </button>
            ) : null}
            {selectedAgentActionableToolRestrictionIds.length > 0 ? (
              <button
                type="button"
                onClick={onApplySelectedAgentToolPolicyRestrictions}
                disabled={isApplying}
                className="inline-flex items-center justify-center rounded-xl border border-amber-200 bg-white px-3 py-2 text-xs font-semibold text-amber-900 hover:bg-amber-100 disabled:opacity-50"
              >
                {text.restrictCoordinatorToolsAction}
              </button>
            ) : null}
            {selectedAgentActionableMcpRestrictionIds.length > 0 ? (
              <button
                type="button"
                onClick={onApplySelectedAgentMcpPolicyRestrictions}
                disabled={isApplying}
                className="inline-flex items-center justify-center rounded-xl border border-sky-200 bg-white px-3 py-2 text-xs font-semibold text-sky-900 hover:bg-sky-100 disabled:opacity-50"
              >
                {text.restrictCoordinatorMcpAction}
              </button>
            ) : null}
            {selectedAgentShouldOpenProjectProviders ? (
              <button
                type="button"
                onClick={onOpenProjectProviders}
                className="inline-flex items-center justify-center rounded-xl border border-amber-200 bg-white px-3 py-2 text-xs font-semibold text-amber-900 hover:bg-amber-100"
              >
                {text.openProjectProvidersAction}
              </button>
            ) : null}
            {selectedAgentShouldOpenProjectMcp ? (
              <button
                type="button"
                onClick={onOpenProjectMcp}
                className="inline-flex items-center justify-center rounded-xl border border-violet-200 bg-white px-3 py-2 text-xs font-semibold text-violet-900 hover:bg-violet-100"
              >
                {text.openProjectMcpAction}
              </button>
            ) : null}
          </div>
        </div>
      ) : null}
      {roleProfileSuggestion ? (
        <div className="rounded-2xl border border-slate-200 bg-white p-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.roleProfileSuggestionLabel}
              </div>
              <div className="mt-1 text-xs leading-5 text-slate-500">{text.roleProfileHint}</div>
              <div className="mt-2 text-sm text-slate-700">{formatRoleProfileDescription(roleProfileId)}</div>
            </div>
            <div className="flex flex-wrap gap-2">
              <span
                className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationContractBadgeClass(
                  roleProfileId
                )}`}
              >
                {formatRoleProfileTitle(roleProfileId)}
              </span>
              {hasRoleProfileApplyAction ? (
                <button
                  type="button"
                  onClick={onApplySelectedAgentRoleProfile}
                  disabled={isApplying}
                  className="inline-flex items-center justify-center rounded-xl border border-cyan-200 bg-white px-3 py-2 text-xs font-semibold text-cyan-950 hover:bg-cyan-100 disabled:opacity-50"
                >
                  {text.applyRoleProfileAction}
                </button>
              ) : null}
              {shouldOpenSkillPoolForRoleProfile && canRequestSelectedAgentRoleProfileSkills ? (
                <button
                  type="button"
                  onClick={onRequestSelectedAgentRoleProfileSkills}
                  disabled={isRequestingSkills}
                  className="inline-flex items-center justify-center rounded-xl border border-rose-200 bg-white px-3 py-2 text-xs font-semibold text-rose-900 hover:bg-rose-100 disabled:opacity-50"
                >
                  {isRequestingSkills ? text.requestingRoleProfileSkills : text.requestRoleProfileSkillsAction}
                </button>
              ) : null}
              {shouldOpenSkillPoolForRoleProfile ? (
                <button
                  type="button"
                  onClick={onOpenSkillPool}
                  className="inline-flex items-center justify-center rounded-xl border border-rose-200 bg-white px-3 py-2 text-xs font-semibold text-rose-900 hover:bg-rose-100"
                >
                  {text.openSkillPoolAction}
                </button>
              ) : null}
            </div>
          </div>
          <div className="mt-3 grid gap-3 xl:grid-cols-3">
            {renderSummaryBucket({
              title: text.roleProfileAvailableSkillsLabel,
              countLabel: formatTemplate(text.skillsCountShort, { count: roleProfileAvailableSkillIds.length }),
              ids: roleProfileAvailableSkillIds,
              lookup: skillTitleById,
              tone: 'cyan',
              emptyLabel: text.none,
              keyPrefix: 'role-profile-available-skills',
            })}
            {renderSummaryBucket({
              title: text.roleProfileMissingSkillsLabel,
              countLabel: formatTemplate(text.skillsCountShort, { count: roleProfileMissingSkillIds.length }),
              ids: roleProfileMissingSkillIds,
              lookup: skillTitleById,
              tone: 'rose',
              emptyLabel: text.none,
              keyPrefix: 'role-profile-missing-skills',
            })}
            {renderSummaryBucket({
              title: text.roleProfileRecommendedToolsLabel,
              countLabel: formatTemplate(text.toolsCountShort, { count: roleProfileSuggestedToolIds.length }),
              ids: roleProfileSuggestedToolIds,
              lookup: toolTitleById,
              tone: 'emerald',
              emptyLabel: text.none,
              keyPrefix: 'role-profile-tools',
            })}
            {renderSummaryBucket({
              title: text.roleProfileRecommendedMcpLabel,
              countLabel: formatTemplate(text.mcpServerCountShort, { count: roleProfileSuggestedMcpServerIds.length }),
              ids: roleProfileSuggestedMcpServerIds,
              lookup: mcpServerTitleById,
              tone: 'violet',
              emptyLabel: text.none,
              keyPrefix: 'role-profile-mcp',
            })}
            {renderSummaryBucket({
              title: text.roleProfileRestrictiveToolsLabel,
              countLabel: formatTemplate(text.toolsCountShort, { count: roleProfileRestrictiveToolIds.length }),
              ids: roleProfileRestrictiveToolIds,
              lookup: toolTitleById,
              tone: 'amber',
              emptyLabel: text.none,
              keyPrefix: 'role-profile-restrictive-tools',
            })}
            {renderSummaryBucket({
              title: text.roleProfileRestrictiveMcpLabel,
              countLabel: formatTemplate(text.mcpServerCountShort, { count: roleProfileRestrictiveMcpServerIds.length }),
              ids: roleProfileRestrictiveMcpServerIds,
              lookup: mcpServerTitleById,
              tone: 'sky',
              emptyLabel: text.none,
              keyPrefix: 'role-profile-restrictive-mcp',
            })}
          </div>
          {selectedAgentRoleProfilePeerDiagnostics.length > 0 ? (
            <div className="mt-4 rounded-2xl border border-violet-200 bg-violet-50/70 p-3">
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-violet-700">
                {text.roleProfilePeerOverlapLabel}
              </div>
              <div className="mt-1 text-xs leading-5 text-violet-800">{text.roleProfilePeerOverlapHint}</div>
              <div className="mt-3 grid gap-3 xl:grid-cols-2">
                {selectedAgentRoleProfilePeerDiagnostics.map((diagnostic) => (
                  <div
                    key={`${selectedAgent.agent_id}-role-profile-peer-${diagnostic.peerAgentId}`}
                    className="rounded-2xl border border-violet-200 bg-white/90 p-3"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <div className="text-sm font-semibold text-slate-900">{diagnostic.peerAgentName}</div>
                        <div className="mt-1 text-xs text-slate-500">{formatRoleProfileTitle(diagnostic.profileId)}</div>
                      </div>
                      {focusableAgentIds.has(diagnostic.peerAgentId) ? (
                        <button
                          type="button"
                          onClick={() => onOpenNodeEditor(diagnostic.peerAgentId)}
                          className="inline-flex items-center justify-center rounded-xl border border-violet-200 bg-white px-3 py-2 text-xs font-semibold text-violet-900 hover:bg-violet-100"
                        >
                          {formatTemplate(text.focusNodeForRecovery, { name: diagnostic.peerAgentName })}
                        </button>
                      ) : null}
                    </div>
                    <div className="mt-3">
                      <div className="text-[11px] font-semibold text-slate-500">{text.roleProfilePeerSharedLanesLabel}</div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {renderPills(
                          diagnostic.sharedLaneIds,
                          EMPTY_LOOKUP,
                          'violet',
                          text.none,
                          `role-profile-peer-shared-${diagnostic.peerAgentId}`,
                          PREVIEW_LIMIT
                        )}
                      </div>
                    </div>
                    <div className="mt-3">
                      <div className="text-[11px] font-semibold text-slate-500">{text.roleProfilePeerFocusLabel}</div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {renderPills(
                          diagnostic.selectedFocusLaneIds,
                          EMPTY_LOOKUP,
                          'cyan',
                          text.roleProfileOverlapNeedsDifferentiation,
                          `role-profile-peer-selected-focus-${diagnostic.peerAgentId}`,
                          PREVIEW_LIMIT
                        )}
                      </div>
                    </div>
                    <div className="mt-3">
                      <div className="text-[11px] font-semibold text-slate-500">{text.roleProfilePeerOtherFocusLabel}</div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {renderPills(
                          diagnostic.peerFocusLaneIds,
                          EMPTY_LOOKUP,
                          'sky',
                          text.none,
                          `role-profile-peer-peer-focus-${diagnostic.peerAgentId}`,
                          PREVIEW_LIMIT
                        )}
                      </div>
                    </div>
                    <div className="mt-3">
                      <div className="text-[11px] font-semibold text-slate-500">{text.roleProfilePeerDistinctToolsLabel}</div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {renderPills(
                          diagnostic.peerUniqueToolIds,
                          toolTitleById,
                          'emerald',
                          text.none,
                          `role-profile-peer-tools-${diagnostic.peerAgentId}`,
                          PREVIEW_LIMIT
                        )}
                      </div>
                    </div>
                    <div className="mt-3">
                      <div className="text-[11px] font-semibold text-slate-500">{text.roleProfilePeerDistinctMcpLabel}</div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {renderPills(
                          diagnostic.peerUniqueMcpServerIds,
                          mcpServerTitleById,
                          'violet',
                          text.none,
                          `role-profile-peer-mcp-${diagnostic.peerAgentId}`,
                          PREVIEW_LIMIT
                        )}
                      </div>
                    </div>
                    <div className="mt-3">
                      <div className="text-[11px] font-semibold text-slate-500">{text.roleProfilePeerDistinctSkillsLabel}</div>
                      <div className="mt-2 flex flex-wrap gap-2">
                        {renderPills(
                          diagnostic.peerUniqueSkillIds,
                          skillTitleById,
                          'cyan',
                          text.none,
                          `role-profile-peer-skills-${diagnostic.peerAgentId}`,
                          PREVIEW_LIMIT
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}
      <div className="rounded-2xl border border-slate-200 bg-white p-3">
        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
          {text.effectiveCapabilityPoolLabel}
        </div>
        <div className="mt-1 text-xs leading-5 text-slate-500">{text.effectiveCapabilityPoolHint}</div>
        <div className="mt-3 grid gap-3 xl:grid-cols-2">
          {renderSummaryBucket({
            title: text.effectiveSkillsLabel,
            countLabel: formatTemplate(text.skillsCountShort, { count: loadedSkillIds.length }),
            ids: loadedSkillIds,
            lookup: skillTitleById,
            tone: 'cyan',
            emptyLabel: text.none,
            keyPrefix: 'effective-skills',
            footer:
              suggestedSkillIds.length > 0 ? (
                <div className="mt-3 border-t border-slate-200/80 pt-3">
                  <div className="text-[11px] font-semibold text-slate-500">{text.suggestedSkillsLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {renderPills(
                      suggestedSkillIds,
                      skillTitleById,
                      'amber',
                      text.none,
                      'effective-skills-suggested',
                      PREVIEW_LIMIT
                    )}
                  </div>
                </div>
              ) : null,
          })}
          {renderSummaryBucket({
            title: text.effectiveToolsLabel,
            countLabel: formatTemplate(text.toolsCountShort, { count: enabledToolIds.length }),
            ids: enabledToolIds,
            lookup: toolTitleById,
            tone: 'emerald',
            emptyLabel: text.none,
            keyPrefix: 'effective-tools',
            footer:
              policyAddedToolIds.length > 0 ? (
                <div className="mt-3 border-t border-slate-200/80 pt-3">
                  <div className="text-[11px] font-semibold text-slate-500">{text.policyAddedToolsLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {renderPills(
                      policyAddedToolIds,
                      toolTitleById,
                      'cyan',
                      text.none,
                      'effective-tools-added',
                      PREVIEW_LIMIT
                    )}
                  </div>
                </div>
              ) : null,
          })}
          {renderSummaryBucket({
            title: text.effectiveMcpServersLabel,
            countLabel: formatTemplate(text.mcpServerCountShort, { count: mcpServerIds.length }),
            ids: mcpServerIds,
            lookup: mcpServerTitleById,
            tone: 'violet',
            emptyLabel: text.noMcpServersConfigured,
            keyPrefix: 'effective-mcp',
            footer:
              policyAddedMcpServerIds.length > 0 ? (
                <div className="mt-3 border-t border-slate-200/80 pt-3">
                  <div className="text-[11px] font-semibold text-slate-500">{text.policyAddedMcpServersLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {renderPills(
                      policyAddedMcpServerIds,
                      mcpServerTitleById,
                      'cyan',
                      text.none,
                      'effective-mcp-added',
                      PREVIEW_LIMIT
                    )}
                  </div>
                </div>
              ) : null,
          })}
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="flex items-center justify-between gap-2">
              <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.capabilityGapsLabel}
              </div>
              <span
                className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${countToneClasses(
                  capabilityGapCount > 0 ? 'amber' : 'slate'
                )}`}
              >
                {formatTemplate(text.capabilityGapCountShort, { count: capabilityGapCount })}
              </span>
            </div>
            {capabilityGapCount > 0 ? (
              <div className="mt-3 space-y-3">
                {renderGapGroup(
                  text.missingSkillsLabel,
                  missingSkillGapIds,
                  skillTitleById,
                  'rose',
                  'capability-gap-skills'
                )}
                {renderGapGroup(
                  text.policyBlockedToolsLabel,
                  selectedAgentPolicyBlockedToolIds,
                  toolTitleById,
                  'amber',
                  'capability-gap-blocked-tools'
                )}
                {renderGapGroup(
                  text.providerLimitedToolsLabel,
                  selectedAgentProviderLimitedToolIds,
                  toolTitleById,
                  'amber',
                  'capability-gap-provider-tools'
                )}
                {renderGapGroup(
                  text.disabledToolsLabel,
                  disabledToolIds,
                  toolTitleById,
                  'slate',
                  'capability-gap-disabled-tools'
                )}
                {renderGapGroup(
                  text.missingMcpServersLabel,
                  selectedAgentMissingMcpServerIds,
                  mcpServerTitleById,
                  'amber',
                  'capability-gap-missing-mcp'
                )}
                {renderGapGroup(
                  text.policyBlockedMcpServersLabel,
                  selectedAgentPolicyBlockedMcpServerIds,
                  mcpServerTitleById,
                  'amber',
                  'capability-gap-blocked-mcp'
                )}
              </div>
            ) : (
              <div className="mt-2 text-xs text-slate-500">{text.noCapabilityGaps}</div>
            )}
          </div>
        </div>
      </div>
      <div className="rounded-2xl border border-slate-200 bg-white p-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
              {text.executionContractLabel}
            </div>
            <div className="mt-1 text-xs leading-5 text-slate-500">{text.executionContractHint}</div>
          </div>
          <div className="flex flex-wrap gap-2">
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                executionContractSkillExecutionMode
              )}`}
            >
              {text.skillGuidanceOnly}
            </span>
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                executionContractToolAccessMode
              )}`}
            >
              {executionContractToolAccessMode === 'direct_execution'
                ? text.toolAccessDirectExecution
                : executionContractToolAccessMode === 'planning_only'
                  ? text.toolAccessPlanningOnly
                  : executionContractToolAccessMode === 'mixed'
                    ? text.toolAccessMixed
                    : text.toolAccessNone}
            </span>
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                executionContractMcpAccessMode
              )}`}
            >
              {executionContractMcpAccessMode === 'planning_only'
                ? text.mcpAccessPlanningOnly
                : text.mcpAccessNone}
            </span>
          </div>
        </div>
        <div className="mt-3 grid gap-3 xl:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              {text.skillExecutionModeLabel}
            </div>
            <div className="mt-2">
              <span
                className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                  executionContractSkillExecutionMode
                )}`}
              >
                {text.skillGuidanceOnly}
              </span>
            </div>
            <div className="mt-3">
              <div className="text-[11px] font-semibold text-slate-500">{text.approvedSkillsLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  executionContractApprovedSkillIds,
                  skillTitleById,
                  'cyan',
                  text.none,
                  'execution-contract-approved-skills',
                  PREVIEW_LIMIT
                )}
              </div>
            </div>
            {executionContractSuggestedSkillIds.length > 0 ? (
              <div className="mt-3">
                <div className="text-[11px] font-semibold text-slate-500">{text.suggestedSkillsLabel}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {renderPills(
                    executionContractSuggestedSkillIds,
                    skillTitleById,
                    'amber',
                    text.none,
                    'execution-contract-suggested-skills',
                    PREVIEW_LIMIT
                  )}
                </div>
              </div>
            ) : null}
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.toolAccessModeLabel}
              </div>
              <span
                className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                  executionContractToolAccessMode
                )}`}
              >
                {executionContractToolAccessMode === 'direct_execution'
                  ? text.toolAccessDirectExecution
                  : executionContractToolAccessMode === 'planning_only'
                    ? text.toolAccessPlanningOnly
                    : executionContractToolAccessMode === 'mixed'
                      ? text.toolAccessMixed
                      : text.toolAccessNone}
              </span>
            </div>
            <div className="mt-3">
              <div className="text-[11px] font-semibold text-slate-500">{text.executableToolsLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  executionContractExecutableToolIds,
                  toolTitleById,
                  'emerald',
                  text.none,
                  'execution-contract-executable-tools',
                  PREVIEW_LIMIT
                )}
              </div>
            </div>
            <div className="mt-3">
              <div className="text-[11px] font-semibold text-slate-500">{text.planningOnlyToolsLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  executionContractPlanningOnlyToolIds,
                  toolTitleById,
                  'amber',
                  text.none,
                  'execution-contract-planning-tools',
                  PREVIEW_LIMIT
                )}
              </div>
            </div>
            {executionContractDisabledToolIds.length > 0 ? (
              <div className="mt-3">
                <div className="text-[11px] font-semibold text-slate-500">{text.disabledToolsLabel}</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {renderPills(
                    executionContractDisabledToolIds,
                    toolTitleById,
                    'slate',
                    text.none,
                    'execution-contract-disabled-tools',
                    PREVIEW_LIMIT
                  )}
                </div>
              </div>
            ) : null}
            {selectedAgentCapabilitySummary.requires_tool_calling ? (
              <div className="mt-3 text-xs leading-5 text-slate-600">
                <span className="font-semibold text-slate-700">{text.contractRequiresToolCallingLabel}:</span>{' '}
                {text.enabled}
              </div>
            ) : null}
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.mcpAccessModeLabel}
              </div>
              <span
                className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                  executionContractMcpAccessMode
                )}`}
              >
                {executionContractMcpAccessMode === 'planning_only'
                  ? text.mcpAccessPlanningOnly
                  : text.mcpAccessNone}
              </span>
            </div>
            <div className="mt-3">
              <div className="text-[11px] font-semibold text-slate-500">{text.planningOnlyMcpServersLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  executionContractPlanningOnlyMcpServerIds,
                  mcpServerTitleById,
                  'violet',
                  text.none,
                  'execution-contract-planning-mcp',
                  PREVIEW_LIMIT
                )}
              </div>
            </div>
            <div className="mt-3">
              <div className="text-[11px] font-semibold text-slate-500">{text.missingMcpServersLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  executionContractMissingMcpServerIds,
                  mcpServerTitleById,
                  'amber',
                  text.none,
                  'execution-contract-missing-mcp',
                  PREVIEW_LIMIT
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="rounded-2xl border border-slate-200 bg-white p-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
              {text.delegationContractLabel}
            </div>
            <div className="mt-1 text-xs leading-5 text-slate-500">{text.delegationContractHint}</div>
          </div>
          <div className="flex flex-wrap gap-2">
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationContractBadgeClass(
                delegationContractPrimaryRoleMode
              )}`}
            >
              {formatDelegationRoleModeLabel(delegationContractPrimaryRoleMode)}
            </span>
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationContractBadgeClass(
                delegationContractWorkStrategy
              )}`}
            >
              {formatDelegationWorkStrategyLabel(delegationContractWorkStrategy)}
            </span>
          </div>
        </div>
        <div className="mt-3 grid gap-3 xl:grid-cols-3">
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              {text.primaryRoleModeLabel}
            </div>
            <div className="mt-2">
              <span
                className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationContractBadgeClass(
                  delegationContractPrimaryRoleMode
                )}`}
              >
                {formatDelegationRoleModeLabel(delegationContractPrimaryRoleMode)}
              </span>
            </div>
            <div className="mt-3">
              <div className="text-[11px] font-semibold text-slate-500">{text.supportingRoleModesLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {delegationContractSupportingRoleModes.length > 0 ? (
                  delegationContractSupportingRoleModes.map((mode) => (
                    <span
                      key={`${selectedAgent.agent_id}-delegation-support-${mode}`}
                      className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationContractBadgeClass(
                        mode
                      )}`}
                    >
                      {formatDelegationRoleModeLabel(mode)}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-slate-500">{text.none}</span>
                )}
              </div>
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.workStrategyLabel}
              </div>
              <span
                className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationContractBadgeClass(
                  delegationContractWorkStrategy
                )}`}
              >
                {formatDelegationWorkStrategyLabel(delegationContractWorkStrategy)}
              </span>
            </div>
            <div className="mt-3 space-y-3">
              <div>
                <div className="text-[11px] font-semibold text-slate-500">
                  {text.delegationFocusLabel}
                </div>
                <div className="mt-2 text-xs leading-5 text-slate-600">
                  {delegationContract?.primary_focus || text.noDelegationFocus}
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <span
                  className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationContractBadgeClass(
                    Boolean(delegationContract?.should_coordinate_parallel_work)
                  )}`}
                >
                  {text.coordinateParallelWorkLabel}: {delegationContract?.should_coordinate_parallel_work ? text.yesLabel : text.noLabel}
                </span>
                <span
                  className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationContractBadgeClass(
                    Boolean(delegationContract?.should_produce_final_output)
                  )}`}
                >
                  {text.finalOutputResponsibilityLabel}: {delegationContract?.should_produce_final_output ? text.yesLabel : text.noLabel}
                </span>
              </div>
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              {text.watchoutsLabel}
            </div>
            <div className="mt-3 space-y-2">
              {delegationContractWatchouts.length > 0 ? (
                delegationContractWatchouts.map((item, index) => (
                  <div
                    key={`${selectedAgent.agent_id}-delegation-watchout-${index + 1}`}
                    className="rounded-xl border border-amber-100 bg-amber-50/80 px-3 py-2 text-xs leading-5 text-amber-900"
                  >
                    {item}
                  </div>
                ))
              ) : (
                <div className="text-xs text-slate-500">{text.none}</div>
              )}
            </div>
          </div>
        </div>
        <div className="mt-3 grid gap-3 xl:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              {text.upstreamAgentsLabel}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {renderAgentPreviewPills(
                delegationContractUpstreamAgents,
                'sky',
                text.none,
                'delegation-upstream',
                PREVIEW_LIMIT
              )}
            </div>
            <div className="mt-3 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              {text.downstreamAgentsLabel}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {renderAgentPreviewPills(
                delegationContractDownstreamAgents,
                'emerald',
                text.none,
                'delegation-downstream',
                PREVIEW_LIMIT
              )}
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              {text.preferredCollaboratorTargetsLabel}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {renderAgentPreviewPills(
                delegationContractPreferredCollaborators,
                'cyan',
                text.none,
                'delegation-preferred',
                PREVIEW_LIMIT
              )}
            </div>
            <div className="mt-3 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              {text.weakHandoffTargetsLabel}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              {renderAgentPreviewPills(
                delegationContractWeakHandoffTargets,
                'amber',
                text.none,
                'delegation-weak-handoff',
                PREVIEW_LIMIT
              )}
            </div>
          </div>
        </div>
      </div>
      <div className="rounded-2xl bg-white p-3">
        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
          {text.availabilityLabel}
        </div>
        {selectedAgentAvailabilityBlockers.length > 0
        || selectedAgentAvailabilityWarnings.length > 0
        || Boolean(selectedAgentCapabilitySummary.requires_tool_calling) ? (
          <div className="mt-2 space-y-2 text-xs leading-5 text-slate-600">
            {selectedAgentAvailabilityBlockers.length > 0 ? (
              <div>
                <span className="font-semibold text-slate-700">{text.availabilityBlockersLabel}:</span>{' '}
                {selectedAgentAvailabilityBlockers.join(' · ')}
              </div>
            ) : null}
            {selectedAgentAvailabilityWarnings.length > 0 ? (
              <div>
                <span className="font-semibold text-slate-700">{text.availabilityWarningsLabel}:</span>{' '}
                {selectedAgentAvailabilityWarnings.join(' · ')}
              </div>
            ) : null}
            {selectedAgentCapabilitySummary.requires_tool_calling ? (
              <div>
                <span className="font-semibold text-slate-700">{text.requireToolCallingLabel}:</span>{' '}
                {text.enabled}
              </div>
            ) : null}
          </div>
        ) : (
          <div className="mt-2 text-xs text-slate-500">{text.noAvailabilityIssues}</div>
        )}
      </div>
      <div className="rounded-2xl bg-white p-3">
        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
          {text.readinessLabel}
        </div>
        {selectedAgentReadinessBlockers.length > 0 || selectedAgentReadinessWarnings.length > 0 ? (
          <div className="mt-2 space-y-2 text-xs leading-5 text-slate-600">
            {selectedAgentReadinessBlockers.length > 0 ? (
              <div>
                <span className="font-semibold text-slate-700">{text.readinessBlockersLabel}:</span>{' '}
                {selectedAgentReadinessBlockers.join(' · ')}
              </div>
            ) : null}
            {selectedAgentReadinessWarnings.length > 0 ? (
              <div>
                <span className="font-semibold text-slate-700">{text.readinessWarningsLabel}:</span>{' '}
                {selectedAgentReadinessWarnings.join(' · ')}
              </div>
            ) : null}
          </div>
        ) : (
          <div className="mt-2 text-xs text-slate-500">{text.noReadinessIssues}</div>
        )}
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.requiredSkillsLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(requiredSkillIds, skillTitleById, 'cyan', text.none, 'required-skills')}
          </div>
          {selectedAgentMissingRequiredSkillIds.length > 0 ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {renderPills(
                selectedAgentMissingRequiredSkillIds,
                skillTitleById,
                'rose',
                text.none,
                'missing-required-skills'
              )}
            </div>
          ) : null}
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.requiredToolsLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(requiredToolIds, toolTitleById, 'emerald', text.none, 'required-tools')}
          </div>
          {missingRequiredToolIds.length > 0 ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {renderPills(
                missingRequiredToolIds,
                toolTitleById,
                'rose',
                text.none,
                'missing-required-tools'
              )}
            </div>
          ) : null}
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.requiredMcpServersLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(requiredMcpServerIds, mcpServerTitleById, 'violet', text.none, 'required-mcp')}
          </div>
          {missingRequiredMcpServerIds.length > 0 ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {renderPills(
                missingRequiredMcpServerIds,
                mcpServerTitleById,
                'amber',
                text.none,
                'missing-required-mcp'
              )}
            </div>
          ) : null}
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.approvedSkillsLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(loadedSkillIds, skillTitleById, 'cyan', text.none, 'approved-skills')}
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.missingSkillsLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(selectedAgentMissingSkillIds, skillTitleById, 'rose', text.none, 'missing-skills')}
          </div>
          {selectedAgentMissingSkillDetails.length > 0 ? (
            <div className="mt-3 space-y-2">
              {selectedAgentMissingSkillDetails.map((detail, detailIndex) => {
                const detailSkillId =
                  typeof detail.skill_id === 'string' && detail.skill_id
                    ? detail.skill_id
                    : `missing-skill-${detailIndex + 1}`;
                const suggestedToolIds = coerceStringList(detail.suggested_tool_ids);
                const suggestedMcpServerIds = coerceStringList(detail.suggested_mcp_server_ids);
                return (
                  <div
                    key={`${selectedAgent.agent_id}-missing-skill-detail-${detailSkillId}`}
                    className="rounded-xl border border-rose-100 bg-rose-50/70 p-3"
                  >
                    <div className="text-sm font-semibold text-slate-900">
                      {typeof detail.title === 'string' && detail.title
                        ? detail.title
                        : formatSkillTitle(detailSkillId, skillTitleById)}
                    </div>
                    {typeof detail.prompt_hint === 'string' && detail.prompt_hint ? (
                      <div className="mt-2 text-xs leading-5 text-slate-600">
                        <span className="font-semibold text-slate-700">{text.promptHintLabel}:</span>{' '}
                        {detail.prompt_hint}
                      </div>
                    ) : null}
                    {suggestedToolIds.length > 0 ? (
                      <div className="mt-2">
                        <div className="text-[11px] font-semibold text-slate-500">
                          {text.suggestedToolsLabel}
                        </div>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {renderPills(
                            suggestedToolIds,
                            toolTitleById,
                            'emerald',
                            text.none,
                            `missing-skill-tools-${detailSkillId}`
                          )}
                        </div>
                      </div>
                    ) : null}
                    {suggestedMcpServerIds.length > 0 ? (
                      <div className="mt-2">
                        <div className="text-[11px] font-semibold text-slate-500">
                          {text.suggestedMcpServersLabel}
                        </div>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {renderPills(
                            suggestedMcpServerIds,
                            mcpServerTitleById,
                            'violet',
                            text.none,
                            `missing-skill-mcp-${detailSkillId}`
                          )}
                        </div>
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.suggestedSkillsLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(suggestedSkillIds, skillTitleById, 'amber', text.none, 'suggested-skills')}
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.enabledToolsLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(enabledToolIds, toolTitleById, 'emerald', text.none, 'enabled-tools')}
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.disabledToolsLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(disabledToolIds, toolTitleById, 'slate', text.none, 'disabled-tools')}
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.toolPolicySummaryLabel}
          </div>
          <div className="mt-2 space-y-3">
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.toolAllowPolicyLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  configuredAllowedToolIds,
                  toolTitleById,
                  'emerald',
                  text.none,
                  'tool-policy-allow'
                )}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.toolDenyPolicyLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  configuredDeniedToolIds,
                  toolTitleById,
                  'rose',
                  text.none,
                  'tool-policy-deny'
                )}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.policyAddedToolsLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(policyAddedToolIds, toolTitleById, 'cyan', text.none, 'tool-policy-added')}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.policyBlockedToolsLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  selectedAgentPolicyBlockedToolIds,
                  toolTitleById,
                  'amber',
                  text.none,
                  'tool-policy-blocked'
                )}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.unknownToolPolicyLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  unknownAllowedToolIds,
                  EMPTY_LOOKUP,
                  'slate',
                  text.none,
                  'tool-policy-unknown'
                )}
              </div>
            </div>
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.mcpPolicySummaryLabel}
          </div>
          <div className="mt-2 space-y-3">
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.mcpAllowPolicyLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  configuredAllowedMcpServerIds,
                  mcpServerTitleById,
                  'violet',
                  text.none,
                  'mcp-policy-allow'
                )}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.mcpDenyPolicyLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  configuredDeniedMcpServerIds,
                  mcpServerTitleById,
                  'amber',
                  text.none,
                  'mcp-policy-deny'
                )}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.policyAddedMcpServersLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  policyAddedMcpServerIds,
                  mcpServerTitleById,
                  'cyan',
                  text.none,
                  'mcp-policy-added'
                )}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.policyBlockedMcpServersLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  selectedAgentPolicyBlockedMcpServerIds,
                  mcpServerTitleById,
                  'amber',
                  text.none,
                  'mcp-policy-blocked'
                )}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.unknownMcpPolicyLabel}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {renderPills(
                  unknownAllowedMcpServerIds,
                  EMPTY_LOOKUP,
                  'slate',
                  text.none,
                  'mcp-policy-unknown'
                )}
              </div>
            </div>
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.providerLimitedToolsLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(
              selectedAgentProviderLimitedToolIds,
              toolTitleById,
              'amber',
              text.none,
              'provider-limited-tools'
            )}
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.mcpServersLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(mcpServerIds, mcpServerTitleById, 'violet', text.noMcpServersConfigured, 'mcp-servers')}
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.missingMcpServersLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(
              selectedAgentMissingMcpServerIds,
              mcpServerTitleById,
              'amber',
              text.noMcpGaps,
              'missing-mcp-servers'
            )}
          </div>
          {selectedAgentMissingMcpServerDetails.length > 0 ? (
            <div className="mt-3 space-y-2">
              {selectedAgentMissingMcpServerDetails.map((detail, detailIndex) => {
                const detailServerId =
                  typeof detail.server_id === 'string' && detail.server_id
                    ? detail.server_id
                    : `missing-mcp-${detailIndex + 1}`;
                const isEnabled = detail.status === 'enabled';
                return (
                  <div
                    key={`${selectedAgent.agent_id}-missing-mcp-detail-${detailServerId}`}
                    className="rounded-xl border border-amber-100 bg-amber-50/70 p-3"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-sm font-semibold text-slate-900">
                        {typeof detail.title === 'string' && detail.title
                          ? detail.title
                          : formatSkillTitle(detailServerId, mcpServerTitleById)}
                      </div>
                      <span
                        className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${
                          isEnabled
                            ? 'bg-emerald-50 text-emerald-800 ring-emerald-200'
                            : 'bg-slate-100 text-slate-700 ring-slate-200'
                        }`}
                      >
                        {isEnabled ? text.enabled : text.disabled}
                      </span>
                    </div>
                    {typeof detail.description === 'string' && detail.description ? (
                      <div className="mt-2 text-xs leading-5 text-slate-600">
                        <span className="font-semibold text-slate-700">{text.descriptionLabel}:</span>{' '}
                        {detail.description}
                      </div>
                    ) : null}
                    {typeof detail.command_preview === 'string' && detail.command_preview ? (
                      <div className="mt-2 text-xs leading-5 text-slate-600">
                        <span className="font-semibold text-slate-700">{text.commandPreviewLabel}:</span>{' '}
                        <code className="rounded bg-white/80 px-1.5 py-0.5 text-[11px] text-slate-700">
                          {detail.command_preview}
                        </code>
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.delegationLanesLabel}
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {renderPills(delegationLaneIds, EMPTY_LOOKUP, 'sky', text.none, 'delegation-lanes')}
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.recommendedCollaboratorsLabel}
          </div>
          <div className="mt-3 space-y-2">
            {recommendedCollaborators.length > 0 ? (
              recommendedCollaborators.map((item, recommendationIndex) => {
                const fit = typeof item.fit === 'string' ? item.fit : 'weak';
                const collaboratorId = typeof item.agent_id === 'string' ? item.agent_id : '';
                const collaboratorName = item.agent_name || collaboratorId || text.none;
                const isConnected =
                  (collaboratorId ? selectedAgentDownstreamTargetIds.has(collaboratorId) : false)
                  || Boolean(item.edge_present);
                const newSkillIds = coerceStringList(item.new_skill_ids);
                const overlapLaneIds = coerceStringList(item.overlap_lane_ids);
                const complementaryLaneIds = coerceStringList(item.complementary_lane_ids);
                const newToolIds = coerceStringList(item.new_tool_ids);
                const newMcpServerIds = coerceStringList(item.new_mcp_server_ids);
                const gapCoverMcpServerIds = coerceStringList(item.gap_cover_mcp_server_ids);
                return (
                  <div
                    key={`${selectedAgent.agent_id}-recommendation-${item.agent_id ?? recommendationIndex}`}
                    className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="text-sm font-semibold text-slate-900">{collaboratorName}</div>
                      <div className="flex flex-wrap items-center justify-end gap-2">
                        {isConnected ? (
                          <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
                            {text.connectedHandoffLabel}
                          </span>
                        ) : null}
                        {renderCollaboratorSignalBadges(
                          item,
                          `recommendation-${collaboratorId || recommendationIndex}`
                        )}
                        <span
                          className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationFitBadgeClass(fit)}`}
                        >
                          {formatDelegationFitLabel(fit, text)}
                        </span>
                      </div>
                    </div>
                    {item.rationale ? (
                      <div className="mt-1 text-xs leading-5 text-slate-600">{item.rationale}</div>
                    ) : null}
                    {overlapLaneIds.length > 0 ? (
                      <div className="mt-2">
                        <div className="text-[11px] font-semibold text-slate-500">{text.overlapLanesLabel}</div>
                        <div className="mt-1 flex flex-wrap gap-2">
                          {renderPills(
                            overlapLaneIds,
                            EMPTY_LOOKUP,
                            'sky',
                            text.none,
                            `recommendation-overlap-${collaboratorId || recommendationIndex}`
                          )}
                        </div>
                      </div>
                    ) : null}
                    {complementaryLaneIds.length > 0 ? (
                      <div className="mt-2">
                        <div className="text-[11px] font-semibold text-slate-500">
                          {text.complementaryLanesLabel}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-2">
                          {renderPills(
                            complementaryLaneIds,
                            EMPTY_LOOKUP,
                            'cyan',
                            text.none,
                            `recommendation-complement-${collaboratorId || recommendationIndex}`
                          )}
                        </div>
                      </div>
                    ) : null}
                    {newSkillIds.length > 0 ? (
                      <div className="mt-2">
                        <div className="text-[11px] font-semibold text-slate-500">
                          {text.collaboratorAddedSkillsLabel}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-2">
                          {renderPills(
                            newSkillIds,
                            skillTitleById,
                            'cyan',
                            text.none,
                            `recommendation-new-skills-${collaboratorId || recommendationIndex}`
                          )}
                        </div>
                      </div>
                    ) : null}
                    {newToolIds.length > 0 ? (
                      <div className="mt-2">
                        <div className="text-[11px] font-semibold text-slate-500">
                          {text.collaboratorAddedToolsLabel}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-2">
                          {renderPills(
                            newToolIds,
                            toolTitleById,
                            'emerald',
                            text.none,
                            `recommendation-new-tools-${collaboratorId || recommendationIndex}`
                          )}
                        </div>
                      </div>
                    ) : null}
                    {newMcpServerIds.length > 0 ? (
                      <div className="mt-2">
                        <div className="text-[11px] font-semibold text-slate-500">
                          {text.collaboratorAddedMcpLabel}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-2">
                          {renderPills(
                            newMcpServerIds,
                            mcpServerTitleById,
                            'violet',
                            text.none,
                            `recommendation-new-mcp-${collaboratorId || recommendationIndex}`
                          )}
                        </div>
                      </div>
                    ) : null}
                    {gapCoverMcpServerIds.length > 0 ? (
                      <div className="mt-2">
                        <div className="text-[11px] font-semibold text-slate-500">
                          {text.collaboratorCoversMissingMcpLabel}
                        </div>
                        <div className="mt-1 flex flex-wrap gap-2">
                          {renderPills(
                            gapCoverMcpServerIds,
                            mcpServerTitleById,
                            'amber',
                            text.none,
                            `recommendation-gap-cover-mcp-${collaboratorId || recommendationIndex}`
                          )}
                        </div>
                      </div>
                    ) : null}
                    {collaboratorId ? (
                      <div className="mt-3 flex flex-wrap gap-2">
                        {focusableAgentIds.has(collaboratorId) ? (
                          <button
                            type="button"
                            onClick={() => onOpenNodeEditor(collaboratorId)}
                            className="inline-flex items-center justify-center rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-100"
                          >
                            {formatTemplate(text.focusNodeForRecovery, { name: collaboratorName })}
                          </button>
                        ) : null}
                        {!isConnected ? (
                          <button
                            type="button"
                            onClick={() =>
                              onApplySuggestedHandoff({
                                sourceAgentId: selectedAgent.agent_id,
                                targetAgentId: collaboratorId,
                              })
                            }
                            disabled={isApplying}
                            className="inline-flex items-center justify-center rounded-xl border border-cyan-200 bg-white px-3 py-2 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                          >
                            {isApplying
                              ? text.applyingSuggestedHandoff
                              : formatTemplate(text.applySuggestedHandoff, { name: collaboratorName })}
                          </button>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                );
              })
            ) : (
              <div className="text-xs text-slate-500">{text.noRecommendedCollaborators}</div>
            )}
          </div>
        </div>
      </div>
      <div className="rounded-2xl bg-white p-3">
        <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
          {text.downstreamHandoffFitLabel}
        </div>
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          {downstreamHandoffScores.length > 0 ? (
            downstreamHandoffScores.map((item, recommendationIndex) => {
              const fit = typeof item.fit === 'string' ? item.fit : 'weak';
              const downstreamAgentId = typeof item.agent_id === 'string' ? item.agent_id : '';
              const downstreamAgentName = item.agent_name || downstreamAgentId || text.none;
              const isConnected =
                (downstreamAgentId ? selectedAgentDownstreamTargetIds.has(downstreamAgentId) : false)
                || Boolean(item.edge_present);
              const newSkillIds = coerceStringList(item.new_skill_ids);
              const newToolIds = coerceStringList(item.new_tool_ids);
              const newMcpServerIds = coerceStringList(item.new_mcp_server_ids);
              const gapCoverMcpServerIds = coerceStringList(item.gap_cover_mcp_server_ids);
              const canAddSuggestedHandoff =
                Boolean(downstreamAgentId) && !isConnected && (fit === 'strong' || fit === 'good');
              const canRewireToPrimarySuggestion =
                Boolean(downstreamAgentId)
                && isConnected
                && fit === 'weak'
                && Boolean(selectedAgentPrimarySuggestedCollaborator?.agent_id)
                && selectedAgentPrimarySuggestedCollaborator?.agent_id !== downstreamAgentId;
              return (
                <div
                  key={`${selectedAgent.agent_id}-downstream-${item.agent_id ?? recommendationIndex}`}
                  className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-sm font-semibold text-slate-900">{downstreamAgentName}</div>
                    <div className="flex flex-wrap items-center justify-end gap-2">
                      {isConnected ? (
                        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
                          {text.connectedHandoffLabel}
                        </span>
                      ) : null}
                      {renderCollaboratorSignalBadges(item, `downstream-${downstreamAgentId || recommendationIndex}`)}
                      <span
                        className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationFitBadgeClass(fit)}`}
                      >
                        {formatDelegationFitLabel(fit, text)}
                      </span>
                    </div>
                  </div>
                  {item.rationale ? (
                    <div className="mt-1 text-xs leading-5 text-slate-600">{item.rationale}</div>
                  ) : null}
                  {newSkillIds.length > 0 ? (
                    <div className="mt-2">
                      <div className="text-[11px] font-semibold text-slate-500">
                        {text.collaboratorAddedSkillsLabel}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {renderPills(
                          newSkillIds,
                          skillTitleById,
                          'cyan',
                          text.none,
                          `downstream-new-skills-${downstreamAgentId || recommendationIndex}`
                        )}
                      </div>
                    </div>
                  ) : null}
                  {newToolIds.length > 0 ? (
                    <div className="mt-2">
                      <div className="text-[11px] font-semibold text-slate-500">
                        {text.collaboratorAddedToolsLabel}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {renderPills(
                          newToolIds,
                          toolTitleById,
                          'emerald',
                          text.none,
                          `downstream-new-tools-${downstreamAgentId || recommendationIndex}`
                        )}
                      </div>
                    </div>
                  ) : null}
                  {newMcpServerIds.length > 0 ? (
                    <div className="mt-2">
                      <div className="text-[11px] font-semibold text-slate-500">
                        {text.collaboratorAddedMcpLabel}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {renderPills(
                          newMcpServerIds,
                          mcpServerTitleById,
                          'violet',
                          text.none,
                          `downstream-new-mcp-${downstreamAgentId || recommendationIndex}`
                        )}
                      </div>
                    </div>
                  ) : null}
                  {gapCoverMcpServerIds.length > 0 ? (
                    <div className="mt-2">
                      <div className="text-[11px] font-semibold text-slate-500">
                        {text.collaboratorCoversMissingMcpLabel}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        {renderPills(
                          gapCoverMcpServerIds,
                          mcpServerTitleById,
                          'amber',
                          text.none,
                          `downstream-gap-cover-mcp-${downstreamAgentId || recommendationIndex}`
                        )}
                      </div>
                    </div>
                  ) : null}
                  {downstreamAgentId ? (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {focusableAgentIds.has(downstreamAgentId) ? (
                        <button
                          type="button"
                          onClick={() => onOpenNodeEditor(downstreamAgentId)}
                          className="inline-flex items-center justify-center rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-100"
                        >
                          {formatTemplate(text.focusNodeForRecovery, { name: downstreamAgentName })}
                        </button>
                      ) : null}
                      {canAddSuggestedHandoff ? (
                        <button
                          type="button"
                          onClick={() =>
                            onApplySuggestedHandoff({
                              sourceAgentId: selectedAgent.agent_id,
                              targetAgentId: downstreamAgentId,
                            })
                          }
                          disabled={isApplying}
                          className="inline-flex items-center justify-center rounded-xl border border-cyan-200 bg-white px-3 py-2 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                        >
                          {isApplying
                            ? text.applyingSuggestedHandoff
                            : formatTemplate(text.applySuggestedHandoff, { name: downstreamAgentName })}
                        </button>
                      ) : null}
                      {canRewireToPrimarySuggestion && selectedAgentPrimarySuggestedCollaborator?.agent_id ? (
                        <button
                          type="button"
                          onClick={() =>
                            onApplySuggestedRewire({
                              sourceAgentId: selectedAgent.agent_id,
                              fromTargetAgentId: downstreamAgentId,
                              toTargetAgentId: selectedAgentPrimarySuggestedCollaborator.agent_id,
                            })
                          }
                          disabled={isApplying}
                          className="inline-flex items-center justify-center rounded-xl border border-cyan-200 bg-white px-3 py-2 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                        >
                          {isApplying
                            ? text.applyingSuggestedRewire
                            : formatTemplate(text.applySuggestedRewire, {
                                name:
                                  selectedAgentPrimarySuggestedCollaborator.agent_name
                                  || selectedAgentPrimarySuggestedCollaborator.agent_id,
                              })}
                        </button>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              );
            })
          ) : (
            <div className="text-xs text-slate-500">{text.noDownstreamHandoffFit}</div>
          )}
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.delegationFocusLabel}
          </div>
          <div className="mt-2 text-sm text-slate-700">
            {selectedAgentCapabilitySummary.delegation_focus || text.noDelegationFocus}
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.capabilityBriefLabel}
          </div>
          <div className="mt-2 text-sm leading-6 text-slate-700">
            {selectedAgentCapabilitySummary.capability_brief || text.none}
          </div>
        </div>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.providerRouteLabel}
          </div>
          <div className="mt-2 text-sm text-slate-700">
            {selectedAgentCapabilitySummary.provider_route || text.none}
          </div>
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.toolExecutionSupportLabel}
          </div>
          <div className="mt-2 text-sm text-slate-700">
            {selectedAgentCapabilitySummary.tool_execution_support === 'supported'
              ? text.toolExecutionSupported
              : selectedAgentCapabilitySummary.tool_execution_support === 'unsupported'
                ? text.toolExecutionUnsupported
                : text.toolExecutionUnknown}
          </div>
          {selectedAgentCapabilitySummary.tool_execution_support_reason ? (
            <div className="mt-2 text-xs leading-5 text-slate-500">
              {selectedAgentCapabilitySummary.tool_execution_support_reason}
            </div>
          ) : null}
        </div>
        <div className="rounded-2xl bg-white p-3">
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
            {text.capabilityReviewPathLabel}
          </div>
          <div className="mt-2 text-sm text-slate-700">
            {selectedAgentCapabilitySummary.review_mode || text.none}
          </div>
        </div>
      </div>
    </div>
  );
}
