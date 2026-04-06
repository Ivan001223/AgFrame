import type { ComponentType, ReactNode } from 'react';
import type { HarnessCanvasAgentDTO, HarnessRunDetailDTO } from '@/domains/harness/hooks';
import { useMessages } from '@/lib/i18n';
import { HARNESS_MESSAGES } from './messages';
import {
  availabilityScopeToneClasses,
  buildRunPreflightScopeCounts,
  capabilityAvailabilityBadgeClass,
  capabilityReadinessBadgeClass,
  delegationFitBadgeClass,
  formatCapabilityAvailabilityLabel,
  formatCapabilityReadinessLabel,
  formatDelegationFitLabel,
  normalizeDelegationFit,
  resolveCapabilityAvailabilityStatus,
  resolveCapabilityReadinessStatus,
  shouldOpenProjectMcpForDiagnostic,
  shouldOpenProjectProvidersForDiagnostic,
  shouldOpenSkillPoolForDiagnostic,
  type AvailabilityScopeSummary,
  type RunPreflightScopeCounts,
} from './diagnostics';
import {
  type AgentPolicyRepairDiagnostic,
  type AgentRoleProfileDiagnostic,
  type RoleProfileScopeSummary,
  computeActionableMcpPolicySuggestionIds,
  computeActionableToolPolicySuggestionIds,
  computeCoordinatorMcpPolicyRestrictionIds,
  computeCoordinatorToolPolicyRestrictionIds,
  type PolicyRepairScopeSummary,
} from './policy-repair';
import { coerceRecord, coerceRecordList, coerceStringList, formatSkillTitle, formatTemplate } from './utils';

type AgentPolicyRepairParams = {
  agentId: string;
  skillIds?: string[];
  toolIds?: string[];
  mcpServerIds?: string[];
  denyToolIds?: string[];
  denyMcpServerIds?: string[];
  forceAllowToolIds?: boolean;
  forceAllowMcpServerIds?: boolean;
};

type DiagnosticTone = 'slate' | 'rose' | 'amber' | 'cyan' | 'sky' | 'violet';
const EMPTY_LOOKUP = new Map<string, string>();

function secondaryActionToneClasses(tone: DiagnosticTone) {
  if (tone === 'rose') {
    return 'border-rose-200 text-rose-900 hover:bg-rose-100';
  }
  if (tone === 'amber') {
    return 'border-amber-200 text-amber-900 hover:bg-amber-100';
  }
  if (tone === 'cyan') {
    return 'border-cyan-200 text-cyan-900 hover:bg-cyan-100';
  }
  if (tone === 'sky') {
    return 'border-sky-200 text-sky-900 hover:bg-sky-100';
  }
  if (tone === 'violet') {
    return 'border-violet-200 text-violet-900 hover:bg-violet-100';
  }
  return 'border-slate-200 text-slate-800 hover:bg-slate-50';
}

function SecondaryActionButton({
  children,
  disabled,
  onClick,
  tone = 'slate',
}: {
  children: ReactNode;
  disabled?: boolean;
  onClick: () => void;
  tone?: DiagnosticTone;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center rounded-xl border bg-white px-3 py-2 text-xs font-semibold disabled:opacity-50 ${secondaryActionToneClasses(tone)}`}
    >
      {children}
    </button>
  );
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

function contractPillToneClasses(tone: 'slate' | 'cyan' | 'emerald' | 'violet' | 'amber') {
  if (tone === 'cyan') {
    return 'bg-cyan-50 text-cyan-800 ring-cyan-200';
  }
  if (tone === 'emerald') {
    return 'bg-emerald-50 text-emerald-800 ring-emerald-200';
  }
  if (tone === 'violet') {
    return 'bg-violet-50 text-violet-800 ring-violet-200';
  }
  if (tone === 'amber') {
    return 'bg-amber-50 text-amber-800 ring-amber-200';
  }
  return 'bg-slate-100 text-slate-700 ring-slate-200';
}

function collaborationRoleBadgeClass(mode: string | null | undefined) {
  if (mode === 'coordinator') {
    return 'bg-cyan-50 text-cyan-800 ring-cyan-200';
  }
  if (mode === 'implementation') {
    return 'bg-emerald-50 text-emerald-800 ring-emerald-200';
  }
  if (mode === 'verification') {
    return 'bg-sky-50 text-sky-800 ring-sky-200';
  }
  if (mode === 'research') {
    return 'bg-violet-50 text-violet-800 ring-violet-200';
  }
  return 'bg-slate-100 text-slate-700 ring-slate-200';
}

function collaborationRiskBadgeClass(severity: string | null | undefined) {
  if (severity === 'high') {
    return 'bg-rose-50 text-rose-800 ring-rose-200';
  }
  if (severity === 'medium') {
    return 'bg-amber-50 text-amber-800 ring-amber-200';
  }
  return 'bg-sky-50 text-sky-800 ring-sky-200';
}

function HandoffOpportunityContext({
  diagnostic,
}: {
  diagnostic: Record<string, unknown>;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const sourceLaneIds = coerceStringList(diagnostic.source_lane_ids);
  const delegationFocus =
    typeof diagnostic.delegation_focus === 'string' && diagnostic.delegation_focus.trim()
      ? diagnostic.delegation_focus.trim()
      : null;

  if (sourceLaneIds.length === 0 && !delegationFocus) {
    return null;
  }

  return (
    <div className="mt-2 space-y-2 text-xs leading-5 text-slate-600">
      {delegationFocus ? (
        <div>
          <span className="font-semibold text-slate-700">{text.delegationFocusLabel}:</span>{' '}
          {delegationFocus}
        </div>
      ) : null}
      {sourceLaneIds.length > 0 ? (
        <div>
          <div className="text-[11px] font-semibold text-slate-500">{text.sourceLanesLabel}</div>
          <div className="mt-1 flex flex-wrap gap-2">
            {sourceLaneIds.map((laneId) => (
              <span
                key={`handoff-context-lane-${laneId}`}
                className="rounded-full bg-sky-50 px-2.5 py-1 text-[10px] font-semibold text-sky-800 ring-1 ring-sky-200"
              >
                {formatSkillTitle(laneId, EMPTY_LOOKUP)}
              </span>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function CollaborationPreviewButtons({
  previews,
  canFocusAgent,
  onFocusAgent,
}: {
  previews: Record<string, unknown>[];
  canFocusAgent?: (agentId: string) => boolean;
  onFocusAgent?: (agentId: string) => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  if (previews.length === 0) {
    return <div className="text-xs text-slate-500">{text.none}</div>;
  }
  return (
    <div className="flex flex-wrap gap-2">
      {previews.map((preview, index) => {
        const agentId =
          typeof preview.agent_id === 'string' && preview.agent_id
            ? preview.agent_id
            : `preview-agent-${index + 1}`;
        const agentName =
          typeof preview.agent_name === 'string' && preview.agent_name
            ? preview.agent_name
            : agentId;
        return canFocusAgent?.(agentId) && onFocusAgent ? (
          <SecondaryActionButton key={`${agentId}-${index}`} tone="slate" onClick={() => onFocusAgent(agentId)}>
            {formatTemplate(text.focusNodeForRecovery, { name: agentName })}
          </SecondaryActionButton>
        ) : (
          <span
            key={`${agentId}-${index}`}
            className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200"
          >
            {agentName}
          </span>
        );
      })}
    </div>
  );
}

function CapabilityPills({
  values,
  tone,
  emptyLabel,
  keyPrefix,
}: {
  values: string[];
  tone: 'slate' | 'cyan' | 'emerald' | 'violet' | 'amber';
  emptyLabel: string;
  keyPrefix: string;
}) {
  if (values.length === 0) {
    return <div className="text-xs text-slate-500">{emptyLabel}</div>;
  }
  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {values.map((value) => (
        <span
          key={`${keyPrefix}-${value}`}
          className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractPillToneClasses(tone)}`}
        >
          {formatSkillTitle(value, EMPTY_LOOKUP)}
        </span>
      ))}
    </div>
  );
}

function RoleProfileRepairButtons({
  diagnostic,
  onApplyAgentPolicyRepair,
  onOpenSkillPool,
  isApplying,
}: {
  diagnostic?: AgentRoleProfileDiagnostic | null;
  onApplyAgentPolicyRepair?: (params: AgentPolicyRepairParams) => void;
  onOpenSkillPool?: () => void;
  isApplying?: boolean;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const agentId = diagnostic?.agentId ?? '';
  const availableSkillIds = diagnostic?.availableSkillIds ?? [];
  const missingSkillIds = diagnostic?.missingSkillIds ?? [];
  const toolIds = diagnostic?.toolIds ?? [];
  const mcpServerIds = diagnostic?.mcpServerIds ?? [];
  const denyToolIds = diagnostic?.denyToolIds ?? [];
  const denyMcpServerIds = diagnostic?.denyMcpServerIds ?? [];
  const hasApplyAction =
    availableSkillIds.length > 0
    || toolIds.length > 0
    || mcpServerIds.length > 0
    || denyToolIds.length > 0
    || denyMcpServerIds.length > 0;

  if (!agentId || (!hasApplyAction && missingSkillIds.length === 0)) {
    return null;
  }

  return (
    <>
      {hasApplyAction && onApplyAgentPolicyRepair ? (
        <SecondaryActionButton
          tone="cyan"
          disabled={isApplying}
          onClick={() =>
            onApplyAgentPolicyRepair({
              agentId,
              skillIds: availableSkillIds,
              toolIds,
              mcpServerIds,
              denyToolIds,
              denyMcpServerIds,
              forceAllowToolIds: toolIds.length > 0,
              forceAllowMcpServerIds: mcpServerIds.length > 0,
            })
          }
        >
          {isApplying ? text.applyingRecoveryRoleProfiles : text.applyRoleProfileAction}
        </SecondaryActionButton>
      ) : null}
      {missingSkillIds.length > 0 && onOpenSkillPool ? (
        <SecondaryActionButton tone="rose" onClick={onOpenSkillPool}>
          {text.openSkillPoolAction}
        </SecondaryActionButton>
      ) : null}
    </>
  );
}

function SummaryPolicyRepairButtons({
  diagnostic,
  onApplyAgentPolicyRepair,
  isApplying,
}: {
  diagnostic?: AgentPolicyRepairDiagnostic | null;
  onApplyAgentPolicyRepair?: (params: AgentPolicyRepairParams) => void;
  isApplying?: boolean;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const agentId = diagnostic?.agentId ?? '';
  const allowToolIds = diagnostic?.allowToolIds ?? [];
  const allowMcpServerIds = diagnostic?.allowMcpServerIds ?? [];
  const denyToolIds = diagnostic?.denyToolIds ?? [];
  const denyMcpServerIds = diagnostic?.denyMcpServerIds ?? [];

  if (
    !agentId
    || !onApplyAgentPolicyRepair
    || (
      allowToolIds.length === 0
      && allowMcpServerIds.length === 0
      && denyToolIds.length === 0
      && denyMcpServerIds.length === 0
    )
  ) {
    return null;
  }

  return (
    <>
      {allowToolIds.length > 0 ? (
        <SecondaryActionButton
          tone="cyan"
          disabled={isApplying}
          onClick={() =>
            onApplyAgentPolicyRepair({
              agentId,
              toolIds: allowToolIds,
            })
          }
        >
          {text.allowSuggestedToolsAction}
        </SecondaryActionButton>
      ) : null}
      {allowMcpServerIds.length > 0 ? (
        <SecondaryActionButton
          tone="cyan"
          disabled={isApplying}
          onClick={() =>
            onApplyAgentPolicyRepair({
              agentId,
              mcpServerIds: allowMcpServerIds,
            })
          }
        >
          {text.allowSuggestedMcpAction}
        </SecondaryActionButton>
      ) : null}
      {denyToolIds.length > 0 ? (
        <SecondaryActionButton
          tone="amber"
          disabled={isApplying}
          onClick={() =>
            onApplyAgentPolicyRepair({
              agentId,
              denyToolIds,
            })
          }
        >
          {text.restrictCoordinatorToolsAction}
        </SecondaryActionButton>
      ) : null}
      {denyMcpServerIds.length > 0 ? (
        <SecondaryActionButton
          tone="amber"
          disabled={isApplying}
          onClick={() =>
            onApplyAgentPolicyRepair({
              agentId,
              denyMcpServerIds,
            })
          }
        >
          {text.restrictCoordinatorMcpAction}
        </SecondaryActionButton>
      ) : null}
    </>
  );
}

function DelegateCandidateCards({
  sourceAgentId,
  candidates,
  keyPrefix,
  canFocusAgent,
  onFocusAgent,
  onApplySuggestedHandoff,
  isApplyingGraphChange,
  roleProfileDiagnosticsByAgentId,
  policyRepairDiagnosticsByAgentId,
  onApplyAgentPolicyRepair,
  onOpenSkillPool,
}: {
  sourceAgentId: string;
  candidates: Record<string, unknown>[];
  keyPrefix: string;
  canFocusAgent?: (agentId: string) => boolean;
  onFocusAgent?: (agentId: string) => void;
  onApplySuggestedHandoff?: (params: {
    sourceAgentId: string;
    targetAgentId: string;
  }) => void;
  isApplyingGraphChange?: boolean;
  roleProfileDiagnosticsByAgentId?: ReadonlyMap<string, AgentRoleProfileDiagnostic>;
  policyRepairDiagnosticsByAgentId?: ReadonlyMap<string, AgentPolicyRepairDiagnostic>;
  onApplyAgentPolicyRepair?: (params: AgentPolicyRepairParams) => void;
  onOpenSkillPool?: () => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);

  if (candidates.length === 0) {
    return null;
  }

  return (
    <div className="mt-3">
      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
        {text.collaborationDelegateCandidatesLabel}
      </div>
      <div className="mt-2 space-y-2">
        {candidates.map((candidate, candidateIndex) => {
          const candidateAgentId =
            typeof candidate.agent_id === 'string' && candidate.agent_id ? candidate.agent_id : '';
          const candidateName =
            typeof candidate.agent_name === 'string' && candidate.agent_name
              ? candidate.agent_name
              : candidateAgentId || `${text.unknownNode} ${candidateIndex + 1}`;
          const candidateFit = normalizeDelegationFit(
            typeof candidate.fit === 'string' ? candidate.fit : 'weak'
          );
          const complementaryLaneIds = coerceStringList(candidate.complementary_lane_ids);
          const newSkillIds = coerceStringList(candidate.new_skill_ids);
          const newToolIds = coerceStringList(candidate.new_tool_ids);
          const newMcpServerIds = coerceStringList(candidate.new_mcp_server_ids);
          const gapCoverMcpServerIds = coerceStringList(candidate.gap_cover_mcp_server_ids);
          const isConnected = Boolean(candidate.edge_present);
          const roleProfileDiagnostic = candidateAgentId
            ? roleProfileDiagnosticsByAgentId?.get(candidateAgentId) ?? null
            : null;
          const policyRepairDiagnostic = candidateAgentId
            ? policyRepairDiagnosticsByAgentId?.get(candidateAgentId) ?? null
            : null;
          const followupMissingSkillIds = roleProfileDiagnostic?.missingSkillIds ?? [];
          const hasRoleProfileApplyAction = Boolean(
            roleProfileDiagnostic
            && (
              roleProfileDiagnostic.availableSkillIds.length > 0
              || roleProfileDiagnostic.toolIds.length > 0
              || roleProfileDiagnostic.mcpServerIds.length > 0
              || roleProfileDiagnostic.denyToolIds.length > 0
              || roleProfileDiagnostic.denyMcpServerIds.length > 0
            )
          );
          const hasPolicyRepairAction = Boolean(
            policyRepairDiagnostic
            && (
              policyRepairDiagnostic.allowToolIds.length > 0
              || policyRepairDiagnostic.allowMcpServerIds.length > 0
              || policyRepairDiagnostic.denyToolIds.length > 0
              || policyRepairDiagnostic.denyMcpServerIds.length > 0
            )
          );
          const hasFollowupActions = Boolean(
            (followupMissingSkillIds.length > 0 && onOpenSkillPool)
            || (hasRoleProfileApplyAction && onApplyAgentPolicyRepair)
            || (hasPolicyRepairAction && onApplyAgentPolicyRepair)
          );
          return (
            <div
              key={`${keyPrefix}-delegate-${candidateAgentId || candidateIndex}`}
              className="rounded-xl border border-slate-200 bg-slate-50/80 p-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="text-sm font-semibold text-slate-900">{candidateName}</div>
                <div className="flex flex-wrap items-center gap-2">
                  {isConnected ? (
                    <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
                      {text.connectedHandoffLabel}
                    </span>
                  ) : null}
                  <span
                    className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationFitBadgeClass(candidateFit)}`}
                  >
                    {formatDelegationFitLabel(candidateFit, text)}
                  </span>
                </div>
              </div>
              {typeof candidate.rationale === 'string' && candidate.rationale ? (
                <div className="mt-2 text-xs leading-5 text-slate-600">{candidate.rationale}</div>
              ) : null}
              {complementaryLaneIds.length > 0 ? (
                <div className="mt-2">
                  <div className="text-[11px] font-semibold text-slate-500">{text.complementaryLanesLabel}</div>
                  <CapabilityPills
                    values={complementaryLaneIds}
                    tone="cyan"
                    emptyLabel={text.none}
                    keyPrefix={`${keyPrefix}-delegate-lanes-${candidateAgentId || candidateIndex}`}
                  />
                </div>
              ) : null}
              {newSkillIds.length > 0 ? (
                <div className="mt-2">
                  <div className="text-[11px] font-semibold text-slate-500">{text.collaboratorAddedSkillsLabel}</div>
                  <CapabilityPills
                    values={newSkillIds}
                    tone="cyan"
                    emptyLabel={text.none}
                    keyPrefix={`${keyPrefix}-delegate-skills-${candidateAgentId || candidateIndex}`}
                  />
                </div>
              ) : null}
              {newToolIds.length > 0 ? (
                <div className="mt-2">
                  <div className="text-[11px] font-semibold text-slate-500">{text.collaboratorAddedToolsLabel}</div>
                  <CapabilityPills
                    values={newToolIds}
                    tone="emerald"
                    emptyLabel={text.none}
                    keyPrefix={`${keyPrefix}-delegate-tools-${candidateAgentId || candidateIndex}`}
                  />
                </div>
              ) : null}
              {newMcpServerIds.length > 0 ? (
                <div className="mt-2">
                  <div className="text-[11px] font-semibold text-slate-500">{text.collaboratorAddedMcpLabel}</div>
                  <CapabilityPills
                    values={newMcpServerIds}
                    tone="violet"
                    emptyLabel={text.none}
                    keyPrefix={`${keyPrefix}-delegate-mcp-${candidateAgentId || candidateIndex}`}
                  />
                </div>
              ) : null}
              {gapCoverMcpServerIds.length > 0 ? (
                <div className="mt-2">
                  <div className="text-[11px] font-semibold text-slate-500">{text.collaboratorCoversMissingMcpLabel}</div>
                  <CapabilityPills
                    values={gapCoverMcpServerIds}
                    tone="amber"
                    emptyLabel={text.none}
                    keyPrefix={`${keyPrefix}-delegate-cover-mcp-${candidateAgentId || candidateIndex}`}
                  />
                </div>
              ) : null}
              {(canFocusAgent?.(candidateAgentId) || (onApplySuggestedHandoff && sourceAgentId && candidateAgentId && !isConnected)) ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {candidateAgentId && canFocusAgent?.(candidateAgentId) && onFocusAgent ? (
                    <SecondaryActionButton tone="slate" onClick={() => onFocusAgent(candidateAgentId)}>
                      {formatTemplate(text.focusNodeForRecovery, { name: candidateName })}
                    </SecondaryActionButton>
                  ) : null}
                  {onApplySuggestedHandoff && sourceAgentId && candidateAgentId && !isConnected ? (
                    <SecondaryActionButton
                      tone="cyan"
                      disabled={isApplyingGraphChange}
                      onClick={() =>
                        onApplySuggestedHandoff({
                          sourceAgentId,
                          targetAgentId: candidateAgentId,
                        })
                      }
                    >
                      {isApplyingGraphChange
                        ? text.applyingSuggestedHandoff
                        : formatTemplate(text.applySuggestedHandoff, { name: candidateName })}
                    </SecondaryActionButton>
                  ) : null}
                </div>
              ) : null}
              {hasFollowupActions ? (
                <div className="mt-3 rounded-xl border border-white/80 bg-white/90 p-3">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {text.collaborationDelegateFollowupLabel}
                  </div>
                  <div className="mt-1 text-xs leading-5 text-slate-600">
                    {text.collaborationDelegateFollowupHint}
                  </div>
                  {followupMissingSkillIds.length > 0 ? (
                    <div className="mt-2 text-xs leading-5 text-slate-600">
                      <span className="font-semibold text-slate-700">{text.roleProfileMissingSkillsRecoveryLabel}:</span>{' '}
                      {followupMissingSkillIds.map((skillId) => formatSkillTitle(skillId, EMPTY_LOOKUP)).join(' · ')}
                    </div>
                  ) : null}
                  <div className="mt-3 flex flex-wrap gap-2">
                    <RoleProfileRepairButtons
                      diagnostic={roleProfileDiagnostic}
                      onApplyAgentPolicyRepair={onApplyAgentPolicyRepair}
                      onOpenSkillPool={onOpenSkillPool}
                      isApplying={isApplyingGraphChange}
                    />
                    <SummaryPolicyRepairButtons
                      diagnostic={policyRepairDiagnostic}
                      onApplyAgentPolicyRepair={onApplyAgentPolicyRepair}
                      isApplying={isApplyingGraphChange}
                    />
                  </div>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DiagnosticPolicyRepairButtons({
  graphAgent,
  diagnosticAgent,
  onApplyAgentPolicyRepair,
  isApplying,
  tone = 'slate',
}: {
  graphAgent?: HarnessCanvasAgentDTO | null;
  diagnosticAgent: Record<string, unknown>;
  onApplyAgentPolicyRepair?: (params: AgentPolicyRepairParams) => void;
  isApplying?: boolean;
  tone?: DiagnosticTone;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const agentId = typeof diagnosticAgent.agent_id === 'string' ? diagnosticAgent.agent_id : '';
  const actionableToolIds = computeActionableToolPolicySuggestionIds(
    graphAgent,
    coerceStringList(diagnosticAgent.policy_blocked_tool_ids)
  );
  const actionableMcpServerIds = computeActionableMcpPolicySuggestionIds(
    graphAgent,
    coerceStringList(diagnosticAgent.policy_blocked_mcp_server_ids)
  );
  const restrictiveToolIds = computeCoordinatorToolPolicyRestrictionIds(graphAgent, diagnosticAgent);
  const restrictiveMcpServerIds = computeCoordinatorMcpPolicyRestrictionIds(graphAgent, diagnosticAgent);

  if (
    !agentId
    || !onApplyAgentPolicyRepair
    || (
      actionableToolIds.length === 0
      && actionableMcpServerIds.length === 0
      && restrictiveToolIds.length === 0
      && restrictiveMcpServerIds.length === 0
    )
  ) {
    return null;
  }

  return (
    <>
      {actionableToolIds.length > 0 ? (
        <SecondaryActionButton
          tone={tone}
          disabled={isApplying}
          onClick={() =>
            onApplyAgentPolicyRepair({
              agentId,
              toolIds: actionableToolIds,
            })
          }
        >
          {text.allowSuggestedToolsAction}
        </SecondaryActionButton>
      ) : null}
      {actionableMcpServerIds.length > 0 ? (
        <SecondaryActionButton
          tone={tone}
          disabled={isApplying}
          onClick={() =>
            onApplyAgentPolicyRepair({
              agentId,
              mcpServerIds: actionableMcpServerIds,
            })
          }
        >
          {text.allowSuggestedMcpAction}
        </SecondaryActionButton>
      ) : null}
      {restrictiveToolIds.length > 0 ? (
        <SecondaryActionButton
          tone={tone}
          disabled={isApplying}
          onClick={() =>
            onApplyAgentPolicyRepair({
              agentId,
              denyToolIds: restrictiveToolIds,
            })
          }
        >
          {text.restrictCoordinatorToolsAction}
        </SecondaryActionButton>
      ) : null}
      {restrictiveMcpServerIds.length > 0 ? (
        <SecondaryActionButton
          tone={tone}
          disabled={isApplying}
          onClick={() =>
            onApplyAgentPolicyRepair({
              agentId,
              denyMcpServerIds: restrictiveMcpServerIds,
            })
          }
        >
          {text.restrictCoordinatorMcpAction}
        </SecondaryActionButton>
      ) : null}
    </>
  );
}

export function HandoffDiagnosticsCard({
  diagnostics,
  emptyLabel,
  onApplyRewire,
  onApplySuggestedHandoff,
  isApplying,
}: {
  diagnostics: Record<string, unknown> | null;
  emptyLabel?: string;
  onApplyRewire?: (params: {
    sourceAgentId: string;
    fromTargetAgentId: string;
    toTargetAgentId: string;
  }) => void;
  onApplySuggestedHandoff?: (params: {
    sourceAgentId: string;
    targetAgentId: string;
  }) => void;
  isApplying?: boolean;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const weakEdges = coerceRecordList(diagnostics?.weak_downstream_edges);
  const bestNextHandoffs = coerceRecordList(diagnostics?.best_next_handoffs);

  if (weakEdges.length === 0 && bestNextHandoffs.length === 0) {
    return <div className="mt-5 text-sm text-slate-500">{emptyLabel || text.noHandoffDiagnostics}</div>;
  }

  return (
    <div className="mt-5 space-y-4">
      {weakEdges.length > 0 ? (
        <div className="space-y-3">
          <div className="text-xs font-semibold uppercase tracking-[0.18em] text-amber-700">{text.weakHandoffEdgesLabel}</div>
          {weakEdges.map((edge, index) => {
            const target = coerceRecord(edge.target);
            const suggestedReplacements = coerceRecordList(edge.suggested_replacements);
            const fit = normalizeDelegationFit(typeof target?.fit === 'string' ? target.fit : 'weak');
            const sourceName =
              typeof edge.source_agent_name === 'string' && edge.source_agent_name
                ? edge.source_agent_name
                : typeof edge.source_agent_id === 'string' && edge.source_agent_id
                  ? edge.source_agent_id
                  : text.unknownNode;
            const targetName =
              typeof target?.agent_name === 'string' && target.agent_name
                ? target.agent_name
                : typeof target?.agent_id === 'string' && target.agent_id
                  ? target.agent_id
                  : text.unknownNode;
            const rationale = typeof target?.rationale === 'string' ? target.rationale : null;
            const primaryReplacement = suggestedReplacements[0] ?? null;
            const replacementAgentId =
              typeof primaryReplacement?.agent_id === 'string' ? primaryReplacement.agent_id : null;
            const replacementAgentName =
              typeof primaryReplacement?.agent_name === 'string' && primaryReplacement.agent_name
                ? primaryReplacement.agent_name
                : replacementAgentId;
            return (
              <div key={`weak-edge-${index}`} className="rounded-2xl border border-amber-200 bg-amber-50/70 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-sm font-semibold text-slate-900">
                    {sourceName} {'->'} {targetName}
                  </div>
                  <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationFitBadgeClass(fit)}`}>
                    {formatDelegationFitLabel(fit, text)}
                  </span>
                </div>
                <HandoffOpportunityContext diagnostic={edge} />
                {rationale ? <div className="mt-2 text-xs leading-5 text-slate-700">{rationale}</div> : null}
                {suggestedReplacements.length > 0 ? (
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">{text.suggestedRewireLabel}</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {suggestedReplacements.map((item, suggestionIndex) => {
                        const suggestionFit = normalizeDelegationFit(typeof item.fit === 'string' ? item.fit : 'weak');
                        const suggestionName =
                          typeof item.agent_name === 'string' && item.agent_name
                            ? item.agent_name
                            : typeof item.agent_id === 'string' && item.agent_id
                              ? item.agent_id
                              : `${text.unknownNode} ${suggestionIndex + 1}`;
                        return (
                          <span
                            key={`weak-edge-${index}-suggestion-${suggestionIndex}`}
                            className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationFitBadgeClass(suggestionFit)}`}
                          >
                            {suggestionName}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                ) : null}
                {onApplyRewire && replacementAgentId ? (
                  <div className="mt-3">
                    <SecondaryActionButton
                      tone="amber"
                      disabled={isApplying}
                      onClick={() =>
                        onApplyRewire({
                          sourceAgentId:
                            typeof edge.source_agent_id === 'string' && edge.source_agent_id
                              ? edge.source_agent_id
                              : '',
                          fromTargetAgentId:
                            typeof target?.agent_id === 'string' && target.agent_id
                              ? target.agent_id
                              : '',
                          toTargetAgentId: replacementAgentId,
                        })
                      }
                    >
                      {isApplying
                        ? text.applyingSuggestedRewire
                        : formatTemplate(text.applySuggestedRewire, { name: replacementAgentName || text.unknownNode })}
                    </SecondaryActionButton>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}

      {bestNextHandoffs.length > 0 ? (
        <div className="space-y-3">
          <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-700">{text.bestNextHandoffsLabel}</div>
          {bestNextHandoffs.map((item, index) => {
            const target = coerceRecord(item.target);
            const fit = normalizeDelegationFit(typeof target?.fit === 'string' ? target.fit : 'weak');
            const sourceAgentId =
              typeof item.source_agent_id === 'string' && item.source_agent_id
                ? item.source_agent_id
                : '';
            const sourceName =
              typeof item.source_agent_name === 'string' && item.source_agent_name
                ? item.source_agent_name
                : sourceAgentId
                  ? sourceAgentId
                  : text.unknownNode;
            const targetAgentId =
              typeof target?.agent_id === 'string' && target.agent_id
                ? target.agent_id
                : '';
            const targetName =
              typeof target?.agent_name === 'string' && target.agent_name
                ? target.agent_name
                : targetAgentId
                  ? targetAgentId
                  : text.unknownNode;
            const rationale = typeof target?.rationale === 'string' ? target.rationale : null;
            return (
              <div key={`best-next-${index}`} className="rounded-2xl border border-cyan-200 bg-cyan-50/60 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="text-sm font-semibold text-slate-900">
                    {sourceName} {'->'} {targetName}
                  </div>
                  <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${delegationFitBadgeClass(fit)}`}>
                    {formatDelegationFitLabel(fit, text)}
                  </span>
                </div>
                <HandoffOpportunityContext diagnostic={item} />
                {rationale ? <div className="mt-2 text-xs leading-5 text-slate-700">{rationale}</div> : null}
                {onApplySuggestedHandoff && sourceAgentId && targetAgentId ? (
                  <div className="mt-3">
                    <SecondaryActionButton
                      tone="cyan"
                      disabled={isApplying}
                      onClick={() =>
                        onApplySuggestedHandoff({
                          sourceAgentId,
                          targetAgentId,
                        })
                      }
                    >
                      {isApplying
                        ? text.applyingSuggestedHandoff
                        : formatTemplate(text.applySuggestedHandoff, { name: targetName })}
                    </SecondaryActionButton>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

export function RunRecoveryGuideCard({
  run,
  blockedAgents,
  availabilityDiagnostics,
  readinessDiagnostics,
  collaborationContractDiagnostics,
  handoffDiagnostics,
  policyRepairSummary,
  roleProfileSummary,
  recoveryMode,
  graphAgentsById,
  canFocusAgent,
  onFocusAgent,
  onOpenSkillPool,
  onOpenProjectProviders,
  onOpenProjectMcpInventory,
  onApplyRewire,
  onApplySuggestedHandoff,
  onApplyAllCollaborationFixes,
  onApplyRoleProfiles,
  onApplyPolicyRepairs,
  onApplyAgentPolicyRepair,
  onRequestRoleProfileSkills,
  onRetry,
  isRetrying,
  isApplyingGraphChange,
  isRequestingSkills,
  StatusPillComponent,
}: {
  run: HarnessRunDetailDTO;
  blockedAgents: Record<string, unknown>[];
  availabilityDiagnostics: Record<string, unknown> | null;
  readinessDiagnostics: Record<string, unknown> | null;
  collaborationContractDiagnostics: Record<string, unknown> | null;
  handoffDiagnostics: Record<string, unknown> | null;
  policyRepairSummary: PolicyRepairScopeSummary;
  roleProfileSummary: RoleProfileScopeSummary;
  recoveryMode: string | null;
  graphAgentsById?: ReadonlyMap<string, HarnessCanvasAgentDTO>;
  canFocusAgent: (agentId: string) => boolean;
  onFocusAgent: (agentId: string) => void;
  onOpenSkillPool: () => void;
  onOpenProjectProviders: () => void;
  onOpenProjectMcpInventory: () => void;
  onApplyRewire: (params: {
    sourceAgentId: string;
    fromTargetAgentId: string;
    toTargetAgentId: string;
  }) => void;
  onApplySuggestedHandoff: (params: {
    sourceAgentId: string;
    targetAgentId: string;
  }) => void;
  onApplyAllCollaborationFixes: () => void;
  onApplyRoleProfiles: () => void;
  onApplyPolicyRepairs: () => void;
  onApplyAgentPolicyRepair?: (params: AgentPolicyRepairParams) => void;
  onRequestRoleProfileSkills: () => void;
  onRetry: () => void;
  isRetrying: boolean;
  isApplyingGraphChange: boolean;
  isRequestingSkills?: boolean;
  StatusPillComponent: ComponentType<{ value?: string | null }>;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const pendingApproval = run.latest_approval?.status === 'pending' ? run.latest_approval : null;
  const rejectedReviewApproval =
    run.latest_approval?.status === 'rejected' && run.latest_approval?.action_type === 'orchestration_review'
      ? run.latest_approval
      : null;
  const reviewApproval = pendingApproval ?? rejectedReviewApproval;
  const reviewPayload = coerceRecord(reviewApproval?.payload_json);
  const reviewStage =
    typeof reviewPayload?.review_stage === 'string' && reviewPayload.review_stage
      ? reviewPayload.review_stage
      : null;
  const reviewAgentId =
    typeof reviewPayload?.agent_id === 'string' && reviewPayload.agent_id
      ? reviewPayload.agent_id
      : null;
  const reviewAgentName =
    typeof reviewPayload?.agent_name === 'string' && reviewPayload.agent_name
      ? reviewPayload.agent_name
      : reviewAgentId;
  const reviewComment =
    reviewApproval && typeof reviewApproval.comment === 'string' && reviewApproval.comment.trim()
      ? reviewApproval.comment.trim()
      : null;

  const blockedSkillAgents = blockedAgents.filter((agent) => {
    const missingSkills = coerceStringList(agent.missing_skills);
    const missingSkillDetails = coerceRecordList(agent.missing_skill_details);
    return missingSkills.length > 0 || missingSkillDetails.length > 0;
  });
  const blockedSkillAgentIds = new Set(
    blockedSkillAgents
      .map((agent) => (typeof agent.agent_id === 'string' ? agent.agent_id : ''))
      .filter(Boolean)
  );

  const availabilityAgents = coerceRecordList(availabilityDiagnostics?.agents);
  const unavailableAgents = availabilityAgents.filter(
    (agent) => resolveCapabilityAvailabilityStatus(agent) === 'unavailable'
  );
  const limitedAvailabilityAgents = availabilityAgents.filter(
    (agent) => resolveCapabilityAvailabilityStatus(agent) === 'limited'
  );
  const primaryAvailabilityAgents = unavailableAgents.length > 0 ? unavailableAgents : limitedAvailabilityAgents;

  const readinessAgents = coerceRecordList(readinessDiagnostics?.agents);
  const blockedReadinessAgents = readinessAgents.filter((agent) => {
    const agentId = typeof agent.agent_id === 'string' ? agent.agent_id : '';
    return resolveCapabilityReadinessStatus(agent) === 'blocked' && !blockedSkillAgentIds.has(agentId);
  });
  const limitedReadinessAgents = readinessAgents.filter(
    (agent) => resolveCapabilityReadinessStatus(agent) === 'limited'
  );
  const primaryReadinessAgents =
    blockedReadinessAgents.length > 0
      ? blockedReadinessAgents
      : blockedSkillAgents.length === 0 && unavailableAgents.length === 0
        ? limitedReadinessAgents
        : [];

  const weakEdges = coerceRecordList(handoffDiagnostics?.weak_downstream_edges);
  const bestNextHandoffs = coerceRecordList(handoffDiagnostics?.best_next_handoffs);
  const collaborationContractRisks = coerceRecordList(collaborationContractDiagnostics?.risks);
  const roleProfileOverlapRisks = coerceRecordList(collaborationContractDiagnostics?.role_profile_overlap_risks);
  const highPriorityContractRisks = collaborationContractRisks.filter(
    (risk) => typeof risk.severity === 'string' && risk.severity === 'high'
  );
  const primaryContractRisks =
    highPriorityContractRisks.length > 0 ? highPriorityContractRisks : collaborationContractRisks.slice(0, 2);
  const hasPolicyRepairs = policyRepairSummary.agentCount > 0;
  const hasRoleProfileRepairs = roleProfileSummary.actionableAgentCount > 0;
  const hasRoleProfileSkillRequests = roleProfileSummary.missingSkillAgentCount > 0;
  const primaryRoleProfileDiagnostics = roleProfileSummary.diagnostics.slice(0, 2);
  const roleProfileDiagnosticsByAgentId = new Map(
    roleProfileSummary.diagnostics.map((diagnostic) => [diagnostic.agentId, diagnostic] as const)
  );
  const policyRepairDiagnosticsByAgentId = new Map(
    policyRepairSummary.diagnostics.map((diagnostic) => [diagnostic.agentId, diagnostic] as const)
  );
  const showRetryAction = Boolean(run.can_retry) && !pendingApproval;
  const hasGuidance =
    Boolean(pendingApproval)
    || Boolean(rejectedReviewApproval)
    || hasRoleProfileRepairs
    || hasRoleProfileSkillRequests
    || blockedSkillAgents.length > 0
    || primaryAvailabilityAgents.length > 0
    || primaryReadinessAgents.length > 0
    || roleProfileOverlapRisks.length > 0
    || primaryContractRisks.length > 0
    || hasPolicyRepairs
    || weakEdges.length > 0
    || bestNextHandoffs.length > 0
    || showRetryAction;

  if (!hasGuidance) {
    return <div className="mt-5 text-sm text-slate-500">{text.noRecoveryGuide}</div>;
  }

  const intro = pendingApproval
    ? text.recoveryGuideApprovalHint
    : hasRoleProfileRepairs
      || hasRoleProfileSkillRequests
      || blockedSkillAgents.length > 0
      || primaryAvailabilityAgents.length > 0
      || primaryReadinessAgents.length > 0
      || hasPolicyRepairs
      ? text.recoveryGuideCapabilityHint
      : roleProfileOverlapRisks.length > 0 || primaryContractRisks.length > 0
        ? text.recoveryGuideContractHint
      : weakEdges.length > 0 || bestNextHandoffs.length > 0
        ? text.recoveryGuideHandoffHint
        : text.recoveryGuideRetryHint;

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="text-sm text-slate-700">{intro}</div>
        <div className="flex flex-wrap gap-2">
          {hasRoleProfileRepairs ? (
            <SecondaryActionButton
              tone="cyan"
              disabled={isApplyingGraphChange}
              onClick={onApplyRoleProfiles}
            >
              {isApplyingGraphChange ? text.applyingRecoveryRoleProfiles : text.applyRecoveryRoleProfiles}
            </SecondaryActionButton>
          ) : null}
          {hasRoleProfileSkillRequests ? (
            <SecondaryActionButton
              tone="rose"
              disabled={isRequestingSkills}
              onClick={onRequestRoleProfileSkills}
            >
              {isRequestingSkills ? text.requestingRecoveryRoleProfileSkills : text.requestRecoveryRoleProfileSkills}
            </SecondaryActionButton>
          ) : null}
          {hasPolicyRepairs ? (
            <SecondaryActionButton
              tone="cyan"
              disabled={isApplyingGraphChange}
              onClick={onApplyPolicyRepairs}
            >
              {isApplyingGraphChange ? text.applyingRecoveryPolicyRepairs : text.applyRecoveryPolicyRepairs}
            </SecondaryActionButton>
          ) : null}
          {weakEdges.length > 0 || bestNextHandoffs.length > 0 ? (
            <SecondaryActionButton
              tone="cyan"
              disabled={isApplyingGraphChange}
              onClick={onApplyAllCollaborationFixes}
            >
              {isApplyingGraphChange ? text.applyingRecoveryCollaborationFixes : text.applyRecoveryCollaborationFixes}
            </SecondaryActionButton>
          ) : null}
        </div>
      </div>

      {primaryRoleProfileDiagnostics.map((diagnostic, index) => {
        const availableSkillIds = coerceStringList(diagnostic.availableSkillIds);
        const missingSkillIds = coerceStringList(diagnostic.missingSkillIds);
        const toolIds = coerceStringList(diagnostic.toolIds);
        const mcpServerIds = coerceStringList(diagnostic.mcpServerIds);
        const denyToolIds = coerceStringList(diagnostic.denyToolIds);
        const denyMcpServerIds = coerceStringList(diagnostic.denyMcpServerIds);
        const hasApplyAction =
          availableSkillIds.length > 0
          || toolIds.length > 0
          || mcpServerIds.length > 0
          || denyToolIds.length > 0
          || denyMcpServerIds.length > 0;
        return (
          <div
            key={`recovery-role-profile-${diagnostic.agentId || index}`}
            className="rounded-2xl border border-cyan-200 bg-cyan-50/75 p-4"
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-cyan-950">{diagnostic.agentName}</div>
                <div className="mt-1 text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">
                  {text.roleProfileRecoveryLabel}
                </div>
                <div className="mt-2 text-sm leading-6 text-cyan-900">{text.roleProfileRecoveryHint}</div>
              </div>
              <span className="rounded-full bg-cyan-100 px-2.5 py-1 text-[10px] font-semibold text-cyan-800 ring-1 ring-cyan-200">
                {text.roleProfileSuggestionLabel}
              </span>
            </div>
            <div className="mt-3 space-y-3">
              {missingSkillIds.length > 0 ? (
                <div>
                  <div className="text-[11px] font-semibold text-slate-500">{text.roleProfileMissingSkillsRecoveryLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {missingSkillIds.map((skillId) => (
                      <span
                        key={`${diagnostic.agentId}-missing-skill-${skillId}`}
                        className="rounded-full bg-rose-50 px-2.5 py-1 text-[10px] font-semibold text-rose-800 ring-1 ring-rose-200"
                      >
                        {formatSkillTitle(skillId, EMPTY_LOOKUP)}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
              {availableSkillIds.length > 0 ? (
                <div>
                  <div className="text-[11px] font-semibold text-slate-500">{text.roleProfileAvailableSkillsLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {availableSkillIds.map((skillId) => (
                      <span
                        key={`${diagnostic.agentId}-available-skill-${skillId}`}
                        className="rounded-full bg-cyan-50 px-2.5 py-1 text-[10px] font-semibold text-cyan-800 ring-1 ring-cyan-200"
                      >
                        {formatSkillTitle(skillId, EMPTY_LOOKUP)}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
              {toolIds.length > 0 ? (
                <div>
                  <div className="text-[11px] font-semibold text-slate-500">{text.roleProfileRecommendedToolsRecoveryLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {toolIds.map((toolId) => (
                      <span
                        key={`${diagnostic.agentId}-tool-${toolId}`}
                        className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200"
                      >
                        {formatSkillTitle(toolId, EMPTY_LOOKUP)}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
              {mcpServerIds.length > 0 ? (
                <div>
                  <div className="text-[11px] font-semibold text-slate-500">{text.roleProfileRecommendedMcpRecoveryLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {mcpServerIds.map((serverId) => (
                      <span
                        key={`${diagnostic.agentId}-mcp-${serverId}`}
                        className="rounded-full bg-violet-50 px-2.5 py-1 text-[10px] font-semibold text-violet-800 ring-1 ring-violet-200"
                      >
                        {formatSkillTitle(serverId, EMPTY_LOOKUP)}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
              {denyToolIds.length > 0 ? (
                <div>
                  <div className="text-[11px] font-semibold text-slate-500">{text.roleProfileRestrictiveToolsRecoveryLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {denyToolIds.map((toolId) => (
                      <span
                        key={`${diagnostic.agentId}-deny-tool-${toolId}`}
                        className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200"
                      >
                        {formatSkillTitle(toolId, EMPTY_LOOKUP)}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
              {denyMcpServerIds.length > 0 ? (
                <div>
                  <div className="text-[11px] font-semibold text-slate-500">{text.roleProfileRestrictiveMcpRecoveryLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {denyMcpServerIds.map((serverId) => (
                      <span
                        key={`${diagnostic.agentId}-deny-mcp-${serverId}`}
                        className="rounded-full bg-sky-50 px-2.5 py-1 text-[10px] font-semibold text-sky-800 ring-1 ring-sky-200"
                      >
                        {formatSkillTitle(serverId, EMPTY_LOOKUP)}
                      </span>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {diagnostic.agentId && canFocusAgent(diagnostic.agentId) ? (
                <SecondaryActionButton tone="cyan" onClick={() => onFocusAgent(diagnostic.agentId)}>
                  {formatTemplate(text.focusNodeForRecovery, { name: diagnostic.agentName })}
                </SecondaryActionButton>
              ) : null}
              {hasApplyAction && onApplyAgentPolicyRepair ? (
                <SecondaryActionButton
                  tone="cyan"
                  disabled={isApplyingGraphChange}
                  onClick={() =>
                    onApplyAgentPolicyRepair({
                      agentId: diagnostic.agentId,
                      skillIds: availableSkillIds,
                      toolIds,
                      mcpServerIds,
                      denyToolIds,
                      denyMcpServerIds,
                      forceAllowToolIds: toolIds.length > 0,
                      forceAllowMcpServerIds: mcpServerIds.length > 0,
                    })
                  }
                >
                  {isApplyingGraphChange ? text.applyingRecoveryRoleProfiles : text.applyRoleProfileAction}
                </SecondaryActionButton>
              ) : null}
              {missingSkillIds.length > 0 ? (
                <SecondaryActionButton tone="rose" onClick={onOpenSkillPool}>
                  {text.openSkillPoolAction}
                </SecondaryActionButton>
              ) : null}
            </div>
          </div>
        );
      })}

      {pendingApproval ? (
        <div className="rounded-2xl border border-amber-200 bg-amber-50/80 p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold text-amber-950">{text.awaitingApproval}</div>
              <div className="mt-2 text-sm leading-6 text-amber-900">
                {pendingApproval.reason || text.legacyApprovalCheckpoint}
              </div>
              {reviewStage === 'cluster_research' ? (
                <div className="mt-2 text-xs text-amber-800">{text.clusterResearchEvidence}</div>
              ) : null}
              {reviewStage === 'agent_output_stream' ? (
                <div className="mt-2 text-xs text-amber-800">{text.liveStreamingOutputGuard}</div>
              ) : null}
            </div>
            <StatusPillComponent value={pendingApproval.status} />
          </div>
          {reviewAgentId && canFocusAgent(reviewAgentId) ? (
            <div className="mt-3">
              <SecondaryActionButton
                tone="amber"
                onClick={() => onFocusAgent(reviewAgentId)}
              >
                {formatTemplate(text.focusNodeForRecovery, { name: reviewAgentName || text.unknownNode })}
              </SecondaryActionButton>
            </div>
          ) : null}
        </div>
      ) : null}

      {blockedSkillAgents.slice(0, 3).map((agent, index) => {
        const agentId = typeof agent.agent_id === 'string' ? agent.agent_id : '';
        const agentName = typeof agent.agent_name === 'string' && agent.agent_name ? agent.agent_name : agentId || text.unknownNode;
        const missingSkillDetails = coerceRecordList(agent.missing_skill_details);
        const detailTitles = missingSkillDetails
          .map((detail) =>
            typeof detail.title === 'string' && detail.title
              ? detail.title
              : typeof detail.skill_id === 'string'
                ? detail.skill_id
                : ''
          )
          .filter(Boolean);
        const missingSkills = coerceStringList(agent.missing_skills);
        const skillPreview = detailTitles.length > 0 ? detailTitles : missingSkills;
        return (
          <div key={`recovery-skill-${agentId || index}`} className="rounded-2xl border border-rose-200 bg-rose-50/75 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-rose-950">{agentName}</div>
                <div className="mt-1 text-xs font-semibold uppercase tracking-[0.16em] text-rose-700">
                  {text.missingSkillRecoveryLabel}
                </div>
                {skillPreview.length > 0 ? (
                  <div className="mt-2 text-sm leading-6 text-rose-900">{skillPreview.join(' · ')}</div>
                ) : (
                  <div className="mt-2 text-sm leading-6 text-rose-900">{text.runSummaryBlocked}</div>
                )}
              </div>
              <StatusPillComponent value="blocked" />
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              <SecondaryActionButton tone="rose" onClick={onOpenSkillPool}>
                {text.openSkillPoolAction}
              </SecondaryActionButton>
              {agentId && canFocusAgent(agentId) ? (
                <SecondaryActionButton tone="rose" onClick={() => onFocusAgent(agentId)}>
                  {formatTemplate(text.focusNodeForRecovery, { name: agentName })}
                </SecondaryActionButton>
              ) : null}
            </div>
          </div>
        );
      })}

      {primaryAvailabilityAgents.slice(0, 3).map((agent, index) => {
        const agentId = typeof agent.agent_id === 'string' ? agent.agent_id : '';
        const agentName = typeof agent.agent_name === 'string' && agent.agent_name ? agent.agent_name : agentId || text.unknownNode;
        const blockers = coerceStringList(agent.blockers);
        const warnings = coerceStringList(agent.warnings);
        const availabilityStatus = resolveCapabilityAvailabilityStatus(agent);
        const skillPoolHint = shouldOpenSkillPoolForDiagnostic(agent);
        const projectMcpHint = shouldOpenProjectMcpForDiagnostic(agent);
        const providerRouteHint = shouldOpenProjectProvidersForDiagnostic(agent);
        return (
          <div key={`recovery-availability-${agentId || index}`} className="rounded-2xl border border-rose-200 bg-rose-50/75 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-rose-950">{agentName}</div>
                <div className="mt-1 text-xs font-semibold uppercase tracking-[0.16em] text-rose-700">
                  {text.definitionBlockersRecoveryLabel}
                </div>
              </div>
              <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityAvailabilityBadgeClass(availabilityStatus)}`}>
                {formatCapabilityAvailabilityLabel(availabilityStatus, text)}
              </span>
            </div>
            <div className="mt-3 space-y-2 text-sm leading-6 text-rose-900">
              {blockers.length > 0 ? <div>{blockers.join(' · ')}</div> : null}
              {warnings.length > 0 ? <div>{warnings.join(' · ')}</div> : null}
            </div>
            {agentId && canFocusAgent(agentId) ? (
              <div className="mt-3 flex flex-wrap gap-2">
                <SecondaryActionButton tone="rose" onClick={() => onFocusAgent(agentId)}>
                  {formatTemplate(text.focusNodeForRecovery, { name: agentName })}
                </SecondaryActionButton>
                {skillPoolHint ? (
                  <SecondaryActionButton tone="rose" onClick={onOpenSkillPool}>
                    {text.openSkillPoolAction}
                  </SecondaryActionButton>
                ) : null}
                {providerRouteHint ? (
                  <SecondaryActionButton tone="rose" onClick={onOpenProjectProviders}>
                    {text.openProjectProvidersAction}
                  </SecondaryActionButton>
                ) : null}
                {projectMcpHint ? (
                  <SecondaryActionButton tone="rose" onClick={onOpenProjectMcpInventory}>
                    {text.openProjectMcpAction}
                  </SecondaryActionButton>
                ) : null}
                <DiagnosticPolicyRepairButtons
                  graphAgent={graphAgentsById?.get(agentId) ?? null}
                  diagnosticAgent={agent}
                  onApplyAgentPolicyRepair={onApplyAgentPolicyRepair}
                  isApplying={isApplyingGraphChange}
                  tone="rose"
                />
              </div>
            ) : skillPoolHint || providerRouteHint || projectMcpHint || onApplyAgentPolicyRepair ? (
              <div className="mt-3 flex flex-wrap gap-2">
                {skillPoolHint ? (
                  <SecondaryActionButton tone="rose" onClick={onOpenSkillPool}>
                    {text.openSkillPoolAction}
                  </SecondaryActionButton>
                ) : null}
                {providerRouteHint ? (
                  <SecondaryActionButton tone="rose" onClick={onOpenProjectProviders}>
                    {text.openProjectProvidersAction}
                  </SecondaryActionButton>
                ) : null}
                {projectMcpHint ? (
                  <SecondaryActionButton tone="rose" onClick={onOpenProjectMcpInventory}>
                    {text.openProjectMcpAction}
                  </SecondaryActionButton>
                ) : null}
                <DiagnosticPolicyRepairButtons
                  graphAgent={graphAgentsById?.get(agentId) ?? null}
                  diagnosticAgent={agent}
                  onApplyAgentPolicyRepair={onApplyAgentPolicyRepair}
                  isApplying={isApplyingGraphChange}
                  tone="rose"
                />
              </div>
            ) : null}
          </div>
        );
      })}

      {primaryReadinessAgents.slice(0, 3).map((agent, index) => {
        const agentId = typeof agent.agent_id === 'string' ? agent.agent_id : '';
        const agentName = typeof agent.agent_name === 'string' && agent.agent_name ? agent.agent_name : agentId || text.unknownNode;
        const blockers = coerceStringList(agent.blockers);
        const warnings = coerceStringList(agent.warnings);
        const readinessStatus = resolveCapabilityReadinessStatus(agent);
        const skillPoolHint = shouldOpenSkillPoolForDiagnostic(agent);
        const projectMcpHint = shouldOpenProjectMcpForDiagnostic(agent);
        const providerRouteHint = shouldOpenProjectProvidersForDiagnostic(agent);
        return (
          <div key={`recovery-readiness-${agentId || index}`} className="rounded-2xl border border-amber-200 bg-amber-50/80 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-amber-950">{agentName}</div>
                <div className="mt-1 text-xs font-semibold uppercase tracking-[0.16em] text-amber-700">
                  {text.runtimeBlockersRecoveryLabel}
                </div>
              </div>
              <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityReadinessBadgeClass(readinessStatus)}`}>
                {formatCapabilityReadinessLabel(readinessStatus, text)}
              </span>
            </div>
            <div className="mt-3 space-y-2 text-sm leading-6 text-amber-900">
              {blockers.length > 0 ? <div>{blockers.join(' · ')}</div> : null}
              {warnings.length > 0 ? <div>{warnings.join(' · ')}</div> : null}
            </div>
            {agentId && canFocusAgent(agentId) ? (
              <div className="mt-3 flex flex-wrap gap-2">
                <SecondaryActionButton tone="amber" onClick={() => onFocusAgent(agentId)}>
                  {formatTemplate(text.focusNodeForRecovery, { name: agentName })}
                </SecondaryActionButton>
                {skillPoolHint ? (
                  <SecondaryActionButton tone="amber" onClick={onOpenSkillPool}>
                    {text.openSkillPoolAction}
                  </SecondaryActionButton>
                ) : null}
                {providerRouteHint ? (
                  <SecondaryActionButton tone="amber" onClick={onOpenProjectProviders}>
                    {text.openProjectProvidersAction}
                  </SecondaryActionButton>
                ) : null}
                {projectMcpHint ? (
                  <SecondaryActionButton tone="amber" onClick={onOpenProjectMcpInventory}>
                    {text.openProjectMcpAction}
                  </SecondaryActionButton>
                ) : null}
                <DiagnosticPolicyRepairButtons
                  graphAgent={graphAgentsById?.get(agentId) ?? null}
                  diagnosticAgent={agent}
                  onApplyAgentPolicyRepair={onApplyAgentPolicyRepair}
                  isApplying={isApplyingGraphChange}
                  tone="amber"
                />
              </div>
            ) : skillPoolHint || providerRouteHint || projectMcpHint || onApplyAgentPolicyRepair ? (
              <div className="mt-3 flex flex-wrap gap-2">
                {skillPoolHint ? (
                  <SecondaryActionButton tone="amber" onClick={onOpenSkillPool}>
                    {text.openSkillPoolAction}
                  </SecondaryActionButton>
                ) : null}
                {providerRouteHint ? (
                  <SecondaryActionButton tone="amber" onClick={onOpenProjectProviders}>
                    {text.openProjectProvidersAction}
                  </SecondaryActionButton>
                ) : null}
                {projectMcpHint ? (
                  <SecondaryActionButton tone="amber" onClick={onOpenProjectMcpInventory}>
                    {text.openProjectMcpAction}
                  </SecondaryActionButton>
                ) : null}
                <DiagnosticPolicyRepairButtons
                  graphAgent={graphAgentsById?.get(agentId) ?? null}
                  diagnosticAgent={agent}
                  onApplyAgentPolicyRepair={onApplyAgentPolicyRepair}
                  isApplying={isApplyingGraphChange}
                  tone="amber"
                />
              </div>
            ) : null}
          </div>
        );
      })}

      {roleProfileOverlapRisks.slice(0, 2).map((risk, index) => {
        const profileId =
          typeof risk.profile_id === 'string' && risk.profile_id ? risk.profile_id : text.roleProfileGeneralistTitle;
        const leftAgent = coerceRecord(risk.left_agent_preview) ?? coerceRecordList(risk.agent_previews)[0] ?? null;
        const rightAgent = coerceRecord(risk.right_agent_preview) ?? coerceRecordList(risk.agent_previews)[1] ?? null;
        const leftAgentId =
          typeof leftAgent?.agent_id === 'string' && leftAgent.agent_id ? leftAgent.agent_id : '';
        const leftAgentName =
          typeof leftAgent?.agent_name === 'string' && leftAgent.agent_name
            ? leftAgent.agent_name
            : leftAgentId || text.unknownNode;
        const rightAgentId =
          typeof rightAgent?.agent_id === 'string' && rightAgent.agent_id ? rightAgent.agent_id : '';
        const rightAgentName =
          typeof rightAgent?.agent_name === 'string' && rightAgent.agent_name
            ? rightAgent.agent_name
            : rightAgentId || text.unknownNode;
        const sharedLaneIds = coerceStringList(risk.shared_lane_ids);
        const leftFocusLaneIds = coerceStringList(risk.left_focus_lane_ids);
        const rightFocusLaneIds = coerceStringList(risk.right_focus_lane_ids);
        const leftUniqueSkillIds = coerceStringList(risk.left_unique_skill_ids);
        const rightUniqueSkillIds = coerceStringList(risk.right_unique_skill_ids);
        const leftUniqueToolIds = coerceStringList(risk.left_unique_tool_ids);
        const rightUniqueToolIds = coerceStringList(risk.right_unique_tool_ids);
        const leftUniqueMcpServerIds = coerceStringList(risk.left_unique_mcp_server_ids);
        const rightUniqueMcpServerIds = coerceStringList(risk.right_unique_mcp_server_ids);
        const hasDifferentiationSignal =
          leftFocusLaneIds.length > 0
          || rightFocusLaneIds.length > 0
          || leftUniqueSkillIds.length > 0
          || rightUniqueSkillIds.length > 0
          || leftUniqueToolIds.length > 0
          || rightUniqueToolIds.length > 0
          || leftUniqueMcpServerIds.length > 0
          || rightUniqueMcpServerIds.length > 0;
        return (
          <div
            key={`recovery-role-profile-overlap-${leftAgentId || rightAgentId || index}`}
            className="rounded-2xl border border-violet-200 bg-violet-50/75 p-4"
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-violet-950">
                  {leftAgentName} {'<->'} {rightAgentName}
                </div>
                <div className="mt-1 text-xs font-semibold uppercase tracking-[0.16em] text-violet-700">
                  {text.roleProfileOverlapRecoveryLabel}
                </div>
                <div className="mt-2 text-sm leading-6 text-violet-900">
                  {formatTemplate(text.roleProfileOverlapRecoveryHint, {
                    profile: formatSkillTitle(profileId, EMPTY_LOOKUP),
                  })}
                </div>
              </div>
              <span className="rounded-full bg-violet-100 px-2.5 py-1 text-[10px] font-semibold text-violet-800 ring-1 ring-violet-200">
                {formatSkillTitle(profileId, EMPTY_LOOKUP)}
              </span>
            </div>
            <div className="mt-3">
              <div className="text-[11px] font-semibold text-slate-500">{text.overlapLanesLabel}</div>
              <CapabilityPills
                values={sharedLaneIds}
                tone="violet"
                emptyLabel={text.none}
                keyPrefix={`role-overlap-shared-${index}`}
              />
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-2">
              {[
                {
                  agentId: leftAgentId,
                  agentName: leftAgentName,
                  focusLaneIds: leftFocusLaneIds,
                  uniqueSkillIds: leftUniqueSkillIds,
                  uniqueToolIds: leftUniqueToolIds,
                  uniqueMcpServerIds: leftUniqueMcpServerIds,
                  side: 'left',
                },
                {
                  agentId: rightAgentId,
                  agentName: rightAgentName,
                  focusLaneIds: rightFocusLaneIds,
                  uniqueSkillIds: rightUniqueSkillIds,
                  uniqueToolIds: rightUniqueToolIds,
                  uniqueMcpServerIds: rightUniqueMcpServerIds,
                  side: 'right',
                },
              ].map((item) => (
                <div
                  key={`role-overlap-agent-${item.side}-${item.agentId || item.agentName}`}
                  className="rounded-xl border border-violet-200 bg-white/80 px-3 py-3"
                >
                  <div className="text-sm font-semibold text-slate-900">{item.agentName}</div>
                  <div className="mt-2">
                    <div className="text-[11px] font-semibold text-slate-500">
                      {text.roleProfileOverlapSuggestedLanesLabel}
                    </div>
                    <CapabilityPills
                      values={item.focusLaneIds}
                      tone="cyan"
                      emptyLabel={text.roleProfileOverlapNeedsDifferentiation}
                      keyPrefix={`role-overlap-focus-${item.side}-${index}`}
                    />
                  </div>
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold text-slate-500">
                      {text.roleProfileOverlapDistinctSkillsLabel}
                    </div>
                    <CapabilityPills
                      values={item.uniqueSkillIds}
                      tone="cyan"
                      emptyLabel={text.none}
                      keyPrefix={`role-overlap-skills-${item.side}-${index}`}
                    />
                  </div>
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold text-slate-500">
                      {text.roleProfileOverlapDistinctToolsLabel}
                    </div>
                    <CapabilityPills
                      values={item.uniqueToolIds}
                      tone="emerald"
                      emptyLabel={text.none}
                      keyPrefix={`role-overlap-tools-${item.side}-${index}`}
                    />
                  </div>
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold text-slate-500">
                      {text.roleProfileOverlapDistinctMcpLabel}
                    </div>
                    <CapabilityPills
                      values={item.uniqueMcpServerIds}
                      tone="violet"
                      emptyLabel={text.none}
                      keyPrefix={`role-overlap-mcp-${item.side}-${index}`}
                    />
                  </div>
                  {item.agentId && canFocusAgent(item.agentId) ? (
                    <div className="mt-3">
                      <SecondaryActionButton tone="violet" onClick={() => onFocusAgent(item.agentId)}>
                        {formatTemplate(text.focusNodeForRecovery, { name: item.agentName })}
                      </SecondaryActionButton>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
            {!hasDifferentiationSignal ? (
              <div className="mt-3 text-sm leading-6 text-violet-900">
                {text.roleProfileOverlapNeedsDifferentiation}
              </div>
            ) : null}
          </div>
        );
      })}

      {primaryContractRisks.map((risk, index) => {
        const severity = typeof risk.severity === 'string' ? risk.severity : 'low';
        const summary =
          typeof risk.summary === 'string' && risk.summary
            ? risk.summary
            : text.noCollaborationContractRisks;
        const recommendedAction =
          typeof risk.recommended_action === 'string' && risk.recommended_action
            ? risk.recommended_action
            : null;
        const agentPreviews = coerceRecordList(risk.agent_previews);
        const sourceAgentId =
          typeof risk.source_agent_id === 'string' && risk.source_agent_id ? risk.source_agent_id : '';
        const delegateToolIds = coerceStringList(risk.delegate_tool_ids);
        const delegateMcpServerIds = coerceStringList(risk.delegate_mcp_server_ids);
        const delegateCandidates = coerceRecordList(risk.delegate_candidates);
        return (
          <div key={`recovery-contract-risk-${index}`} className="rounded-2xl border border-sky-200 bg-sky-50/75 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-sky-950">{summary}</div>
                <div className="mt-1 text-xs font-semibold uppercase tracking-[0.16em] text-sky-700">
                  {text.collaborationContractsEvidence}
                </div>
              </div>
              <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${collaborationRiskBadgeClass(severity)}`}>
                {severity === 'high' ? text.priorityHighLabel : severity === 'medium' ? text.priorityMediumLabel : text.priorityLowLabel}
              </span>
            </div>
            {recommendedAction ? (
              <div className="mt-2 text-sm leading-6 text-sky-900">
                <span className="font-semibold text-sky-950">{text.collaborationContractRecommendedActionLabel}:</span>{' '}
                {recommendedAction}
              </div>
            ) : null}
            {agentPreviews.length > 0 ? (
              <div className="mt-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                  {text.collaborationContractRisksLabel}
                </div>
                <div className="mt-2">
                  <CollaborationPreviewButtons
                    previews={agentPreviews}
                    canFocusAgent={canFocusAgent}
                    onFocusAgent={onFocusAgent}
                  />
                </div>
              </div>
            ) : null}
            {delegateToolIds.length > 0 ? (
              <div className="mt-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                  {text.collaborationDelegateToolsLabel}
                </div>
                <CapabilityPills
                  values={delegateToolIds}
                  tone="emerald"
                  emptyLabel={text.none}
                  keyPrefix={`recovery-contract-tools-${index}`}
                />
              </div>
            ) : null}
            {delegateMcpServerIds.length > 0 ? (
              <div className="mt-3">
                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                  {text.collaborationDelegateMcpLabel}
                </div>
                <CapabilityPills
                  values={delegateMcpServerIds}
                  tone="violet"
                  emptyLabel={text.none}
                  keyPrefix={`recovery-contract-mcp-${index}`}
                />
              </div>
            ) : null}
            <DelegateCandidateCards
              sourceAgentId={sourceAgentId}
              candidates={delegateCandidates}
              keyPrefix={`recovery-contract-risk-${index}`}
              canFocusAgent={canFocusAgent}
              onFocusAgent={onFocusAgent}
              onApplySuggestedHandoff={onApplySuggestedHandoff}
              isApplyingGraphChange={isApplyingGraphChange}
              roleProfileDiagnosticsByAgentId={roleProfileDiagnosticsByAgentId}
              policyRepairDiagnosticsByAgentId={policyRepairDiagnosticsByAgentId}
              onApplyAgentPolicyRepair={onApplyAgentPolicyRepair}
              onOpenSkillPool={onOpenSkillPool}
            />
          </div>
        );
      })}

      {weakEdges.slice(0, 2).map((edge, index) => {
        const target = coerceRecord(edge.target);
        const suggestedReplacements = coerceRecordList(edge.suggested_replacements);
        const primaryReplacement = suggestedReplacements[0] ?? null;
        const sourceAgentId =
          typeof edge.source_agent_id === 'string' && edge.source_agent_id
            ? edge.source_agent_id
            : '';
        const sourceName =
          typeof edge.source_agent_name === 'string' && edge.source_agent_name
            ? edge.source_agent_name
            : sourceAgentId || text.unknownNode;
        const targetAgentId =
          typeof target?.agent_id === 'string' && target.agent_id
            ? target.agent_id
            : '';
        const targetName =
          typeof target?.agent_name === 'string' && target.agent_name
            ? target.agent_name
            : targetAgentId || text.unknownNode;
        const replacementAgentId =
          typeof primaryReplacement?.agent_id === 'string' && primaryReplacement.agent_id
            ? primaryReplacement.agent_id
            : '';
        const replacementAgentName =
          typeof primaryReplacement?.agent_name === 'string' && primaryReplacement.agent_name
            ? primaryReplacement.agent_name
            : replacementAgentId || text.unknownNode;
        const rationale = typeof target?.rationale === 'string' ? target.rationale : null;
        return (
          <div key={`recovery-weak-edge-${index}`} className="rounded-2xl border border-cyan-200 bg-cyan-50/75 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-cyan-950">
                  {sourceName} {'->'} {targetName}
                </div>
                <div className="mt-1 text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">
                  {text.collaborationRecoveryLabel}
                </div>
                {rationale ? <div className="mt-2 text-sm leading-6 text-cyan-900">{rationale}</div> : null}
              </div>
              <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
                {text.weakHandoffEdgesLabel}
              </span>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {sourceAgentId && canFocusAgent(sourceAgentId) ? (
                <SecondaryActionButton tone="cyan" onClick={() => onFocusAgent(sourceAgentId)}>
                  {formatTemplate(text.focusNodeForRecovery, { name: sourceName })}
                </SecondaryActionButton>
              ) : null}
              {sourceAgentId && targetAgentId && replacementAgentId ? (
                <SecondaryActionButton
                  tone="cyan"
                  disabled={isApplyingGraphChange}
                  onClick={() =>
                    onApplyRewire({
                      sourceAgentId,
                      fromTargetAgentId: targetAgentId,
                      toTargetAgentId: replacementAgentId,
                    })
                  }
                >
                  {isApplyingGraphChange
                    ? text.applyingSuggestedRewire
                    : formatTemplate(text.applySuggestedRewire, { name: replacementAgentName })}
                </SecondaryActionButton>
              ) : null}
            </div>
          </div>
        );
      })}

      {bestNextHandoffs.slice(0, 2).map((item, index) => {
        const target = coerceRecord(item.target);
        const sourceAgentId =
          typeof item.source_agent_id === 'string' && item.source_agent_id
            ? item.source_agent_id
            : '';
        const sourceName =
          typeof item.source_agent_name === 'string' && item.source_agent_name
            ? item.source_agent_name
            : sourceAgentId || text.unknownNode;
        const targetAgentId =
          typeof target?.agent_id === 'string' && target.agent_id
            ? target.agent_id
            : '';
        const targetName =
          typeof target?.agent_name === 'string' && target.agent_name
            ? target.agent_name
            : targetAgentId || text.unknownNode;
        const rationale = typeof target?.rationale === 'string' ? target.rationale : null;
        return (
          <div key={`recovery-best-next-${index}`} className="rounded-2xl border border-cyan-200 bg-cyan-50/75 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-cyan-950">
                  {sourceName} {'->'} {targetName}
                </div>
                <div className="mt-1 text-xs font-semibold uppercase tracking-[0.16em] text-cyan-700">
                  {text.bestNextHandoffsLabel}
                </div>
                {rationale ? <div className="mt-2 text-sm leading-6 text-cyan-900">{rationale}</div> : null}
              </div>
              <span className="rounded-full bg-cyan-100 px-2.5 py-1 text-[10px] font-semibold text-cyan-800 ring-1 ring-cyan-200">
                {text.bestNextHandoffsLabel}
              </span>
            </div>
            <div className="mt-3 flex flex-wrap gap-2">
              {sourceAgentId && canFocusAgent(sourceAgentId) ? (
                <SecondaryActionButton tone="cyan" onClick={() => onFocusAgent(sourceAgentId)}>
                  {formatTemplate(text.focusNodeForRecovery, { name: sourceName })}
                </SecondaryActionButton>
              ) : null}
              {sourceAgentId && targetAgentId ? (
                <SecondaryActionButton
                  tone="cyan"
                  disabled={isApplyingGraphChange}
                  onClick={() =>
                    onApplySuggestedHandoff({
                      sourceAgentId,
                      targetAgentId,
                    })
                  }
                >
                  {isApplyingGraphChange
                    ? text.applyingSuggestedHandoff
                    : formatTemplate(text.applySuggestedHandoff, { name: targetName })}
                </SecondaryActionButton>
              ) : null}
            </div>
          </div>
        );
      })}

      {rejectedReviewApproval ? (
        <div className="rounded-2xl border border-sky-200 bg-sky-50/80 p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold text-sky-950">{text.retryRun}</div>
              <div className="mt-2 text-sm leading-6 text-sky-900">
                {recoveryMode === 'continue_without_research' ? text.nextRunWithoutResearchHint : text.nextRunFromRollbackHint}
              </div>
              {reviewComment ? (
                <div className="mt-3 rounded-2xl border border-sky-200 bg-white/80 p-3 text-sm text-sky-950">
                  <div className="text-xs font-semibold uppercase tracking-[0.16em] text-sky-700">{text.reviewerComment}</div>
                  <div className="mt-2">{reviewComment}</div>
                </div>
              ) : null}
            </div>
            <StatusPillComponent value={rejectedReviewApproval.status} />
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {reviewAgentId && canFocusAgent(reviewAgentId) ? (
              <SecondaryActionButton tone="sky" onClick={() => onFocusAgent(reviewAgentId)}>
                {formatTemplate(text.focusNodeForRecovery, { name: reviewAgentName || text.unknownNode })}
              </SecondaryActionButton>
            ) : null}
            {showRetryAction ? (
              <button
                type="button"
                onClick={onRetry}
                disabled={isRetrying}
                className="inline-flex items-center justify-center rounded-xl bg-sky-600 px-3 py-2 text-xs font-semibold text-white hover:bg-sky-500 disabled:opacity-50"
              >
                {isRetrying
                  ? text.starting
                  : reviewStage === 'cluster_research'
                    ? text.continueWithoutResearch
                    : text.continueFromRollback}
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      {!rejectedReviewApproval && showRetryAction ? (
        <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
          <div className="text-sm font-semibold text-slate-950">{text.retryRun}</div>
          <div className="mt-2 text-sm leading-6 text-slate-700">{text.recoveryGuideRetryHint}</div>
          <div className="mt-3">
            <button
              type="button"
              onClick={onRetry}
              disabled={isRetrying}
              className="inline-flex items-center justify-center rounded-xl bg-sky-600 px-3 py-2 text-xs font-semibold text-white hover:bg-sky-500 disabled:opacity-50"
            >
              {isRetrying ? text.starting : text.retryRun}
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function AvailabilityPreflightCard({
  summary,
  emptyLabel,
}: {
  summary: AvailabilityScopeSummary;
  emptyLabel: string;
}) {
  const text = useMessages(HARNESS_MESSAGES);

  if (summary.totalCount === 0) {
    return <div className="mt-5 text-sm text-slate-500">{emptyLabel}</div>;
  }

  const hint =
    summary.unavailableCount > 0
      ? text.preflightAvailabilityBlockedHint
      : summary.limitedCount > 0
        ? text.preflightAvailabilityLimitedHint
        : text.preflightAvailabilityReadyHint;

  return (
    <div className="mt-5 space-y-4">
      <div className="text-sm text-slate-700">{hint}</div>
      <div className="flex flex-wrap gap-2">
        <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
          {formatTemplate(text.availabilityAvailableCountLabel, { count: summary.availableCount })}
        </span>
        <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
          {formatTemplate(text.availabilityLimitedCountLabel, { count: summary.limitedCount })}
        </span>
        <span className="rounded-full bg-rose-50 px-2.5 py-1 text-[10px] font-semibold text-rose-800 ring-1 ring-rose-200">
          {formatTemplate(text.availabilityUnavailableCountLabel, { count: summary.unavailableCount })}
        </span>
      </div>
      {summary.flaggedAgents.length > 0 ? (
        <div className="space-y-3">
          {summary.flaggedAgents.map((agent) => (
            <div key={agent.agentId} className="rounded-2xl border border-slate-200 bg-white/90 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="text-sm font-semibold text-slate-900">{agent.agentName}</div>
                <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityAvailabilityBadgeClass(agent.status)}`}>
                  {formatCapabilityAvailabilityLabel(agent.status, text)}
                </span>
              </div>
              {agent.blockers.length > 0 || agent.warnings.length > 0 ? (
                <div className="mt-2 space-y-2 text-xs leading-5 text-slate-600">
                  {agent.blockers.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.availabilityBlockersLabel}:</span>{' '}
                      {agent.blockers.join(' · ')}
                    </div>
                  ) : null}
                  {agent.warnings.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.availabilityWarningsLabel}:</span>{' '}
                      {agent.warnings.join(' · ')}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-sm text-slate-500">{text.noAvailabilityPreflightIssues}</div>
      )}
    </div>
  );
}

export function RunPreflightSummaryCard({
  metadata,
}: {
  metadata: Record<string, unknown> | null;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const graphSummary = (() => {
    const summary = buildRunPreflightScopeCounts(metadata, 'graph');
    return summary;
  })();
  const scopeSummary = (() => {
    const summary = buildRunPreflightScopeCounts(metadata, 'scope');
    return summary;
  })();
  const scopeMode =
    typeof metadata?.handoff_diagnostic_scope === 'string' && metadata.handoff_diagnostic_scope
      ? metadata.handoff_diagnostic_scope
      : 'all_agents';

  if (!graphSummary && !scopeSummary) {
    return <div className="text-sm text-slate-500">{text.noRunPreflightSummary}</div>;
  }

  const cards = [
    graphSummary
      ? {
          key: 'graph',
          heading: text.savedGraphScopeLabel,
          subheading: text.runAllScopeLabel,
          summary: graphSummary,
        }
      : null,
    scopeSummary
      ? {
          key: 'scope',
          heading: text.runScopeSummaryLabel,
          subheading: scopeMode === 'selected_agents' ? text.runSelectedScopeLabel : text.runAllScopeLabel,
          summary: scopeSummary,
        }
      : null,
  ].filter(
    (item): item is {
      key: string;
      heading: string;
      subheading: string;
      summary: RunPreflightScopeCounts;
    } => item !== null
  );

  return (
    <div className="space-y-4">
      <div className="text-sm text-slate-700">{text.runPreflightSummary}</div>
      <div className={`grid gap-4 ${cards.length > 1 ? 'xl:grid-cols-2' : ''}`}>
        {cards.map((card) => {
          const tone = availabilityScopeToneClasses({
            limitedCount: card.summary.limitedAvailabilityCount,
            unavailableCount: card.summary.unavailableCount,
          });
          return (
            <div key={card.key} className={`rounded-2xl border p-4 ${tone.panel}`}>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className={`text-xs font-semibold uppercase tracking-[0.18em] ${tone.accent}`}>{card.heading}</div>
                  <div className="mt-1 text-sm font-semibold text-slate-900">{card.subheading}</div>
                </div>
                <div className="text-xs text-slate-600">
                  {formatTemplate(text.agentCountShort, { count: card.summary.totalCount })}
                </div>
              </div>
              <div className="mt-4 flex flex-wrap gap-2">
                <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
                  {formatTemplate(text.weakEdgesCountLabel, { count: card.summary.weakEdgeCount })}
                </span>
                <span className="rounded-full bg-cyan-50 px-2.5 py-1 text-[10px] font-semibold text-cyan-800 ring-1 ring-cyan-200">
                  {formatTemplate(text.bestNextCountLabel, { count: card.summary.bestNextCount })}
                </span>
              </div>
              <div className="mt-4 space-y-3">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">{text.readinessLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
                      {formatTemplate(text.readinessReadyCountLabel, { count: card.summary.readyCount })}
                    </span>
                    <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
                      {formatTemplate(text.readinessLimitedCountLabel, { count: card.summary.limitedReadinessCount })}
                    </span>
                    <span className="rounded-full bg-rose-50 px-2.5 py-1 text-[10px] font-semibold text-rose-800 ring-1 ring-rose-200">
                      {formatTemplate(text.readinessBlockedCountLabel, { count: card.summary.blockedCount })}
                    </span>
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">{text.availabilityLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
                      {formatTemplate(text.availabilityAvailableCountLabel, { count: card.summary.availableCount })}
                    </span>
                    <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
                      {formatTemplate(text.availabilityLimitedCountLabel, { count: card.summary.limitedAvailabilityCount })}
                    </span>
                    <span className="rounded-full bg-rose-50 px-2.5 py-1 text-[10px] font-semibold text-rose-800 ring-1 ring-rose-200">
                      {formatTemplate(text.availabilityUnavailableCountLabel, { count: card.summary.unavailableCount })}
                    </span>
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">{text.executionContractLabel}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
                      {formatTemplate(text.directExecutionAgentCountLabel, {
                        count: card.summary.directExecutionAgentCount,
                      })}
                    </span>
                    <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
                      {formatTemplate(text.planningOnlyToolContractsCountLabel, {
                        count: card.summary.planningOnlyToolAgentCount,
                      })}
                    </span>
                    <span className="rounded-full bg-violet-50 px-2.5 py-1 text-[10px] font-semibold text-violet-800 ring-1 ring-violet-200">
                      {formatTemplate(text.planningOnlyMcpContractsCountLabel, {
                        count: card.summary.planningOnlyMcpAgentCount,
                      })}
                    </span>
                  </div>
                </div>
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">{text.collaborationContractsEvidence}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <span className="rounded-full bg-cyan-50 px-2.5 py-1 text-[10px] font-semibold text-cyan-800 ring-1 ring-cyan-200">
                      {formatTemplate(text.coordinatorAgentCountLabel, {
                        count: card.summary.coordinatorAgentCount,
                      })}
                    </span>
                    <span className="rounded-full bg-sky-50 px-2.5 py-1 text-[10px] font-semibold text-sky-800 ring-1 ring-sky-200">
                      {formatTemplate(text.parallelCoordinatorAgentCountLabel, {
                        count: card.summary.parallelCoordinatorAgentCount,
                      })}
                    </span>
                    <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
                      {formatTemplate(text.finalOutputAgentCountLabel, {
                        count: card.summary.finalOutputAgentCount,
                      })}
                    </span>
                    <span className="rounded-full bg-violet-50 px-2.5 py-1 text-[10px] font-semibold text-violet-800 ring-1 ring-violet-200">
                      {formatTemplate(text.verificationAgentCountLabel, {
                        count: card.summary.verificationAgentCount,
                      })}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

type EvidenceCardSharedProps = {
  diagnostics: Record<string, unknown> | null;
  graphAgentsById?: ReadonlyMap<string, HarnessCanvasAgentDTO>;
  policyRepairSummary?: PolicyRepairScopeSummary;
  roleProfileSummary?: RoleProfileScopeSummary;
  canFocusAgent?: (agentId: string) => boolean;
  onFocusAgent?: (agentId: string) => void;
  onOpenSkillPool?: () => void;
  onOpenProjectProviders?: () => void;
  onOpenProjectMcpInventory?: () => void;
  onApplyAgentPolicyRepair?: (params: AgentPolicyRepairParams) => void;
  isApplyingPolicyRepair?: boolean;
  onApplySuggestedHandoff?: (params: {
    sourceAgentId: string;
    targetAgentId: string;
  }) => void;
  isApplyingGraphChange?: boolean;
};

export function CapabilityAvailabilityEvidenceCard({
  diagnostics,
  graphAgentsById,
  canFocusAgent,
  onFocusAgent,
  onOpenSkillPool,
  onOpenProjectProviders,
  onOpenProjectMcpInventory,
  onApplyAgentPolicyRepair,
  isApplyingPolicyRepair,
}: EvidenceCardSharedProps) {
  const text = useMessages(HARNESS_MESSAGES);
  const agents = coerceRecordList(diagnostics?.agents);
  const unavailableCount =
    typeof diagnostics?.unavailable_count === 'number'
      ? diagnostics.unavailable_count
      : agents.filter((agent) => resolveCapabilityAvailabilityStatus(agent) === 'unavailable').length;
  const limitedCount =
    typeof diagnostics?.limited_count === 'number'
      ? diagnostics.limited_count
      : agents.filter((agent) => resolveCapabilityAvailabilityStatus(agent) === 'limited').length;
  const agentCount = typeof diagnostics?.agent_count === 'number' ? diagnostics.agent_count : agents.length;

  if (agents.length === 0) {
    return <div className="mt-5 text-sm text-slate-500">{text.noCapabilityAvailabilityEvidence}</div>;
  }

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap gap-2">
        <span className="rounded-full bg-rose-50 px-2.5 py-1 text-[10px] font-semibold text-rose-800 ring-1 ring-rose-200">
          {formatTemplate(text.availabilityUnavailableCountLabel, { count: unavailableCount })}
        </span>
        <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
          {formatTemplate(text.availabilityLimitedCountLabel, { count: limitedCount })}
        </span>
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
          {formatTemplate(text.availabilityFlaggedAgentCountLabel, { count: agentCount })}
        </span>
      </div>
      <div className="space-y-3">
        {agents.map((agent, index) => {
          const agentId =
            typeof agent.agent_id === 'string' && agent.agent_id
              ? agent.agent_id
              : `availability-agent-${index + 1}`;
          const agentName =
            typeof agent.agent_name === 'string' && agent.agent_name
              ? agent.agent_name
              : agentId;
          const availabilityStatus = resolveCapabilityAvailabilityStatus(agent);
          const blockers = coerceStringList(agent.blockers);
          const warnings = coerceStringList(agent.warnings);
          const skillPoolHint = shouldOpenSkillPoolForDiagnostic(agent);
          const projectMcpHint = shouldOpenProjectMcpForDiagnostic(agent);
          const providerRouteHint = shouldOpenProjectProvidersForDiagnostic(agent);
          return (
            <div key={`${agentId}-${index}`} className="rounded-2xl border border-slate-200 bg-white/90 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="text-sm font-semibold text-slate-950">{agentName}</div>
                <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityAvailabilityBadgeClass(availabilityStatus)}`}>
                  {formatCapabilityAvailabilityLabel(availabilityStatus, text)}
                </span>
              </div>
              {blockers.length > 0 || warnings.length > 0 ? (
                <div className="mt-3 space-y-2 text-xs leading-5 text-slate-600">
                  {blockers.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.availabilityBlockersLabel}:</span>{' '}
                      {blockers.join(' · ')}
                    </div>
                  ) : null}
                  {warnings.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.availabilityWarningsLabel}:</span>{' '}
                      {warnings.join(' · ')}
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="mt-3 text-xs text-slate-500">{text.noAvailabilityIssues}</div>
              )}
              {canFocusAgent || onOpenSkillPool || onOpenProjectProviders || onOpenProjectMcpInventory || onApplyAgentPolicyRepair ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {agentId && canFocusAgent?.(agentId) && onFocusAgent ? (
                    <SecondaryActionButton tone="slate" onClick={() => onFocusAgent(agentId)}>
                      {formatTemplate(text.focusNodeForRecovery, { name: agentName })}
                    </SecondaryActionButton>
                  ) : null}
                  {skillPoolHint && onOpenSkillPool ? (
                    <SecondaryActionButton tone="slate" onClick={onOpenSkillPool}>
                      {text.openSkillPoolAction}
                    </SecondaryActionButton>
                  ) : null}
                  {providerRouteHint && onOpenProjectProviders ? (
                    <SecondaryActionButton tone="slate" onClick={onOpenProjectProviders}>
                      {text.openProjectProvidersAction}
                    </SecondaryActionButton>
                  ) : null}
                  {projectMcpHint && onOpenProjectMcpInventory ? (
                    <SecondaryActionButton tone="slate" onClick={onOpenProjectMcpInventory}>
                      {text.openProjectMcpAction}
                    </SecondaryActionButton>
                  ) : null}
                  <DiagnosticPolicyRepairButtons
                    graphAgent={graphAgentsById?.get(agentId) ?? null}
                    diagnosticAgent={agent}
                    onApplyAgentPolicyRepair={onApplyAgentPolicyRepair}
                    isApplying={isApplyingPolicyRepair}
                    tone="slate"
                  />
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function CapabilityReadinessEvidenceCard({
  diagnostics,
  graphAgentsById,
  canFocusAgent,
  onFocusAgent,
  onOpenSkillPool,
  onOpenProjectProviders,
  onOpenProjectMcpInventory,
  onApplyAgentPolicyRepair,
  isApplyingPolicyRepair,
}: EvidenceCardSharedProps) {
  const text = useMessages(HARNESS_MESSAGES);
  const agents = coerceRecordList(diagnostics?.agents);
  const blockedCount =
    typeof diagnostics?.blocked_count === 'number'
      ? diagnostics.blocked_count
      : agents.filter((agent) => resolveCapabilityReadinessStatus(agent) === 'blocked').length;
  const limitedCount =
    typeof diagnostics?.limited_count === 'number'
      ? diagnostics.limited_count
      : agents.filter((agent) => resolveCapabilityReadinessStatus(agent) === 'limited').length;
  const agentCount = typeof diagnostics?.agent_count === 'number' ? diagnostics.agent_count : agents.length;

  if (agents.length === 0) {
    return <div className="mt-5 text-sm text-slate-500">{text.noCapabilityReadinessEvidence}</div>;
  }

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap gap-2">
        <span className="rounded-full bg-rose-50 px-2.5 py-1 text-[10px] font-semibold text-rose-800 ring-1 ring-rose-200">
          {formatTemplate(text.readinessBlockedCountLabel, { count: blockedCount })}
        </span>
        <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
          {formatTemplate(text.readinessLimitedCountLabel, { count: limitedCount })}
        </span>
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
          {formatTemplate(text.readinessFlaggedAgentCountLabel, { count: agentCount })}
        </span>
      </div>
      <div className="space-y-3">
        {agents.map((agent, index) => {
          const agentId =
            typeof agent.agent_id === 'string' && agent.agent_id
              ? agent.agent_id
              : `readiness-agent-${index + 1}`;
          const agentName =
            typeof agent.agent_name === 'string' && agent.agent_name
              ? agent.agent_name
              : agentId;
          const readinessStatus = resolveCapabilityReadinessStatus(agent);
          const blockers = coerceStringList(agent.blockers);
          const warnings = coerceStringList(agent.warnings);
          const skillPoolHint = shouldOpenSkillPoolForDiagnostic(agent);
          const projectMcpHint = shouldOpenProjectMcpForDiagnostic(agent);
          const providerRouteHint = shouldOpenProjectProvidersForDiagnostic(agent);
          return (
            <div key={`${agentId}-${index}`} className="rounded-2xl border border-slate-200 bg-white/90 p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="text-sm font-semibold text-slate-950">{agentName}</div>
                <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityReadinessBadgeClass(readinessStatus)}`}>
                  {formatCapabilityReadinessLabel(readinessStatus, text)}
                </span>
              </div>
              {blockers.length > 0 || warnings.length > 0 ? (
                <div className="mt-3 space-y-2 text-xs leading-5 text-slate-600">
                  {blockers.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.readinessBlockersLabel}:</span>{' '}
                      {blockers.join(' · ')}
                    </div>
                  ) : null}
                  {warnings.length > 0 ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.readinessWarningsLabel}:</span>{' '}
                      {warnings.join(' · ')}
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="mt-3 text-xs text-slate-500">{text.noReadinessIssues}</div>
              )}
              {canFocusAgent || onOpenSkillPool || onOpenProjectProviders || onOpenProjectMcpInventory || onApplyAgentPolicyRepair ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {agentId && canFocusAgent?.(agentId) && onFocusAgent ? (
                    <SecondaryActionButton tone="slate" onClick={() => onFocusAgent(agentId)}>
                      {formatTemplate(text.focusNodeForRecovery, { name: agentName })}
                    </SecondaryActionButton>
                  ) : null}
                  {skillPoolHint && onOpenSkillPool ? (
                    <SecondaryActionButton tone="slate" onClick={onOpenSkillPool}>
                      {text.openSkillPoolAction}
                    </SecondaryActionButton>
                  ) : null}
                  {providerRouteHint && onOpenProjectProviders ? (
                    <SecondaryActionButton tone="slate" onClick={onOpenProjectProviders}>
                      {text.openProjectProvidersAction}
                    </SecondaryActionButton>
                  ) : null}
                  {projectMcpHint && onOpenProjectMcpInventory ? (
                    <SecondaryActionButton tone="slate" onClick={onOpenProjectMcpInventory}>
                      {text.openProjectMcpAction}
                    </SecondaryActionButton>
                  ) : null}
                  <DiagnosticPolicyRepairButtons
                    graphAgent={graphAgentsById?.get(agentId) ?? null}
                    diagnosticAgent={agent}
                    onApplyAgentPolicyRepair={onApplyAgentPolicyRepair}
                    isApplying={isApplyingPolicyRepair}
                    tone="slate"
                  />
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function CapabilityExecutionContractEvidenceCard({
  diagnostics,
  canFocusAgent,
  onFocusAgent,
  onOpenProjectProviders,
  onOpenProjectMcpInventory,
}: Pick<
  EvidenceCardSharedProps,
  'diagnostics' | 'canFocusAgent' | 'onFocusAgent' | 'onOpenProjectProviders' | 'onOpenProjectMcpInventory'
>) {
  const text = useMessages(HARNESS_MESSAGES);
  const agents = coerceRecordList(diagnostics?.agents);
  const agentCount = typeof diagnostics?.agent_count === 'number' ? diagnostics.agent_count : agents.length;
  const directExecutionAgentCount =
    typeof diagnostics?.direct_execution_agent_count === 'number'
      ? diagnostics.direct_execution_agent_count
      : agents.filter((agent) => {
          const mode = typeof agent.tool_access_mode === 'string' ? agent.tool_access_mode : '';
          return mode === 'direct_execution' || mode === 'mixed';
        }).length;
  const planningOnlyToolAgentCount =
    typeof diagnostics?.planning_only_tool_agent_count === 'number'
      ? diagnostics.planning_only_tool_agent_count
      : agents.filter((agent) => coerceStringList(agent.planning_only_tool_ids).length > 0).length;
  const planningOnlyMcpAgentCount =
    typeof diagnostics?.planning_only_mcp_agent_count === 'number'
      ? diagnostics.planning_only_mcp_agent_count
      : agents.filter((agent) => coerceStringList(agent.planning_only_mcp_server_ids).length > 0).length;

  if (agents.length === 0) {
    return <div className="mt-5 text-sm text-slate-500">{text.noCapabilityExecutionContractEvidence}</div>;
  }

  const renderPills = (
    ids: string[],
    lookup: Map<string, string>,
    tone: 'slate' | 'cyan' | 'emerald' | 'violet' | 'amber',
    emptyLabel: string,
    keyPrefix: string
  ) => {
    if (ids.length === 0) {
      return <span className="text-xs text-slate-500">{emptyLabel}</span>;
    }
    return (
      <div className="mt-2 flex flex-wrap gap-2">
        {ids.map((id) => (
          <span
            key={`${keyPrefix}-${id}`}
            className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractPillToneClasses(tone)}`}
          >
            {formatSkillTitle(id, lookup)}
          </span>
        ))}
      </div>
    );
  };

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap gap-2">
        <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
          {formatTemplate(text.directExecutionAgentCountLabel, { count: directExecutionAgentCount })}
        </span>
        <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
          {formatTemplate(text.planningOnlyToolContractsCountLabel, { count: planningOnlyToolAgentCount })}
        </span>
        <span className="rounded-full bg-violet-50 px-2.5 py-1 text-[10px] font-semibold text-violet-800 ring-1 ring-violet-200">
          {formatTemplate(text.planningOnlyMcpContractsCountLabel, { count: planningOnlyMcpAgentCount })}
        </span>
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
          {formatTemplate(text.executionContractFlaggedAgentCountLabel, { count: agentCount })}
        </span>
      </div>
      <div className="space-y-3">
        {agents.map((agent, index) => {
          const agentId =
            typeof agent.agent_id === 'string' && agent.agent_id
              ? agent.agent_id
              : `execution-contract-agent-${index + 1}`;
          const agentName =
            typeof agent.agent_name === 'string' && agent.agent_name
              ? agent.agent_name
              : agentId;
          const skillExecutionMode =
            typeof agent.skill_execution_mode === 'string' && agent.skill_execution_mode === 'guidance_only'
              ? agent.skill_execution_mode
              : 'guidance_only';
          const toolAccessMode =
            typeof agent.tool_access_mode === 'string'
              ? agent.tool_access_mode
              : coerceStringList(agent.executable_tool_ids).length > 0
                ? 'direct_execution'
                : coerceStringList(agent.planning_only_tool_ids).length > 0
                  ? 'planning_only'
                  : 'none';
          const mcpAccessMode =
            typeof agent.mcp_access_mode === 'string'
              ? agent.mcp_access_mode
              : coerceStringList(agent.planning_only_mcp_server_ids).length > 0
                ? 'planning_only'
                : 'none';
          const approvedSkillIds = coerceStringList(agent.approved_skill_ids);
          const suggestedSkillIds = coerceStringList(agent.suggested_skill_ids);
          const executableToolIds = coerceStringList(agent.executable_tool_ids);
          const planningOnlyToolIds = coerceStringList(agent.planning_only_tool_ids);
          const disabledToolIds = coerceStringList(agent.disabled_tool_ids);
          const planningOnlyMcpServerIds = coerceStringList(agent.planning_only_mcp_server_ids);
          const missingMcpServerIds = coerceStringList(agent.missing_mcp_server_ids);
          const toolExecutionSupportReason =
            typeof agent.tool_execution_support_reason === 'string' && agent.tool_execution_support_reason
              ? agent.tool_execution_support_reason
              : null;
          const providerRoute =
            typeof agent.provider_route === 'string' && agent.provider_route ? agent.provider_route : null;
          const requiresToolCalling = Boolean(agent.requires_tool_calling);
          const recoveryActions = coerceRecord(agent.recovery_actions);
          const shouldOpenProjectProviders =
            Boolean(recoveryActions?.open_project_providers)
            || planningOnlyToolIds.length > 0
            || (requiresToolCalling && toolAccessMode !== 'direct_execution');
          const shouldOpenProjectMcpInventory =
            Boolean(recoveryActions?.open_project_mcp_inventory) || missingMcpServerIds.length > 0;

          return (
            <div key={`${agentId}-${index}`} className="rounded-2xl border border-slate-200 bg-white/90 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-semibold text-slate-950">{agentName}</div>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <span
                      className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                        skillExecutionMode
                      )}`}
                    >
                      {text.skillGuidanceOnly}
                    </span>
                    <span
                      className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                        toolAccessMode
                      )}`}
                    >
                      {toolAccessMode === 'direct_execution'
                        ? text.toolAccessDirectExecution
                        : toolAccessMode === 'planning_only'
                          ? text.toolAccessPlanningOnly
                          : toolAccessMode === 'mixed'
                            ? text.toolAccessMixed
                            : text.toolAccessNone}
                    </span>
                    <span
                      className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                        mcpAccessMode
                      )}`}
                    >
                      {mcpAccessMode === 'planning_only' ? text.mcpAccessPlanningOnly : text.mcpAccessNone}
                    </span>
                  </div>
                </div>
              </div>
              <div className="mt-4 grid gap-3 xl:grid-cols-3">
                <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {text.skillExecutionModeLabel}
                  </div>
                  <div className="mt-2">
                    <span
                      className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                        skillExecutionMode
                      )}`}
                    >
                      {text.skillGuidanceOnly}
                    </span>
                  </div>
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold text-slate-500">{text.approvedSkillsLabel}</div>
                    {renderPills(approvedSkillIds, EMPTY_LOOKUP, 'cyan', text.none, `${agentId}-approved-skills`)}
                  </div>
                  {suggestedSkillIds.length > 0 ? (
                    <div className="mt-3">
                      <div className="text-[11px] font-semibold text-slate-500">{text.suggestedSkillsLabel}</div>
                      {renderPills(
                        suggestedSkillIds,
                        EMPTY_LOOKUP,
                        'amber',
                        text.none,
                        `${agentId}-suggested-skills`
                      )}
                    </div>
                  ) : null}
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {text.toolAccessModeLabel}
                  </div>
                  <div className="mt-2">
                    <span
                      className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                        toolAccessMode
                      )}`}
                    >
                      {toolAccessMode === 'direct_execution'
                        ? text.toolAccessDirectExecution
                        : toolAccessMode === 'planning_only'
                          ? text.toolAccessPlanningOnly
                          : toolAccessMode === 'mixed'
                            ? text.toolAccessMixed
                            : text.toolAccessNone}
                    </span>
                  </div>
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold text-slate-500">{text.executableToolsLabel}</div>
                    {renderPills(
                      executableToolIds,
                      EMPTY_LOOKUP,
                      'emerald',
                      text.none,
                      `${agentId}-executable-tools`
                    )}
                  </div>
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold text-slate-500">{text.planningOnlyToolsLabel}</div>
                    {renderPills(
                      planningOnlyToolIds,
                      EMPTY_LOOKUP,
                      'amber',
                      text.none,
                      `${agentId}-planning-tools`
                    )}
                  </div>
                  {disabledToolIds.length > 0 ? (
                    <div className="mt-3">
                      <div className="text-[11px] font-semibold text-slate-500">{text.disabledToolsLabel}</div>
                      {renderPills(
                        disabledToolIds,
                        EMPTY_LOOKUP,
                        'slate',
                        text.none,
                        `${agentId}-disabled-tools`
                      )}
                    </div>
                  ) : null}
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {text.mcpAccessModeLabel}
                  </div>
                  <div className="mt-2">
                    <span
                      className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractModeBadgeClass(
                        mcpAccessMode
                      )}`}
                    >
                      {mcpAccessMode === 'planning_only' ? text.mcpAccessPlanningOnly : text.mcpAccessNone}
                    </span>
                  </div>
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold text-slate-500">{text.planningOnlyMcpServersLabel}</div>
                    {renderPills(
                      planningOnlyMcpServerIds,
                      EMPTY_LOOKUP,
                      'violet',
                      text.none,
                      `${agentId}-planning-mcp`
                    )}
                  </div>
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold text-slate-500">{text.missingMcpServersLabel}</div>
                    {renderPills(
                      missingMcpServerIds,
                      EMPTY_LOOKUP,
                      'amber',
                      text.none,
                      `${agentId}-missing-mcp`
                    )}
                  </div>
                </div>
              </div>
              {requiresToolCalling || providerRoute || toolExecutionSupportReason ? (
                <div className="mt-3 space-y-2 text-xs leading-5 text-slate-600">
                  {requiresToolCalling ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.contractRequiresToolCallingLabel}:</span>{' '}
                      {text.enabled}
                    </div>
                  ) : null}
                  {providerRoute ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.providerRouteLabel}:</span>{' '}
                      {providerRoute}
                    </div>
                  ) : null}
                  {toolExecutionSupportReason ? (
                    <div>
                      <span className="font-semibold text-slate-700">{text.toolExecutionSupportLabel}:</span>{' '}
                      {toolExecutionSupportReason}
                    </div>
                  ) : null}
                </div>
              ) : null}
              {canFocusAgent || onOpenProjectProviders || onOpenProjectMcpInventory ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {agentId && canFocusAgent?.(agentId) && onFocusAgent ? (
                    <SecondaryActionButton tone="slate" onClick={() => onFocusAgent(agentId)}>
                      {formatTemplate(text.focusNodeForRecovery, { name: agentName })}
                    </SecondaryActionButton>
                  ) : null}
                  {shouldOpenProjectProviders && onOpenProjectProviders ? (
                    <SecondaryActionButton tone="slate" onClick={onOpenProjectProviders}>
                      {text.openProjectProvidersAction}
                    </SecondaryActionButton>
                  ) : null}
                  {shouldOpenProjectMcpInventory && onOpenProjectMcpInventory ? (
                    <SecondaryActionButton tone="slate" onClick={onOpenProjectMcpInventory}>
                      {text.openProjectMcpAction}
                    </SecondaryActionButton>
                  ) : null}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function CollaborationContractEvidenceCard({
  diagnostics,
  canFocusAgent,
  onFocusAgent,
  policyRepairSummary,
  roleProfileSummary,
  onOpenSkillPool,
  onApplyAgentPolicyRepair,
  onApplySuggestedHandoff,
  isApplyingGraphChange,
}: Pick<
  EvidenceCardSharedProps,
  | 'diagnostics'
  | 'canFocusAgent'
  | 'onFocusAgent'
  | 'policyRepairSummary'
  | 'roleProfileSummary'
  | 'onOpenSkillPool'
  | 'onApplyAgentPolicyRepair'
  | 'onApplySuggestedHandoff'
  | 'isApplyingGraphChange'
>) {
  const text = useMessages(HARNESS_MESSAGES);
  const agents = coerceRecordList(diagnostics?.agents);
  const risks = coerceRecordList(diagnostics?.risks);
  const roleProfileDiagnosticsByAgentId = roleProfileSummary
    ? new Map(roleProfileSummary.diagnostics.map((diagnostic) => [diagnostic.agentId, diagnostic] as const))
    : undefined;
  const policyRepairDiagnosticsByAgentId = policyRepairSummary
    ? new Map(policyRepairSummary.diagnostics.map((diagnostic) => [diagnostic.agentId, diagnostic] as const))
    : undefined;
  const agentCount = typeof diagnostics?.agent_count === 'number' ? diagnostics.agent_count : agents.length;
  const coordinatorAgentCount =
    typeof diagnostics?.coordinator_agent_count === 'number'
      ? diagnostics.coordinator_agent_count
      : agents.filter((agent) => agent.primary_role_mode === 'coordinator').length;
  const parallelCoordinatorAgentCount =
    typeof diagnostics?.parallel_coordinator_agent_count === 'number'
      ? diagnostics.parallel_coordinator_agent_count
      : agents.filter((agent) => Boolean(agent.should_coordinate_parallel_work)).length;
  const finalOutputAgentCount =
    typeof diagnostics?.final_output_agent_count === 'number'
      ? diagnostics.final_output_agent_count
      : agents.filter((agent) => Boolean(agent.should_produce_final_output)).length;
  const verificationAgentCount =
    typeof diagnostics?.verification_agent_count === 'number'
      ? diagnostics.verification_agent_count
      : agents.filter((agent) => {
          const primaryRoleMode = typeof agent.primary_role_mode === 'string' ? agent.primary_role_mode : '';
          return (
            primaryRoleMode === 'verification'
            || coerceStringList(agent.supporting_role_modes).includes('verification')
            || agent.work_strategy === 'verify_and_close'
          );
        }).length;
  const coordinatorAgents = coerceRecordList(diagnostics?.coordinator_agents);
  const parallelCoordinatorAgents = coerceRecordList(diagnostics?.parallel_coordinator_agents);
  const finalOutputAgents = coerceRecordList(diagnostics?.final_output_agents);
  const verificationAgents = coerceRecordList(diagnostics?.verification_agents);

  if (agents.length === 0 && risks.length === 0) {
    return <div className="mt-5 text-sm text-slate-500">{text.noCollaborationContractEvidence}</div>;
  }

  const formatRoleLabel = (mode: string | null | undefined) => {
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

  const formatWorkStrategyLabel = (strategy: string | null | undefined) => {
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

  const renderCapabilityPills = (
    ids: string[],
    tone: 'slate' | 'cyan' | 'emerald' | 'violet' | 'amber',
    keyPrefix: string
  ) => {
    if (ids.length === 0) {
      return null;
    }
    return (
      <div className="mt-2 flex flex-wrap gap-2">
        {ids.map((id) => (
          <span
            key={`${keyPrefix}-${id}`}
            className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${contractPillToneClasses(tone)}`}
          >
            {formatSkillTitle(id, EMPTY_LOOKUP)}
          </span>
        ))}
      </div>
    );
  };

  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap gap-2">
        <span className="rounded-full bg-cyan-50 px-2.5 py-1 text-[10px] font-semibold text-cyan-800 ring-1 ring-cyan-200">
          {formatTemplate(text.coordinatorAgentCountLabel, { count: coordinatorAgentCount })}
        </span>
        <span className="rounded-full bg-sky-50 px-2.5 py-1 text-[10px] font-semibold text-sky-800 ring-1 ring-sky-200">
          {formatTemplate(text.parallelCoordinatorAgentCountLabel, { count: parallelCoordinatorAgentCount })}
        </span>
        <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
          {formatTemplate(text.finalOutputAgentCountLabel, { count: finalOutputAgentCount })}
        </span>
        <span className="rounded-full bg-violet-50 px-2.5 py-1 text-[10px] font-semibold text-violet-800 ring-1 ring-violet-200">
          {formatTemplate(text.verificationAgentCountLabel, { count: verificationAgentCount })}
        </span>
        <span className="rounded-full bg-amber-50 px-2.5 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-amber-200">
          {formatTemplate(text.collaborationContractRiskCountLabel, { count: risks.length })}
        </span>
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
          {formatTemplate(text.collaborationContractFlaggedAgentCountLabel, { count: agentCount })}
        </span>
      </div>
      <div className="grid gap-3 xl:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white/90 p-4">
          <div className="space-y-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.coordinatorOwnersLabel}
              </div>
              <div className="mt-2">
                <CollaborationPreviewButtons previews={coordinatorAgents} canFocusAgent={canFocusAgent} onFocusAgent={onFocusAgent} />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.parallelCoordinatorOwnersLabel}
              </div>
              <div className="mt-2">
                <CollaborationPreviewButtons previews={parallelCoordinatorAgents} canFocusAgent={canFocusAgent} onFocusAgent={onFocusAgent} />
              </div>
            </div>
          </div>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-white/90 p-4">
          <div className="space-y-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.finalOutputOwnersLabel}
              </div>
              <div className="mt-2">
                <CollaborationPreviewButtons previews={finalOutputAgents} canFocusAgent={canFocusAgent} onFocusAgent={onFocusAgent} />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.verificationOwnersLabel}
              </div>
              <div className="mt-2">
                <CollaborationPreviewButtons previews={verificationAgents} canFocusAgent={canFocusAgent} onFocusAgent={onFocusAgent} />
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="space-y-3">
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
          {text.collaborationContractRisksLabel}
        </div>
        {risks.length > 0 ? (
          risks.map((risk, index) => {
            const severity = typeof risk.severity === 'string' ? risk.severity : 'low';
            const summary = typeof risk.summary === 'string' && risk.summary ? risk.summary : text.none;
            const recommendedAction =
              typeof risk.recommended_action === 'string' && risk.recommended_action
                ? risk.recommended_action
                : null;
            const agentPreviews = coerceRecordList(risk.agent_previews);
            const sourceAgentId =
              typeof risk.source_agent_id === 'string' && risk.source_agent_id ? risk.source_agent_id : '';
            const delegateToolIds = coerceStringList(risk.delegate_tool_ids);
            const delegateMcpServerIds = coerceStringList(risk.delegate_mcp_server_ids);
            const delegateCandidates = coerceRecordList(risk.delegate_candidates);
            return (
              <div key={`collaboration-risk-${index}`} className="rounded-2xl border border-slate-200 bg-white/90 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0 flex-1 text-sm font-semibold text-slate-950">{summary}</div>
                  <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${collaborationRiskBadgeClass(severity)}`}>
                    {severity === 'high' ? text.priorityHighLabel : severity === 'medium' ? text.priorityMediumLabel : text.priorityLowLabel}
                  </span>
                </div>
                {recommendedAction ? (
                  <div className="mt-2 text-xs leading-5 text-slate-600">
                    <span className="font-semibold text-slate-700">{text.collaborationContractRecommendedActionLabel}:</span>{' '}
                    {recommendedAction}
                  </div>
                ) : null}
                {agentPreviews.length > 0 ? (
                  <div className="mt-3">
                    <CollaborationPreviewButtons previews={agentPreviews} canFocusAgent={canFocusAgent} onFocusAgent={onFocusAgent} />
                  </div>
                ) : null}
                {delegateToolIds.length > 0 ? (
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                      {text.collaborationDelegateToolsLabel}
                    </div>
                    {renderCapabilityPills(delegateToolIds, 'emerald', `risk-tools-${index}`)}
                  </div>
                ) : null}
                {delegateMcpServerIds.length > 0 ? (
                  <div className="mt-3">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                      {text.collaborationDelegateMcpLabel}
                    </div>
                    {renderCapabilityPills(delegateMcpServerIds, 'violet', `risk-mcp-${index}`)}
                  </div>
                ) : null}
                <DelegateCandidateCards
                  sourceAgentId={sourceAgentId}
                  candidates={delegateCandidates}
                  keyPrefix={`risk-${index}`}
                  canFocusAgent={canFocusAgent}
                  onFocusAgent={onFocusAgent}
                  onApplySuggestedHandoff={onApplySuggestedHandoff}
                  isApplyingGraphChange={isApplyingGraphChange}
                  roleProfileDiagnosticsByAgentId={roleProfileDiagnosticsByAgentId}
                  policyRepairDiagnosticsByAgentId={policyRepairDiagnosticsByAgentId}
                  onApplyAgentPolicyRepair={onApplyAgentPolicyRepair}
                  onOpenSkillPool={onOpenSkillPool}
                />
              </div>
            );
          })
        ) : (
          <div className="text-sm text-slate-500">{text.noCollaborationContractRisks}</div>
        )}
      </div>
      {agents.length > 0 ? (
        <div className="space-y-3">
          {agents.map((agent, index) => {
            const agentId =
              typeof agent.agent_id === 'string' && agent.agent_id
                ? agent.agent_id
                : `collaboration-agent-${index + 1}`;
            const agentName =
              typeof agent.agent_name === 'string' && agent.agent_name
                ? agent.agent_name
                : agentId;
            const primaryRoleMode = typeof agent.primary_role_mode === 'string' ? agent.primary_role_mode : 'generalist';
            const workStrategy = typeof agent.work_strategy === 'string' ? agent.work_strategy : 'flexible';
            const supportingRoleModes = coerceStringList(agent.supporting_role_modes);
            const executableToolIds = coerceStringList(agent.executable_tool_ids);
            const planningOnlyMcpServerIds = coerceStringList(agent.planning_only_mcp_server_ids);
            const watchouts = coerceStringList(agent.watchouts);
            const upstreamAgents = coerceRecordList(agent.upstream_agents);
            const downstreamAgents = coerceRecordList(agent.downstream_agents);
            return (
              <div key={`${agentId}-${index}`} className="rounded-2xl border border-slate-200 bg-white/90 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-semibold text-slate-950">{agentName}</div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${collaborationRoleBadgeClass(primaryRoleMode)}`}>
                        {formatRoleLabel(primaryRoleMode)}
                      </span>
                      <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${collaborationRoleBadgeClass(primaryRoleMode)}`}>
                        {formatWorkStrategyLabel(workStrategy)}
                      </span>
                      {Boolean(agent.should_coordinate_parallel_work) ? (
                        <span className="rounded-full bg-cyan-50 px-2.5 py-1 text-[10px] font-semibold text-cyan-800 ring-1 ring-cyan-200">
                          {text.coordinateParallelWorkLabel}
                        </span>
                      ) : null}
                      {Boolean(agent.should_produce_final_output) ? (
                        <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[10px] font-semibold text-emerald-800 ring-1 ring-emerald-200">
                          {text.finalOutputResponsibilityLabel}
                        </span>
                      ) : null}
                    </div>
                  </div>
                  {canFocusAgent?.(agentId) && onFocusAgent ? (
                    <SecondaryActionButton tone="slate" onClick={() => onFocusAgent(agentId)}>
                      {formatTemplate(text.focusNodeForRecovery, { name: agentName })}
                    </SecondaryActionButton>
                  ) : null}
                </div>
                {supportingRoleModes.length > 0 ? (
                  <div className="mt-3 text-xs leading-5 text-slate-600">
                    <span className="font-semibold text-slate-700">{text.supportingRoleModesLabel}:</span>{' '}
                    {supportingRoleModes.map((mode) => formatRoleLabel(mode)).join(' · ')}
                  </div>
                ) : null}
                {typeof agent.primary_focus === 'string' && agent.primary_focus ? (
                  <div className="mt-2 text-xs leading-5 text-slate-600">
                    <span className="font-semibold text-slate-700">{text.delegationFocusLabel}:</span>{' '}
                    {agent.primary_focus}
                  </div>
                ) : null}
                {executableToolIds.length > 0 ? (
                  <div className="mt-2 text-xs leading-5 text-slate-600">
                    <span className="font-semibold text-slate-700">{text.executableToolsLabel}:</span>{' '}
                    {executableToolIds.join(' · ')}
                  </div>
                ) : null}
                {planningOnlyMcpServerIds.length > 0 ? (
                  <div className="mt-2 text-xs leading-5 text-slate-600">
                    <span className="font-semibold text-slate-700">{text.planningOnlyMcpServersLabel}:</span>{' '}
                    {planningOnlyMcpServerIds.join(' · ')}
                  </div>
                ) : null}
                {(upstreamAgents.length > 0 || downstreamAgents.length > 0) ? (
                  <div className="mt-3 grid gap-3 xl:grid-cols-2">
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                        {text.upstreamAgentsLabel}
                      </div>
                      <div className="mt-2">
                        <CollaborationPreviewButtons previews={upstreamAgents} canFocusAgent={canFocusAgent} onFocusAgent={onFocusAgent} />
                      </div>
                    </div>
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                        {text.downstreamAgentsLabel}
                      </div>
                      <div className="mt-2">
                        <CollaborationPreviewButtons previews={downstreamAgents} canFocusAgent={canFocusAgent} onFocusAgent={onFocusAgent} />
                      </div>
                    </div>
                  </div>
                ) : null}
                {watchouts.length > 0 ? (
                  <div className="mt-3 space-y-2">
                    {watchouts.map((watchout, watchoutIndex) => (
                      <div
                        key={`${agentId}-watchout-${watchoutIndex + 1}`}
                        className="rounded-xl border border-amber-100 bg-amber-50/80 px-3 py-2 text-xs leading-5 text-amber-900"
                      >
                        {watchout}
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}
