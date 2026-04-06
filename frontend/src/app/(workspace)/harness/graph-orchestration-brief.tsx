import { useMessages } from '@/lib/i18n';
import type {
  OrchestrationAgentRoutingSummary,
  CoordinationAgentPreview,
  OrchestrationBriefSummary,
  OrchestrationPhaseId,
  OrchestrationPhaseSummary,
  OrchestrationRepairPriority,
  OrchestrationRepairPriorityId,
} from './diagnostics';
import { HARNESS_MESSAGES } from './messages';
import { formatSkillTitle, formatTemplate } from './utils';

const PREVIEW_LIMIT = 6;
const SYNTHETIC_REVIEW_AGENT_ID = '__review_agent__';

function formatAgentPreview(agents: CoordinationAgentPreview[], text: Record<string, string>) {
  const names = agents.map((agent) => agent.agentName);
  if (names.length === 0) {
    return '';
  }
  const visibleNames = names.slice(0, 4);
  const remainingCount = names.length - visibleNames.length;
  if (remainingCount <= 0) {
    return visibleNames.join(', ');
  }
  return `${visibleNames.join(', ')} (${formatTemplate(text.additionalItemsCount, { count: remainingCount })})`;
}

function phaseLabel(phaseId: OrchestrationPhaseId, text: Record<string, string>) {
  if (phaseId === 'research') {
    return text.researchPhaseLabel;
  }
  if (phaseId === 'synthesis') {
    return text.synthesisPhaseLabel;
  }
  if (phaseId === 'implementation') {
    return text.implementationPhaseLabel;
  }
  return text.verificationPhaseLabel;
}

function phaseHint(phaseId: OrchestrationPhaseId, text: Record<string, string>) {
  if (phaseId === 'research') {
    return text.researchPhaseHint;
  }
  if (phaseId === 'synthesis') {
    return text.synthesisPhaseHint;
  }
  if (phaseId === 'implementation') {
    return text.implementationPhaseHint;
  }
  return text.verificationPhaseHint;
}

function phaseModeLabel(phase: OrchestrationPhaseSummary, reviewEnabled: boolean, text: Record<string, string>) {
  if (phase.phaseId === 'verification') {
    return reviewEnabled ? text.phaseModeGuardrail : text.phaseModeManual;
  }
  if (phase.phaseId === 'synthesis') {
    return phase.agentCount > 1 ? text.phaseModeCoordinated : text.phaseModeSerial;
  }
  return phase.agentCount > 1 ? text.phaseModeParallel : text.phaseModeSerial;
}

function priorityLabel(priorityId: OrchestrationRepairPriorityId, text: Record<string, string>) {
  if (priorityId === 'availability') {
    return text.repairPriorityAvailabilityLabel;
  }
  if (priorityId === 'capability_gaps') {
    return text.repairPriorityCapabilityGapsLabel;
  }
  if (priorityId === 'policy_repair') {
    return text.repairPriorityPolicyLabel;
  }
  if (priorityId === 'role_profile_alignment') {
    return text.repairPriorityRoleProfileLabel;
  }
  if (priorityId === 'weak_handoffs') {
    return text.repairPriorityWeakHandoffsLabel;
  }
  if (priorityId === 'best_next_handoffs') {
    return text.repairPriorityBestNextLabel;
  }
  if (priorityId === 'connectivity') {
    return text.repairPriorityConnectivityLabel;
  }
  if (priorityId === 'single_owner_capabilities') {
    return text.repairPrioritySingleOwnerLabel;
  }
  return text.repairPriorityReviewLabel;
}

function priorityHint(priorityId: OrchestrationRepairPriorityId, text: Record<string, string>) {
  if (priorityId === 'availability') {
    return text.repairPriorityAvailabilityHint;
  }
  if (priorityId === 'capability_gaps') {
    return text.repairPriorityCapabilityGapsHint;
  }
  if (priorityId === 'policy_repair') {
    return text.repairPriorityPolicyHint;
  }
  if (priorityId === 'role_profile_alignment') {
    return text.repairPriorityRoleProfileHint;
  }
  if (priorityId === 'weak_handoffs') {
    return text.repairPriorityWeakHandoffsHint;
  }
  if (priorityId === 'best_next_handoffs') {
    return text.repairPriorityBestNextHint;
  }
  if (priorityId === 'connectivity') {
    return text.repairPriorityConnectivityHint;
  }
  if (priorityId === 'single_owner_capabilities') {
    return text.repairPrioritySingleOwnerHint;
  }
  return text.repairPriorityReviewHint;
}

function prioritySeverityLabel(severity: OrchestrationRepairPriority['severity'], text: Record<string, string>) {
  if (severity === 'high') {
    return text.priorityHighLabel;
  }
  if (severity === 'medium') {
    return text.priorityMediumLabel;
  }
  return text.priorityLowLabel;
}

function priorityToneClassName(severity: OrchestrationRepairPriority['severity']) {
  if (severity === 'high') {
    return 'bg-rose-50 text-rose-800 ring-rose-200';
  }
  if (severity === 'medium') {
    return 'bg-amber-50 text-amber-800 ring-amber-200';
  }
  return 'bg-sky-50 text-sky-800 ring-sky-200';
}

function SequencePills({
  values,
  emptyLabel,
}: {
  values: string[];
  emptyLabel: string;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  if (values.length === 0) {
    return <div className="text-sm text-slate-500">{emptyLabel}</div>;
  }
  const visibleValues = values.slice(0, PREVIEW_LIMIT);
  const remainingCount = values.length - visibleValues.length;
  return (
    <div className="flex flex-wrap gap-2">
      {visibleValues.map((value, index) => (
        <span
          key={`${index}-${value}`}
          className="rounded-full bg-white px-3 py-1 text-xs font-medium text-slate-700 ring-1 ring-slate-200"
        >
          {index + 1}. {value}
        </span>
      ))}
      {remainingCount > 0 ? (
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
          {formatTemplate(text.additionalItemsCount, { count: remainingCount })}
        </span>
      ) : null}
    </div>
  );
}

function AgentPills({
  agents,
  emptyLabel,
  focusableAgentIds,
  onFocusAgent,
}: {
  agents: CoordinationAgentPreview[];
  emptyLabel: string;
  focusableAgentIds: Set<string>;
  onFocusAgent: (agentId: string) => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  if (agents.length === 0) {
    return <div className="text-xs text-slate-500">{emptyLabel}</div>;
  }
  const visibleAgents = agents.slice(0, PREVIEW_LIMIT);
  const remainingCount = agents.length - visibleAgents.length;
  return (
    <div className="flex flex-wrap gap-2">
      {visibleAgents.map((agent) =>
        focusableAgentIds.has(agent.agentId) ? (
          <button
            key={agent.agentId}
            type="button"
            onClick={() => onFocusAgent(agent.agentId)}
            className="inline-flex items-center justify-center rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-700 hover:bg-slate-100"
          >
            {agent.agentName}
          </button>
        ) : (
          <span
            key={agent.agentId}
            className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200"
          >
            {agent.agentName}
          </span>
        )
      )}
      {remainingCount > 0 ? (
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
          {formatTemplate(text.additionalItemsCount, { count: remainingCount })}
        </span>
      ) : null}
    </div>
  );
}

function capabilityKindLabel(kind: OrchestrationBriefSummary['singleOwnerCapabilityRisks'][number]['kind'], text: Record<string, string>) {
  if (kind === 'tool') {
    return text.capabilityRiskToolLabel;
  }
  if (kind === 'mcp') {
    return text.capabilityRiskMcpLabel;
  }
  return text.capabilityRiskSkillLabel;
}

function capabilityKindTone(kind: OrchestrationBriefSummary['singleOwnerCapabilityRisks'][number]['kind']) {
  if (kind === 'tool') {
    return 'bg-emerald-50 text-emerald-800 ring-emerald-200';
  }
  if (kind === 'mcp') {
    return 'bg-violet-50 text-violet-800 ring-violet-200';
  }
  return 'bg-cyan-50 text-cyan-800 ring-cyan-200';
}

function resolveCapabilityTitle(
  risk: OrchestrationBriefSummary['singleOwnerCapabilityRisks'][number],
  {
    skillTitleById,
    toolTitleById,
    mcpServerTitleById,
  }: {
    skillTitleById: Map<string, string>;
    toolTitleById: Map<string, string>;
    mcpServerTitleById: Map<string, string>;
  }
) {
  if (risk.kind === 'tool') {
    return formatSkillTitle(risk.capabilityId, toolTitleById);
  }
  if (risk.kind === 'mcp') {
    return formatSkillTitle(risk.capabilityId, mcpServerTitleById);
  }
  return formatSkillTitle(risk.capabilityId, skillTitleById);
}

function SingleOwnerCapabilityRows({
  risks,
  skillTitleById,
  toolTitleById,
  mcpServerTitleById,
  focusableAgentIds,
  onFocusAgent,
}: {
  risks: OrchestrationBriefSummary['singleOwnerCapabilityRisks'];
  skillTitleById: Map<string, string>;
  toolTitleById: Map<string, string>;
  mcpServerTitleById: Map<string, string>;
  focusableAgentIds: Set<string>;
  onFocusAgent: (agentId: string) => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  if (risks.length === 0) {
    return <div className="text-xs text-slate-500">{text.noSingleOwnerCapabilityRisks}</div>;
  }
  const visibleRisks = risks.slice(0, PREVIEW_LIMIT);
  const remainingCount = risks.length - visibleRisks.length;
  return (
    <div className="space-y-2">
      {visibleRisks.map((risk) => {
        const owner = risk.ownerAgents[0] ?? null;
        return (
          <div
            key={`${risk.kind}:${risk.capabilityId}`}
            className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white px-3 py-3"
          >
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${capabilityKindTone(risk.kind)}`}>
                  {capabilityKindLabel(risk.kind, text)}
                </span>
                <div className="min-w-0 text-sm font-semibold text-slate-900">
                  {resolveCapabilityTitle(risk, { skillTitleById, toolTitleById, mcpServerTitleById })}
                </div>
              </div>
              <div className="mt-1 text-xs text-slate-500">{owner?.agentName || text.unknownNode}</div>
            </div>
            {owner && focusableAgentIds.has(owner.agentId) ? (
              <button
                type="button"
                onClick={() => onFocusAgent(owner.agentId)}
                className="inline-flex items-center justify-center rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-100"
              >
                {formatTemplate(text.focusNodeForRecovery, { name: owner.agentName })}
              </button>
            ) : null}
          </div>
        );
      })}
      {remainingCount > 0 ? (
        <div className="text-xs text-slate-500">{formatTemplate(text.additionalItemsCount, { count: remainingCount })}</div>
      ) : null}
    </div>
  );
}

function PhaseBoard({
  phases,
  reviewEnabled,
  reviewAgentName,
  focusableAgentIds,
  onFocusAgent,
}: {
  phases: OrchestrationBriefSummary['phases'];
  reviewEnabled: boolean;
  reviewAgentName: string;
  focusableAgentIds: Set<string>;
  onFocusAgent: (agentId: string) => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);

  return (
    <div className="space-y-2">
      {phases.map((phase) => {
        const phaseAgents =
          phase.phaseId === 'verification' && reviewEnabled
            ? [{ agentId: SYNTHETIC_REVIEW_AGENT_ID, agentName: reviewAgentName }, ...phase.agents]
            : phase.agents;
        const dedupedPhaseAgents = Array.from(
          new Map(phaseAgents.map((agent) => [agent.agentId, agent])).values()
        );
        return (
          <div
            key={phase.phaseId}
            className="rounded-xl border border-slate-200 bg-white px-3 py-3"
          >
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-slate-900">{phaseLabel(phase.phaseId, text)}</div>
                <div className="mt-1 text-xs leading-5 text-slate-500">{phaseHint(phase.phaseId, text)}</div>
              </div>
              <div className="flex flex-wrap gap-2">
                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
                  {formatTemplate(text.agentCountShort, { count: dedupedPhaseAgents.length })}
                </span>
                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
                  {phaseModeLabel(phase, reviewEnabled, text)}
                </span>
              </div>
            </div>
            <div className="mt-3">
              <AgentPills
                agents={dedupedPhaseAgents}
                emptyLabel={text.noPhaseAnchorsHint}
                focusableAgentIds={focusableAgentIds}
                onFocusAgent={onFocusAgent}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function RepairPriorityRows({
  priorities,
}: {
  priorities: OrchestrationBriefSummary['repairPriorities'];
}) {
  const text = useMessages(HARNESS_MESSAGES);
  if (priorities.length === 0) {
    return <div className="text-xs text-slate-500">{text.noRepairPriorities}</div>;
  }
  return (
    <div className="space-y-2">
      {priorities.map((priority) => (
        <div
          key={priority.priorityId}
          className="rounded-xl border border-slate-200 bg-white px-3 py-3"
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="text-sm font-semibold text-slate-900">{priorityLabel(priority.priorityId, text)}</div>
              <div className="mt-1 text-xs leading-5 text-slate-500">{priorityHint(priority.priorityId, text)}</div>
            </div>
            <div className="flex flex-wrap gap-2">
              <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${priorityToneClassName(priority.severity)}`}>
                {prioritySeverityLabel(priority.severity, text)}
              </span>
              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
                {formatTemplate(text.priorityCountShort, { count: priority.count })}
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function routingLabel(
  routingId: keyof OrchestrationAgentRoutingSummary,
  text: Record<string, string>
) {
  if (routingId === 'coordinatorAnchors') {
    return text.coordinatorAnchorsLabel;
  }
  if (routingId === 'researchAnchors') {
    return text.researchAnchorsLabel;
  }
  if (routingId === 'implementationAnchors') {
    return text.implementationAnchorsLabel;
  }
  if (routingId === 'verificationAnchors') {
    return text.verificationAnchorsLabel;
  }
  if (routingId === 'skillCapableAnchors') {
    return text.skillCapableAnchorsLabel;
  }
  if (routingId === 'toolCapableAnchors') {
    return text.toolCapableAnchorsLabel;
  }
  return text.mcpCapableAnchorsLabel;
}

function routingHint(
  routingId: keyof OrchestrationAgentRoutingSummary,
  text: Record<string, string>
) {
  if (routingId === 'coordinatorAnchors') {
    return text.coordinatorAnchorsHint;
  }
  if (routingId === 'researchAnchors') {
    return text.researchAnchorsHint;
  }
  if (routingId === 'implementationAnchors') {
    return text.implementationAnchorsHint;
  }
  if (routingId === 'verificationAnchors') {
    return text.verificationAnchorsHint;
  }
  if (routingId === 'skillCapableAnchors') {
    return text.skillCapableAnchorsHint;
  }
  if (routingId === 'toolCapableAnchors') {
    return text.toolCapableAnchorsHint;
  }
  return text.mcpCapableAnchorsHint;
}

function AgentRoutingRows({
  routing,
  focusableAgentIds,
  onFocusAgent,
}: {
  routing: OrchestrationBriefSummary['agentRouting'];
  focusableAgentIds: Set<string>;
  onFocusAgent: (agentId: string) => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const entries: Array<{
    routingId: keyof OrchestrationAgentRoutingSummary;
    agents: CoordinationAgentPreview[];
  }> = [
    { routingId: 'coordinatorAnchors', agents: routing.coordinatorAnchors },
    { routingId: 'researchAnchors', agents: routing.researchAnchors },
    { routingId: 'implementationAnchors', agents: routing.implementationAnchors },
    { routingId: 'verificationAnchors', agents: routing.verificationAnchors },
    { routingId: 'skillCapableAnchors', agents: routing.skillCapableAnchors },
    { routingId: 'toolCapableAnchors', agents: routing.toolCapableAnchors },
    { routingId: 'mcpCapableAnchors', agents: routing.mcpCapableAnchors },
  ];

  return (
    <div className="grid gap-2 xl:grid-cols-2">
      {entries.map((entry) => (
        <div
          key={entry.routingId}
          className="rounded-xl border border-slate-200 bg-white px-3 py-3"
        >
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-slate-900">{routingLabel(entry.routingId, text)}</div>
              <div className="mt-1 text-xs leading-5 text-slate-500">{routingHint(entry.routingId, text)}</div>
            </div>
            <span className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200">
              {formatTemplate(text.agentCountShort, { count: entry.agents.length })}
            </span>
          </div>
          <div className="mt-3">
            <AgentPills
              agents={entry.agents}
              emptyLabel={text.noRoutingAnchorsHint}
              focusableAgentIds={focusableAgentIds}
              onFocusAgent={onFocusAgent}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export function GraphOrchestrationBrief({
  scopeLabel,
  previewLabel,
  previewSteps,
  emptyLabel,
  summary,
  reviewAgentName,
  applyRoleProfilesLabel = null,
  requestRoleProfileSkillsLabel = null,
  applyPolicyRepairsLabel = null,
  applyCollaborationFixesLabel = null,
  isApplying = false,
  isRequestingRoleProfileSkills = false,
  onApplyRoleProfiles,
  onRequestRoleProfileSkills,
  onApplyPolicyRepairs,
  onApplyCollaborationFixes,
  skillTitleById,
  toolTitleById,
  mcpServerTitleById,
  focusableAgentIds,
  onFocusAgent,
}: {
  scopeLabel: string;
  previewLabel: string;
  previewSteps: string[];
  emptyLabel: string;
  summary: OrchestrationBriefSummary;
  reviewAgentName: string;
  applyRoleProfilesLabel?: string | null;
  requestRoleProfileSkillsLabel?: string | null;
  applyPolicyRepairsLabel?: string | null;
  applyCollaborationFixesLabel?: string | null;
  isApplying?: boolean;
  isRequestingRoleProfileSkills?: boolean;
  onApplyRoleProfiles?: (() => void) | null;
  onRequestRoleProfileSkills?: (() => void) | null;
  onApplyPolicyRepairs?: (() => void) | null;
  onApplyCollaborationFixes?: (() => void) | null;
  skillTitleById: Map<string, string>;
  toolTitleById: Map<string, string>;
  mcpServerTitleById: Map<string, string>;
  focusableAgentIds: Set<string>;
  onFocusAgent: (agentId: string) => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const hasScope = summary.totalAgentCount > 0;
  const tone =
    !hasScope
      ? { panel: 'border-slate-200 bg-slate-50/80', accent: 'text-slate-500' }
      : summary.readiness === 'blocked'
        ? { panel: 'border-rose-200 bg-rose-50/70', accent: 'text-rose-700' }
        : summary.readiness === 'repair'
          ? { panel: 'border-amber-200 bg-amber-50/70', accent: 'text-amber-700' }
          : summary.readiness === 'watch'
            ? { panel: 'border-sky-200 bg-sky-50/70', accent: 'text-sky-700' }
            : { panel: 'border-emerald-200 bg-emerald-50/70', accent: 'text-emerald-700' };
  const launchHint =
    summary.startAgents.length > 1
      ? formatTemplate(text.parallelLaunchHint, { names: formatAgentPreview(summary.startAgents, text) })
      : summary.startAgents.length === 1
        ? formatTemplate(text.singleAnchorLaunchHint, { name: summary.startAgents[0]?.agentName || text.unknownNode })
        : text.noLaunchEntrypointHint;
  const terminalHint =
    summary.terminalAgents.length > 0
      ? formatTemplate(text.terminalFlowHint, { names: formatAgentPreview(summary.terminalAgents, text) })
      : text.noTerminalFlowHint;
  const statusHint =
    !hasScope
      ? emptyLabel
      : summary.readiness === 'blocked'
        ? text.orchestrationBriefBlockedHint
        : summary.readiness === 'repair'
          ? text.orchestrationBriefRepairHint
          : summary.readiness === 'watch'
            ? text.orchestrationBriefWatchHint
            : text.orchestrationBriefReadyHint;

  return (
    <div className={`rounded-[14px] border p-4 ${tone.panel}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className={`text-[11px] font-semibold uppercase tracking-[0.18em] ${tone.accent}`}>{scopeLabel}</div>
          <div className="mt-1 text-sm font-semibold text-slate-900">{text.orchestrationBriefLabel}</div>
          <div className="mt-1 text-sm text-slate-600">{statusHint}</div>
        </div>
        {hasScope ? (
          <div className="flex flex-wrap gap-2">
            <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
              {formatTemplate(text.agentCountShort, { count: summary.totalAgentCount })}
            </span>
            <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
              {formatTemplate(text.executionStepsCountShort, { count: summary.executionStepCount })}
            </span>
            <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
              {formatTemplate(text.launchStartCountShort, { count: summary.startAgents.length })}
            </span>
            <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
              {summary.reviewEnabled ? text.reviewEnabledShort : text.reviewDisabledShort}
            </span>
          </div>
        ) : null}
      </div>

      {hasScope ? (
        <>
          <div className="mt-4 grid gap-3 xl:grid-cols-2">
            <div className="rounded-[12px] border border-white/80 bg-white/80 p-4">
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">{previewLabel}</div>
              <div className="mt-3">
                <SequencePills values={previewSteps} emptyLabel={emptyLabel} />
              </div>
              <div className="mt-4 space-y-3">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {text.launchStartersLabel}
                  </div>
                  <div className="mt-2">
                    <AgentPills
                      agents={summary.startAgents}
                      emptyLabel={text.noLaunchEntrypointHint}
                      focusableAgentIds={focusableAgentIds}
                      onFocusAgent={onFocusAgent}
                    />
                  </div>
                  <div className="mt-2 text-xs leading-5 text-slate-500">{launchHint}</div>
                </div>
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {text.terminalAgentsLabel}
                  </div>
                  <div className="mt-2">
                    <AgentPills
                      agents={summary.terminalAgents}
                      emptyLabel={text.noTerminalFlowHint}
                      focusableAgentIds={focusableAgentIds}
                      onFocusAgent={onFocusAgent}
                    />
                  </div>
                  <div className="mt-2 text-xs leading-5 text-slate-500">{terminalHint}</div>
                </div>
                <div className="text-xs leading-5 text-slate-500">
                  {summary.reviewEnabled
                    ? formatTemplate(text.reviewLoopAttachedNamedHint, { name: reviewAgentName })
                    : text.reviewLoopDetachedHint}
                </div>
              </div>
            </div>

            <div className="rounded-[12px] border border-white/80 bg-white/80 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {text.orchestrationPhaseBoardLabel}
                  </div>
                  <div className="mt-1 text-xs leading-5 text-slate-500">{text.orchestrationPhaseBoardHint}</div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                    {formatTemplate(text.phaseCountShort, { count: summary.phases.length })}
                  </span>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                    {formatTemplate(text.priorityCountShort, { count: summary.repairPriorities.length })}
                  </span>
                </div>
              </div>
              <div className="mt-3">
                <PhaseBoard
                  phases={summary.phases}
                  reviewEnabled={summary.reviewEnabled}
                  reviewAgentName={reviewAgentName}
                  focusableAgentIds={focusableAgentIds}
                  onFocusAgent={onFocusAgent}
                />
              </div>
            </div>
          </div>

          <div className="mt-4 grid gap-3 xl:grid-cols-2">
            <div className="rounded-[12px] border border-white/80 bg-white/80 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {text.repairPriorityBoardLabel}
                  </div>
                  <div className="mt-1 text-xs leading-5 text-slate-500">{text.repairPriorityBoardHint}</div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {applyRoleProfilesLabel && onApplyRoleProfiles ? (
                    <button
                      type="button"
                      onClick={onApplyRoleProfiles}
                      disabled={isApplying}
                      className="inline-flex items-center justify-center rounded-[10px] border border-cyan-300 bg-white px-3 py-1.5 text-xs font-semibold text-cyan-900 hover:bg-cyan-100 disabled:opacity-50"
                    >
                      {isApplying ? text.applyingRoleProfiles : applyRoleProfilesLabel}
                    </button>
                  ) : null}
                  {requestRoleProfileSkillsLabel && onRequestRoleProfileSkills ? (
                    <button
                      type="button"
                      onClick={onRequestRoleProfileSkills}
                      disabled={isRequestingRoleProfileSkills}
                      className="inline-flex items-center justify-center rounded-[10px] border border-rose-300 bg-white px-3 py-1.5 text-xs font-semibold text-rose-900 hover:bg-rose-100 disabled:opacity-50"
                    >
                      {isRequestingRoleProfileSkills
                        ? text.requestingRoleProfileSkills
                        : requestRoleProfileSkillsLabel}
                    </button>
                  ) : null}
                  {applyPolicyRepairsLabel && onApplyPolicyRepairs ? (
                    <button
                      type="button"
                      onClick={onApplyPolicyRepairs}
                      disabled={isApplying}
                      className="inline-flex items-center justify-center rounded-[10px] border border-amber-300 bg-white px-3 py-1.5 text-xs font-semibold text-amber-900 hover:bg-amber-100 disabled:opacity-50"
                    >
                      {isApplying ? text.applyingPolicyRepairs : applyPolicyRepairsLabel}
                    </button>
                  ) : null}
                  {applyCollaborationFixesLabel && onApplyCollaborationFixes ? (
                    <button
                      type="button"
                      onClick={onApplyCollaborationFixes}
                      disabled={isApplying}
                      className="inline-flex items-center justify-center rounded-[10px] border border-emerald-300 bg-white px-3 py-1.5 text-xs font-semibold text-emerald-900 hover:bg-emerald-100 disabled:opacity-50"
                    >
                      {isApplying ? text.applyingSuggestedCollaborationFixes : applyCollaborationFixesLabel}
                    </button>
                  ) : null}
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                    {formatTemplate(text.priorityCountShort, { count: summary.repairPriorities.length })}
                  </span>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                    {formatTemplate(text.singleOwnerCapabilityCountShort, { count: summary.singleOwnerCapabilityCount })}
                  </span>
                </div>
              </div>
              <div className="mt-3">
                <RepairPriorityRows priorities={summary.repairPriorities} />
              </div>
            </div>

            <div className="rounded-[12px] border border-white/80 bg-white/80 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                    {text.agentRoutingBoardLabel}
                  </div>
                  <div className="mt-1 text-xs leading-5 text-slate-500">{text.agentRoutingBoardHint}</div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                    {formatTemplate(text.routingBucketsCountShort, { count: 7 })}
                  </span>
                  <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                    {formatTemplate(text.singleOwnerCapabilityCountShort, { count: summary.singleOwnerCapabilityCount })}
                  </span>
                </div>
              </div>
              <div className="mt-3">
                <AgentRoutingRows
                  routing={summary.agentRouting}
                  focusableAgentIds={focusableAgentIds}
                  onFocusAgent={onFocusAgent}
                />
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-[12px] border border-white/80 bg-white/80 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                  {text.singleOwnerCapabilityWatchlistLabel}
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                  {formatTemplate(text.singleOwnerCapabilityCountShort, { count: summary.singleOwnerCapabilityCount })}
                </span>
                <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
                  {formatTemplate(text.capabilityGapCountShort, { count: summary.capabilityGapCount })}
                </span>
              </div>
            </div>
            <div className="mt-3">
              <SingleOwnerCapabilityRows
                risks={summary.singleOwnerCapabilityRisks}
                skillTitleById={skillTitleById}
                toolTitleById={toolTitleById}
                mcpServerTitleById={mcpServerTitleById}
                focusableAgentIds={focusableAgentIds}
                onFocusAgent={onFocusAgent}
              />
            </div>
          </div>
        </>
      ) : (
        <div className="mt-4 rounded-[12px] border border-dashed border-slate-300 bg-white/80 px-4 py-5 text-sm text-slate-500">
          {emptyLabel}
        </div>
      )}
    </div>
  );
}
