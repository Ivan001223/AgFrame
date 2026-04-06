import { useMessages } from '@/lib/i18n';
import {
  type CollaborationScopeSummary,
  type CoordinationTopologySummary,
} from './diagnostics';
import { HARNESS_MESSAGES } from './messages';
import { formatSkillTitle, formatTemplate } from './utils';

const EMPTY_LOOKUP = new Map<string, string>();
const PREVIEW_LIMIT = 5;

function PreviewPills({
  values,
  emptyLabel,
  toneClassName,
}: {
  values: string[];
  emptyLabel: string;
  toneClassName: string;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  if (values.length === 0) {
    return <div className="text-xs text-slate-500">{emptyLabel}</div>;
  }
  const visibleValues = values.slice(0, PREVIEW_LIMIT);
  const remainingCount = values.length - visibleValues.length;
  return (
    <div className="flex flex-wrap gap-2">
      {visibleValues.map((value) => (
        <span
          key={value}
          className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ring-1 ${toneClassName}`}
        >
          {formatSkillTitle(value, EMPTY_LOOKUP)}
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

function AgentPreviewButtons({
  agents,
  emptyLabel,
  focusableAgentIds,
  onFocusAgent,
}: {
  agents: CoordinationTopologySummary['isolatedAgents'];
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
            className="inline-flex items-center justify-center rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-800 hover:bg-slate-100"
          >
            {formatTemplate(text.focusNodeForRecovery, { name: agent.agentName })}
          </button>
        ) : (
          <span
            key={agent.agentId}
            className="rounded-full bg-slate-100 px-2.5 py-1 text-[10px] font-semibold text-slate-700 ring-1 ring-slate-200"
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

export function GraphCollaborationSummary({
  collaborationSummary,
  topologySummary,
  focusableAgentIds,
  onFocusAgent,
}: {
  collaborationSummary: CollaborationScopeSummary;
  topologySummary: CoordinationTopologySummary;
  focusableAgentIds: Set<string>;
  onFocusAgent: (agentId: string) => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);

  return (
    <div className="mt-4 rounded-[12px] border border-slate-200 bg-slate-50/80 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-900">{text.collaborationTopologyLabel}</div>
          <div className="mt-1 text-sm text-slate-600">{text.collaborationTopologyHint}</div>
        </div>
        <div className="flex flex-wrap gap-2">
          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
            {formatTemplate(text.collaborationSourceAgentsCountLabel, {
              count: collaborationSummary.actionableSourceAgentCount,
            })}
          </span>
          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
            {formatTemplate(text.collaborationLaneCountLabel, { count: topologySummary.totalLaneCount })}
          </span>
          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
            {formatTemplate(text.weakEdgesCountLabel, { count: collaborationSummary.weakEdgeCount })}
          </span>
          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
            {formatTemplate(text.bestNextCountLabel, { count: collaborationSummary.bestNextCount })}
          </span>
        </div>
      </div>
      <div className="mt-4 grid gap-3 xl:grid-cols-2">
        <div className="rounded-[12px] border border-white/80 bg-white/80 p-4">
          <div className="flex flex-wrap gap-2">
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
              {text.sharedDelegationLanesLabel} · {topologySummary.sharedLaneCount}
            </span>
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
              {text.singleOwnerLanesLabel} · {topologySummary.singleOwnerLaneCount}
            </span>
          </div>
          <div className="mt-3 space-y-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.sharedDelegationLanesLabel}
              </div>
              <div className="mt-2">
                <PreviewPills
                  values={topologySummary.sharedLaneIds}
                  emptyLabel={text.noSharedDelegationLanes}
                  toneClassName="bg-sky-50 text-sky-800 ring-sky-200"
                />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.singleOwnerLanesLabel}
              </div>
              <div className="mt-2">
                <PreviewPills
                  values={topologySummary.singleOwnerLaneIds}
                  emptyLabel={text.noSingleOwnerLanes}
                  toneClassName="bg-amber-50 text-amber-800 ring-amber-200"
                />
              </div>
            </div>
          </div>
        </div>
        <div className="rounded-[12px] border border-white/80 bg-white/80 p-4">
          <div className="flex flex-wrap gap-2">
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
              {text.isolatedAgentsLabel} · {topologySummary.isolatedAgentCount}
            </span>
            <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
              {text.underconnectedAgentsLabel} · {topologySummary.underconnectedAgentCount}
            </span>
          </div>
          <div className="mt-3 space-y-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.isolatedAgentsLabel}
              </div>
              <div className="mt-2">
                <AgentPreviewButtons
                  agents={topologySummary.isolatedAgents}
                  emptyLabel={text.noIsolatedAgents}
                  focusableAgentIds={focusableAgentIds}
                  onFocusAgent={onFocusAgent}
                />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.underconnectedAgentsLabel}
              </div>
              <div className="mt-2">
                <AgentPreviewButtons
                  agents={topologySummary.underconnectedAgents}
                  emptyLabel={text.noUnderconnectedAgents}
                  focusableAgentIds={focusableAgentIds}
                  onFocusAgent={onFocusAgent}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="mt-4 text-xs leading-5 text-slate-500">
        <span className="font-semibold text-slate-700">{text.collaborationFocusPreviewLabel}:</span>{' '}
        {collaborationSummary.focusPreview.length > 0
          ? collaborationSummary.focusPreview.join(' · ')
          : text.noCollaborationFocusPreview}
      </div>
    </div>
  );
}
