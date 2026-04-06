import { useMessages } from '@/lib/i18n';
import type { CapabilityCoverageSummary } from './diagnostics';
import { HARNESS_MESSAGES } from './messages';
import { formatSkillTitle, formatTemplate } from './utils';

const PREVIEW_LIMIT = 5;

function CapabilityPills({
  values,
  lookup,
  emptyLabel,
  toneClassName,
}: {
  values: string[];
  lookup: Map<string, string>;
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
          {formatSkillTitle(value, lookup)}
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

function SingleOwnerRows({
  entries,
  lookup,
  emptyLabel,
  focusableAgentIds,
  onFocusAgent,
}: {
  entries: CapabilityCoverageSummary['singleOwnerSkills'];
  lookup: Map<string, string>;
  emptyLabel: string;
  focusableAgentIds: Set<string>;
  onFocusAgent: (agentId: string) => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  if (entries.length === 0) {
    return <div className="text-xs text-slate-500">{emptyLabel}</div>;
  }
  const visibleEntries = entries.slice(0, PREVIEW_LIMIT);
  const remainingCount = entries.length - visibleEntries.length;
  return (
    <div className="space-y-2">
      {visibleEntries.map((entry) => {
        const owner = entry.ownerAgents[0] ?? null;
        return (
          <div
            key={entry.capabilityId}
            className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2"
          >
            <div className="min-w-0">
              <div className="text-sm font-semibold text-slate-900">
                {formatSkillTitle(entry.capabilityId, lookup)}
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
        <div className="text-xs text-slate-500">
          {formatTemplate(text.additionalItemsCount, { count: remainingCount })}
        </div>
      ) : null}
    </div>
  );
}

export function GraphCapabilityCoverageSummary({
  summary,
  skillTitleById,
  toolTitleById,
  mcpServerTitleById,
  focusableAgentIds,
  onFocusAgent,
}: {
  summary: CapabilityCoverageSummary;
  skillTitleById: Map<string, string>;
  toolTitleById: Map<string, string>;
  mcpServerTitleById: Map<string, string>;
  focusableAgentIds: Set<string>;
  onFocusAgent: (agentId: string) => void;
}) {
  const text = useMessages(HARNESS_MESSAGES);
  const gapCount =
    summary.missingSkillIds.length + summary.blockedToolIds.length + summary.missingMcpServerIds.length;

  return (
    <div className="mt-4 rounded-[12px] border border-slate-200 bg-slate-50/80 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-slate-900">{text.capabilityCoverageLabel}</div>
          <div className="mt-1 text-sm text-slate-600">{text.capabilityCoverageHint}</div>
        </div>
        <div className="flex flex-wrap gap-2">
          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
            {formatTemplate(text.skillsCountShort, { count: summary.totalSkillCount })}
          </span>
          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
            {formatTemplate(text.toolsCountShort, { count: summary.totalToolCount })}
          </span>
          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
            {formatTemplate(text.mcpServerCountShort, { count: summary.totalMcpCount })}
          </span>
          <span className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-slate-200">
            {formatTemplate(text.capabilityGapCountShort, { count: gapCount })}
          </span>
        </div>
      </div>
      <div className="mt-4 grid gap-3 xl:grid-cols-2">
        <div className="rounded-[12px] border border-white/80 bg-white/80 p-4">
          <div className="space-y-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.sharedSkillCoverageLabel}
              </div>
              <div className="mt-2">
                <CapabilityPills
                  values={summary.sharedSkillIds}
                  lookup={skillTitleById}
                  emptyLabel={text.noSharedSkillCoverage}
                  toneClassName="bg-cyan-50 text-cyan-800 ring-cyan-200"
                />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.sharedToolCoverageLabel}
              </div>
              <div className="mt-2">
                <CapabilityPills
                  values={summary.sharedToolIds}
                  lookup={toolTitleById}
                  emptyLabel={text.noSharedToolCoverage}
                  toneClassName="bg-emerald-50 text-emerald-800 ring-emerald-200"
                />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.sharedMcpCoverageLabel}
              </div>
              <div className="mt-2">
                <CapabilityPills
                  values={summary.sharedMcpServerIds}
                  lookup={mcpServerTitleById}
                  emptyLabel={text.noSharedMcpCoverage}
                  toneClassName="bg-violet-50 text-violet-800 ring-violet-200"
                />
              </div>
            </div>
          </div>
        </div>
        <div className="rounded-[12px] border border-white/80 bg-white/80 p-4">
          <div className="space-y-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.singleOwnerSkillCoverageLabel}
              </div>
              <div className="mt-2">
                <SingleOwnerRows
                  entries={summary.singleOwnerSkills}
                  lookup={skillTitleById}
                  emptyLabel={text.noSingleOwnerSkillCoverage}
                  focusableAgentIds={focusableAgentIds}
                  onFocusAgent={onFocusAgent}
                />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.singleOwnerToolCoverageLabel}
              </div>
              <div className="mt-2">
                <SingleOwnerRows
                  entries={summary.singleOwnerTools}
                  lookup={toolTitleById}
                  emptyLabel={text.noSingleOwnerToolCoverage}
                  focusableAgentIds={focusableAgentIds}
                  onFocusAgent={onFocusAgent}
                />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                {text.singleOwnerMcpCoverageLabel}
              </div>
              <div className="mt-2">
                <SingleOwnerRows
                  entries={summary.singleOwnerMcpServers}
                  lookup={mcpServerTitleById}
                  emptyLabel={text.noSingleOwnerMcpCoverage}
                  focusableAgentIds={focusableAgentIds}
                  onFocusAgent={onFocusAgent}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="mt-4 rounded-[12px] border border-white/80 bg-white/80 p-4">
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
          {text.scopeCapabilityGapsLabel}
        </div>
        {gapCount > 0 ? (
          <div className="mt-3 space-y-3">
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.missingSkillsLabel}</div>
              <div className="mt-2">
                <CapabilityPills
                  values={summary.missingSkillIds}
                  lookup={skillTitleById}
                  emptyLabel={text.none}
                  toneClassName="bg-rose-50 text-rose-800 ring-rose-200"
                />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.policyBlockedToolsLabel}</div>
              <div className="mt-2">
                <CapabilityPills
                  values={summary.blockedToolIds}
                  lookup={toolTitleById}
                  emptyLabel={text.none}
                  toneClassName="bg-amber-50 text-amber-800 ring-amber-200"
                />
              </div>
            </div>
            <div>
              <div className="text-[11px] font-semibold text-slate-500">{text.missingMcpServersLabel}</div>
              <div className="mt-2">
                <CapabilityPills
                  values={summary.missingMcpServerIds}
                  lookup={mcpServerTitleById}
                  emptyLabel={text.none}
                  toneClassName="bg-amber-50 text-amber-800 ring-amber-200"
                />
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-2 text-xs text-slate-500">{text.noScopeCapabilityGaps}</div>
        )}
      </div>
    </div>
  );
}
