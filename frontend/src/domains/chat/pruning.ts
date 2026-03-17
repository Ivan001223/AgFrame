export type SavingsSummary = {
  saved?: number;
  saved_ratio?: number;
};

export type PruningBlockSummary = {
  method?: string;
  scoring_source?: string;
  ratio?: number;
  items?: number;
  items_pruned?: number;
  char_count_before?: number;
  char_count_after?: number;
  char_savings?: SavingsSummary;
  line_savings?: SavingsSummary;
};

export type PruningStageSummary = {
  focus_hint?: string;
  method?: string;
  scoring_source?: string;
  docs?: PruningBlockSummary;
  memories?: PruningBlockSummary;
};

export type ContextPruningSummary = {
  focus_hint?: string;
  candidatePruning?: PruningBlockSummary;
  promptPruning?: PruningStageSummary;
};

export type PruningDisplayStats = {
  keptText: string;
  savedText: string;
  itemsText: string | null;
};

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : null;
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function normalizeSavings(value: unknown): SavingsSummary | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  return {
    saved: asNumber(record.saved),
    saved_ratio: asNumber(record.saved_ratio),
  };
}

function normalizePruningBlock(value: unknown): PruningBlockSummary | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  return {
    method: asString(record.method),
    scoring_source: asString(record.scoring_source),
    ratio: asNumber(record.ratio),
    items: asNumber(record.items),
    items_pruned: asNumber(record.items_pruned),
    char_count_before: asNumber(record.char_count_before),
    char_count_after: asNumber(record.char_count_after),
    char_savings: normalizeSavings(record.char_savings),
    line_savings: normalizeSavings(record.line_savings),
  };
}

function normalizePruningStage(value: unknown): PruningStageSummary | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  return {
    focus_hint: asString(record.focus_hint),
    method: asString(record.method),
    scoring_source: asString(record.scoring_source),
    docs: normalizePruningBlock(record.docs),
    memories: normalizePruningBlock(record.memories),
  };
}

export function extractContextPruning(payload: unknown): ContextPruningSummary | null {
  const root = asRecord(payload);
  const output = asRecord(root?.output) ?? root;
  const contextRecord = asRecord(output?.context);
  if (!contextRecord) {
    return null;
  }
  const promptPruning = normalizePruningStage(contextRecord.context_pruning);
  const retrievalDebug = asRecord(contextRecord.retrieval_debug);
  const candidatePruning = normalizePruningBlock(retrievalDebug?.candidate_pruning);

  if (!promptPruning && !candidatePruning) {
    return null;
  }

  return {
    focus_hint:
      promptPruning?.focus_hint ||
      asString(asRecord(retrievalDebug?.candidate_pruning)?.focus_hint) ||
      '',
    candidatePruning,
    promptPruning,
  };
}

export function formatPruningBlock(block: PruningBlockSummary | undefined): PruningDisplayStats {
  const before = block?.char_count_before ?? 0;
  const after = block?.char_count_after ?? 0;
  const saved = block?.char_savings?.saved ?? 0;
  const savedRatio = Math.round((block?.char_savings?.saved_ratio ?? 0) * 100);
  const items = block?.items;
  const itemsPruned = block?.items_pruned;
  return {
    keptText: `${after} / ${before} chars`,
    savedText: `${saved} chars · ${savedRatio}%`,
    itemsText:
      items === undefined && itemsPruned === undefined
        ? null
        : `${itemsPruned ?? 0} / ${items ?? 0} items`,
  };
}
