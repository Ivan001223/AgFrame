'use client';

import Link from 'next/link';
import { useMemo, useState } from 'react';
import { CheckCircle2, Clock3, PlusCircle, RefreshCw, RotateCcw, ShieldCheck, ShieldX, Sparkles, Workflow } from 'lucide-react';
import {
  HarnessApprovalDTO,
  HarnessEventDTO,
  HarnessPolicyDTO,
  HarnessRunSummaryDTO,
  useHarnessApprovalMutation,
  useHarnessCreateRunMutation,
  useHarnessPoliciesQuery,
  useHarnessRunDetailQuery,
  useHarnessRetryRunMutation,
  useHarnessRunsQuery,
} from '@/domains/harness/hooks';

function formatTimestamp(value?: number | null) {
  if (!value) {
    return 'Not recorded';
  }
  const normalized = value > 1_000_000_000_000 ? value : value * 1000;
  return new Date(normalized).toLocaleString();
}

function parseOptionalObject(value: string): Record<string, unknown> | undefined {
  const text = value.trim();
  if (!text) {
    return undefined;
  }
  const parsed = JSON.parse(text) as unknown;
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Metadata must be a JSON object.');
  }
  return parsed as Record<string, unknown>;
}

function statusTone(status?: string | null) {
  switch (status) {
    case 'completed':
    case 'approved':
    case 'pass':
      return 'bg-emerald-100 text-emerald-900 ring-emerald-200';
    case 'failed':
    case 'rejected':
    case 'fail':
      return 'bg-rose-100 text-rose-900 ring-rose-200';
    case 'waiting_approval':
    case 'pending':
    case 'partial':
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

function ApprovalSummary({ approval }: { approval?: HarnessApprovalDTO | null }) {
  if (!approval) {
    return <p className="text-sm text-slate-500">No approval record yet.</p>;
  }

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
      {approval.reason ? (
        <div className="rounded-2xl bg-amber-50 p-4 text-sm text-amber-900">{approval.reason}</div>
      ) : null}
      {approval.comment ? (
        <div className="rounded-2xl bg-slate-100 p-4 text-sm text-slate-700">{approval.comment}</div>
      ) : null}
    </div>
  );
}

function EventRow({ event }: { event: HarnessEventDTO }) {
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
    </button>
  );
}

export default function HarnessPage() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [approvalComment, setApprovalComment] = useState('');
  const [createTaskType, setCreateTaskType] = useState('document_ingest');
  const [createFilePath, setCreateFilePath] = useState('');
  const [createSessionId, setCreateSessionId] = useState('');
  const [createMetadata, setCreateMetadata] = useState('');
  const [createError, setCreateError] = useState<string | null>(null);
  const runsQuery = useHarnessRunsQuery();
  const policiesQuery = useHarnessPoliciesQuery();
  const approvalMutation = useHarnessApprovalMutation();
  const createRunMutation = useHarnessCreateRunMutation();
  const retryRunMutation = useHarnessRetryRunMutation();
  const runs = useMemo(() => runsQuery.data?.runs ?? [], [runsQuery.data?.runs]);
  const policies = useMemo(() => policiesQuery.data?.policies ?? [], [policiesQuery.data?.policies]);
  const activeRunId = useMemo(() => {
    if (!runs.length) {
      return null;
    }
    if (selectedRunId && runs.some((run) => run.run_id === selectedRunId)) {
      return selectedRunId;
    }
    return runs[0].run_id;
  }, [runs, selectedRunId]);
  const detailQuery = useHarnessRunDetailQuery(activeRunId);

  const selectedRun = detailQuery.data;
  const pendingApproval = useMemo(() => {
    if (!selectedRun?.latest_approval) {
      return null;
    }
    return selectedRun.latest_approval.status === 'pending' ? selectedRun.latest_approval : null;
  }, [selectedRun]);
  const selectedPolicy = useMemo(
    () => policies.find((policy) => policy.task_type === createTaskType) ?? null,
    [policies, createTaskType]
  );

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

  const handleCreateRun = () => {
    setCreateError(null);
    try {
      const metadata = parseOptionalObject(createMetadata);
      const cleanSessionId = createSessionId.trim();
      const input =
        createTaskType === 'session_resume_approval'
          ? { session_id: cleanSessionId }
          : { file_path: createFilePath.trim() };
      const missingRequired =
        createTaskType === 'session_resume_approval' ? !cleanSessionId : !String(input.file_path || '').trim();
      if (missingRequired) {
        setCreateError(
          createTaskType === 'session_resume_approval'
            ? 'Session resume approval requires a session id.'
            : 'Document ingest requires a file path.'
        );
        return;
      }
      createRunMutation.mutate(
        {
          taskType: createTaskType,
          input,
          sessionId: cleanSessionId || undefined,
          metadata,
        },
        {
          onSuccess: (payload) => {
            setSelectedRunId(payload.run_id);
            setApprovalComment('');
          },
          onError: (error) => {
            setCreateError(error instanceof Error ? error.message : 'Failed to create harness run.');
          },
        }
      );
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : 'Invalid metadata JSON.');
    }
  };

  const handleRetry = () => {
    if (!selectedRun) {
      return;
    }
    retryRunMutation.mutate(
      { runId: selectedRun.run_id },
      {
        onSuccess: (payload) => {
          setSelectedRunId(payload.run_id);
        },
      }
    );
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.18),_transparent_28%),linear-gradient(180deg,#f8fafc_0%,#ecfeff_42%,#f8fafc_100%)]">
      <div className="mx-auto max-w-7xl px-6 py-10 lg:px-8">
        <div className="flex flex-col gap-4 rounded-[32px] border border-white/60 bg-white/80 p-8 shadow-[0_24px_80px_-48px_rgba(15,23,42,0.45)] backdrop-blur md:flex-row md:items-end md:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full bg-slate-900 px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] text-cyan-200">
              <Workflow className="h-3.5 w-3.5" />
              Harness Control Plane
            </div>
            <h1 className="mt-4 font-serif text-4xl text-slate-950">Runs, approvals, verification, and evidence in one place.</h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
              This dashboard turns the agent runtime into something inspectable: which run is blocked, what got verified, and how the lifecycle actually moved.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link
              href="/chat"
              className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-700 hover:bg-slate-50"
            >
              Back to chat
            </Link>
            <button
              type="button"
              onClick={() => {
                runsQuery.refetch();
                detailQuery.refetch();
              }}
              className="inline-flex items-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white hover:bg-slate-800"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </button>
          </div>
        </div>

        <div className="mt-8 grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
          <section className="rounded-[28px] border border-slate-200 bg-white/90 p-5 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
            <div className="rounded-[28px] border border-cyan-200 bg-cyan-50/80 p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold text-slate-900">Create run</div>
                  <div className="mt-1 text-sm text-slate-600">Start a harness-managed task with explicit policy and evidence.</div>
                </div>
                <PlusCircle className="h-5 w-5 text-cyan-600" />
              </div>
              <div className="mt-4 space-y-3">
                <label className="block text-sm font-medium text-slate-800">
                  Task type
                  <select
                    value={createTaskType}
                    onChange={(event) => setCreateTaskType(event.target.value)}
                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                  >
                    {(policies.length ? policies : [{ task_type: 'document_ingest' }, { task_type: 'session_resume_approval' } as HarnessPolicyDTO]).map((policy) => (
                      <option key={policy.task_type} value={policy.task_type}>
                        {policy.task_type}
                      </option>
                    ))}
                  </select>
                </label>
                {createTaskType === 'session_resume_approval' ? (
                  <label className="block text-sm font-medium text-slate-800">
                    Session id
                    <input
                      value={createSessionId}
                      onChange={(event) => setCreateSessionId(event.target.value)}
                      placeholder="resume target session id"
                      className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                    />
                  </label>
                ) : (
                  <label className="block text-sm font-medium text-slate-800">
                    File path
                    <input
                      value={createFilePath}
                      onChange={(event) => setCreateFilePath(event.target.value)}
                      placeholder="/absolute/path/to/document.pdf"
                      className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                    />
                  </label>
                )}
                <label className="block text-sm font-medium text-slate-800">
                  Metadata JSON
                  <textarea
                    value={createMetadata}
                    onChange={(event) => setCreateMetadata(event.target.value)}
                    rows={4}
                    placeholder='{"requested_by":"dashboard"}'
                    className="mt-2 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-200"
                  />
                </label>
                {selectedPolicy ? (
                  <div className="rounded-2xl bg-white/90 p-4 text-sm text-slate-700">
                    <div>approval_required: <span className="font-semibold text-slate-900">{String(selectedPolicy.approval_required)}</span></div>
                    <div className="mt-1">retry_budget: <span className="font-semibold text-slate-900">{selectedPolicy.retry_budget ?? 0}</span></div>
                    <div className="mt-1">allowed_tools: <span className="font-semibold text-slate-900">{selectedPolicy.allowed_tools?.join(', ') || 'none'}</span></div>
                  </div>
                ) : null}
                {createError ? <div className="text-sm text-rose-700">{createError}</div> : null}
                <button
                  type="button"
                  onClick={handleCreateRun}
                  disabled={createRunMutation.isPending}
                  className="inline-flex w-full items-center justify-center gap-2 rounded-2xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white hover:bg-slate-800 disabled:opacity-50"
                >
                  <PlusCircle className="h-4 w-4" />
                  {createRunMutation.isPending ? 'Creating...' : 'Create harness run'}
                </button>
              </div>
            </div>

            <div className="mt-5 flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-slate-900">Active runs</div>
                <div className="mt-1 text-sm text-slate-500">{runs.length} visible harness runs</div>
              </div>
              <Sparkles className="h-5 w-5 text-cyan-500" />
            </div>

            {runsQuery.isLoading ? (
              <div className="mt-6 text-sm text-slate-500">Loading harness runs...</div>
            ) : runs.length === 0 ? (
              <div className="mt-6 rounded-3xl border border-dashed border-slate-300 bg-slate-50 p-8 text-sm text-slate-500">
                No harness runs yet. Once a task enters the control plane, it will appear here with approval and verification state.
              </div>
            ) : (
              <div className="mt-6 space-y-3">
                {runs.map((run) => (
                  <RunRow
                    key={run.run_id}
                    run={run}
                    selected={run.run_id === activeRunId}
                    onSelect={() => setSelectedRunId(run.run_id)}
                  />
                ))}
              </div>
            )}
          </section>

          <section className="space-y-6">
            {!activeRunId || !selectedRun ? (
              <div className="rounded-[28px] border border-dashed border-slate-300 bg-white/70 p-10 text-sm text-slate-500">
                Select a run to inspect its approval state, verification output, and event evidence.
              </div>
            ) : (
              <>
                <div className="rounded-[28px] border border-slate-200 bg-white/90 p-6 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                  <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                    <div>
                      <div className="text-xs uppercase tracking-[0.24em] text-slate-400">Run detail</div>
                      <h2 className="mt-2 text-2xl font-semibold text-slate-950">{selectedRun.task_type || 'unknown_task'}</h2>
                      <div className="mt-2 text-sm text-slate-500">{selectedRun.run_id}</div>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <StatusPill value={selectedRun.status} />
                      <StatusPill value={selectedRun.latest_verification?.status || selectedRun.verification_status} />
                      {selectedRun.latest_approval ? <StatusPill value={selectedRun.latest_approval.status} /> : null}
                      {selectedRun.can_retry ? <StatusPill value="retry_available" /> : null}
                    </div>
                  </div>

                  <dl className="mt-6 grid gap-4 text-sm text-slate-700 md:grid-cols-2 xl:grid-cols-4">
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <dt className="text-slate-500">Session</dt>
                      <dd className="mt-1 font-medium text-slate-900">{selectedRun.session_id || 'n/a'}</dd>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <dt className="text-slate-500">Current step</dt>
                      <dd className="mt-1 font-medium text-slate-900">{selectedRun.current_step || 'idle'}</dd>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <dt className="text-slate-500">Policy</dt>
                      <dd className="mt-1 font-medium text-slate-900">{selectedRun.policy_id || 'unknown'}</dd>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <dt className="text-slate-500">Resume count</dt>
                      <dd className="mt-1 font-medium text-slate-900">{selectedRun.resume_count || 0}</dd>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <dt className="text-slate-500">Retry count</dt>
                      <dd className="mt-1 font-medium text-slate-900">{selectedRun.retry_count || 0}</dd>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <dt className="text-slate-500">Retry budget</dt>
                      <dd className="mt-1 font-medium text-slate-900">{selectedRun.policy?.retry_budget ?? 0}</dd>
                    </div>
                    <div className="rounded-2xl bg-slate-50 p-4">
                      <dt className="text-slate-500">Allowed tools</dt>
                      <dd className="mt-1 font-medium text-slate-900">{selectedRun.policy?.allowed_tools?.join(', ') || 'none'}</dd>
                    </div>
                  </dl>
                  {selectedRun.can_retry ? (
                    <div className="mt-6">
                      <button
                        type="button"
                        onClick={handleRetry}
                        disabled={retryRunMutation.isPending}
                        className="inline-flex items-center gap-2 rounded-2xl bg-sky-600 px-4 py-3 text-sm font-semibold text-white hover:bg-sky-500 disabled:opacity-50"
                      >
                        <RotateCcw className="h-4 w-4" />
                        {retryRunMutation.isPending ? 'Retrying...' : 'Retry run'}
                      </button>
                    </div>
                  ) : null}
                </div>

                <div className="grid gap-6 lg:grid-cols-2">
                  <div className="rounded-[28px] border border-slate-200 bg-white/90 p-6 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
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
                            Approve
                          </button>
                          <button
                            type="button"
                            disabled={approvalMutation.isPending}
                            onClick={() => handleDecision(false)}
                            className="inline-flex flex-1 items-center justify-center gap-2 rounded-2xl bg-rose-600 px-4 py-3 text-sm font-semibold text-white hover:bg-rose-500 disabled:opacity-50"
                          >
                            <ShieldX className="h-4 w-4" />
                            Reject
                          </button>
                        </div>
                      </div>
                    ) : null}
                  </div>

                  <div className="rounded-[28px] border border-slate-200 bg-white/90 p-6 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                      <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                      Verification summary
                    </div>

                    {!selectedRun.latest_verification ? (
                      <div className="mt-5 text-sm text-slate-500">No verification result recorded yet.</div>
                    ) : (
                      <div className="mt-5 space-y-4">
                        <div className="flex items-center justify-between gap-3">
                          <div className="text-sm font-semibold text-slate-900">{selectedRun.latest_verification.summary || 'Verification recorded'}</div>
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
                        </dl>
                        <JsonBlock value={selectedRun.latest_verification.artifacts_json} />
                      </div>
                    )}
                  </div>
                </div>

                <div className="grid gap-6 lg:grid-cols-2">
                  <div className="rounded-[28px] border border-slate-200 bg-white/90 p-6 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                      <Clock3 className="h-4 w-4 text-sky-600" />
                      Input payload
                    </div>
                    <div className="mt-5">
                      <JsonBlock value={selectedRun.input_json} />
                    </div>
                  </div>
                  <div className="rounded-[28px] border border-slate-200 bg-white/90 p-6 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                      <Sparkles className="h-4 w-4 text-fuchsia-600" />
                      Metadata
                    </div>
                    <div className="mt-5">
                      <JsonBlock value={selectedRun.metadata_json || null} />
                    </div>
                  </div>
                </div>

                <div className="rounded-[28px] border border-slate-200 bg-white/90 p-6 shadow-[0_18px_60px_-45px_rgba(15,23,42,0.45)]">
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
          </section>
        </div>
      </div>
    </div>
  );
}
