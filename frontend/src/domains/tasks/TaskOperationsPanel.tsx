'use client';

import { useState } from 'react';
import { Activity, AlertTriangle, Archive, Clock, Filter } from 'lucide-react';
import { useTaskIncidentsQuery, useTaskSummaryQuery } from '@/domains/tasks/hooks';
import { formatMessage, useMessages } from '@/lib/i18n';

const MESSAGES = {
  title: { en: 'Task Operations', zh: '任务运维' },
  description: {
    en: 'Monitor backend async jobs such as document ingestion and re-indexing. This is an operator panel, not your business task list.',
    zh: '监控文档摄取、重建索引等后台异步任务。这是运维面板，不是你的业务任务列表。',
  },
  explanation: {
    en: 'It helps you judge whether uploads and indexing are healthy, whether files are stuck, and whether someone needs to intervene.',
    zh: '它帮助你判断上传和索引链路是否健康、文件有没有卡住，以及是否需要人工介入。',
  },
  openIncidents: { en: 'Open Incidents', zh: '未处理事件' },
  handledIncidents: { en: 'Handled Incidents', zh: '已处理事件' },
  trackedTasks: { en: 'Tracked Tasks', zh: '已跟踪任务' },
  running: { en: 'Running', zh: '运行中' },
  queued: { en: 'Queued', zh: '排队中' },
  failed: { en: 'Failed', zh: '失败' },
  loading: { en: 'Loading task operations...', zh: '正在加载任务运维数据...' },
  loadFailed: { en: 'Failed to load task operations.', zh: '加载任务运维数据失败。' },
  noIncidents: { en: 'No incidents found.', zh: '没有发现异常事件。' },
  taskIncident: { en: 'Task incident', zh: '任务事件' },
  handled: { en: 'handled', zh: '已处理' },
  open: { en: 'open', zh: '未处理' },
  task: { en: 'Task', zh: '任务' },
  code: { en: 'Code', zh: '代码' },
  noMessage: { en: 'No user-facing message provided.', zh: '没有提供用户可读说明。' },
  suspectedTimeouts: { en: 'Suspected timeouts', zh: '疑似超时' },
  inSettings: {
    en: 'Task operations now live inside Settings so operators can manage workspace behavior and backend job health in one place.',
    zh: '任务运维现在并入了设置页，方便把工作区配置和后台任务健康度放在同一个地方查看。',
  },
  timeoutAge: { en: '{seconds}s old', zh: '已等待 {seconds} 秒' },
} as const;

export function TaskOperationsPanel({ compact = false }: { compact?: boolean }) {
  const [showHandled, setShowHandled] = useState(false);
  const { data: summary, isLoading, isError } = useTaskSummaryQuery();
  const { data: incidents } = useTaskIncidentsQuery({ handled: showHandled, archived: false });
  const text = useMessages(MESSAGES);

  return (
    <div className={compact ? '' : 'p-8'}>
      <div className="mb-8 flex items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">{text.title}</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">{text.description}</p>
        </div>

        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-gray-500" />
          <select
            value={showHandled ? 'handled' : 'open'}
            onChange={(event) => setShowHandled(event.target.value === 'handled')}
            className="block w-40 rounded-md border-gray-300 py-2 pl-3 pr-10 text-base focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 sm:text-sm dark:border-gray-700 dark:bg-gray-800 dark:text-white"
          >
            <option value="open">{text.openIncidents}</option>
            <option value="handled">{text.handledIncidents}</option>
          </select>
        </div>
      </div>

      <div className="mb-6 rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900 dark:border-sky-900/40 dark:bg-sky-950/30 dark:text-sky-100">
        <div className="font-medium">{text.explanation}</div>
        <div className="mt-1 opacity-90">{text.inSettings}</div>
      </div>

      <div className="mb-6 grid gap-4 md:grid-cols-4">
        <div className="rounded-lg bg-white p-4 shadow dark:bg-gray-800">
          <div className="text-sm text-gray-500 dark:text-gray-400">{text.trackedTasks}</div>
          <div className="mt-2 text-2xl font-semibold text-gray-900 dark:text-white">{summary?.total ?? 0}</div>
        </div>
        <div className="rounded-lg bg-white p-4 shadow dark:bg-gray-800">
          <div className="text-sm text-gray-500 dark:text-gray-400">{text.running}</div>
          <div className="mt-2 text-2xl font-semibold text-blue-600">{summary?.status_counts.running ?? 0}</div>
        </div>
        <div className="rounded-lg bg-white p-4 shadow dark:bg-gray-800">
          <div className="text-sm text-gray-500 dark:text-gray-400">{text.queued}</div>
          <div className="mt-2 text-2xl font-semibold text-gray-900 dark:text-white">{summary?.status_counts.queued ?? 0}</div>
        </div>
        <div className="rounded-lg bg-white p-4 shadow dark:bg-gray-800">
          <div className="text-sm text-gray-500 dark:text-gray-400">{text.failed}</div>
          <div className="mt-2 text-2xl font-semibold text-red-600">{summary?.status_counts.failed ?? 0}</div>
        </div>
      </div>

      <div className="overflow-hidden rounded-lg bg-white shadow dark:bg-gray-800">
        <ul className="divide-y divide-gray-200 dark:divide-gray-700">
          {isLoading ? (
            <li className="p-6 text-center text-gray-500 dark:text-gray-400">{text.loading}</li>
          ) : isError ? (
            <li className="p-6 text-center text-red-500">{text.loadFailed}</li>
          ) : !incidents || incidents.length === 0 ? (
            <li className="p-6 text-center text-gray-500">{text.noIncidents}</li>
          ) : (
            incidents.map((incident) => (
              <li key={incident.incident_id || incident.task_id} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                <div className="flex items-center justify-between">
                  <div className="flex min-w-0 items-center gap-4 pr-4">
                    <div className="flex-shrink-0">
                      {incident.handled ? (
                        <Archive className="h-5 w-5 text-gray-400" />
                      ) : (
                        <AlertTriangle className="h-5 w-5 text-red-500" />
                      )}
                    </div>
                    <div>
                      <h3 className="truncate text-sm font-medium text-gray-900 dark:text-white">
                        {incident.title || incident.error_code || text.taskIncident}
                      </h3>
                      <div className="mt-1 flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
                        <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                          incident.handled
                            ? 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
                            : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                        }`}>
                          {incident.handled ? text.handled : text.open}
                        </span>
                        <span>{text.task}: <code className="font-mono">{incident.task_id.slice(0, 8)}...</code></span>
                        {incident.error_code ? <span>{text.code}: {incident.error_code}</span> : null}
                      </div>
                    </div>
                  </div>
                </div>

                {incident.user_message || incident.suggested_action ? (
                  <div className="mt-3 rounded-md border border-red-100 bg-red-50 p-3 dark:border-red-900/30 dark:bg-red-900/10">
                    <p className="flex items-start gap-2 break-words text-sm text-red-700 dark:text-red-400">
                      <Activity className="mt-0.5 h-4 w-4 shrink-0" />
                      <span>{incident.user_message || text.noMessage}</span>
                    </p>
                    {incident.suggested_action ? (
                      <p className="mt-2 pl-6 text-xs text-red-600 dark:text-red-300">{incident.suggested_action}</p>
                    ) : null}
                  </div>
                ) : null}
              </li>
            ))
          )}
        </ul>
      </div>

      {summary && summary.suspected_timeouts.length > 0 ? (
        <div className="mt-6 rounded-lg bg-amber-50 p-4 text-sm text-amber-900 dark:bg-amber-950/30 dark:text-amber-200">
          <div className="mb-2 flex items-center gap-2 font-medium">
            <Clock className="h-4 w-4" />
            {text.suspectedTimeouts}
          </div>
          <div className="space-y-1">
            {summary.suspected_timeouts.map((task) => (
              <div key={task.task_id}>
                <code className="font-mono">{task.task_id}</code>{' '}
                {formatMessage(text.timeoutAge, { seconds: task.age_seconds ?? 0 })}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
