'use client';

import { useState } from 'react';
import { useTaskIncidentsQuery, useTaskSummaryQuery } from '@/domains/tasks/hooks';
import { Clock, AlertTriangle, Filter, Activity, Archive } from 'lucide-react';

export default function TasksPage() {
  const [showHandled, setShowHandled] = useState(false);
  const { data: summary, isLoading, isError } = useTaskSummaryQuery();
  const { data: incidents } = useTaskIncidentsQuery({ handled: showHandled, archived: false });

  return (
    <div className="p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Task Operations</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Monitor backend task health and active incidents (auto-refreshes every 10s).
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-gray-500" />
          <select
            value={showHandled ? 'handled' : 'open'}
            onChange={(e) => setShowHandled(e.target.value === 'handled')}
            className="block w-40 rounded-md border-gray-300 py-2 pl-3 pr-10 text-base focus:border-indigo-500 focus:outline-none focus:ring-indigo-500 sm:text-sm dark:bg-gray-800 dark:border-gray-700 dark:text-white"
          >
            <option value="open">Open Incidents</option>
            <option value="handled">Handled Incidents</option>
          </select>
        </div>
      </div>

      <div className="mb-6 grid gap-4 md:grid-cols-4">
        <div className="rounded-lg bg-white p-4 shadow dark:bg-gray-800">
          <div className="text-sm text-gray-500 dark:text-gray-400">Tracked Tasks</div>
          <div className="mt-2 text-2xl font-semibold text-gray-900 dark:text-white">{summary?.total ?? 0}</div>
        </div>
        <div className="rounded-lg bg-white p-4 shadow dark:bg-gray-800">
          <div className="text-sm text-gray-500 dark:text-gray-400">Running</div>
          <div className="mt-2 text-2xl font-semibold text-blue-600">{summary?.status_counts.running ?? 0}</div>
        </div>
        <div className="rounded-lg bg-white p-4 shadow dark:bg-gray-800">
          <div className="text-sm text-gray-500 dark:text-gray-400">Queued</div>
          <div className="mt-2 text-2xl font-semibold text-gray-900 dark:text-white">{summary?.status_counts.queued ?? 0}</div>
        </div>
        <div className="rounded-lg bg-white p-4 shadow dark:bg-gray-800">
          <div className="text-sm text-gray-500 dark:text-gray-400">Failed</div>
          <div className="mt-2 text-2xl font-semibold text-red-600">{summary?.status_counts.failed ?? 0}</div>
        </div>
      </div>

      <div className="overflow-hidden rounded-lg bg-white shadow dark:bg-gray-800">
        <ul className="divide-y divide-gray-200 dark:divide-gray-700">
          {isLoading ? (
            <li className="p-6 text-center text-gray-500 dark:text-gray-400">Loading tasks...</li>
          ) : isError ? (
            <li className="p-6 text-center text-red-500">Failed to load tasks</li>
          ) : !incidents || incidents.length === 0 ? (
            <li className="p-6 text-center text-gray-500">No incidents found.</li>
          ) : (
            incidents.map((incident) => (
              <li key={incident.incident_id || incident.task_id} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-700/50">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4 min-w-0 pr-4">
                    <div className="flex-shrink-0">
                      {incident.handled ? (
                        <Archive className="w-5 h-5 text-gray-400" />
                      ) : (
                        <AlertTriangle className="w-5 h-5 text-red-500" />
                      )}
                    </div>
                    <div>
                      <h3 className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {incident.title || incident.error_code || 'Task incident'}
                      </h3>
                      <div className="mt-1 flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
                        <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                          incident.handled
                            ? 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
                            : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
                        }`}>
                          {incident.handled ? 'handled' : 'open'}
                        </span>
                        <span>Task: <code className="font-mono">{incident.task_id.slice(0, 8)}...</code></span>
                        {incident.error_code && <span>Code: {incident.error_code}</span>}
                      </div>
                    </div>
                  </div>
                </div>
                
                {(incident.user_message || incident.suggested_action) && (
                  <div className="mt-3 bg-red-50 dark:bg-red-900/10 p-3 rounded-md border border-red-100 dark:border-red-900/30">
                    <p className="flex items-start gap-2 text-sm text-red-700 dark:text-red-400 break-words">
                      <Activity className="mt-0.5 h-4 w-4 shrink-0" />
                      <span>{incident.user_message || 'No user-facing message provided.'}</span>
                    </p>
                    {incident.suggested_action && (
                      <p className="mt-2 pl-6 text-xs text-red-600 dark:text-red-300">
                        {incident.suggested_action}
                      </p>
                    )}
                  </div>
                )}
              </li>
            ))
          )}
        </ul>
      </div>

      {summary && summary.suspected_timeouts.length > 0 && (
        <div className="mt-6 rounded-lg bg-amber-50 p-4 text-sm text-amber-900 dark:bg-amber-950/30 dark:text-amber-200">
          <div className="mb-2 flex items-center gap-2 font-medium">
            <Clock className="h-4 w-4" />
            Suspected timeouts
          </div>
          <div className="space-y-1">
            {summary.suspected_timeouts.map((task) => (
              <div key={task.task_id}>
                <code className="font-mono">{task.task_id}</code> has been running for {task.age_seconds ?? 0}s
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
