from __future__ import annotations


class RuntimeWorkerAdapter:
    def build_execution_plan(self, *, run_id: str, task_type: str) -> dict[str, str]:
        return {
            "run_id": run_id,
            "task_type": task_type,
            "governance_phase": "load_run_and_authorize",
            "runtime_phase": "execute_runtime_command",
            "completion_phase": "report_result_to_governance",
        }
