from __future__ import annotations


class RuntimeWorkerAdapter:
    _PHASE_DESCRIPTIONS = {
        "load_run_and_authorize": "Load the run envelope and authorize the lifecycle transition via GovernanceService.",
        "execute_runtime_command": "Dispatch the RuntimeCommandV1 through RuntimeApplicationService and execute the orchestration graph.",
        "report_result_to_governance": "Record the verification result and transition the run to a terminal lifecycle status.",
    }

    def build_execution_plan(self, *, run_id: str, task_type: str, command_type: str = "start") -> dict[str, object]:
        phases = [
            "load_run_and_authorize",
            "execute_runtime_command",
            "report_result_to_governance",
        ]
        return {
            "run_id": run_id,
            "task_type": task_type,
            "command_type": command_type,
            "governance_phase": phases[0],
            "runtime_phase": phases[1],
            "completion_phase": phases[2],
            "phases": [
                {"name": name, "description": self._PHASE_DESCRIPTIONS[name]}
                for name in phases
            ],
        }
