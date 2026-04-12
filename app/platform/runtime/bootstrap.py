from __future__ import annotations

from app.platform.contracts.runtime_protocol import RuntimeCommandV1
from app.platform.runtime.worker_adapter import RuntimeWorkerAdapter


def build_runtime_execution_plan(*, run_id: str, task_type: str) -> dict[str, str]:
    return RuntimeWorkerAdapter().build_execution_plan(run_id=run_id, task_type=task_type)


def build_runtime_command_for_run(*, run_id: str, task_type: str) -> RuntimeCommandV1:
    return RuntimeCommandV1(
        command_id=f"runtime-command-{run_id}",
        run_id=run_id,
        command_type="start",
        payload={"task_type": task_type, "run_scope": task_type},
    )
