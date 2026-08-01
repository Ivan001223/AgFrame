from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.platform.contracts.runtime_protocol import (
    RuntimeCommandV1,
    RuntimeResultV1,
    RuntimeResumePoint,
    runtime_resume_point_from_payload,
)
from app.platform.runtime.worker_adapter import RuntimeWorkerAdapter

if TYPE_CHECKING:
    from app.harness.runtime.run_service import HarnessRunService


_COMMAND_HANDLERS = ("start", "resume", "step", "cancel")


class RuntimeApplicationService:
    def __init__(
        self,
        *,
        run_service: HarnessRunService | None = None,
        worker_adapter: RuntimeWorkerAdapter | None = None,
    ):
        self._run_service = run_service
        self.worker_adapter = worker_adapter or RuntimeWorkerAdapter()

    @property
    def run_service(self) -> HarnessRunService:
        if self._run_service is None:
            from app.harness.runtime.run_service import build_run_service

            self._run_service = build_run_service()
        return self._run_service

    # -- public API ----------------------------------------------------------

    def accept(self, command: RuntimeCommandV1) -> RuntimeResultV1:
        handlers = {
            "start": self._handle_start,
            "resume": self._handle_resume,
            "step": self._handle_step,
            "cancel": self._handle_cancel,
        }
        handler = handlers.get(command.command_type)
        if handler is None:
            return self._reject(command, f"unsupported command_type: {command.command_type}")
        return handler(command)

    # -- command handlers ----------------------------------------------------

    def _handle_start(self, command: RuntimeCommandV1) -> RuntimeResultV1:
        run = self.run_service.get_run(command.run_id)
        if run is None:
            return self._error(command, "run_not_found", "run not found", resumable=False)

        task_type = str(run.get("task_type") or "")
        run_status = str(run.get("status") or "").strip()
        if run_status in {"completed", "failed"}:
            return self._reject(
                command,
                f"run is already in terminal state: {run_status}",
                resumable=False,
            )

        plan = self.worker_adapter.build_execution_plan(
            run_id=command.run_id,
            task_type=task_type,
        )

        return RuntimeResultV1(
            command_id=command.command_id,
            run_id=command.run_id,
            result_type="execution_ready",
            payload={
                "plan": plan,
                "task_type": task_type,
                "from_status": run_status,
                "input_keys": sorted(self._input_keys(run)),
            },
            resumable=True,
        )

    def _handle_resume(self, command: RuntimeCommandV1) -> RuntimeResultV1:
        run = self.run_service.get_run(command.run_id)
        if run is None:
            return self._error(command, "run_not_found", "run not found", resumable=False)

        task_type = str(run.get("task_type") or "")
        run_status = str(run.get("status") or "").strip()

        resume_payload = dict(command.payload or {}).get("resume_point") or dict(command.payload or {}).get("orchestration_resume")
        resume_point: RuntimeResumePoint | None = None
        if isinstance(resume_payload, dict):
            resume_point = runtime_resume_point_from_payload(resume_payload)

        next_step_index = resume_point.next_step_index if resume_point else 0
        rollback_state = resume_point.rollback_state if resume_point else None

        return RuntimeResultV1(
            command_id=command.command_id,
            run_id=command.run_id,
            result_type="resume_ready",
            payload={
                "task_type": task_type,
                "from_status": run_status,
                "next_step_index": next_step_index,
                "has_rollback_state": rollback_state is not None,
            },
            resumable=True,
        )

    def _handle_step(self, command: RuntimeCommandV1) -> RuntimeResultV1:
        run = self.run_service.get_run(command.run_id)
        if run is None:
            return self._error(command, "run_not_found", "run not found", resumable=False)

        step_payload = dict(command.payload or {})
        step_index = step_payload.get("step_index")

        return RuntimeResultV1(
            command_id=command.command_id,
            run_id=command.run_id,
            result_type="step_acknowledged",
            payload={
                "task_type": str(run.get("task_type") or ""),
                "step_index": step_index,
            },
            resumable=True,
        )

    def _handle_cancel(self, command: RuntimeCommandV1) -> RuntimeResultV1:
        run = self.run_service.get_run(command.run_id)
        if run is None:
            return self._error(command, "run_not_found", "run not found", resumable=False)

        run_status = str(run.get("status") or "").strip()
        if run_status in {"completed", "failed"}:
            return self._reject(
                command,
                f"run is already in terminal state: {run_status}",
                resumable=False,
            )

        self.run_service.mark_failed(
            command.run_id,
            verification_status="fail",
        )

        return RuntimeResultV1(
            command_id=command.command_id,
            run_id=command.run_id,
            result_type="cancelled",
            payload={"from_status": run_status},
            resumable=False,
        )

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _input_keys(run: dict[str, Any]) -> list[str]:
        input_json = run.get("input_json")
        return list(input_json.keys()) if isinstance(input_json, dict) else []

    @staticmethod
    def _reject(
        command: RuntimeCommandV1,
        reason: str,
        *,
        resumable: bool | None = None,
    ) -> RuntimeResultV1:
        return RuntimeResultV1(
            command_id=command.command_id,
            run_id=command.run_id,
            result_type="rejected",
            payload={"reason": reason},
            resumable=resumable,
        )

    @staticmethod
    def _error(
        command: RuntimeCommandV1,
        error_type: str,
        message: str,
        *,
        resumable: bool | None = None,
    ) -> RuntimeResultV1:
        return RuntimeResultV1(
            command_id=command.command_id,
            run_id=command.run_id,
            result_type="error",
            error_type=error_type,
            payload={"message": message},
           resumable=resumable,
        )

