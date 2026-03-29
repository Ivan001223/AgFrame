from __future__ import annotations


class VerificationService:
    def build_document_ingest_result(
        self,
        *,
        ok: bool,
        stage: str | None,
        error_code: str | None,
        error_message: str | None,
    ) -> dict[str, object]:
        return {
            "status": "pass" if ok else "fail",
            "checks_run": ["document_ingest_result"],
            "artifacts": {
                "stage": stage,
                "error_code": error_code,
            },
            "summary": "document ingest succeeded" if ok else (error_message or "document ingest failed"),
        }

    def build_approval_checkpoint_result(
        self,
        *,
        ok: bool,
        session_id: str | None,
        approved: bool,
        interrupted: bool | None,
        error_code: str | None,
        error_message: str | None,
    ) -> dict[str, object]:
        return {
            "status": "pass" if ok else "fail",
            "checks_run": ["approval_checkpoint_ready"],
            "artifacts": {
                "session_id": session_id,
                "approved": approved,
                "interrupted": interrupted,
                "error_code": error_code,
            },
            "summary": "approval checkpoint ready" if ok else (error_message or "approval checkpoint not ready"),
        }

    def build_session_resume_result(
        self,
        *,
        ok: bool,
        session_id: str | None,
        interrupted: bool | None,
        error_code: str | None,
        error_message: str | None,
    ) -> dict[str, object]:
        return {
            "status": "pass" if ok else "fail",
            "checks_run": ["session_resume_execution"],
            "artifacts": {
                "session_id": session_id,
                "interrupted": interrupted,
                "error_code": error_code,
            },
            "summary": "session resume succeeded" if ok else (error_message or "session resume failed"),
        }

    def build_agent_orchestration_result(
        self,
        *,
        ok: bool,
        active_agent_ids: list[str],
        blocked_agents: list[dict[str, object]],
        loop_count: int,
        review_agent_enabled: bool,
        error_code: str | None,
        error_message: str | None,
        agent_outputs: dict[str, str] | None = None,
        output_artifacts: dict[str, dict[str, object]] | None = None,
        recovery_mode: str | None = None,
    ) -> dict[str, object]:
        return {
            "status": "pass" if ok else "fail",
            "checks_run": ["agent_orchestration_cycle"],
            "artifacts": {
                "active_agent_ids": active_agent_ids,
                "blocked_agents": blocked_agents,
                "loop_count": loop_count,
                "review_agent_enabled": review_agent_enabled,
                "error_code": error_code,
                "agent_outputs": agent_outputs or {},
                "output_artifacts": output_artifacts or {},
                "recovery_mode": recovery_mode,
            },
            "summary": "agent orchestration completed" if ok else (error_message or "agent orchestration failed"),
        }
