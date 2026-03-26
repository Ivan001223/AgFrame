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
