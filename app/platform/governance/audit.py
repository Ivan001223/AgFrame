from __future__ import annotations


def build_lifecycle_event_details(
    *,
    run_id: str,
    actor: str | None,
    contract_version: str,
    from_status: str | None,
    to_status: str,
    triggered_by: str,
    correlation_id: str | None = None,
) -> dict[str, object]:
    details: dict[str, object] = {
        "run_id": run_id,
        "actor": actor,
        "contract_version": contract_version,
        "from_status": from_status,
        "to_status": to_status,
        "triggered_by": triggered_by,
    }
    if correlation_id is not None:
        details["correlation_id"] = correlation_id
    return details
