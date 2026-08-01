from __future__ import annotations

import uuid

from app.harness.persistence.stores import HarnessEventStore
from app.infrastructure.database.schema import ensure_schema_if_possible
from app.platform.contracts.event import EventEnvelopeV1


class HarnessEventService:
    def __init__(
        self,
        *,
        event_store: HarnessEventStore | None = None,
        database_optional: bool = False,
    ):
        self.event_store = event_store or HarnessEventStore()
        self.database_optional = database_optional

    def _database_available(self) -> bool:
        if not self.database_optional:
            return True
        return ensure_schema_if_possible()

    def record(
        self,
        *,
        event_type: str,
        event_source: str,
        user_id: str,
        session_id: str | None = None,
        run_id: str | None = None,
        actor: str | None = None,
        details: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if not self._database_available():
            return None
        return self.event_store.create_event(
            event_id=f"he_{uuid.uuid4()}",
            event_type=event_type,
            event_source=event_source,
            user_id=user_id,
            session_id=session_id,
            run_id=run_id,
            actor=actor,
            details_json=details,
        )

    def record_runtime_event(
        self,
        envelope: EventEnvelopeV1,
        *,
        user_id: str = "",
    ) -> dict[str, object] | None:
        """Persist a canonical ``EventEnvelopeV1`` as a harness event."""
        if not self._database_available():
            return None
        return self.event_store.create_event(
            event_id=f"he_{uuid.uuid4()}",
            event_type=envelope.event_type,
            event_source=envelope.source or "platform.runtime",
            user_id=user_id,
            run_id=envelope.aggregate_id or None,
            actor=envelope.actor,
            details_json=dict(envelope.payload),
        )

    def list_for_run(self, *, run_id: str, user_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        if not self._database_available():
            return []
        return self.event_store.list_events(run_id=run_id, user_id=user_id, limit=limit)

    def list_for_session(
        self,
        *,
        session_id: str,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        if not self._database_available():
            return []
        return self.event_store.list_events(session_id=session_id, user_id=user_id, limit=limit)
