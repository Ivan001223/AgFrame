from __future__ import annotations

from pydantic import BaseModel

from app.platform.contracts.versioning import EVENT_CONTRACT_VERSION


class EventEnvelopeV1(BaseModel):
    version: str = EVENT_CONTRACT_VERSION
    event_id: str
    event_type: str
    aggregate_id: str
    payload: dict[str, object]
    actor: str | None = None
    source: str | None = None
