from __future__ import annotations

from sqlalchemy import text

from app.core.database.models import Base
from app.core.database.orm import get_engine


def ensure_schema() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def is_database_ready() -> bool:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def ensure_schema_if_possible() -> bool:
    if not is_database_ready():
        return False
    ensure_schema()
    return True
