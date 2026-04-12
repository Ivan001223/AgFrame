from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine instance."""
    global _engine
    if _engine is not None:
        return _engine

    db_config = settings.database
    explicit_url = str(db_config.url or os.getenv("DATABASE_URL") or "").strip()
    if explicit_url:
        url = explicit_url
    else:
        db_type = db_config.type
        host = db_config.host
        port = db_config.port
        user = db_config.user
        password = db_config.password
        db_name = db_config.db_name

        if db_type in {"postgres", "postgresql"}:
            url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db_name}"
        elif db_type == "mysql":
            url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"
        else:
            raise ValueError(f"Unsupported database.type: {db_type}")

    _engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        future=True,
    )
    return _engine


def get_sessionmaker() -> sessionmaker[Session]:
    """Return the shared SQLAlchemy sessionmaker."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
    return _SessionLocal


@contextmanager
def get_session() -> Iterator[Session]:
    """
    Return a transactional database session context manager.

    Usage:
        with get_session() as session:
            session.add(obj)
    """
    session_factory = get_sessionmaker()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception as exc:
        logger.debug("Database session error, rolling back: %s", exc)
        session.rollback()
        raise
    finally:
        session.close()
