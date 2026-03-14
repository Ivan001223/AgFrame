from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine

logger = logging.getLogger(__name__)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.infrastructure.config.settings import settings

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Engine:
    """获取全局唯一的 SQLAlchemy Engine 实例"""
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
        elif db_type in {"mysql"}:
            url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"
        else:
            raise ValueError(f"不支持的 database.type: {db_type}")

    _engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        future=True,
    )
    return _engine


def get_sessionmaker() -> sessionmaker:
    """获取 Session 工厂"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), 
            autoflush=False, 
            autocommit=False, 
            expire_on_commit=False, 
            future=True
        )
    return _SessionLocal


@contextmanager
def get_session() -> Iterator[Session]:
    """
    获取数据库会话的上下文管理器。
    自动处理事务提交和回滚。
    
    Usage:
        with get_session() as session:
            session.add(obj)
            # 自动 commit
        # 自动 close
    """
    SessionLocal = get_sessionmaker()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.debug(f"Database session error, rolling back: {e}")
        session.rollback()
        raise
    finally:
        session.close()

