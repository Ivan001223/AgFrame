from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config.config_manager import config_manager


_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """获取全局唯一的 SQLAlchemy Engine 实例"""
    global _engine
    if _engine is not None:
        return _engine

    db_config = config_manager.get_config().get("database", {})
    host = db_config.get("host", "localhost")
    port = int(db_config.get("port", 3306))
    user = db_config.get("user", "root")
    password = db_config.get("password", "password")
    db_name = db_config.get("db_name", "agent_app")

    url = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{db_name}"
    _engine = create_engine(
        url,
        pool_pre_ping=True,  # 自动检测并回收失效连接
        pool_recycle=3600,   # 连接最大生命周期
        future=True,
    )
    return _engine


def get_sessionmaker() -> sessionmaker:
    """获取 Session 工厂"""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, future=True)
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
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

