from __future__ import annotations

from sqlalchemy import text

from app.core.database.models import Base
from app.core.database.orm import get_engine


def ensure_schema() -> None:
    """初始化数据库表结构 (create_all)"""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def is_database_ready() -> bool:
    """检查数据库连接是否可用"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def ensure_schema_if_possible() -> bool:
    """
    如果数据库可用，则确保表结构已创建。
    
    Returns:
        bool: 数据库是否可用且初始化成功
    """
    if not is_database_ready():
        return False
    ensure_schema()
    return True
