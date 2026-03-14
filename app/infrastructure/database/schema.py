from __future__ import annotations

import logging
import time

from sqlalchemy import text

from app.infrastructure.database.models import Base
from app.infrastructure.database.orm import get_engine

logger = logging.getLogger(__name__)


def ensure_schema() -> None:
    """初始化数据库表结构 (create_all)"""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


_db_ready_cache = None
_last_check_time = 0
CHECK_INTERVAL = 30  # seconds


def is_database_ready() -> bool:
    """检查数据库连接是否可用（带简单缓存）"""
    global _db_ready_cache, _last_check_time

    # 如果缓存有效且是 True，直接返回
    if _db_ready_cache is True and (time.time() - _last_check_time < CHECK_INTERVAL):
        return True

    try:
        # 复用 get_engine() 单例
        engine = get_engine()
        # 尝试快速连接检查
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        _db_ready_cache = True
        _last_check_time = time.time()
        return True
    except Exception as e:
        logger.debug(f"Database readiness check failed: {e}")
        _db_ready_cache = False
        _last_check_time = time.time()
        return False


def ensure_schema_if_possible() -> bool:
    """
    如果数据库可用，则确保表结构已创建。

    Returns:
        bool: 数据库是否可用且初始化成功
    """
    if not is_database_ready():
        return False
    # ensure_schema() 操作较重（虽然是幂等的），也应该避免每次都调
    # 这里我们假设只要 is_database_ready() 返回 True，且我们至少调过一次 ensure_schema，就不必每次都调
    # 但为了简单起见，且 create_all 本身会检查表是否存在，开销尚可接受。
    # 也可以优化为只在 _db_ready_cache 从 False 变 True 时调用一次。
    try:
        ensure_schema()
        return True
    except Exception as e:
        logger.warning(f"Failed to ensure schema: {e}")
        return False
