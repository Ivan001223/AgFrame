import time

import mysql.connector

from app.infrastructure.config.settings import settings


class DatabaseManager:
    """
    原生 MySQL 数据库连接池管理器（单例模式）。
    提供基础的数据库连接获取和 SQL 执行能力。
    主要用于一些不需要 ORM 的底层操作或旧代码兼容。
    """
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_pool()
        return cls._instance

    def _init_pool(self):
        """初始化连接池。支持重试机制。"""
        db_config = settings.database

        # 数据库连接重试逻辑
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._pool = mysql.connector.pooling.MySQLConnectionPool(
                    pool_name="agent_pool",
                    pool_size=5,
                    host=db_config.host,
                    port=db_config.port,
                    user=db_config.user,
                    password=db_config.password,
                    database=db_config.db_name,
                    autocommit=True
                )
                print("MySQL 连接池初始化成功。")
                break
            except mysql.connector.Error as err:
                print(f"初始化数据库连接池失败（第 {attempt+1}/{max_retries} 次尝试）：{err}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print("连接 MySQL 数据库失败。")
                    self._pool = None

    def get_connection(self):
        """从连接池获取一个数据库连接。"""
        if not self._pool:
            self._init_pool()
            if not self._pool:
                raise Exception("Database connection pool is not initialized.")
        
        return self._pool.get_connection()

    def execute_query(self, query, params=None):
        """
        执行查询语句 (SELECT) 并返回字典格式的结果列表。
        
        Args:
            query: SQL 查询语句
            params: SQL 参数元组
            
        Returns:
            list[dict]: 查询结果列表
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            result = cursor.fetchall()
            return result
        except mysql.connector.Error as err:
            print(f"查询执行错误：{err}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def execute_update(self, query, params=None):
        """
        执行更新语句 (INSERT/UPDATE/DELETE) 并返回最后插入的行 ID。
        
        Args:
            query: SQL 更新语句
            params: SQL 参数元组
            
        Returns:
            int: lastrowid (对于 INSERT 语句)
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()
            return cursor.lastrowid
        except mysql.connector.Error as err:
            print(f"更新执行错误：{err}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

# 全局实例
db_manager = DatabaseManager()
