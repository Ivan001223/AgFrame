import mysql.connector
from mysql.connector import pooling
import time
from app.core.config.config_manager import config_manager

class DatabaseManager:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._init_pool()
        return cls._instance

    def _init_pool(self):
        """初始化连接池。"""
        db_config = config_manager.get_config().get("database", {})
        
        # 数据库连接重试逻辑
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._pool = mysql.connector.pooling.MySQLConnectionPool(
                    pool_name="agent_pool",
                    pool_size=5,
                    host=db_config.get("host", "localhost"),
                    port=db_config.get("port", 3306),
                    user=db_config.get("user", "root"),
                    password=db_config.get("password", "password"),
                    database=db_config.get("db_name", "agent_app"),
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
        """从连接池获取连接。"""
        if not self._pool:
            self._init_pool()
            if not self._pool:
                raise Exception("Database connection pool is not initialized.")
        
        return self._pool.get_connection()

    def execute_query(self, query, params=None):
        """执行查询并返回结果（用于 SELECT）。"""
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
        """执行更新语句（INSERT/UPDATE/DELETE）并返回最后插入的行 ID。"""
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
