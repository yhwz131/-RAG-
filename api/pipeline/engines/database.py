"""
数据库数据源引擎
支持 MySQL / PostgreSQL 数据接入
"""
from typing import List, Dict, Any, Optional
from ..adapter import DatabaseSource
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("pipeline.database")


class MySQLSource(DatabaseSource):
    """MySQL 数据源"""

    def __init__(self, host: str = "", port: int = 3306, user: str = "",
                 password: str = "", database: str = "", table: str = "",
                 text_columns: Optional[List[str]] = None):
        self._host = host or settings.db_host
        self._port = port or settings.db_port
        self._user = user or settings.db_user
        self._password = password or settings.db_password
        self._database = database or settings.db_name
        self._table = table or settings.db_table
        self._text_columns = text_columns or settings.db_text_columns
        self._conn = None

    @property
    def source_type(self) -> str:
        return "mysql"

    def connect(self) -> bool:
        """测试 MySQL 连接"""
        try:
            import pymysql
            self._conn = pymysql.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                database=self._database,
                charset="utf8mb4",
                connect_timeout=10,
            )
            logger.info(f"MySQL 连接成功: {self._host}:{self._port}/{self._database}")
            return True
        except Exception as e:
            logger.error(f"MySQL 连接失败: {e}")
            return False

    def fetch_data(self, query: Optional[str] = None, limit: int = 10000) -> List[Dict[str, Any]]:
        """从 MySQL 获取数据"""
        if not self._conn:
            if not self.connect():
                raise ConnectionError("MySQL 连接失败")

        try:
            import pymysql
            if query:
                sql = query
            elif self._table and self._text_columns:
                cols = ", ".join(self._text_columns)
                sql = f"SELECT {cols} FROM `{self._table}` LIMIT {limit}"
            else:
                raise ValueError("未指定 query 或 table + text_columns")

            logger.info(f"执行 MySQL 查询: {sql[:200]}")

            with self._conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()

            logger.info(f"MySQL 查询返回 {len(rows)} 行")
            return rows

        except Exception as e:
            logger.error(f"MySQL 查询失败: {e}")
            raise

    def get_table_info(self) -> Dict[str, Any]:
        """获取 MySQL 表结构信息"""
        if not self._conn:
            if not self.connect():
                raise ConnectionError("MySQL 连接失败")

        try:
            import pymysql
            info = {"database": self._database, "tables": []}

            with self._conn.cursor(pymysql.cursors.DictCursor) as cursor:
                # 获取所有表
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()

                for table_row in tables:
                    table_name = list(table_row.values())[0]
                    cursor.execute(f"SELECT COUNT(*) as cnt FROM `{table_name}`")
                    count = cursor.fetchone()["cnt"]
                    cursor.execute(f"DESCRIBE `{table_name}`")
                    columns = cursor.fetchall()
                    info["tables"].append({
                        "name": table_name,
                        "row_count": count,
                        "columns": [{"name": c["Field"], "type": c["Type"]} for c in columns],
                    })

            return info

        except Exception as e:
            logger.error(f"获取表信息失败: {e}")
            raise

    def close(self):
        """关闭连接"""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


class PostgreSQLSource(DatabaseSource):
    """PostgreSQL 数据源"""

    def __init__(self, host: str = "", port: int = 5432, user: str = "",
                 password: str = "", database: str = "", table: str = "",
                 text_columns: Optional[List[str]] = None):
        self._host = host or settings.db_host
        self._port = port or settings.db_port
        self._user = user or settings.db_user
        self._password = password or settings.db_password
        self._database = database or settings.db_name
        self._table = table or settings.db_table
        self._text_columns = text_columns or settings.db_text_columns
        self._conn = None

    @property
    def source_type(self) -> str:
        return "postgresql"

    def connect(self) -> bool:
        """测试 PostgreSQL 连接"""
        try:
            import psycopg2
            self._conn = psycopg2.connect(
                host=self._host,
                port=self._port,
                user=self._user,
                password=self._password,
                dbname=self._database,
                connect_timeout=10,
            )
            logger.info(f"PostgreSQL 连接成功: {self._host}:{self._port}/{self._database}")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL 连接失败: {e}")
            return False

    def fetch_data(self, query: Optional[str] = None, limit: int = 10000) -> List[Dict[str, Any]]:
        """从 PostgreSQL 获取数据"""
        if not self._conn:
            if not self.connect():
                raise ConnectionError("PostgreSQL 连接失败")

        try:
            import psycopg2
            import psycopg2.extras

            if query:
                sql = query
            elif self._table and self._text_columns:
                cols = ", ".join(self._text_columns)
                sql = f'SELECT {cols} FROM "{self._table}" LIMIT {limit}'
            else:
                raise ValueError("未指定 query 或 table + text_columns")

            logger.info(f"执行 PostgreSQL 查询: {sql[:200]}")

            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(sql)
                rows = [dict(r) for r in cursor.fetchall()]

            logger.info(f"PostgreSQL 查询返回 {len(rows)} 行")
            return rows

        except Exception as e:
            logger.error(f"PostgreSQL 查询失败: {e}")
            raise

    def get_table_info(self) -> Dict[str, Any]:
        """获取 PostgreSQL 表结构信息"""
        if not self._conn:
            if not self.connect():
                raise ConnectionError("PostgreSQL 连接失败")

        try:
            import psycopg2
            import psycopg2.extras

            info = {"database": self._database, "tables": []}

            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' ORDER BY table_name
                """)
                tables = cursor.fetchall()

                for t in tables:
                    table_name = t["table_name"]
                    cursor.execute(f'SELECT COUNT(*) as cnt FROM "{table_name}"')
                    count = cursor.fetchone()["cnt"]
                    cursor.execute(f"""
                        SELECT column_name, data_type FROM information_schema.columns
                        WHERE table_name = '{table_name}' ORDER BY ordinal_position
                    """)
                    columns = cursor.fetchall()
                    info["tables"].append({
                        "name": table_name,
                        "row_count": count,
                        "columns": [{"name": c["column_name"], "type": c["data_type"]} for c in columns],
                    })

            return info

        except Exception as e:
            logger.error(f"获取表信息失败: {e}")
            raise

    def close(self):
        """关闭连接"""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


def create_database_source(db_type: str = "") -> Optional[DatabaseSource]:
    """
    工厂方法：根据配置创建数据库数据源
    
    Args:
        db_type: 数据库类型 (mysql / postgresql)，为空则从 settings 读取
        
    Returns:
        DatabaseSource 实例，或 None（未配置时）
    """
    db_type = db_type or settings.db_type
    if not db_type:
        return None

    if db_type == "mysql":
        return MySQLSource()
    elif db_type == "postgresql":
        return PostgreSQLSource()
    else:
        logger.warning(f"不支持的数据库类型: {db_type}")
        return None
