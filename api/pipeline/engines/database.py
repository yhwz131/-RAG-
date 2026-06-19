"""
数据库数据源引擎
支持 MySQL / PostgreSQL 数据接入（含 Text-to-SQL 安全执行）
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from ..adapter import DatabaseSource
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("pipeline.database")


# ========== SQL 安全校验 ==========

# 允许的 SQL 语句前缀（只读操作）
_ALLOWED_SQL_PREFIXES = ("select", "with")
# 禁止的关键词（防止注入）
_DANGEROUS_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "rename", "replace", "merge", "call", "exec",
    "execute", "xp_", "sp_", "into outfile", "into dumpfile",
    "lock table", "unlock table",
]


def validate_sql(sql: str) -> Tuple[bool, str]:
    """校验 SQL 是否为安全的只读查询
    
    Returns:
        (is_safe, error_message): is_safe=True 表示安全
    """
    sql_clean = sql.strip().rstrip(";").strip().lower()
    
    # 检查是否以允许的前缀开头
    if not any(sql_clean.startswith(prefix) for prefix in _ALLOWED_SQL_PREFIXES):
        return False, f"只允许 SELECT 查询，不允许: {sql_clean[:30]}..."
    
    # 检查是否包含危险关键词
    for kw in _DANGEROUS_KEYWORDS:
        if re.search(r'\b' + kw + r'\b', sql_clean):
            return False, f"SQL 包含禁止的操作: {kw}"
    
    return True, ""


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
        self._text_columns = text_columns or settings.db_text_columns_list
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

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """安全执行 SQL 查询（只读，带校验）
        
        Args:
            sql: SQL 查询语句
            
        Returns:
            {"columns": [...], "rows": [...], "row_count": int}
        """
        is_safe, err = validate_sql(sql)
        if not is_safe:
            raise ValueError(f"SQL 安全校验失败: {err}")
        
        if not self._conn:
            if not self.connect():
                raise ConnectionError("MySQL 连接失败")
        
        try:
            import pymysql
            with self._conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
            
            columns = list(rows[0].keys()) if rows else []
            safe_rows = []
            for row in rows:
                safe_row = {}
                for k, v in row.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        safe_row[k] = v
                    else:
                        safe_row[k] = str(v)
                safe_rows.append(safe_row)
            
            logger.info(f"SQL 执行成功: {len(safe_rows)} 行, 列={columns}")
            return {"columns": columns, "rows": safe_rows, "row_count": len(safe_rows)}
            
        except Exception as e:
            logger.error(f"SQL 执行失败: {e}")
            raise

    def get_schema_for_llm(self) -> str:
        """获取表结构的 LLM 可读描述（用于 SQL 生成 Prompt）"""
        info = self.get_table_info()
        parts = []
        for table in info.get("tables", []):
            cols = ", ".join(
                f"{c['name']} {c['type']}" for c in table.get("columns", [])
            )
            parts.append(f"表 {table['name']}（{table['row_count']} 行）: {cols}")
        return "\n".join(parts)


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
        self._text_columns = text_columns or settings.db_text_columns_list
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

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        """安全执行 SQL 查询（只读，带校验）"""
        is_safe, err = validate_sql(sql)
        if not is_safe:
            raise ValueError(f"SQL 安全校验失败: {err}")
        
        if not self._conn:
            if not self.connect():
                raise ConnectionError("PostgreSQL 连接失败")
        
        try:
            import psycopg2
            import psycopg2.extras
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                cursor.execute(sql)
                rows = [dict(r) for r in cursor.fetchall()]
            
            columns = list(rows[0].keys()) if rows else []
            safe_rows = []
            for row in rows:
                safe_row = {}
                for k, v in row.items():
                    if isinstance(v, (int, float, str, bool, type(None))):
                        safe_row[k] = v
                    else:
                        safe_row[k] = str(v)
                safe_rows.append(safe_row)
            
            logger.info(f"SQL 执行成功: {len(safe_rows)} 行, 列={columns}")
            return {"columns": columns, "rows": safe_rows, "row_count": len(safe_rows)}
            
        except Exception as e:
            logger.error(f"SQL 执行失败: {e}")
            raise

    def get_schema_for_llm(self) -> str:
        """获取表结构的 LLM 可读描述（用于 SQL 生成 Prompt）"""
        info = self.get_table_info()
        parts = []
        for table in info.get("tables", []):
            cols = ", ".join(
                f"{c['name']} {c['type']}" for c in table.get("columns", [])
            )
            parts.append(f"表 {table['name']}（{table['row_count']} 行）: {cols}")
        return "\n".join(parts)


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
