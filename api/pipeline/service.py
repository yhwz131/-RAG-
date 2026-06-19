"""
管线服务层
编排业务流程：文件保存 → 处理引擎 → 自动入库
路由层只调用此服务，不直接接触引擎或数据库
"""
import os
import json
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from .schema import PipelineResult, TaskStatus
from .adapter import PipelineEngine, DatabaseSource
from .engines.simple import SimpleEngine
from .engines.spark_engine import SparkEngine
from .engines.database import create_database_source
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("pipeline_service")


class PipelineService:
    """
    管线服务 — 业务流程编排器
    
    职责：
    1. 管理处理引擎（Simple / Spark）
    2. 管理数据库数据源（MySQL / PostgreSQL）
    3. 执行处理流程（文件保存 → 处理 → 入库）
    4. 管理任务状态和历史
    """

    def __init__(self):
        self._engines: Dict[str, PipelineEngine] = {}
        self._db_source: Optional[DatabaseSource] = None
        self._tasks: Dict[str, TaskStatus] = {}
        self._history: List[Dict[str, Any]] = []
        self._retriever = None
        self._mm_retriever = None

        # 注册默认引擎
        self._engines["simple"] = SimpleEngine()
        self._engines["spark"] = SparkEngine()

        # 尝试创建数据库数据源
        self._db_source = create_database_source()

        # 加载历史记录
        self._load_history()

    def set_retriever(self, retriever):
        """注入向量检索器（由 main.py 调用）"""
        self._retriever = retriever

    def set_mm_retriever(self, retriever):
        """注入多模态检索器（由 main.py 调用）"""
        self._mm_retriever = retriever

    # ========== 引擎管理 ==========

    def get_engine(self, name: str) -> Optional[PipelineEngine]:
        """获取处理引擎"""
        return self._engines.get(name)

    def get_available_engines(self) -> List[Dict[str, str]]:
        """获取可用引擎列表"""
        engines = []
        for name, engine in self._engines.items():
            engines.append({
                "name": name,
                "label": self._get_engine_label(name),
                "description": self._get_engine_description(name),
            })
        return engines

    def _get_engine_label(self, name: str) -> str:
        labels = {
            "simple": "快速入库",
            "spark": "大数据处理",
        }
        return labels.get(name, name)

    def _get_engine_description(self, name: str) -> str:
        descriptions = {
            "simple": "直接解析入库，几秒完成。适合日常补充文档。",
            "spark": "Spark 批量清洗、去重、统计后自动入库。处理时间较长。",
        }
        return descriptions.get(name, "")

    # ========== 数据库数据源 ==========

    def get_db_source(self) -> Optional[DatabaseSource]:
        """获取数据库数据源"""
        return self._db_source

    def get_db_status(self) -> Dict[str, Any]:
        """获取数据库连接状态"""
        if not self._db_source:
            return {"connected": False, "type": None, "message": "未配置数据库"}

        try:
            connected = self._db_source.connect()
            return {
                "connected": connected,
                "type": self._db_source.source_type,
                "host": settings.db_host,
                "database": settings.db_name,
                "table": settings.db_table,
            }
        except Exception as e:
            return {"connected": False, "type": self._db_source.source_type, "error": str(e)}

    def save_db_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        保存数据库配置到 .env 文件并重建数据源
        """
        import re

        env_path = ".env"
        env_lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                env_lines = f.readlines()

        # 需要写入的配置项
        db_keys = [
            "db_type", "db_host", "db_port", "db_user",
            "db_password", "db_name", "db_table", "db_text_columns",
        ]

        # 构建新配置行
        new_lines = []
        for key in db_keys:
            value = config.get(key, "")
            # db_text_columns 可能是 list（前端传来的），需要转为逗号分隔字符串
            if key == "db_text_columns" and isinstance(value, list):
                value = ",".join(value)
            new_lines.append(f"{key.upper()}={value}\n")

        # 替换或追加
        updated_keys = set()
        for i, line in enumerate(env_lines):
            for key in db_keys:
                if line.strip().startswith(f"{key.upper()}="):
                    env_lines[i] = f"{key.upper()}={config.get(key, '')}\n"
                    if key == "db_text_columns":
                        cols = config.get(key, "")
                        if isinstance(cols, list):
                            cols = ",".join(cols)
                        env_lines[i] = f"DB_TEXT_COLUMNS={cols}\n"
                    updated_keys.add(key)
                    break

        # 追加未找到的 key
        for key in db_keys:
            if key not in updated_keys:
                value = config.get(key, "")
                env_lines.append(f"{key.upper()}={value}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(env_lines)

        # 更新运行时 settings
        settings.db_type = config.get("db_type", "mysql")
        settings.db_host = config.get("db_host", "localhost")
        settings.db_port = config.get("db_port", 3306)
        settings.db_user = config.get("db_user", "")
        settings.db_password = config.get("db_password", "")
        settings.db_name = config.get("db_name", "")
        settings.db_table = config.get("db_table", "")
        # db_text_columns 存储为逗号分隔字符串
        cols = config.get("db_text_columns", "")
        if isinstance(cols, list):
            settings.db_text_columns = ",".join(cols)
        else:
            settings.db_text_columns = str(cols)

        # 重建数据源
        self._db_source = create_database_source(config.get("db_type", "mysql"))
        logger.info(f"数据库配置已保存并重建数据源: type={config.get('db_type')}")

        return {"status": "ok", "message": "数据库配置已保存"}

    def fetch_from_database(self, query: Optional[str] = None) -> PipelineResult:
        """数据库数据不再向量化入库，改用 Text-to-SQL 直接查询"""
        return PipelineResult(
            run_id="",
            engine="database",
            success=False,
            error="数据库数据已改用 Text-to-SQL 模式，请通过聊天界面直接提问查询数据库",
        )

    # ========== 处理流程 ==========

    def process_files(self, engine_name: str, input_dir: str, output_dir: str,
                      auto_import: bool = True) -> PipelineResult:
        """
        执行文件处理流程
        
        Args:
            engine_name: 引擎名称 (simple / spark)
            input_dir: 输入目录
            output_dir: 输出目录
            auto_import: 是否自动导入 Milvus
            
        Returns:
            PipelineResult: 处理结果
        """
        engine = self._engines.get(engine_name)
        if not engine:
            return PipelineResult(
                run_id="",
                engine=engine_name,
                success=False,
                error=f"未知引擎: {engine_name}",
            )

        # 1. 执行处理
        result = engine.run(input_dir, output_dir)
        if not result.success:
            return result

        # 2. 自动导入 Milvus
        if auto_import and result.chunks_file and self._retriever:
            imported = self._import_to_milvus(result.chunks_file)
            logger.info(f"自动导入 Milvus: {imported} 条")

        # 3. 保存历史记录
        self._save_to_history(result)

        return result

    async def process_files_async(self, engine_name: str, input_dir: str, output_dir: str,
                                   auto_import: bool = True) -> PipelineResult:
        """异步执行文件处理（用于后台任务）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.process_files, engine_name, input_dir, output_dir, auto_import
        )

    def save_uploaded_files(self, files_data: List[tuple]) -> List[str]:
        """
        保存上传的文件到 raw 目录
        
        Args:
            files_data: [(filename, content), ...] 文件数据列表
            
        Returns:
            保存后的文件路径列表
        """
        os.makedirs(settings.upload_dir, exist_ok=True)
        saved_paths = []

        for filename, content in files_data:
            file_id = str(uuid.uuid4())[:8]
            save_name = f"{file_id}_{filename}"
            save_path = os.path.join(settings.upload_dir, save_name)
            with open(save_path, "wb") as f:
                f.write(content)
            saved_paths.append(save_path)
            logger.info(f"文件已保存: {save_path}")

        return saved_paths

    # ========== 任务管理 ==========

    def create_task(self, files_count: int) -> str:
        """创建异步任务"""
        task_id = str(uuid.uuid4())[:8]
        task = TaskStatus(
            task_id=task_id,
            status="processing",
            files_count=files_count,
        )
        self._tasks[task_id] = task
        return task_id

    def update_task(self, task_id: str, status: str, **kwargs):
        """更新任务状态"""
        task = self._tasks.get(task_id)
        if task:
            task.status = status
            if status in ("completed", "failed"):
                task.completed_at = datetime.now().isoformat()
            for k, v in kwargs.items():
                setattr(task, k, v)

    def get_task(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[TaskStatus]:
        """获取所有任务"""
        return list(self._tasks.values())

    # ========== 状态查询 ==========

    def get_latest_stats(self) -> Dict[str, Any]:
        """获取最近一次处理的统计摘要"""
        stats_file = os.path.join(settings.processed_dir, "stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告"""
        quality_file = os.path.join(settings.processed_dir, "quality_report.json")
        if os.path.exists(quality_file):
            with open(quality_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_history(self) -> List[Dict[str, Any]]:
        """获取处理历史"""
        return self._history

    def get_milvus_stats(self) -> Dict[str, Any]:
        """获取 Milvus 统计信息"""
        if not self._retriever:
            return {}
        try:
            client = self._retriever._client
            collection = self._retriever.collection_name
            stats = client.get_collection_stats(collection)
            return {
                "collection": collection,
                "row_count": stats.get("row_count", 0),
            }
        except Exception as e:
            logger.warning(f"获取 Milvus 统计失败: {e}")
            return {}

    # ========== 内部方法 ==========

    def _import_to_milvus(self, chunks_file: str) -> int:
        """将切片数据导入 Milvus"""
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            if not chunks:
                return 0
            # 字段映射：pipeline ChunkData 用 "source"，Milvus 用 "filename"
            for c in chunks:
                if "filename" not in c and "source" in c:
                    c["filename"] = self._strip_uuid_prefix(c["source"])
            count = self._retriever.insert_documents(chunks)
            return count
        except Exception as e:
            logger.error(f"导入 Milvus 失败: {e}")
            return 0

    @staticmethod
    def _strip_uuid_prefix(name: str) -> str:
        """循环去除 8位hex_ 前缀（如 257dab77_2639779b_xxx → xxx）"""
        while True:
            parts = name.split("_", 1)
            if len(parts) == 2 and len(parts[0]) == 8 and all(ch in "0123456789abcdef" for ch in parts[0]):
                name = parts[1]
            else:
                break
        return name

    def _save_to_history(self, result: PipelineResult):
        """保存处理历史"""
        record = {
            "run_id": result.run_id,
            "engine": result.engine,
            "timestamp": result.timestamp,
            "total_files": result.total_files,
            "total_chunks": result.total_chunks,
            "success": result.success,
        }
        self._history.append(record)

        # 保存到文件
        history_file = os.path.join(settings.processed_dir, "history.json")
        os.makedirs(settings.processed_dir, exist_ok=True)
        try:
            existing = []
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            existing.append(record)
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存历史记录失败: {e}")

    def _load_history(self):
        """加载历史记录"""
        history_file = os.path.join(settings.processed_dir, "history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except Exception:
                self._history = []


# 全局单例
pipeline_service = PipelineService()
