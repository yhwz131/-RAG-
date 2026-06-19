"""
数据管线 API 路由
提供管线状态查询、处理模式列表、数据库接入等功能
"""
import os
import uuid
import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from config.settings import settings
from utils.logger import get_logger
from api.pipeline.service import pipeline_service

logger = get_logger("routes_pipeline")

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

# 全局 retriever 引用（由 main.py 注入）
_retriever = None
_mm_retriever = None


def set_pipeline_retriever(retriever):
    """注入检索器到管线服务"""
    global _retriever
    _retriever = retriever
    pipeline_service.set_retriever(retriever)


def set_pipeline_mm_retriever(retriever):
    """注入多模态检索器到管线服务"""
    global _mm_retriever
    _mm_retriever = retriever
    pipeline_service.set_mm_retriever(retriever)


# ========== 请求/响应模型 ==========

class ProcessRequest(BaseModel):
    """处理请求"""
    engine: str = "simple"  # simple / spark
    auto_import: bool = True  # 是否自动导入 Milvus


class DatabaseQueryRequest(BaseModel):
    """数据库查询请求"""
    query: Optional[str] = None
    limit: int = 10000


class DatabaseConfigRequest(BaseModel):
    """数据库配置请求"""
    db_type: str = "mysql"
    db_host: str
    db_port: int = 3306
    db_user: str
    db_password: str
    db_name: str
    db_table: str = ""
    db_text_columns: List[str] = []


class ProcessResponse(BaseModel):
    """处理响应"""
    task_id: Optional[str] = None
    status: str
    message: str
    engine: Optional[str] = None


# ========== 引擎接口 ==========

@router.get("/engines")
async def get_engines():
    """获取可用处理引擎列表（供前端模式选择使用）"""
    engines = pipeline_service.get_available_engines()
    return {"engines": engines}


# ========== 处理接口 ==========

@router.post("/process", response_model=ProcessResponse)
async def process_files(
    background_tasks: BackgroundTasks,
    engine: str = "simple",
    files: List[UploadFile] = File(...)
):
    """
    大数据处理模式入口：
    1. 保存上传文件到 data/staging/（暂存）
    2. 后台异步触发处理管线（扫描 staging 目录）
    3. 处理完成后自动导入 Milvus，并将文件移至 data/raw/
    4. 立即返回 task_id
    """
    # 1. 保存文件到暂存目录
    staging_dir = settings.staging_dir
    os.makedirs(staging_dir, exist_ok=True)
    files_data = []
    for file in files:
        content = await file.read()
        files_data.append((file.filename or "unknown", content))

    saved_paths = []
    for filename, content in files_data:
        file_id = str(uuid.uuid4())[:8]
        save_name = f"{file_id}_{filename}"
        save_path = os.path.join(staging_dir, save_name)
        with open(save_path, "wb") as f:
            f.write(content)
        saved_paths.append(save_path)
        logger.info(f"暂存文件: {save_path}")

    # 2. 创建异步任务
    task_id = pipeline_service.create_task(files_count=len(saved_paths))

    # 3. 后台执行处理（Spark 失败自动降级到 simple）
    async def run_background():
        import shutil
        try:
            result = await pipeline_service.process_files_async(
                engine_name=engine,
                input_dir=staging_dir,
                output_dir=settings.processed_dir,
                auto_import=True,
            )
            # Spark 引擎返回失败时自动降级到 simple
            if not result.success and engine == "spark":
                logger.warning(f"Spark 引擎失败({result.error})，自动降级到 simple 引擎")
                result = await pipeline_service.process_files_async(
                    engine_name="simple",
                    input_dir=staging_dir,
                    output_dir=settings.processed_dir,
                    auto_import=True,
                )
        except Exception as e:
            # Spark 引擎抛异常（如 Java 未安装）时自动降级到 simple
            if engine == "spark":
                logger.warning(f"Spark 引擎异常({e})，自动降级到 simple 引擎")
                try:
                    result = await pipeline_service.process_files_async(
                        engine_name="simple",
                        input_dir=staging_dir,
                        output_dir=settings.processed_dir,
                        auto_import=True,
                    )
                except Exception as e2:
                    logger.error(f"simple 引擎也失败: {e2}")
                    pipeline_service.update_task(task_id, "failed", error=str(e2))
                    return
            else:
                logger.error(f"后台处理失败: {e}")
                pipeline_service.update_task(task_id, "failed", error=str(e))
                return

        # 处理成功后将 staging 中所有文件移到 raw（标记为已处理）
        if result.success:
            for p in saved_paths:
                if os.path.exists(p):
                    fname = os.path.basename(p)
                    dest = os.path.join(settings.upload_dir, fname)
                    shutil.move(p, dest)
            # 同时清理 staging 中可能残留的历史文件
            for leftover in os.listdir(staging_dir):
                leftover_path = os.path.join(staging_dir, leftover)
                if os.path.isfile(leftover_path):
                    dest = os.path.join(settings.upload_dir, leftover)
                    shutil.move(leftover_path, dest)
                    logger.info(f"清理残留文件: {leftover}")
            logger.info(f"已将文件从 staging 移至 raw，staging 已清空")
            pipeline_service.update_task(
                task_id, "completed",
                chunks_count=result.total_chunks,
                result=result,
            )
        else:
            pipeline_service.update_task(task_id, "failed", error=result.error)

    background_tasks.add_task(run_background)

    return ProcessResponse(
        task_id=task_id,
        status="processing",
        message=f"处理任务已提交（ID: {task_id}），共 {len(saved_paths)} 个文件",
        engine=engine,
    )


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """查询异步任务状态"""
    task = pipeline_service.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return task.model_dump()


@router.get("/tasks")
async def get_all_tasks():
    """获取所有任务列表"""
    tasks = pipeline_service.get_all_tasks()
    return {"tasks": [t.model_dump() for t in tasks]}


# ========== 状态查询接口 ==========

@router.get("/status")
async def get_pipeline_status():
    """获取管线状态（最近处理统计 + Milvus 统计）"""
    latest_stats = pipeline_service.get_latest_stats()
    milvus_stats = pipeline_service.get_milvus_stats()
    db_status = pipeline_service.get_db_status()

    return {
        "last_run": latest_stats,
        "milvus_stats": milvus_stats,
        "database_status": db_status,
    }


@router.get("/history")
async def get_pipeline_history():
    """获取处理历史记录"""
    history = pipeline_service.get_history()
    return {"runs": history}


@router.get("/quality")
async def get_quality_report():
    """获取数据质量详细报告"""
    report = pipeline_service.get_quality_report()
    return report


# ========== 数据库接入接口 ==========

@router.get("/database/status")
async def get_database_status():
    """获取数据库连接状态"""
    return pipeline_service.get_db_status()


@router.get("/database/tables")
async def get_database_tables():
    """获取数据库表结构信息"""
    db_source = pipeline_service.get_db_source()
    if not db_source:
        raise HTTPException(status_code=400, detail="未配置数据库数据源")

    try:
        info = db_source.get_table_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取表信息失败: {e}")


@router.post("/database/import")
async def import_from_database(request: DatabaseQueryRequest):
    """数据库数据已改用 Text-to-SQL 模式，不再支持向量化导入"""
    return {
        "status": "deprecated",
        "message": "数据库数据已改用 Text-to-SQL 模式，请通过聊天界面直接提问查询数据库",
    }


@router.post("/database/test")
async def test_database_connection(config: Optional[DatabaseConfigRequest] = None):
    """测试数据库连接（可传入临时配置测试）"""
    try:
        if config:
            # 用传入的配置临时测试
            from api.pipeline.engines.database import MySQLSource, PostgreSQLSource
            if config.db_type == "mysql":
                source = MySQLSource(
                    host=config.db_host, port=config.db_port,
                    user=config.db_user, password=config.db_password,
                    database=config.db_name, table=config.db_table,
                    text_columns=config.db_text_columns or None,
                )
            else:
                source = PostgreSQLSource(
                    host=config.db_host, port=config.db_port,
                    user=config.db_user, password=config.db_password,
                    database=config.db_name, table=config.db_table,
                    text_columns=config.db_text_columns or None,
                )
            connected = source.connect()
            source.close()
        else:
            db_source = pipeline_service.get_db_source()
            if not db_source:
                raise HTTPException(status_code=400, detail="未配置数据库数据源")
            connected = db_source.connect()

        if connected:
            return {"status": "connected", "type": config.db_type if config else "mysql"}
        else:
            raise HTTPException(status_code=500, detail="连接失败")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"连接测试失败: {e}")


@router.post("/database/config")
async def save_database_config(config: DatabaseConfigRequest):
    """保存数据库配置到 .env 文件并重建数据源"""
    try:
        result = pipeline_service.save_db_config(config.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存配置失败: {e}")


@router.get("/database/config")
async def get_database_config():
    """获取当前数据库配置（密码脱敏）"""
    from config.settings import settings
    return {
        "db_type": settings.db_type or "mysql",
        "db_host": settings.db_host,
        "db_port": settings.db_port,
        "db_user": settings.db_user,
        "db_password": "***" if settings.db_password else "",
        "db_name": settings.db_name,
        "db_table": settings.db_table,
        "db_text_columns": settings.db_text_columns_list,
    }
