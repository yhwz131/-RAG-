"""
健康检查 API 路由
"""
from fastapi import APIRouter
from config.settings import settings
from utils.logger import get_logger
import time

logger = get_logger("routes_health")

router = APIRouter(tags=["health"])

# 记录服务启动时间
_start_time = time.time()


@router.get("/health")
async def health_check():
    """健康检查接口"""
    uptime = time.time() - _start_time
    return {
        "status": "ok",
        "uptime_seconds": round(uptime, 2),
        "version": "1.0.0"
    }


@router.get("/health/config")
async def config_check():
    """配置检查接口（不暴露敏感信息）"""
    return {
        "llm_model": settings.llm_model_name,
        "embedding_model": settings.embedding_model_name,
        "embedding_dim": settings.embedding_dim,
        "collection_name": settings.collection_name,
        "milvus_uri": settings.milvus_db_path,
        "chunk_size": settings.chunk_size,
        "top_k": settings.retriever_top_k
    }