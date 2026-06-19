"""
管线模块
提供数据处理管线的抽象接口和多种引擎实现
"""
from .schema import PipelineResult, TaskStatus, QualityReport
from .adapter import PipelineEngine, DatabaseSource
from .service import PipelineService

__all__ = [
    "PipelineResult", "TaskStatus", "QualityReport",
    "PipelineEngine", "DatabaseSource",
    "PipelineService",
]
