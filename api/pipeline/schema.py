"""
管线数据契约（Schema）
定义管线各环节的输入/输出格式，确保模块间解耦
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class ChunkData(BaseModel):
    """切片数据 — 管线输出的标准化格式"""
    chunk_id: str = Field(..., description="切片唯一ID")
    content: str = Field(..., description="切片文本内容")
    source: str = Field(default="unknown", description="来源文件名")
    page_number: int = Field(default=0, description="页码")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")


class CleanStats(BaseModel):
    """清洗统计"""
    total_raw: int = Field(default=0, description="原始文档数")
    empty_removed: int = Field(default=0, description="空文档移除数")
    duplicate_removed: int = Field(default=0, description="重复文档移除数")
    short_removed: int = Field(default=0, description="过短文档移除数")
    encoding_fixed: int = Field(default=0, description="编码修复数")
    total_clean: int = Field(default=0, description="清洗后文档数")


class ChunkStats(BaseModel):
    """切片统计"""
    total_chunks: int = Field(default=0, description="总切片数")
    avg_length: float = Field(default=0, description="平均切片长度")
    min_length: int = Field(default=0, description="最短切片长度")
    max_length: int = Field(default=0, description="最长切片长度")
    length_distribution: Dict[str, int] = Field(default_factory=dict, description="长度分布")


class KeywordStat(BaseModel):
    """关键词统计"""
    word: str
    count: int


class QualityReport(BaseModel):
    """数据质量报告"""
    clean_stats: CleanStats = Field(default_factory=CleanStats)
    chunk_stats: ChunkStats = Field(default_factory=ChunkStats)
    top_keywords: List[KeywordStat] = Field(default_factory=list, description="高频关键词Top N")
    format_breakdown: Dict[str, int] = Field(default_factory=dict, description="文件格式分布")
    failed_files: List[Dict[str, str]] = Field(default_factory=list, description="失败文件列表")


class PipelineResult(BaseModel):
    """管线处理结果 — 引擎输出的标准化格式"""
    run_id: str = Field(..., description="运行ID")
    engine: str = Field(..., description="处理引擎名称")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="处理时间")
    input_dir: str = Field(default="", description="输入目录")
    output_dir: str = Field(default="", description="输出目录")
    chunks_file: str = Field(default="", description="切片数据文件路径")
    stats_file: str = Field(default="", description="统计文件路径")
    total_files: int = Field(default=0, description="处理文件数")
    total_chunks: int = Field(default=0, description="生成切片数")
    quality_report: QualityReport = Field(default_factory=QualityReport)
    success: bool = Field(default=True, description="是否成功")
    error: Optional[str] = Field(default=None, description="错误信息")


class TaskStatus(BaseModel):
    """异步任务状态"""
    task_id: str
    status: str = "pending"  # pending / processing / completed / failed
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    files_count: int = 0
    chunks_count: int = 0
    error: Optional[str] = None
    result: Optional[PipelineResult] = None


class DataSourceConfig(BaseModel):
    """数据源配置"""
    source_type: str  # file / mysql / postgresql
    config: Dict[str, Any] = Field(default_factory=dict)
