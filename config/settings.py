"""
全局配置模块
从 .env 文件和环境变量读取配置
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, model_validator
from typing import Optional
import os


class Settings(BaseSettings):
    """系统配置类，自动从 .env 文件读取配置"""
    
    # ========== LLM 配置 ==========
    llm_api_key: str = ""
    llm_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"
    llm_model_name: str = "mimo-v2.5"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_timeout: float = 120.0
    
    # ========== 多模态 LLM 配置（mimo-v2-omni） ==========
    mm_llm_model_name: str = "mimo-v2-omni"
    
    # ========== Embedding 配置 ==========
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model_name: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024

    # ========== 多模态 Embedding 配置 ==========
    multimodal_embedding_model: str = "Qwen/Qwen3-VL-Embedding-8B"
    multimodal_embedding_dim: int = 4096
    multimodal_collection_name: str = "knowledge_base_mm"
    
    # ========== 文档切片配置 ==========
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # ========== 检索配置 ==========
    retriever_top_k: int = 5
    similarity_threshold: float = 0.3
    rrf_threshold: float = 0.015  # RRF 融合分数阈值，低于此值的结果被过滤
    max_context_tokens: int = 3000  # 上下文最大 token 数
    rrf_k: int = 60  # RRF (Reciprocal Rank Fusion) 参数
    
    # ========== 对话记忆配置 ==========
    max_history_rounds: int = 10
    max_history_tokens: int = 2000  # 对话历史最大 token 数
    
    # ========== Milvus 配置 ==========
    milvus_db_path: str = "./data/milvus.db"
    collection_name: str = "knowledge_base"
    
    # ========== Spark 配置 ==========
    spark_app_name: str = "KnowledgeQA"
    spark_master: str = "local[*]"
    
    # ========== 数据库接入配置 ==========
    db_type: str = ""  # mysql / postgresql / 空表示不启用
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = ""
    db_password: str = ""
    db_name: str = ""
    db_table: str = ""  # 要导入的表名
    db_text_columns: str = ""  # 文本列名，逗号分隔，如 "title,content"
    db_query: str = ""  # 自定义 SQL 查询（可选）

    @property
    def db_text_columns_list(self) -> list:
        """将逗号分隔的文本列名解析为列表"""
        if not self.db_text_columns:
            return []
        return [c.strip() for c in self.db_text_columns.split(",") if c.strip()]
    
    # ========== 管线配置 ==========
    pipeline_engine: str = "simple"  # simple / spark
    pipeline_min_text_length: int = 50  # 最短文本长度过滤
    
    # ========== 服务器配置 ==========
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # ========== 日志配置 ==========
    log_level: str = "INFO"
    log_file: str = "./data/app.log"
    
    # ========== 文件上传配置 ==========
    upload_dir: str = "./data/raw"
    staging_dir: str = "./data/staging"  # 批量入库暂存目录
    processed_dir: str = "./data/processed"
    sessions_dir: str = "./data/sessions"
    max_file_size_mb: int = 100
    allowed_extensions: list = [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".txt", ".md", ".csv", ".xlsx", ".xls"]
    
    # ========== 配置校验 ==========
    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError(f"llm_temperature 必须在 0~2 之间，当前值: {v}")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"similarity_threshold 必须在 0~1 之间，当前值: {v}")
        return v

    @field_validator("chunk_size", "retriever_top_k", "max_context_tokens",
                     "max_history_tokens", "rrf_k", "max_file_size_mb", "llm_max_tokens")
    @classmethod
    def validate_positive_int(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} 必须大于 0，当前值: {v}")
        return v

    @model_validator(mode="after")
    def validate_model(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) 必须小于 chunk_size ({self.chunk_size})"
            )
        if self.max_history_rounds <= 0:
            raise ValueError(f"max_history_rounds 必须大于 0，当前值: {self.max_history_rounds}")
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # ========== 兼容属性（供已有代码使用） ==========
    @property
    def llm_api_url(self) -> str:
        """LLM API 完整地址（chat completions）"""
        return f"{self.llm_base_url.rstrip('/')}/chat/completions"
    
    @property
    def llm_model(self) -> str:
        """LLM 模型名（兼容属性）"""
        return self.llm_model_name
    
    @property
    def mm_llm_api_url(self) -> str:
        """多模态 LLM API 完整地址（与 LLM 共用端点）"""
        return f"{self.llm_base_url.rstrip('/')}/chat/completions"
    
    @property
    def mm_llm_model(self) -> str:
        """多模态 LLM 模型名"""
        return self.mm_llm_model_name
    
    @property
    def embedding_dimension(self) -> int:
        """Embedding 维度（兼容属性）"""
        return self.embedding_dim
    
    @property
    def milvus_uri(self) -> str:
        """Milvus 连接 URI（本地文件模式）"""
        return self.milvus_db_path
    
    @property
    def milvus_collection(self) -> str:
        """Milvus 集合名（兼容属性）"""
        return self.collection_name
    
    @property
    def top_k(self) -> int:
        """检索 Top-K（兼容属性）"""
        return self.retriever_top_k


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取全局配置实例"""
    return settings