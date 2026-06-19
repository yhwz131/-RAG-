"""
管线引擎抽象接口
定义处理引擎的统一接口，后端 API 通过此接口调用，不直接依赖具体实现
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from .schema import PipelineResult


class PipelineEngine(ABC):
    """处理引擎抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""
        pass

    @abstractmethod
    def run(self, input_dir: str, output_dir: str) -> PipelineResult:
        """
        执行处理管线
        
        Args:
            input_dir: 原始文件目录
            output_dir: 处理结果输出目录
            
        Returns:
            PipelineResult: 标准化处理结果
        """
        pass


class DatabaseSource(ABC):
    """数据源抽象基类（数据库接入）"""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """数据源类型"""
        pass

    @abstractmethod
    def connect(self) -> bool:
        """测试连接"""
        pass

    @abstractmethod
    def fetch_data(self, query: Optional[str] = None, limit: int = 10000) -> List[Dict[str, Any]]:
        """
        从数据库获取数据
        
        Args:
            query: 自定义 SQL 查询
            limit: 最大返回行数
            
        Returns:
            List[Dict]: 查询结果，每行为一个字典
        """
        pass

    @abstractmethod
    def get_table_info(self) -> Dict[str, Any]:
        """
        获取表结构信息
        
        Returns:
            Dict: 包含列名、行数等信息
        """
        pass

    def close(self):
        """关闭连接"""
        pass
