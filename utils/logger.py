"""
日志配置模块
提供统一的日志配置和获取接口
"""
import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "knowledge_qa",
    log_level: str = "INFO",
    log_file: str = "./data/app.log"
) -> logging.Logger:
    """
    配置并返回日志记录器
    
    Args:
        name: 日志记录器名称
        log_level: 日志级别
        log_file: 日志文件路径
    
    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"无法创建日志文件 {log_file}: {e}")
    
    return logger


# 全局日志实例
logger = setup_logger()


def get_logger(name: str = "knowledge_qa") -> logging.Logger:
    """获取日志记录器"""
    if name == "knowledge_qa":
        return logging.getLogger(name)
    # 子模块 logger 继承 knowledge_qa 的 handler
    return logging.getLogger(f"knowledge_qa.{name}")