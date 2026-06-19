"""
Spark 数据处理模块
用于大规模文档数据处理：文本提取、清洗、切片、向量化
依赖 PySpark（可选，需单独安装 pyspark）

使用方法：
    python -m spark.processor --input ./data/raw --output ./data/processed
"""
import os
import sys
import json
import argparse
from typing import List, Dict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from utils.logger import get_logger
from utils.file_parser import FileParser
from embeddings.chunker import TextChunker

logger = get_logger("spark_processor")


def process_local(input_dir: str, output_dir: str):
    """
    本地模式处理（不依赖 Spark），用于小规模数据或 Spark 不可用时
    
    Args:
        input_dir: 输入文档目录
        output_dir: 输出切片目录
    """
    os.makedirs(output_dir, exist_ok=True)
    parser = FileParser()
    chunker = TextChunker()
    
    supported_ext = settings.allowed_extensions
    files = []
    for f in os.listdir(input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in supported_ext:
            files.append(os.path.join(input_dir, f))
    
    logger.info(f"找到 {len(files)} 个待处理文件")
    
    total_chunks = 0
    for filepath in files:
        try:
            filename = os.path.basename(filepath)
            logger.info(f"处理文件: {filename}")
            
            # 解析文件
            text = parser.parse(filepath)
            if not text.strip():
                logger.warning(f"文件内容为空: {filename}")
                continue
            
            # 切片
            chunks = chunker.chunk(text, source=filename)
            total_chunks += len(chunks)
            
            # 保存切片
            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_chunks.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            
            logger.info(f"文件 {filename} 切分为 {len(chunks)} 个片段 -> {output_file}")
        
        except Exception as e:
            logger.error(f"处理文件 {filepath} 失败: {e}")
    
    logger.info(f"处理完成: {len(files)} 个文件, {total_chunks} 个切片")


def process_spark(input_dir: str, output_dir: str):
    """
    Spark 模式处理大规模数据
    
    Args:
        input_dir: 输入文档目录（支持 HDFS/S3）
        output_dir: 输出切片目录
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        logger.error("PySpark 未安装，请先安装: pip install pyspark")
        logger.info("回退到本地模式处理...")
        process_local(input_dir, output_dir)
        return
    
    spark = SparkSession.builder \
        .appName(settings.spark_app_name) \
        .master(settings.spark_master) \
        .getOrCreate()
    
    logger.info(f"Spark 会话已创建: {spark.sparkContext.appName}")
    
    parser = FileParser()
    chunker = TextChunker()
    
    # 获取文件列表
    supported_ext = settings.allowed_extensions
    files = []
    for f in os.listdir(input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in supported_ext:
            files.append(os.path.join(input_dir, f))
    
    # 并行处理
    rdd = spark.sparkContext.parallelize(files, numSlices=min(len(files), 8))
    
    def process_file(filepath):
        try:
            filename = os.path.basename(filepath)
            text = parser.parse(filepath)
            if not text.strip():
                return []
            chunks = chunker.chunk(text, source=filename)
            return chunks
        except Exception as e:
            logger.error(f"处理文件失败: {filepath}, 错误: {e}")
            return []
    
    results = rdd.map(process_file).collect()
    
    # 合并结果
    all_chunks = []
    for chunks in results:
        all_chunks.extend(chunks)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "all_chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Spark 处理完成: {len(files)} 个文件, {len(all_chunks)} 个切片")
    
    spark.stop()


def main():
    parser = argparse.ArgumentParser(description="知识库文档数据处理")
    parser.add_argument("--input", type=str, default=settings.upload_dir, help="输入目录")
    parser.add_argument("--output", type=str, default=settings.processed_dir, help="输出目录")
    parser.add_argument("--mode", type=str, choices=["local", "spark"], default="local", help="处理模式")
    args = parser.parse_args()
    
    logger.info(f"开始处理: input={args.input}, output={args.output}, mode={args.mode}")
    
    if args.mode == "spark":
        process_spark(args.input, args.output)
    else:
        process_local(args.input, args.output)


if __name__ == "__main__":
    main()