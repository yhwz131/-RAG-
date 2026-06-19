"""
PySpark 处理引擎
适用于大规模数据的分布式处理
"""
import os
import json
import hashlib
from typing import List, Dict, Any
from datetime import datetime

from ..adapter import PipelineEngine
from ..schema import (
    PipelineResult, ChunkData, CleanStats, ChunkStats,
    KeywordStat, QualityReport
)
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("pipeline.spark")


class SparkEngine(PipelineEngine):
    """PySpark 分布式处理引擎"""

    @property
    def name(self) -> str:
        return "spark"

    def run(self, input_dir: str, output_dir: str) -> PipelineResult:
        """执行 Spark 处理管线"""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"[SparkEngine] 开始处理: input={input_dir}, output={output_dir}, run_id={run_id}")

        try:
            from pyspark.sql import SparkSession
        except ImportError:
            logger.error("PySpark 未安装，无法使用 SparkEngine")
            return PipelineResult(
                run_id=run_id,
                engine=self.name,
                success=False,
                error="PySpark 未安装",
            )

        os.makedirs(output_dir, exist_ok=True)

        # 创建 Spark 会话
        spark = SparkSession.builder \
            .appName(settings.spark_app_name) \
            .master(settings.spark_master) \
            .getOrCreate()

        logger.info(f"Spark 会话已创建: {spark.sparkContext.appName}")

        try:
            result = self._process_with_spark(spark, input_dir, output_dir, run_id)
            return result
        except Exception as e:
            logger.error(f"Spark 处理失败: {e}")
            return PipelineResult(
                run_id=run_id,
                engine=self.name,
                input_dir=input_dir,
                output_dir=output_dir,
                success=False,
                error=str(e),
            )
        finally:
            spark.stop()
            logger.info("Spark 会话已关闭")

    def _process_with_spark(self, spark, input_dir: str, output_dir: str, run_id: str) -> PipelineResult:
        """使用 Spark 执行处理"""
        from utils.file_parser import FileParser
        from embeddings.chunker import TextChunker

        parser = FileParser()
        chunker = TextChunker()

        # 1. 文件扫描
        files = self._scan_files(input_dir)
        logger.info(f"扫描到 {len(files)} 个文件")

        if not files:
            return PipelineResult(
                run_id=run_id,
                engine=self.name,
                input_dir=input_dir,
                output_dir=output_dir,
                total_files=0,
                total_chunks=0,
                success=True,
            )

        # 2. 并行解析
        sc = spark.sparkContext
        rdd = sc.parallelize(files, numSlices=min(len(files), 8))

        # 解析文件 → (source, text, page_number)
        def parse_file(filepath):
            try:
                filename = os.path.basename(filepath)
                pages = parser.parse_with_pages(filepath)
                results = []
                for p in pages:
                    results.append({
                        "source": filename,
                        "text": p.get("text", ""),
                        "page_number": p.get("page_number", 0),
                        "ext": os.path.splitext(filepath)[1].lower(),
                    })
                return results
            except Exception as e:
                logger.warning(f"解析失败: {filepath}: {e}")
                return []

        parsed_rdd = rdd.flatMap(parse_file)
        parsed_list = parsed_rdd.collect()

        # 3. 数据清洗（单机执行，数据量已缩小）
        min_len = settings.pipeline_min_text_length
        seen_hashes = set()
        clean_stats = CleanStats(total_raw=len(files))
        format_breakdown: Dict[str, int] = {}
        failed_files = []
        cleaned = []

        for item in parsed_list:
            ext = item.get("ext", "")
            format_breakdown[ext] = format_breakdown.get(ext, 0) + 1

            text = item.get("text", "").strip()
            if not text:
                clean_stats.empty_removed += 1
                continue
            if len(text) < min_len:
                clean_stats.short_removed += 1
                continue

            content_hash = hashlib.md5(text.encode()).hexdigest()
            if content_hash in seen_hashes:
                clean_stats.duplicate_removed += 1
                continue
            seen_hashes.add(content_hash)

            cleaned.append(item)

        clean_stats.total_clean = len(cleaned)

        # 4. 并行切片
        def chunk_item(item):
            source = item["source"]
            text = item["text"]
            page_num = item["page_number"]
            raw_chunks = chunker.chunk(text, source=source)
            results = []
            for i, raw_chunk in enumerate(raw_chunks):
                chunk_text = raw_chunk["content"] if isinstance(raw_chunk, dict) else raw_chunk
                results.append(ChunkData(
                    chunk_id=f"{hashlib.md5(f'{source}_{page_num}_{i}_{chunk_text[:50]}'.encode()).hexdigest()[:16]}",
                    content=chunk_text,
                    source=source,
                    page_number=page_num,
                ))
            return results

        cleaned_rdd = sc.parallelize(cleaned, numSlices=min(len(cleaned), 8))
        chunks_rdd = cleaned_rdd.flatMap(chunk_item)
        all_chunks = chunks_rdd.collect()

        # 5. 统计分析
        chunk_stats = self._analyze_chunks(all_chunks)
        top_keywords = self._extract_keywords(all_chunks)

        quality_report = QualityReport(
            clean_stats=clean_stats,
            chunk_stats=chunk_stats,
            top_keywords=top_keywords,
            format_breakdown=format_breakdown,
            failed_files=failed_files,
        )

        # 6. 输出结果
        chunks_file = os.path.join(output_dir, "all_chunks.json")
        stats_file = os.path.join(output_dir, "stats.json")
        quality_file = os.path.join(output_dir, "quality_report.json")

        chunks_data = [c.model_dump() for c in all_chunks]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        stats_data = {
            "run_id": run_id,
            "engine": self.name,
            "timestamp": datetime.now().isoformat(),
            "input_dir": input_dir,
            "output_dir": output_dir,
            "files_scanned": len(files),
            "files_parsed": len(files) - len(failed_files),
            "files_failed": len(failed_files),
            "total_chunks": len(all_chunks),
            "avg_chunk_length": chunk_stats.avg_length,
            "format_breakdown": format_breakdown,
            "top_keywords": [{"word": kw.word, "count": kw.count} for kw in top_keywords[:10]],
        }
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)

        with open(quality_file, "w", encoding="utf-8") as f:
            json.dump(quality_report.model_dump(), f, ensure_ascii=False, indent=2)

        logger.info(f"[SparkEngine] 处理完成: {len(all_chunks)} 个切片")

        return PipelineResult(
            run_id=run_id,
            engine=self.name,
            input_dir=input_dir,
            output_dir=output_dir,
            chunks_file=chunks_file,
            stats_file=stats_file,
            total_files=len(files),
            total_chunks=len(all_chunks),
            quality_report=quality_report,
            success=True,
        )

    def _scan_files(self, input_dir: str, deduplicate: bool = True) -> List[str]:
        """扫描目录中的支持文件，可选按文件内容 MD5 去重"""
        files = []
        if not os.path.exists(input_dir):
            return files
        for f in os.listdir(input_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in settings.allowed_extensions:
                files.append(os.path.join(input_dir, f))

        if not deduplicate:
            return sorted(files)

        # 文件级去重：按内容 MD5 去重，保留第一个（文件名最小的）
        seen_hashes = set()
        unique_files = []
        for fp in sorted(files):
            try:
                with open(fp, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                if file_hash not in seen_hashes:
                    seen_hashes.add(file_hash)
                    unique_files.append(fp)
                else:
                    logger.info(f"[SparkEngine] 跳过重复文件: {os.path.basename(fp)}")
            except Exception as e:
                logger.warning(f"[SparkEngine] 读取文件失败，保留: {fp}: {e}")
                unique_files.append(fp)

        duplicates = len(files) - len(unique_files)
        if duplicates > 0:
            logger.info(f"[SparkEngine] 文件级去重: {len(files)} → {len(unique_files)} (去除 {duplicates} 个重复)")
        return unique_files

    def _analyze_chunks(self, chunks: List[ChunkData]) -> ChunkStats:
        """分析切片统计"""
        if not chunks:
            return ChunkStats()
        lengths = [len(c.content) for c in chunks]
        dist = {"50-100": 0, "101-200": 0, "201-300": 0, "301-500": 0, "500+": 0}
        for l in lengths:
            if l <= 100:
                dist["50-100"] += 1
            elif l <= 200:
                dist["101-200"] += 1
            elif l <= 300:
                dist["201-300"] += 1
            elif l <= 500:
                dist["301-500"] += 1
            else:
                dist["500+"] += 1
        return ChunkStats(
            total_chunks=len(chunks),
            avg_length=round(sum(lengths) / len(lengths), 1),
            min_length=min(lengths),
            max_length=max(lengths),
            length_distribution=dist,
        )

    def _extract_keywords(self, chunks: List[ChunkData], top_n: int = 20) -> List[KeywordStat]:
        """提取高频关键词"""
        try:
            import jieba
        except ImportError:
            return []
        word_count: Dict[str, int] = {}
        for chunk in chunks:
            words = jieba.cut(chunk.content)
            for w in words:
                w = w.strip()
                if len(w) >= 2 and not w.isdigit():
                    word_count[w] = word_count.get(w, 0) + 1
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [KeywordStat(word=w, count=c) for w, c in sorted_words[:top_n]]
