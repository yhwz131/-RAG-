"""
简单处理引擎（本地处理，无 Spark 依赖）
适用于小规模数据或 Spark 不可用时
"""
import os
import json
import hashlib
import uuid
from typing import List, Dict, Any
from datetime import datetime

from ..adapter import PipelineEngine
from ..schema import (
    PipelineResult, ChunkData, CleanStats, ChunkStats,
    KeywordStat, QualityReport
)
from config.settings import settings
from utils.logger import get_logger
from utils.file_parser import FileParser
from embeddings.chunker import TextChunker

logger = get_logger("pipeline.simple")


class SimpleEngine(PipelineEngine):
    """本地处理引擎 — 解析 → 清洗 → 切片 → 统计 → 输出"""

    @property
    def name(self) -> str:
        return "simple"

    def run(self, input_dir: str, output_dir: str) -> PipelineResult:
        """执行本地处理管线"""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"[SimpleEngine] 开始处理: input={input_dir}, output={output_dir}, run_id={run_id}")

        os.makedirs(output_dir, exist_ok=True)
        parser = FileParser()
        chunker = TextChunker()

        # 1. 文件扫描
        files = self._scan_files(input_dir)
        logger.info(f"扫描到 {len(files)} 个文件")

        # 2. 解析 + 清洗（按文件分组，保留 doc_type）
        raw_pages = []
        failed_files = []
        format_breakdown: Dict[str, int] = {}
        file_doc_types: Dict[str, str] = {}  # filename -> doc_type

        for filepath in files:
            ext = os.path.splitext(filepath)[1].lower()
            format_breakdown[ext] = format_breakdown.get(ext, 0) + 1
            filename = os.path.basename(filepath)
            try:
                pages = parser.parse_with_pages(filepath)
                if pages:
                    # 记录 doc_type（去掉前导点）
                    doc_type = ext.lstrip(".") if ext else ""
                    file_doc_types[filename] = doc_type
                    for p in pages:
                        p["_source"] = filename
                        p["_doc_type"] = doc_type
                    raw_pages.extend(pages)
            except Exception as e:
                logger.warning(f"解析失败: {filepath}: {e}")
                failed_files.append({"file": filename, "reason": str(e)})

        # 3. 数据清洗
        clean_pages, clean_stats = self._clean_documents(raw_pages, len(files))
        clean_stats.total_raw = len(files)

        # 4. 文本切片（按文件分组，PPT 走一页一切）
        all_chunks: List[ChunkData] = []

        # 按文件分组
        from collections import defaultdict
        pages_by_file: Dict[str, list] = defaultdict(list)
        for page in clean_pages:
            source = page.get("_source", "unknown")
            pages_by_file[source].append(page)

        for source, file_pages in pages_by_file.items():
            doc_type = file_pages[0].get("_doc_type", "") if file_pages else ""

            if doc_type in ("pptx", "ppt"):
                # PPT：一页一切片
                raw_chunks = chunker.chunk_with_pages(
                    [{"text": p.get("text", ""), "page": p.get("page", 1)} for p in file_pages],
                    source=source, doc_type=doc_type,
                )
                for raw_chunk in raw_chunks:
                    chunk = ChunkData(
                        chunk_id=raw_chunk["chunk_id"],
                        content=raw_chunk["content"],
                        source=source,
                        page_number=raw_chunk.get("page_number", 0),
                    )
                    all_chunks.append(chunk)
            else:
                # 其他格式：逐页切片（结构感知）
                for page in file_pages:
                    source_name = page.get("_source", "unknown")
                    page_num = page.get("page", 0)  # file_parser 返回 "page" 字段
                    text = page.get("text", "").strip()
                    if not text:
                        continue
                    raw_chunks = chunker.chunk(text, source=source_name)
                    for i, raw_chunk in enumerate(raw_chunks):
                        chunk_text = raw_chunk["content"] if isinstance(raw_chunk, dict) else raw_chunk
                        chunk = ChunkData(
                            chunk_id=f"{hashlib.md5(f'{source_name}_{page_num}_{i}_{chunk_text[:50]}'.encode()).hexdigest()[:16]}",
                            content=chunk_text,
                            source=source_name,
                            page_number=page_num,
                        )
                        all_chunks.append(chunk)

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

        # 输出切片数据
        chunks_data = [c.model_dump() for c in all_chunks]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        # 输出统计摘要
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

        # 输出质量报告
        with open(quality_file, "w", encoding="utf-8") as f:
            json.dump(quality_report.model_dump(), f, ensure_ascii=False, indent=2)

        logger.info(f"[SimpleEngine] 处理完成: {len(all_chunks)} 个切片")

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
            logger.warning(f"输入目录不存在: {input_dir}")
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
                    logger.info(f"[SimpleEngine] 跳过重复文件: {os.path.basename(fp)}")
            except Exception as e:
                logger.warning(f"[SimpleEngine] 读取文件失败，保留: {fp}: {e}")
                unique_files.append(fp)

        duplicates = len(files) - len(unique_files)
        if duplicates > 0:
            logger.info(f"[SimpleEngine] 文件级去重: {len(files)} → {len(unique_files)} (去除 {duplicates} 个重复)")
        return unique_files

    def _clean_documents(self, pages: List[Dict], total_files: int) -> tuple:
        """数据清洗：去空、去重、编码标准化、长度过滤"""
        stats = CleanStats(total_raw=total_files)
        seen_hashes = set()
        cleaned = []
        min_len = settings.pipeline_min_text_length

        for page in pages:
            text = page.get("text", "")

            # 去空
            if not text or not text.strip():
                stats.empty_removed += 1
                continue

            # 编码标准化（去除不可见字符）
            text_clean = text.strip()
            if text_clean != text:
                stats.encoding_fixed += 1

            # 长度过滤
            if len(text_clean) < min_len:
                stats.short_removed += 1
                continue

            # 去重（基于内容哈希）
            content_hash = hashlib.md5(text_clean.encode()).hexdigest()
            if content_hash in seen_hashes:
                stats.duplicate_removed += 1
                continue
            seen_hashes.add(content_hash)

            page["text"] = text_clean
            cleaned.append(page)

        stats.total_clean = len(cleaned)
        logger.info(f"清洗完成: 原始 {stats.total_raw} 文件, "
                    f"空文档 {stats.empty_removed}, 重复 {stats.duplicate_removed}, "
                    f"过短 {stats.short_removed}, 保留 {stats.total_clean} 页")
        return cleaned, stats

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
        """提取高频关键词（jieba 分词）"""
        try:
            import jieba
        except ImportError:
            logger.warning("jieba 未安装，跳过关键词提取")
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
