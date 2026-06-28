"""
文档切片模块
将长文本按指定大小切分为带重叠的片段
"""
from typing import List, Dict, Optional
import hashlib
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("chunker")


class TextChunker:
    """文本切片器，支持按字符数切分，带重叠区域"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        logger.info(f"初始化切片器: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def chunk(self, text: str, source: str = "unknown") -> List[Dict]:
        """
        将文本切分为多个片段
        
        Args:
            text: 原始文本
            source: 文档来源标识
        
        Returns:
            切片列表，每个切片包含 content, source, chunk_id
        """
        if not text or not text.strip():
            return []
        
        # 清理文本
        text = text.strip()
        
        chunks = []
        start = 0
        idx = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # 句子边界对齐：尽量在句号/换行处切断，避免截断句子
            if end < len(text):
                for sep in ['\n', '。', '！', '？', '.', '!', '?', '；', '，', ',']:
                    last_sep = text.rfind(sep, start + self.chunk_size // 2, end)
                    if last_sep > start:
                        end = last_sep + 1
                        break
            
            content = text[start:end]
            
            # 生成唯一ID
            chunk_id = hashlib.md5(
                f"{source}_{idx}_{content[:50]}".encode()
            ).hexdigest()[:12]
            
            chunks.append({
                "content": content,
                "source": source,
                "filename": source,  # 兼容 retriever
                "chunk_id": chunk_id,
                "index": idx,
                "chunk_index": idx  # 兼容 retriever
            })
            
            idx += 1
            start += self.chunk_size - self.chunk_overlap
        
        logger.info(f"文档 [{source}] 切分为 {len(chunks)} 个片段")
        return chunks

    def chunk_with_pages(
        self, pages: List[Dict], source: str = "unknown"
    ) -> List[Dict]:
        """
        将带页码的文本块切分为多个片段，保留页码信息
        
        Args:
            pages: [{"text": "页面文本", "page": 1}, ...]
                   page 为 None 时（非 PDF）统一标记为第 1 页
            source: 文档来源标识
        
        Returns:
            切片列表，每项包含 content/source/chunk_id/page_number 等字段
        """
        if not pages:
            return []
        
        # 拼接全文，记录每页的字符偏移范围
        # 使用特殊分隔符标记页边界，便于回溯定位
        PAGE_SEP = "\x00"
        full_text = ""
        page_ranges = []   # [(start, end, page_num), ...]
        
        for p in pages:
            text = p.get("text", "").strip()
            page_num = p.get("page") or 1
            if not text:
                continue
            
            start = len(full_text)
            if full_text:
                full_text += PAGE_SEP
                start = len(full_text)
            full_text += text
            end = len(full_text)
            page_ranges.append((start, end, page_num))
        
        if not full_text.strip():
            return []
        
        # 滑动窗口切分
        chunks = []
        start = 0
        idx = 0
        
        while start < len(full_text):
            end = min(start + self.chunk_size, len(full_text))
            content = full_text[start:end]
            
            # 去除分隔符
            content = content.replace(PAGE_SEP, "").strip()
            if not content:
                start += self.chunk_size - self.chunk_overlap
                continue
            
            # 定位该切片覆盖的页码范围
            page_number = self._find_page(start, page_ranges)
            end_page = self._find_page(
                end - 1 if end > start else start, page_ranges
            )
            
            chunk_id = hashlib.md5(
                f"{source}_{idx}_{content[:50]}".encode()
            ).hexdigest()[:12]
            
            chunk = {
                "content": content,
                "source": source,
                "filename": source,
                "chunk_id": chunk_id,
                "index": idx,
                "chunk_index": idx,
                "page_number": page_number,
            }
            # 跨页切片记录页码范围
            if end_page and end_page != page_number:
                chunk["page_end"] = end_page
            
            chunks.append(chunk)
            idx += 1
            start += self.chunk_size - self.chunk_overlap
        
        logger.info(
            f"文档 [{source}] 切分为 {len(chunks)} 个片段（含页码信息）"
        )
        return chunks
    
    @staticmethod
    def _find_page(
        char_pos: int, page_ranges: List[tuple]
    ) -> Optional[int]:
        """根据字符偏移量查找所属页码"""
        for start, end, page_num in page_ranges:
            if start <= char_pos < end:
                return page_num
        # 超出范围时返回最后一页
        if page_ranges:
            return page_ranges[-1][2]
        return None