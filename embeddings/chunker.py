"""
文档切片模块
将长文本按指定大小切分为带重叠的片段
支持文档类型差异化切片：PPT 一页一切、Markdown 按标题切、其他结构感知切片
"""
from typing import List, Dict, Optional
import hashlib
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("chunker")


# 结构感知切片的分隔符优先级（中文标点靠前）
_RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", "，", ", ", " ", ""]


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
        self, pages: List[Dict], source: str = "unknown", doc_type: str = None
    ) -> List[Dict]:
        """
        将带页码的文本块切分为多个片段，保留页码信息
        
        根据 doc_type 自动选择切片策略：
        - pptx: 一页一切片（每页是天然语义单元）
        - 其他: 结构感知切片（递归分割符优先级）
        
        Args:
            pages: [{"text": "页面文本", "page": 1}, ...]
            source: 文档来源标识
            doc_type: 文档类型（pptx/md/pdf/docx 等）
        
        Returns:
            切片列表，每项包含 content/source/chunk_id/page_number 等字段
        """
        if not pages:
            return []
        
        # PPT 类型：每页一个 chunk，不拆分
        if doc_type and doc_type.lower() in ("pptx", "ppt"):
            return self._chunk_ppt_by_page(pages, source)
        
        # 其他类型：结构感知切片
        return self._chunk_with_pages_recursive(pages, source)
    
    def _chunk_ppt_by_page(self, pages: List[Dict], source: str) -> List[Dict]:
        """PPT 一页一切片——每页是天然语义单元，不拆分"""
        chunks = []
        for p in pages:
            text = p.get("text", "").strip()
            page_num = p.get("page") or 1
            if not text:
                continue
            
            chunk_id = hashlib.md5(
                f"{source}_{page_num}_{text[:50]}".encode()
            ).hexdigest()[:12]
            
            chunks.append({
                "content": text,
                "source": source,
                "filename": source,
                "chunk_id": chunk_id,
                "index": len(chunks),
                "chunk_index": len(chunks),
                "page_number": page_num,
            })
        
        logger.info(f"PPT [{source}] 一页一切: {len(chunks)} 个切片")
        return chunks
    
    def _chunk_with_pages_recursive(
        self, pages: List[Dict], source: str
    ) -> List[Dict]:
        """带页码的结构感知切片（递归分割符优先级）"""
        PAGE_SEP = "\x00"
        full_text = ""
        page_ranges = []
        
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
        
        # 结构感知切片
        raw_chunks = self._recursive_split(full_text)
        
        # 回溯页码
        chunks = []
        offset = 0
        for content in raw_chunks:
            # 在 full_text 中定位该 chunk 的位置
            chunk_start = full_text.find(content, offset)
            if chunk_start == -1:
                chunk_start = offset
            chunk_end = chunk_start + len(content)
            
            content_clean = content.replace(PAGE_SEP, "").strip()
            if not content_clean:
                offset = chunk_end
                continue
            
            page_number = self._find_page(chunk_start, page_ranges)
            end_page = self._find_page(
                chunk_end - 1 if chunk_end > chunk_start else chunk_start,
                page_ranges,
            )
            
            chunk_id = hashlib.md5(
                f"{source}_{len(chunks)}_{content_clean[:50]}".encode()
            ).hexdigest()[:12]
            
            chunk = {
                "content": content_clean,
                "source": source,
                "filename": source,
                "chunk_id": chunk_id,
                "index": len(chunks),
                "chunk_index": len(chunks),
                "page_number": page_number,
            }
            if end_page and end_page != page_number:
                chunk["page_end"] = end_page
            
            chunks.append(chunk)
            offset = chunk_end
        
        logger.info(
            f"文档 [{source}] 结构感知切片: {len(chunks)} 个片段（含页码信息）"
        )
        return chunks
    
    def _recursive_split(self, text: str) -> List[str]:
        """结构感知递归切片（RecursiveCharacterTextSplitter 思路）
        
        按分隔符优先级递归切分：段落 > 句子 > 字符
        优先在高层级分隔符处切断，避免截断句子
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        # 尝试每个分隔符
        for sep in _RECURSIVE_SEPARATORS:
            if sep == "":
                # 最后兜底：硬切
                chunks = []
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunks.append(text[start:end])
                    start += self.chunk_size - self.chunk_overlap
                return chunks
            
            parts = text.split(sep)
            if len(parts) <= 1:
                continue
            
            # 合并小片段
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # 如果单个 part 就超长，递归用更细的分隔符切
                    if len(part) > self.chunk_size:
                        chunks.extend(self._recursive_split(part))
                        current = ""
                    else:
                        current = part
            if current:
                chunks.append(current)
            
            # 加上 overlap
            if self.chunk_overlap > 0 and len(chunks) > 1:
                overlapped = [chunks[0]]
                for i in range(1, len(chunks)):
                    overlap_text = chunks[i - 1][-self.chunk_overlap:]
                    overlapped.append(overlap_text + chunks[i])
                chunks = overlapped
            
            return chunks
        
        # 不会走到这里，但保险起见
        return [text]
    
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