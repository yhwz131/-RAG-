"""
Reranker 精排模块
使用 bge-reranker-v2-m3 cross-encoder 对召回结果重新打分排序
调用 SiliconFlow /rerank API（与 embedding 同账号）
"""
import httpx
from typing import List, Dict, Optional
from config.settings import settings
from utils.logger import get_logger
logger = get_logger("reranker")


class Reranker:
    """bge-reranker-v2-m3 精排器"""

    def __init__(self):
        self.api_key = settings.embedding_api_key
        self.base_url = settings.embedding_base_url.rstrip("/")
        self.model = settings.reranker_model

    def rerank(
        self,
        query: str,
        docs: List[Dict],
        top_k: int = None,
    ) -> List[Dict]:
        """
        对召回结果精排
        
        Args:
            query: 用户查询
            docs: 召回文档列表（每个 dict 需有 "content" 字段）
            top_k: 返回数量，默认从配置读取
        
        Returns:
            精排后的文档列表，按 rerank_score 降序排列
        """
        if not docs:
            return []
        
        top_k = top_k or settings.reranker_top_k

        # 单条时直接返回（不需要 cross-encoder 打分）
        if len(docs) == 1:
            docs[0]["rerank_score"] = 1.0
            return docs

        pairs = [[query, d.get("content", "")] for d in docs]
        scores = self._call_rerank_api(pairs)

        if scores is None:
            # API 调用失败，降级返回原始顺序
            logger.warning("Reranker API 调用失败，降级为原始排序")
            for doc in docs:
                doc["rerank_score"] = doc.get("score", 0.0)
            return docs[:top_k]

        for doc, score in zip(docs, scores):
            doc["rerank_score"] = score

        docs.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        top_docs = docs[:top_k]

        logger.info(
            f"Reranker 精排: 输入 {len(docs)} 条 → 输出 {len(top_docs)} 条, "
            f"top1={top_docs[0]['rerank_score']:.4f}, "
            f"last={top_docs[-1]['rerank_score']:.4f}"
        )
        return top_docs

    def _call_rerank_api(
        self, pairs: List[List[str]]
    ) -> Optional[List[float]]:
        """
        调用 SiliconFlow /rerank API
        
        Args:
            pairs: [[query, doc], ...] 对列表
        
        Returns:
            相关性分数列表，失败返回 None
        """
        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "query": pairs[0][0],  # query 对所有 pair 相同
            "documents": [p[1] for p in pairs],
            "top_n": len(pairs),
            "return_documents": False,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            # API 返回按 index 排序的结果
            scores = [0.0] * len(pairs)
            for r in results:
                idx = r.get("index", 0)
                scores[idx] = r.get("relevance_score", 0.0)
            return scores

        except Exception as e:
            logger.error(f"Reranker API 错误: {e}")
            return None


def adaptive_filter(results: List[Dict], ratio: float = None) -> List[Dict]:
    """
    相对阈值过滤：保留 top1 * ratio 以上结果
    
    比绝对阈值更稳健——无论检索结果整体偏高还是偏低，都能自适应。
    
    Args:
        results: 已排序的结果列表（第一个分数最高）
        ratio: 保留比例，默认从配置读取
    
    Returns:
        过滤后的结果列表
    """
    if not results:
        return []
    
    ratio = ratio if ratio is not None else settings.adaptive_filter_ratio
    top1_score = results[0].get("rerank_score") or results[0].get("score", 0)
    threshold = top1_score * ratio

    filtered = [
        r for r in results
        if (r.get("rerank_score") or r.get("score", 0)) >= threshold
    ]

    if len(filtered) < len(results):
        logger.info(
            f"相对阈值过滤: {len(results)} → {len(filtered)} 条 "
            f"(top1={top1_score:.4f}, threshold={threshold:.4f}, ratio={ratio})"
        )
    return filtered
