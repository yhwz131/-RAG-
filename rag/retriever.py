"""
检索模块
基于 Milvus Lite 本地向量数据库进行相似度搜索，支持混合检索（向量 + BM25）
"""
import os
import time
import hashlib
import base64
import httpx
from typing import List, Dict, Optional
from pymilvus import MilvusClient
from config.settings import settings
from utils.logger import get_logger
from embeddings.embedder import EmbeddingClient, MultimodalEmbedder
from rag.reranker import Reranker, adaptive_filter

logger = get_logger("retriever")


class VectorRetriever:
    """向量检索器，连接 Milvus Lite 本地数据库（使用 pymilvus 3.0 MilvusClient API）"""
    
    _client: MilvusClient
    
    def __init__(self, embedder: Optional[EmbeddingClient] = None):
        self.embedder = embedder or EmbeddingClient()
        self.collection_name = settings.milvus_collection
        self._connect()
        self._ensure_collection()
        self._bm25_docs: List[Dict] = []  # 缓存文档用于BM25检索
        self._bm25_instance = None  # BM25Okapi 实例缓存
        self._corpus_tokens: List[List[str]] = []  # 分词结果缓存
        self._reranker = Reranker() if settings.reranker_enabled else None
        self._update_bm25_cache()

    def close(self):
        """关闭底层资源（EmbeddingClient 连接池 + Milvus 连接）"""
        try:
            if hasattr(self, 'embedder') and self.embedder:
                self.embedder.close()
            if hasattr(self, '_client') and self._client:
                self._client.close()
            logger.info("VectorRetriever 资源已释放")
        except Exception as e:
            logger.warning(f"VectorRetriever 关闭异常: {e}")
    
    def _connect(self):
        """连接 Milvus Lite（本地文件模式）"""
        try:
            uri = settings.milvus_uri
            self._client = MilvusClient(uri=uri)
            logger.info(f"已连接 Milvus Lite: {uri}")
        except Exception as e:
            logger.error(f"Milvus 连接失败: {e}")
            raise
    
    def _ensure_collection(self):
        """确保 Collection 存在"""
        if self._client.has_collection(self.collection_name):
            self._client.load_collection(self.collection_name)
            logger.info(f"已加载 Collection: {self.collection_name}")
            return
        
        # 创建 Collection（使用简化 API，自动创建 schema 和索引）
        self._client.create_collection(
            collection_name=self.collection_name,
            dimension=settings.embedding_dim,
            metric_type="COSINE",
            auto_id=True,
            vector_field_name="embedding",
        )
        self._client.load_collection(self.collection_name)
        logger.info(f"已创建 Collection: {self.collection_name}")
    
    def _update_bm25_cache(self):
        """更新 BM25 检索缓存（分批加载，突破单次查询限制）"""
        try:
            self._client.load_collection(self.collection_name)
            all_results = []
            offset = 0
            batch_size = 10000
            while True:
                batch = self._client.query(
                    self.collection_name,
                    filter='chunk_id != ""',
                    output_fields=["content", "filename", "chunk_id", "page_number"],
                    limit=batch_size,
                    offset=offset
                )
                if not batch:
                    break
                all_results.extend(batch)
                if len(batch) < batch_size:
                    break
                offset += batch_size
            self._bm25_docs = all_results
            # 构建 BM25 实例（分词 + 构建一次性完成）
            try:
                import jieba
                from rank_bm25 import BM25Okapi
                self._corpus_tokens = [list(jieba.cut(doc.get("content", ""))) for doc in self._bm25_docs]
                self._bm25_instance = BM25Okapi(self._corpus_tokens)
            except ImportError:
                logger.warning("jieba/rank_bm25 未安装，BM25 检索不可用")
                self._bm25_instance = None
            logger.info(f"BM25 缓存更新完成: {len(self._bm25_docs)} 条文档")
        except Exception as e:
            logger.warning(f"BM25 缓存更新失败: {e}")
            self._bm25_docs = []
            self._bm25_instance = None
            self._corpus_tokens = []
    
    def insert_documents(self, chunks: List[Dict]) -> int:
        """将文档切片插入 Milvus
        
        Args:
            chunks: 切片列表，每个切片包含 content, filename, chunk_id, doc_id, chunk_index
        
        Returns:
            插入的切片数量
        """
        if not chunks:
            return 0
        
        # 批量生成向量
        contents = [c["content"] for c in chunks]
        embeddings = self.embedder.embed_documents(contents)
        
        # 构造 dict 列表（MilvusClient.insert 接受 List[Dict]）
        data = []
        for i, c in enumerate(chunks):
            data.append({
                "chunk_id": c["chunk_id"],
                "doc_id": c.get("doc_id", ""),
                "filename": c.get("filename", "未知"),
                "content": c["content"],
                "chunk_index": c.get("chunk_index", 0),
                "page_number": c.get("page_number", 0),
                "embedding": embeddings[i],
            })
        
        self._client.insert(self.collection_name, data)
        self._client.flush(self.collection_name)
        
        count = len(chunks)
        logger.info(f"已插入 {count} 个文档切片到 Milvus")
        
        # 更新 BM25 缓存
        self._update_bm25_cache()
        
        return count
    
    def list_documents(self) -> List[Dict]:
        """列出知识库中的所有文档（按文件名分组）
        
        Returns:
            文档列表，每项包含 filename 和 chunk_count
        """
        try:
            self._client.load_collection(self.collection_name)
            results = self._client.query(
                self.collection_name,
                filter='chunk_id != ""',
                output_fields=["filename"],
                limit=10000
            )
            # 按文件名分组统计
            doc_map: Dict[str, int] = {}
            for r in results:
                fname = r.get("filename", "未知")
                doc_map[fname] = doc_map.get(fname, 0) + 1
            
            docs = [{"filename": k, "chunk_count": v} for k, v in doc_map.items()]
            docs.sort(key=lambda x: x["filename"])
            return docs
        except Exception as e:
            logger.error(f"列出文档失败: {e}")
            return []
    
    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除所有关联的向量切片
        
        Args:
            filename: 要删除的文件名
        
        Returns:
            删除的切片数量
        """
        try:
            self._client.load_collection(self.collection_name)
            # 先查询该文件有多少条记录
            results = self._client.query(
                self.collection_name,
                filter=f'filename == "{filename}"',
                output_fields=["chunk_id"],
                limit=10000
            )
            if not results:
                logger.warning(f"未找到文件 [{filename}] 的向量数据")
                return 0
            
            count = len(results)
            # MilvusClient.delete 使用 filter 表达式删除
            self._client.delete(
                self.collection_name,
                filter=f'filename == "{filename}"',
            )
            self._client.flush(self.collection_name)
            
            # 更新 BM25 缓存
            self._update_bm25_cache()
            
            logger.info(f"已删除文件 [{filename}] 的 {count} 个向量切片")
            return count
        except Exception as e:
            logger.error(f"删除文件 [{filename}] 向量数据失败: {e}")
            raise
    
    def _bm25_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25 关键词检索（使用缓存的 BM25 实例，仅分词查询）"""
        if not self._bm25_docs or self._bm25_instance is None:
            return []
        
        try:
            import jieba
            
            # 仅对查询分词（语料已在 _update_bm25_cache 中预分词）
            query_tokens = list(jieba.cut(query))
            
            # 使用缓存的 BM25 实例
            scores = self._bm25_instance.get_scores(query_tokens)
            
            # 获取 Top-K 结果
            import numpy as np
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append({
                        "content": self._bm25_docs[idx].get("content", ""),
                        "source": self._bm25_docs[idx].get("filename", "未知"),
                        "chunk_id": self._bm25_docs[idx].get("chunk_id", ""),
                        "page_number": self._bm25_docs[idx].get("page_number", 0),
                        "score": float(scores[idx]),
                        "source_type": "keyword"
                    })

            # 调试日志
            for i, r in enumerate(results):
                logger.info(
                    f"  BM25[{i+1}] 来源={r['source']}, 页码={r['page_number']}, "
                    f"bm25_score={r['score']:.4f}"
                )
            return results
        except Exception as e:
            logger.warning(f"BM25 检索失败: {e}")
            return []
    
    def _vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """向量相似度检索
        
        注意: Milvus COSINE metric 返回的 distance 是 1 - cosine_similarity，
        值域 [0, 2]，其中 0 表示完全相同，1 表示正交，2 表示相反。
        本方法将其转换为 similarity = 1 - distance，值域 [-1, 1]，越大越相似。
        """
        query_vector = self.embedder.embed_query(query)
        
        results = self._client.search(
            self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["content", "filename", "chunk_id", "chunk_index", "page_number"],
            search_params={"metric_type": "COSINE"},
        )
        
        docs = []
        hits = results[0] if results else []
        for hit in hits:
            entity = hit.get("entity", {})
            # COSINE metric: distance = 1 - cosine_similarity
            # 转换为 similarity = 1 - distance，值域 [-1, 1]，越大越相似
            distance = hit.get("distance", 0.0)
            similarity = 1.0 - distance
            docs.append({
                "content": entity.get("content"),
                "source": entity.get("filename", "未知"),
                "chunk_id": entity.get("chunk_id", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "page_number": entity.get("page_number", 0),
                "score": similarity,
                "source_type": "vector"
            })

        # 调试日志
        for i, d in enumerate(docs):
            logger.info(
                f"  向量[{i+1}] 来源={d['source']}, 页码={d['page_number']}, "
                f"cosine_sim={d['score']:.4f}"
            )
        return docs
    
    def _rrf_fusion(self, vector_results: List[Dict], keyword_results: List[Dict], k: int = None) -> List[Dict]:
        """RRF (Reciprocal Rank Fusion) 融合排序（加权版）
        
        向量检索权重更高（语义匹配更可靠），BM25 权重较低（关键词匹配噪音多）。
        
        Args:
            k: RRF 参数，默认从配置读取
        """
        k = k or settings.rrf_k
        VECTOR_WEIGHT = 1.5   # 向量检索权重
        KEYWORD_WEIGHT = 0.8  # BM25 权重
        scores = {}
        doc_map = {}
        
        for rank, doc in enumerate(vector_results):
            key = doc.get("chunk_id") or doc.get("content", "")[:100]
            scores[key] = scores.get(key, 0) + VECTOR_WEIGHT / (k + rank + 1)
            doc_map[key] = doc
        
        for rank, doc in enumerate(keyword_results):
            key = doc.get("chunk_id") or doc.get("content", "")[:100]
            scores[key] = scores.get(key, 0) + KEYWORD_WEIGHT / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc
        
        # 按 RRF 分数排序
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        results = []
        for key in sorted_keys:
            doc = doc_map[key].copy()
            doc["score"] = scores[key]
            doc["source_type"] = "fusion"
            results.append(doc)
        
        return results
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """混合检索：向量检索 + BM25 + RRF 融合 + Reranker 精排
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            检索结果列表
        """
        top_k = top_k or settings.top_k
        candidate_k = top_k * 3  # 扩大候选池
        
        # 1. 向量检索
        vector_results = self._vector_search(query, top_k=candidate_k)
        
        # 2. BM25 关键词检索
        keyword_results = self._bm25_search(query, top_k=candidate_k)
        
        # 3. RRF 融合排序 + 分路径阈值过滤
        if keyword_results:
            fused_results = self._rrf_fusion(vector_results, keyword_results)
            # RRF 路径：用 rrf_threshold（值域 ~0.01-0.05）
            rrf_threshold = settings.rrf_threshold
            before = len(fused_results)
            if rrf_threshold > 0:
                fused_results = [r for r in fused_results if r.get("score", 0) >= rrf_threshold]
                if len(fused_results) < before:
                    logger.info(
                        f"RRF 阈值过滤: {before} → {len(fused_results)} 条 "
                        f"(rrf_threshold={rrf_threshold})"
                    )
        else:
            # 纯向量路径：用 similarity_threshold（值域 ~0.3-0.9）
            sim_threshold = settings.similarity_threshold
            before = len(vector_results)
            fused_results = [r for r in vector_results if r.get("score", 0) >= sim_threshold]
            if len(fused_results) < before:
                logger.info(
                    f"相似度阈值过滤: {before} → {len(fused_results)} 条 "
                    f"(similarity_threshold={sim_threshold})"
                )
        
        # 4. Reranker 精排（cross-encoder 重打分）
        if self._reranker and fused_results:
            fused_results = self._reranker.rerank(query, fused_results, top_k=top_k)
            # 相对阈值过滤
            fused_results = adaptive_filter(fused_results)
        
        results = fused_results[:top_k]
        logger.info(f"检索完成: 向量={len(vector_results)}, 关键词={len(keyword_results)}, 最终={len(results)}")
        # 调试日志：打印最终结果详情
        for i, r in enumerate(results):
            score_key = "rerank_score" if "rerank_score" in r else "score"
            logger.info(
                f"  [{i+1}] 来源={r.get('source', '?')}, "
                f"页码={r.get('page_number', 0)}, "
                f"分数={r.get(score_key, 0):.4f}, "
                f"类型={r.get('source_type', '?')}"
            )
        return results
    
    def multi_search(self, queries: List[str], original_query: str = "", top_k: int = None, skip_reranker: bool = False) -> List[Dict]:
        """多查询检索：对每个子查询独立检索，合并去重后统一重排

        用于跨文档查询——拆分为多个聚焦子查询分别检索，
        避免单个查询 embedding 被主导话题"吃掉"。

        Args:
            queries: 子查询列表
            original_query: 原始查询（用于 reranker 重排）
            top_k: 最终返回数量
            skip_reranker: 是否跳过 reranker（跨文档场景用 RRF+多样性代替，
                避免 reranker 偏向主查询主题导致次要主题文档被排挤）

        Returns:
            合并去重后的检索结果
        """
        top_k = top_k or settings.top_k
        candidate_k = top_k * 4  # 子查询各取更多候选，合并后再截断

        # 分别对每个子查询做 RRF 融合，然后合并
        all_fused = []
        for i, q in enumerate(queries):
            logger.info(f"  子查询[{i+1}]: {q}")
            vec_res = self._vector_search(q, top_k=candidate_k)
            kw_res = self._bm25_search(q, top_k=candidate_k)
            if kw_res:
                fused = self._rrf_fusion(vec_res, kw_res)
            else:
                fused = vec_res
            logger.info(f"    子查询[{i+1}] 候选: vec={len(vec_res)}, kw={len(kw_res)}, fused={len(fused)}")
            all_fused.extend(fused)

        # 全局去重：按 chunk_id 去重，保留最高分
        seen = {}
        for doc in all_fused:
            key = doc.get("chunk_id") or doc.get("content", "")[:100]
            if key not in seen or doc.get("score", 0) > seen[key].get("score", 0):
                seen[key] = doc
        merged = list(seen.values())
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        logger.info(f"    去重详情: {len(all_fused)} → {len(merged)} unique keys")
        for i, m in enumerate(merged[:8]):
            logger.info(f"      [{i+1}] {m.get('source','?')} key={m.get('chunk_id','')[:20] or m.get('content','')[:30]} score={m.get('score',0):.4f}")

        # Reranker 精排
        # 跨文档场景跳过 reranker：用 RRF 分数 + 来源多样性代替
        # 原因：reranker 用原始查询打分时偏向主查询主题，次要主题文档被排挤
        if not skip_reranker:
            rerank_query = original_query or queries[0]
            if self._reranker and merged:
                merged = self._reranker.rerank(rerank_query, merged, top_k=top_k * 3)
        else:
            logger.info("  跨文档模式：跳过 reranker，使用 RRF+多样性")

        # 保证来源多样性：优先从不同文档中取结果
        results = self._diversify_by_source(merged, top_k)
        logger.info(
            f"多查询检索完成: {len(queries)} 个子查询, "
            f"融合候选={len(all_fused)}, 去重后={len(merged)}, 最终={len(results)}"
        )
        for i, r in enumerate(results):
            score_key = "rerank_score" if "rerank_score" in r else "score"
            logger.info(
                f"  [{i+1}] 来源={r.get('source', '?')}, "
                f"页码={r.get('page_number', 0)}, "
                f"分数={r.get(score_key, 0):.4f}"
            )
        return results

    @staticmethod
    def _diversify_by_source(sorted_docs: List[Dict], top_k: int) -> List[Dict]:
        """保证来源多样性的选取策略。

        对于跨文档查询，reranker 可能把来自同一文档的所有切片排在前面，
        导致返回结果全是同一份文档。此方法在选取时优先从不同来源中各取一条，
        确保 MQR 拆分出的每个子查询对应的文档都有机会出现在最终结果中。

        Args:
            sorted_docs: 已按 rerank_score 降序排列的文档列表
            top_k: 最终返回数量

        Returns:
            来源多样化的 top_k 条结果
        """
        if len(sorted_docs) <= top_k:
            return sorted_docs

        # 按 source 分组，保持组内 rerank_score 顺序
        from collections import defaultdict
        by_source: Dict[str, List[Dict]] = defaultdict(list)
        for doc in sorted_docs:
            src = doc.get("source", "未知")
            by_source[src].append(doc)

        sources = list(by_source.keys())
        results = []
        seen_sources_round = set()  # 当前轮次已选的来源

        # 轮询选取：每轮从每个来源中各取一条（跳过已耗尽的来源）
        round_idx = 0
        while len(results) < top_k:
            added = False
            for src in sources:
                if len(results) >= top_k:
                    break
                if round_idx < len(by_source[src]):
                    results.append(by_source[src][round_idx])
                    added = True
            round_idx += 1
            if not added:
                break  # 所有来源都已耗尽

        logger.debug(
            f"来源多样化: {len(sorted_docs)} → {len(results)} "
            f"({len(sources)} 个来源, {[f'{s}×{len(by_source[s])}' for s in sources]})"
        )
        return results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """根据文档ID删除所有相关切片"""
        try:
            expr = f'doc_id == "{doc_id}"'
            # 先查询要删除的数量
            results = self._client.query(self.collection_name, filter=expr, output_fields=["id"], limit=10000)
            count = len(results)
            
            if count > 0:
                self._client.delete(self.collection_name, filter=expr)
                self._client.flush(self.collection_name)
                logger.info(f"已删除文档 {doc_id} 的 {count} 个切片")
                self._update_bm25_cache()
            
            return count
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise
    
    def get_document_count(self) -> int:
        """获取文档切片总数"""
        try:
            stats = self._client.get_collection_stats(self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def list_all_files(self) -> List[Dict]:
        """列出知识库中所有文件（枚举查询，不走向量检索）

        Returns:
            [{"filename": "xxx.pdf", "chunk_count": 5}, ...]
        """
        try:
            self._client.load_collection(self.collection_name)
            all_docs = self._client.query(
                self.collection_name,
                filter='chunk_id != ""',
                output_fields=["filename"],
                limit=10000,
            )
            # 统计每个文件的切片数
            file_counts: dict[str, int] = {}
            for doc in all_docs:
                fname = doc.get("filename", "未知")
                file_counts[fname] = file_counts.get(fname, 0) + 1
            result = [
                {"filename": fname, "chunk_count": count}
                for fname, count in sorted(file_counts.items())
            ]
            logger.info(f"枚举查询: 共 {len(result)} 个文件, {len(all_docs)} 个切片")
            return result
        except Exception as e:
            logger.error(f"枚举文件列表失败: {e}")
            return []


class MultimodalRetriever:
    """多模态检索器，独立 Collection (knowledge_base_mm)，4096 维向量

    与 VectorRetriever 接口一致，可直接注入 RAGChain。
    额外支持图片切片入库（insert_images）。
    """

    def __init__(self, embedder: Optional[MultimodalEmbedder] = None):
        self.embedder = embedder or MultimodalEmbedder()
        self.collection_name = settings.multimodal_collection_name
        self._connect()
        self._ensure_collection()
        self._bm25_docs: List[Dict] = []
        self._bm25_instance = None  # BM25Okapi 实例缓存
        self._corpus_tokens: List[List[str]] = []  # 分词结果缓存
        self._reranker = Reranker() if settings.reranker_enabled else None
        # 图片描述 LLM 专用连接池（复用连接，避免每次调用新建）
        self._mm_llm_client = httpx.Client(
            timeout=30,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=3, keepalive_expiry=30),
        )
        self._update_bm25_cache()

    def close(self):
        """关闭底层资源（MultimodalEmbedder 连接池 + Milvus 连接 + LLM httpx）"""
        try:
            if hasattr(self, '_mm_llm_client') and self._mm_llm_client:
                self._mm_llm_client.close()
            if hasattr(self, 'embedder') and self.embedder:
                self.embedder.close()
            if hasattr(self, '_client') and self._client:
                self._client.close()
            logger.info("MultimodalRetriever 资源已释放")
        except Exception as e:
            logger.warning(f"MultimodalRetriever 关闭异常: {e}")

    def _connect(self):
        """连接 Milvus Lite"""
        try:
            self._client = MilvusClient(uri=settings.milvus_uri)
            logger.info(f"多模态检索器已连接 Milvus Lite: {settings.milvus_uri}")
        except Exception as e:
            logger.error(f"多模态检索器 Milvus 连接失败: {e}")
            raise

    def _ensure_collection(self):
        """确保多模态 Collection 存在（4096 维）"""
        if self._client.has_collection(self.collection_name):
            self._client.load_collection(self.collection_name)
            logger.info(f"已加载多模态 Collection: {self.collection_name}")
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            dimension=settings.multimodal_embedding_dim,
            metric_type="COSINE",
            auto_id=True,
            vector_field_name="embedding",
        )
        self._client.load_collection(self.collection_name)
        logger.info(f"已创建多模态 Collection: {self.collection_name}")

    def _update_bm25_cache(self):
        """更新 BM25 检索缓存（分批加载，突破单次查询限制）"""
        try:
            self._client.load_collection(self.collection_name)
            all_results = []
            offset = 0
            batch_size = 10000
            while True:
                batch = self._client.query(
                    self.collection_name,
                    filter='chunk_id != ""',
                    output_fields=["content", "filename", "chunk_id", "page_number", "has_image", "image_url"],
                    limit=batch_size,
                    offset=offset
                )
                if not batch:
                    break
                all_results.extend(batch)
                if len(batch) < batch_size:
                    break
                offset += batch_size
            self._bm25_docs = all_results
            # 构建 BM25 实例（分词 + 构建一次性完成）
            try:
                import jieba
                from rank_bm25 import BM25Okapi
                self._corpus_tokens = [list(jieba.cut(doc.get("content", ""))) for doc in self._bm25_docs]
                self._bm25_instance = BM25Okapi(self._corpus_tokens)
            except ImportError:
                logger.warning("jieba/rank_bm25 未安装，多模态 BM25 检索不可用")
                self._bm25_instance = None
            logger.info(f"多模态 BM25 缓存更新完成: {len(self._bm25_docs)} 条")
        except Exception as e:
            logger.warning(f"多模态 BM25 缓存更新失败: {e}")
            self._bm25_docs = []
            self._bm25_instance = None
            self._corpus_tokens = []

    def insert_documents(self, chunks: List[Dict]) -> int:
        """将文档切片插入多模态 Collection

        文本内容用多模态 Embedder 向量化。
        如果 chunk 含 image_b64 字段，用 embed_image() 向量化。
        """
        if not chunks:
            return 0

        # 逐条判断是否有图片
        embeddings = []
        for c in chunks:
            if c.get("image_b64"):
                emb = self.embedder.embed_image(
                    c["image_b64"], description=c.get("content", "")
                )
            else:
                emb = self.embedder.embed_text(c["content"])
            embeddings.append(emb)

        data = []
        for i, c in enumerate(chunks):
            data.append({
                "chunk_id": c["chunk_id"],
                "doc_id": c.get("doc_id", ""),
                "filename": c.get("filename", "未知"),
                "content": c["content"],
                "chunk_index": c.get("chunk_index", 0),
                "page_number": c.get("page_number", 0),
                "has_image": bool(c.get("image_b64")),
                "image_url": c.get("image_url", ""),
                "embedding": embeddings[i],
            })

        self._client.insert(self.collection_name, data)
        self._client.flush(self.collection_name)

        count = len(chunks)
        logger.info(f"已插入 {count} 个切片到多模态 Collection")
        self._update_bm25_cache()
        return count

    def insert_images(self, images: List[Dict], source: str = "unknown") -> int:
        """将 PDF 提取的图片入库到多模态 Collection

        Args:
            images: [{"page": 1, "path": "/path/img.png", "b64": "..."}]
            source: 来源文件名
        """
        if not images:
            return 0

        data = []
        for img in images:
            page = img.get("page", 0)
            b64 = img.get("b64", "")
            img_path = img.get("path", "")

            # 用 embed_image 向量化图片
            emb = self.embedder.embed_image(b64)

            # 图片去重：基于内容 hash
            content_hash = hashlib.md5(
                (b64[:256] if len(b64) > 256 else b64).encode()
            ).hexdigest()[:12]
            chunk_id = hashlib.md5(
                f"{source}_img_p{page}_{img_path}".encode()
            ).hexdigest()[:12]

            # 检查是否已存在（基于 chunk_id 去重）
            try:
                existing = self._client.query(
                    self.collection_name,
                    filter=f'chunk_id == "{chunk_id}"',
                    limit=1
                )
                if existing:
                    logger.debug(f"图片已存在，跳过: {chunk_id} (来源: {source}, 第{page}页)")
                    continue
            except Exception:
                pass  # 查询失败不阻塞入库

            # P0: 用 LLM 生成图片内容描述
            description = self._generate_image_description(b64, source, page)

            data.append({
                "chunk_id": chunk_id,
                "doc_id": "",
                "filename": source,
                "content": description,
                "chunk_index": -1,  # -1 表示图片块
                "page_number": page,
                "has_image": True,
                "image_url": img_path,
                "embedding": emb,
            })

        if not data:
            logger.info("所有图片均已存在，无新图片入库")
            return 0

        self._client.insert(self.collection_name, data)
        self._client.flush(self.collection_name)

        logger.info(f"已插入 {len(data)} 张图片到多模态 Collection")
        self._update_bm25_cache()
        return len(data)

    def refresh_image_descriptions(self, source: str = None) -> int:
        """刷新图片描述：为没有有效描述的图片重新生成描述

        Args:
            source: 指定来源文件名（可选），为空则刷新所有图片

        Returns:
            成功刷新的图片数量
        """
        # 查询所有图片记录（chunk_index == -1 表示图片块）
        filter_expr = 'chunk_index == -1'
        if source:
            filter_expr = f'chunk_index == -1 and filename == "{source}"'

        try:
            results = self._client.query(
                self.collection_name,
                filter=filter_expr,
                output_fields=["id", "chunk_id", "filename", "content", "page_number", "image_url", "embedding", "has_image", "doc_id", "chunk_index"],
                limit=1000,
            )
        except Exception as e:
            logger.error(f"查询图片记录失败: {e}")
            return 0

        if not results:
            logger.info("未找到需要刷新的图片记录")
            return 0

        # 识别需要刷新的图片：content 为默认标签或包含失败关键词
        failure_keywords = ["无法看到", "无法识别", "抱歉", "对不起", "请重新上传"]
        need_refresh = []
        for r in results:
            content = r.get("content", "")
            # 默认标签格式: "[图片] 来源: xxx, 第x页" (无实际描述)
            is_default = content.startswith("[图片] 来源:") and "（来源:" not in content
            # 包含失败关键词
            is_failure = any(kw in content for kw in failure_keywords)
            if is_default or is_failure:
                need_refresh.append(r)

        if not need_refresh:
            logger.info(f"所有 {len(results)} 张图片均已有有效描述，无需刷新")
            return 0

        logger.info(f"找到 {len(need_refresh)} 张需要刷新的图片（共 {len(results)} 张）")

        # 从 image_url 读取图片文件并重新生成描述
        refreshed = 0
        for r in need_refresh:
            chunk_id = r.get("chunk_id", "")
            filename = r.get("filename", "unknown")
            page = r.get("page_number", 0)
            image_url = r.get("image_url", "")

            if not image_url:
                logger.warning(f"图片无 image_url，跳过: {chunk_id}")
                continue

            # 读取图片文件并转 base64
            img_path = image_url
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.getcwd(), img_path)

            if not os.path.exists(img_path):
                logger.warning(f"图片文件不存在，跳过: {img_path}")
                continue

            try:
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
            except Exception as e:
                logger.warning(f"读取图片文件失败: {img_path}, {e}")
                continue

            # 重新生成描述
            new_desc = self._generate_image_description(b64, filename, page)

            # 更新记录：直接用原始记录（已包含 id、embedding 等全部字段）
            try:
                r["content"] = new_desc
                self._client.upsert(self.collection_name, [r])
                refreshed += 1
                logger.info(f"刷新图片描述成功: {filename} 第{page}页")
            except Exception as e:
                logger.warning(f"更新图片描述失败: {chunk_id}, {e}")

        logger.info(f"图片描述刷新完成: {refreshed}/{len(need_refresh)} 成功")
        self._update_bm25_cache()
        return refreshed

    def _generate_image_description(self, image_b64: str, source: str, page: int, max_retries: int = 2) -> str:
        """用 LLM 生成图片内容描述（vision 能力）

        Args:
            image_b64: 图片的 base64 编码
            source: 来源文件名
            page: 页码
            max_retries: 最大重试次数（默认 2，共尝试 3 次）

        Returns:
            图片描述文本，失败时降级为默认标签
        """
        # 失败关键词列表：LLM 无法识别图片时的常见回复
        failure_keywords = [
            "无法看到", "无法识别", "无法查看", "无法分析",
            "没有看到", "没有收到", "没有找到", "没有办法",
            "未收到", "未提供",
            "抱歉", "对不起", "请重新上传", "请提供",
            "cannot see", "cannot identify", "unable to",
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.llm_api_key}",
        }
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "请用中文简要描述这张图片的内容（50字以内），只描述图片核心内容，不要添加额外说明。"
                        ),
                    },
                ],
            }
        ]
        payload = {
            "model": settings.mm_llm_model,
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0,
        }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._mm_llm_client.post(
                    settings.mm_llm_api_url, json=payload, headers=headers
                )
                resp.raise_for_status()
                desc = resp.json()["choices"][0]["message"]["content"].strip()

                # 验证描述是否有效（过滤 LLM 无法识别的回复）
                desc_lower = desc.lower()
                is_failure = any(kw in desc_lower for kw in failure_keywords)
                if is_failure:
                    last_error = f"LLM 返回无效描述: {desc[:50]}"
                    logger.warning(
                        f"图片描述生成返回无效内容 (尝试 {attempt + 1}/{max_retries + 1}): "
                        f"{source} 第{page}页 -> {desc[:80]}"
                    )
                    if attempt < max_retries:
                        time.sleep(1 * (2 ** attempt))  # 指数退避: 1s, 2s
                    continue  # 重试

                # 有效描述
                logger.info(
                    f"图片描述生成成功: {source} 第{page}页 -> {desc[:50]}..."
                )
                return f"[图片] {desc}（来源: {source}, 第{page}页）"

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"图片描述生成异常 (尝试 {attempt + 1}/{max_retries + 1}): "
                    f"{source} 第{page}页 -> {e}"
                )
                if attempt < max_retries:
                    time.sleep(1 * (2 ** attempt))  # 指数退避: 1s, 2s
                continue  # 重试

        # 所有重试均失败，降级为默认标签
        logger.warning(
            f"图片描述生成最终失败，使用默认标签: {source} 第{page}页, 最后错误: {last_error}"
        )
        return f"[图片] 来源: {source}, 第{page}页"

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """多模态混合检索：向量检索 + BM25 关键词 + RRF 融合 + 相似度阈值

        策略：
        1. Qwen3-VL 4096d embedding 做语义相似度检索
        2. BM25 关键词检索补充精确匹配能力
        3. RRF 融合两路结果
        4. 相似度阈值过滤低质量结果

        去重策略：同一来源文件只保留分数最高的 1 条，避免单文件霸榜。
        """
        top_k = top_k or settings.top_k
        candidate_k = max(top_k * 10, 50)

        # 1. 向量检索（用多模态模型编码查询）
        query_vector = self.embedder.embed_text(query)
        results = self._client.search(
            self.collection_name,
            data=[query_vector],
            limit=candidate_k,
            output_fields=[
                "content", "filename", "chunk_id",
                "chunk_index", "page_number", "has_image", "image_url"
            ],
            search_params={"metric_type": "COSINE"},
        )

        all_docs = []
        hits = results[0] if results else []
        for hit in hits:
            entity = hit.get("entity", {})
            distance = hit.get("distance", 0.0)
            similarity = 1.0 - distance
            all_docs.append({
                "content": entity.get("content"),
                "source": entity.get("filename", "未知"),
                "chunk_id": entity.get("chunk_id", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "page_number": entity.get("page_number", 0),
                "has_image": entity.get("has_image", False),
                "image_url": entity.get("image_url", ""),
                "score": similarity,
                "source_type": "vector"
            })

        # 2. BM25 关键词检索（补充精确匹配）
        keyword_results = self._bm25_search(query, top_k=candidate_k)

        # 3. 融合排序
        if keyword_results:
            fused_results = self._rrf_fusion(all_docs, keyword_results)
            # RRF 路径：用 rrf_threshold
            rrf_threshold = settings.rrf_threshold
            before_fusion = len(fused_results)
            if rrf_threshold > 0:
                fused_results = [r for r in fused_results if r.get("score", 0) >= rrf_threshold]
                if len(fused_results) < before_fusion:
                    logger.info(f"多模态 RRF 阈值过滤: {before_fusion} → {len(fused_results)} 条")
        else:
            # 纯向量路径：用 similarity_threshold
            sim_threshold = settings.similarity_threshold
            before_filter = len(all_docs)
            fused_results = [r for r in all_docs if r.get("score", 0) >= sim_threshold]
            if len(fused_results) < before_filter:
                logger.info(
                    f"多模态相似度阈值过滤: {before_filter} → {len(fused_results)} 条 "
                    f"(similarity_threshold={sim_threshold})"
                )

        # 4. Reranker 精排（cross-encoder 重打分）
        if self._reranker and fused_results:
            fused_results = self._reranker.rerank(query, fused_results, top_k=top_k * 2)
            # 相对阈值过滤
            fused_results = adaptive_filter(fused_results)

        # 5. 文件来源去重：同一文件只保留分数最高的 1 条
        seen_files: set[str] = set()
        vector_docs = []
        for doc in fused_results:
            fname = doc["source"]
            if fname not in seen_files:
                seen_files.add(fname)
                vector_docs.append(doc)
            if len(vector_docs) >= top_k:
                break

        logger.info(
            f"多模态检索完成: 候选={len(all_docs)}, "
            f"关键词={len(keyword_results)}, "
            f"去重丢弃={len(fused_results) - len(vector_docs)}, "
            f"最终={len(vector_docs)}"
        )
        for i, r in enumerate(vector_docs):
            score_key = "rerank_score" if "rerank_score" in r else "score"
            logger.info(
                f"  [{i+1}] 来源={r.get('source', '?')}, "
                f"页码={r.get('page_number', 0)}, "
                f"分数={r.get(score_key, 0):.4f}, "
                f"类型={r.get('source_type', '?')}, "
                f"has_image={r.get('has_image', False)}"
            )
        return vector_docs

    def _bm25_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25 关键词检索（使用缓存的 BM25 实例，仅分词查询）"""
        if not self._bm25_docs or self._bm25_instance is None:
            return []
        try:
            import jieba
            query_tokens = list(jieba.cut(query))
            scores = self._bm25_instance.get_scores(query_tokens)

            import numpy as np
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append({
                        "content": self._bm25_docs[idx].get("content", ""),
                        "source": self._bm25_docs[idx].get("filename", "未知"),
                        "chunk_id": self._bm25_docs[idx].get("chunk_id", ""),
                        "page_number": self._bm25_docs[idx].get("page_number", 0),
                        "has_image": self._bm25_docs[idx].get("has_image", False),
                        "image_url": self._bm25_docs[idx].get("image_url", ""),
                        "score": float(scores[idx]),
                        "source_type": "keyword"
                    })
            return results
        except Exception as e:
            logger.warning(f"多模态 BM25 检索失败: {e}")
            return []

    @staticmethod
    def _rrf_fusion(
        vector_results: List[Dict], keyword_results: List[Dict], k: int = None
    ) -> List[Dict]:
        """RRF 融合排序（与 VectorRetriever 逻辑一致）"""
        k = k or settings.rrf_k
        scores = {}
        doc_map = {}

        for rank, doc in enumerate(vector_results):
            key = doc.get("chunk_id") or doc.get("content", "")[:100]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(keyword_results):
            key = doc.get("chunk_id") or doc.get("content", "")[:100]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            if key not in doc_map:
                doc_map[key] = doc

        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        results = []
        for key in sorted_keys:
            doc = doc_map[key].copy()
            doc["score"] = scores[key]
            doc["source_type"] = "fusion"
            results.append(doc)
        return results

    def delete_by_filename(self, filename: str) -> int:
        """按文件名删除多模态 Collection 中的切片"""
        try:
            results = self._client.query(
                self.collection_name,
                filter=f'filename == "{filename}"',
                output_fields=["chunk_id"],
                limit=10000
            )
            if not results:
                return 0
            count = len(results)
            self._client.delete(
                self.collection_name, filter=f'filename == "{filename}"'
            )
            self._client.flush(self.collection_name)
            self._update_bm25_cache()
            logger.info(f"已从多模态 Collection 删除 [{filename}] 的 {count} 个切片")
            return count
        except Exception as e:
            logger.error(f"多模态删除失败: {e}")
            raise

    def list_documents(self) -> List[Dict]:
        """列出多模态 Collection 中的文档"""
        try:
            self._client.load_collection(self.collection_name)
            results = self._client.query(
                self.collection_name,
                filter='chunk_id != ""',
                output_fields=["filename"],
                limit=10000
            )
            doc_map: Dict[str, int] = {}
            for r in results:
                fname = r.get("filename", "未知")
                doc_map[fname] = doc_map.get(fname, 0) + 1
            docs = [{"filename": k, "chunk_count": v} for k, v in doc_map.items()]
            docs.sort(key=lambda x: x["filename"])
            return docs
        except Exception as e:
            logger.error(f"多模态列出文档失败: {e}")
            return []