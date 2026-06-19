"""
检索模块
基于 Milvus Lite 本地向量数据库进行相似度搜索，支持混合检索（向量 + BM25）
"""
import hashlib
import httpx
from typing import List, Dict, Optional
from pymilvus import MilvusClient
from config.settings import settings
from utils.logger import get_logger
from embeddings.embedder import EmbeddingClient, MultimodalEmbedder

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
        self._update_bm25_cache()
    
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
            logger.info(f"BM25 缓存更新完成: {len(self._bm25_docs)} 条文档")
        except Exception as e:
            logger.warning(f"BM25 缓存更新失败: {e}")
            self._bm25_docs = []
    
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
        """BM25 关键词检索"""
        if not self._bm25_docs:
            return []
        
        try:
            import jieba
            from rank_bm25 import BM25Okapi
            
            # 分词
            query_tokens = list(jieba.cut(query))
            corpus_tokens = [list(jieba.cut(doc.get("content", ""))) for doc in self._bm25_docs]
            
            # BM25 检索
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
            
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
        for hit in results[0]:
            entity = hit.get("entity", {})
            # COSINE metric: distance = 1 - cosine_similarity
            # 转换为 similarity = 1 - distance，值域 [-1, 1]，越大越相似
            distance = hit.get("distance", 1.0)
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
        
        return docs
    
    def _rrf_fusion(self, vector_results: List[Dict], keyword_results: List[Dict], k: int = None) -> List[Dict]:
        """RRF (Reciprocal Rank Fusion) 融合排序
        
        Args:
            k: RRF 参数，默认从配置读取
        """
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
        """混合检索：向量检索 + BM25 + RRF 融合
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            检索结果列表（已过滤低于阈值的结果）
        """
        top_k = top_k or settings.top_k
        
        # 1. 向量检索
        vector_results = self._vector_search(query, top_k=top_k)
        
        # 2. 相似度阈值过滤（在 RRF 融合前，基于余弦相似度过滤）
        #    score 已在 _vector_search 中转换为 similarity（越大越相似）
        threshold = settings.similarity_threshold
        if threshold > 0 and vector_results:
            before = len(vector_results)
            vector_results = [r for r in vector_results if r.get("score", 0) >= threshold]
            if len(vector_results) < before:
                logger.info(
                    f"向量阈值过滤: {before} → {len(vector_results)} 条 "
                    f"(similarity阈值={threshold})"
                )
        
        # 3. BM25 关键词检索
        keyword_results = self._bm25_search(query, top_k=top_k)
        
        # 4. RRF 融合排序
        if keyword_results:
            fused_results = self._rrf_fusion(vector_results, keyword_results)
            results = fused_results[:top_k]
        else:
            results = vector_results[:top_k]
        
        logger.info(f"检索完成: 向量={len(vector_results)}, 关键词={len(keyword_results)}, 最终={len(results)}")
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
            self.collection.flush()
            return self.collection.num_entities
        except Exception:
            return 0


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
        self._update_bm25_cache()

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
            logger.info(f"多模态 BM25 缓存更新完成: {len(self._bm25_docs)} 条")
        except Exception as e:
            logger.warning(f"多模态 BM25 缓存更新失败: {e}")
            self._bm25_docs = []

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

    def _generate_image_description(self, image_b64: str, source: str, page: int) -> str:
        """用 LLM 生成图片内容描述（vision 能力）

        Args:
            image_b64: 图片的 base64 编码
            source: 来源文件名
            page: 页码

        Returns:
            图片描述文本，失败时降级为默认标签
        """
        try:
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
                                f"请用中文简要描述这张图片的内容（50字以内），"
                                f"这是文档《{source}》第{page}页的图片。"
                                f"只描述图片核心内容，不要添加额外说明。"
                            ),
                        },
                    ],
                }
            ]
            payload = {
                "model": settings.llm_model_name,
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0,
            }
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    settings.llm_api_url, json=payload, headers=headers
                )
                resp.raise_for_status()
                desc = resp.json()["choices"][0]["message"]["content"].strip()
                logger.info(
                    f"图片描述生成成功: {source} 第{page}页 -> {desc[:50]}..."
                )
                return f"[图片] {desc}（来源: {source}, 第{page}页）"
        except Exception as e:
            logger.warning(f"图片描述生成失败，使用默认标签: {e}")
            return f"[图片] 来源: {source}, 第{page}页"

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """多模态混合检索（与 VectorRetriever.search 接口一致）"""
        top_k = top_k or settings.top_k

        # 1. 向量检索（用多模态模型编码查询）
        query_vector = self.embedder.embed_text(query)
        results = self._client.search(
            self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=[
                "content", "filename", "chunk_id",
                "chunk_index", "page_number", "has_image", "image_url"
            ],
            search_params={"metric_type": "COSINE"},
        )

        vector_docs = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            # COSINE metric: distance = 1 - cosine_similarity
            # 转换为 similarity = 1 - distance，值域 [-1, 1]，越大越相似
            distance = hit.get("distance", 1.0)
            similarity = 1.0 - distance
            vector_docs.append({
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

        # 1.5 向量阈值过滤（在 RRF 融合前，基于余弦相似度过滤）
        #    score 已转换为 similarity（越大越相似）
        threshold = settings.similarity_threshold
        if threshold > 0 and vector_docs:
            before = len(vector_docs)
            vector_docs = [r for r in vector_docs if r.get("score", 0) >= threshold]
            if len(vector_docs) < before:
                logger.info(
                    f"多模态向量阈值过滤: {before} → {len(vector_docs)} 条 "
                    f"(similarity阈值={threshold})"
                )

        # 2. BM25（仅文本部分）
        keyword_docs = []
        if self._bm25_docs:
            try:
                import jieba
                import numpy as np
                from rank_bm25 import BM25Okapi

                query_tokens = list(jieba.cut(query))
                corpus_tokens = [
                    list(jieba.cut(d.get("content", "")))
                    for d in self._bm25_docs
                ]
                bm25 = BM25Okapi(corpus_tokens)
                scores = bm25.get_scores(query_tokens)
                top_indices = np.argsort(scores)[::-1][:top_k]

                for idx in top_indices:
                    if scores[idx] > 0:
                        keyword_docs.append({
                            "content": self._bm25_docs[idx].get("content", ""),
                            "source": self._bm25_docs[idx].get("filename", "未知"),
                            "chunk_id": self._bm25_docs[idx].get("chunk_id", ""),
                            "page_number": self._bm25_docs[idx].get("page_number", 0),
                            "score": float(scores[idx]),
                            "source_type": "keyword"
                        })
            except Exception as e:
                logger.warning(f"多模态 BM25 检索失败: {e}")

        # 3. RRF 融合
        if keyword_docs:
            results_final = self._rrf_fusion(vector_docs, keyword_docs)[:top_k]
        else:
            results_final = vector_docs[:top_k]

        logger.info(
            f"多模态检索完成: 向量={len(vector_docs)}, "
            f"关键词={len(keyword_docs)}, 最终={len(results_final)}"
        )
        return results_final

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