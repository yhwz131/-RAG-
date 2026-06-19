"""
Embedding 模块
调用远程 Embedding API 将文本转为向量
支持纯文本模型 (bge-large-zh) 和多模态模型 (Qwen3-VL-Embedding)
"""
import time
import httpx
from typing import List, Dict, Optional
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("embedder")


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """指数退避重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"{func.__name__} 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}. "
                            f"{delay:.1f}秒后重试..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} 调用失败，已重试 {max_retries} 次: {e}")
            raise last_exception
        return wrapper
    return decorator


class EmbeddingClient:
    """Embedding API 客户端，使用 OpenAI 兼容接口"""
    
    def __init__(self):
        self.api_key = settings.embedding_api_key
        self.base_url = settings.embedding_base_url.rstrip("/")
        self.model = settings.embedding_model_name
        self.dimension = settings.embedding_dimension
    
    def embed_query(self, text: str) -> List[float]:
        """将单条文本转为向量"""
        return self._call_api([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将多条文本批量转为向量"""
        return self._call_api(texts)
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用 Embedding API（带重试机制）"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts
        }
        
        with httpx.Client(timeout=60) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            logger.info(f"Embedding 完成: {len(texts)} 条文本 -> {len(embeddings)} 个向量")
            return embeddings


class MultimodalEmbedder:
    """多模态 Embedding 客户端，调用 Qwen3-VL-Embedding-8B

    支持纯文本和图片两种输入模式，图片通过 base64 编码传入。
    API 格式：{"model": "...", "input": [{"text": "..."}, {"image": "data:image/...;base64,..."}]}
    """

    def __init__(self):
        self.api_key = settings.embedding_api_key
        self.base_url = settings.embedding_base_url.rstrip("/")
        self.model = settings.multimodal_embedding_model
        self.dimension = settings.multimodal_embedding_dim

    def embed_text(self, text: str) -> List[float]:
        """纯文本向量化（使用多模态模型）"""
        payload = {
            "model": self.model,
            "input": [{"text": text}]
        }
        return self._call_api(payload)[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量纯文本向量化"""
        payload = {
            "model": self.model,
            "input": [{"text": t} for t in texts]
        }
        return self._call_api(payload)

    def embed_image(self, image_b64: str, description: str = "", mime_type: str = "png") -> List[float]:
        """图片向量化，可附带文字描述

        Args:
            image_b64: 图片的 base64 编码（不含 data:image 前缀）
            description: 图片描述文字（可选）
            mime_type: 图片 MIME 类型（png, jpeg, gif, webp 等），默认 png
        """
        # 支持多种图片格式
        valid_mime_types = {"png", "jpeg", "jpg", "gif", "webp", "bmp", "tiff"}
        if mime_type.lower() not in valid_mime_types:
            logger.warning(f"不常见的图片格式: {mime_type}，默认使用 png")
            mime_type = "png"
        # 统一 jpg -> jpeg
        if mime_type.lower() == "jpg":
            mime_type = "jpeg"
        
        content = [{"image": f"data:image/{mime_type};base64,{image_b64}"}]
        if description:
            content.append({"text": description})
        payload = {
            "model": self.model,
            "input": content
        }
        return self._call_api(payload)[0]

    def embed_mixed(self, items: List[Dict]) -> List[List[float]]:
        """批量混合向量化

        Args:
            items: [{"text": "..."}, {"image": "base64...", "text": "描述"}, ...]
        """
        payload = {
            "model": self.model,
            "input": items
        }
        return self._call_api(payload)

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _call_api(self, payload: dict) -> List[List[float]]:
        """调用多模态 Embedding API（带重试机制）"""
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        with httpx.Client(timeout=60) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            embeddings = [item["embedding"] for item in data["data"]]
            logger.info(
                f"多模态 Embedding 完成: {len(payload['input'])} 项 -> "
                f"{len(embeddings)} 个向量, 维度={len(embeddings[0]) if embeddings else 0}"
            )
            return embeddings