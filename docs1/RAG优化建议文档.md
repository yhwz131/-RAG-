# 知识问答系统 RAG 优化建议文档

## 文档信息

| 项目 | 内容 |
|------|------|
| **文档名称** | RAG 优化建议文档 |
| **版本** | v2.0 |
| **日期** | 2026-06-19 |
| **仓库地址** | [https://github.com/yhwz131/-RAG-](https://github.com/yhwz131/-RAG-) |
| **定位** | 针对当前系统 RAG 功能链路的全面分析与优化方案，含 LangGraph 多 Agent 升级规划 |

---

## 目录

- [1. 现状概览](#1-现状概览)
- [2. 问题总览与优先级](#2-问题总览与优先级)
- [3. 检索优化](#3-检索优化)
- [4. 切片优化（迁移 LangGraph 时一并解决）](#4-切片优化迁移-langgraph-时一并解决)
- [5. 上下文构建优化](#5-上下文构建优化)
- [6. Prompt 优化](#6-prompt-优化)
- [7. 对话记忆优化](#7-对话记忆优化)
- [8. API 稳定性优化](#8-api-稳定性优化)
- [9. 多模态链路优化](#9-多模态链路优化)
- [10. 文档解析优化](#10-文档解析优化)
- [11. 配置与治理优化](#11-配置与治理优化)
- [12. 查询路由优化](#12-查询路由优化)
- [13. 多知识库架构](#13-多知识库架构)
- [14. LangGraph 多 Agent 架构升级规划](#14-langgraph-多-agent-架构升级规划)
- [15. 优化路线图](#15-优化路线图)

---

## 1. 现状概览

### 1.1 当前 RAG 链路

```
用户提问
  │
  ▼
Query Embedding (bge-large-zh)
  │
  ▼
┌──────────────────────────────────┐
│  Milvus 向量检索 (top_k=5)       │
│  + BM25 关键词检索 (jieba分词)   │
│  → RRF 融合                      │
└──────────────────────────────────┘
  │
  ▼
拼接检索结果为"参考资料"
  │
  ▼
构建 Messages (system + 历史 + 用户问题)
  │
  ▼
调用 mimo-v2.5 生成回答（流式/非流式）
  │
  ▼
返回 answer + references
```

### 1.2 已实现能力

| 能力 | 状态 | 说明 |
|------|------|------|
| 混合检索（向量 + BM25） | ✅ 已实现 | RRF 融合，alpha=0.7 |
| 多模态检索 | ✅ 已实现 | Qwen3-VL-Embedding 独立链路 |
| 流式输出 | ✅ 已实现 | SSE 风格，先发 sources 再流 token |
| 多轮对话 | ✅ 已实现 | JSON 文件持久化，滑动窗口 |
| 批量上传 | ✅ 已实现 | 支持多文件并发上传 |
| 文档管理 | ✅ 已实现 | 列表/删除/清空/统计 |

### 1.3 关键瓶颈

当前系统**功能链路已跑通**，但在以下方面存在明显短板：

| 瓶颈 | 影响 |
|------|------|
| 切片策略过于粗放 | 语义完整性差，检索精度低（计划迁移 LangGraph 时用 LangChain 切片器统一解决） |
| 上下文无长度控制 | Prompt 可能超长导致 LLM 报错或截断 |
| Prompt 分散管理 | 维护困难，约束不一致 |
| Embedding 无重试 | 网络抖动直接失败 |
| 相似度阈值方向可能有误 | `distance` vs `similarity` 语义混乱 |
| 单链式 RAG 架构 | 无法灵活编排多步骤推理，计划升级为 LangGraph 多 Agent 平台 |
| 多模态链路重复代码多 | 可维护性差 |

---

## 2. 问题总览与优先级

### P0 — 必须修复（直接影响正确性）

| # | 问题 | 所在文件 | 影响 |
|---|------|----------|------|
| 1 | 向量检索 `distance` 与 `similarity` 语义未确认，阈值过滤可能反了 | `rag/retriever.py` | ✅ 已修复：COSINE 距离转相似度 `similarity = 1.0 - distance` |
| 2 | 上下文拼接无 token 长度控制 | `rag/chain.py` | ✅ 已修复：`_build_context()` 支持 `max_context_tokens` 参数 |
| 3 | Prompt 模板分散（chain.py 自带 SYSTEM_PROMPT vs prompt_template.py 模板） | `rag/chain.py` + `rag/prompt_template.py` | ✅ 已修复：统一使用 `prompt_template.py` 中的 `SYSTEM_PROMPT` |

### P1 — 建议改进（显著影响质量）

| # | 问题 | 所在文件 | 影响 |
|---|------|----------|------|
| 4 | 按字符切分，非语义切分 | `embeddings/chunker.py` | **暂缓**：迁移 LangGraph 时用 LangChain `RecursiveCharacterTextSplitter` 统一解决 |
| 5 | BM25 缓存限制 10000 条 | `rag/retriever.py` | ✅ 已修复：改为批量循环加载，无上限 |
| 6 | Embedding 无重试/超时机制 | `embeddings/embedder.py` | ✅ 已修复：添加 `retry_with_backoff` 装饰器（3次重试） |
| 7 | 对话记忆按条数而非 token 控制 | `rag/memory.py` | ✅ 已修复：双重截断（轮次 + token），`_trim_by_tokens()` 方法 |
| 8 | 无查询路由，闲聊也走 RAG 检索 | `rag/chain.py` | ✅ 已修复：实现混合路由（规则+LLM），区分 rag/chitchat/general |

### P2 — 可选优化（提升体验和健壮性）

| # | 问题 | 所在文件 | 影响 |
|---|------|----------|------|
| 9 | 文件大小校验未实际生效 | `api/routes_docs.py` | ✅ 已修复：上传时校验文件大小，超限返回 413 |
| 10 | 上传失败时可能残留脏文件 | `api/routes_docs.py` | ✅ 已修复：失败时自动清理已保存的文件 |
| 11 | RRF k 参数硬编码为 60 | `rag/retriever.py` | ✅ 已修复：`settings.rrf_k` 可配置 |
| 12 | 多模态图片 MIME 固定为 png | `embeddings/embedder.py` | ✅ 已修复：`embed_image()` 支持 `mime_type` 参数 |
| 13 | 单一知识库，所有数据混在一起 | Milvus / config | 无法按类型分离管理不同领域的 RAG 资料 |

---

## 3. 检索优化

### 3.1 确认相似度语义（P0）

> **✅ 已实现**：统一使用 COSINE metric，`similarity = 1.0 - distance` 转换。

**现状问题**：

```python
# rag/retriever.py
score: hit.get("distance", 0)
if r.get("score", 0) >= threshold:  # 阈值过滤
```

`distance` 在 Milvus 中通常是**距离**（越小越相似），但代码将其当作**相似度**（越大越相似）使用。如果 metric 是 L2，当前的 `>= threshold` 逻辑会导致**过滤掉最相关的结果**。

**优化方案**：

```python
# 方案 A：明确使用 IP（内积）或 COSINE，此时 distance 就是 similarity
search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
# COSINE 下 distance 值域 [-1, 1]，越大越相似，>= threshold 正确

# 方案 B：如果用 L2，需要反转逻辑
if r.get("score", 0) <= threshold:  # L2 距离越小越好
```

**建议**：统一使用 `COSINE` 作为 metric type，与 Embedding 模型输出对齐。

---

### 3.2 检索结果重排（P1）

**现状问题**：RRF 融合后直接取 top_k，没有经过精排。

**优化方案**：添加 Cross-Encoder 重排阶段：

```
向量检索 (top_k=20)
    + BM25 检索 (top_k=20)
         │
         ▼
    RRF 融合 → 候选集 (top_k=20)
         │
         ▼
    Cross-Encoder 重排
         │
         ▼
    精选结果 (top_k=5)
```

可选方案：
- **轻量级**：使用 `bge-reranker-base`，API 调用
- **本地**：`sentence-transformers` 的 `CrossEncoder`
- **无额外模型**：利用 LLM 做 LLM-based reranking（成本较高）

```python
# 示例：基于 bge-reranker 的重排
class Reranker:
    def __init__(self, api_url: str, model: str = "BAAI/bge-reranker-base"):
        self.api_url = api_url
        self.model = model

    def rerank(self, query: str, documents: list[str], top_k: int = 5):
        # 调用 reranker API，返回按相关性排序的 documents
        ...
```

---

### 3.3 查询预处理（P1）

> **✅ 已实现**：`preprocess_query()` 轻量清洗，仅 RAG 路径生效（路由后、检索前）。

**现状问题**：Query 直接进入检索，未做任何清洗。

**实际实现**：

```python
# rag/router.py — preprocess_query()
def preprocess_query(query: str) -> str:
    """RAG 路径查询预处理（轻量清洗，不改变语义）

    仅做以下处理：
    1. 去除首尾空白
    2. 合并连续空格为单个空格
    3. 全角英数字 → 半角（提升 embedding 和 BM25 匹配率）
    4. 连续重复标点压缩为单个（如 !!! → !）

    不做：不去停用词、不改写、不改变语序
    """
    text = query.strip()
    text = re.sub(r'\s+', ' ', text)
    # 全角英数字 → 半角
    text = text.translate(str.maketrans(...))
    # 全角标点 → 半角
    text = text.translate(str.maketrans('！？，；：、', '!?,;:,'))
    # 连续重复标点压缩
    text = re.sub(r'([!\uff1f??.。,，;；:：])\1+', r'\1', text)
    return text

# rag/chain.py — 仅 RAG 路径使用
def chat(self, query, session_id, stream=False):
    query_type = route_query(query)  # 路由用原始 query
    ...
    retrieval_query = preprocess_query(query)  # 检索用预处理后的 query
    docs = self.retriever.search(retrieval_query)
    # LLM 生成仍用原始 query
```

---

### 3.4 BM25 缓存优化（P1）

> **✅ 已实现**：分批循环加载，无上限。

**现状问题**：`limit=10000`，超过部分无法被 BM25 检索。

**实际实现**：

```python
# rag/retriever.py — _update_bm25_cache()
offset = 0
batch_size = 10000
all_results = []
while True:
    results = self._client.query(
        collection_name=self.collection_name,
        filter='id >= 0',
        offset=offset,
        limit=batch_size,
        output_fields=["chunk_id", "filename", "content", "chunk_index", "page_number"]
    )
    if not results:
        break
    all_results.extend(results)
    offset += batch_size
```

---

### 3.5 RRF 参数可配置化（P2）

> **✅ 已实现**：`settings.rrf_k` 可配置，默认 60。

**实际实现**：

```python
# config/settings.py
rrf_k: int = 60  # RRF (Reciprocal Rank Fusion) 参数

# rag/retriever.py
def _rrf_fusion(self, vector_results, bm25_results, k=None):
    k = k or settings.rrf_k
    ...
```

---

## 4. 切片优化

### 4.1 语义切片（P1）

**现状问题**：按固定字符数（500）切分，不考虑语义边界。

```python
# 当前：纯字符切割
for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
```

**优化方案**：分层切片策略

```
Level 1: 按段落/标题分割
    │
    ▼
Level 2: 段落过长时按句子分割
    │
    ▼
Level 3: 合并过短的相邻切片
    │
    ▼
Level 4: 保留 overlap 确保上下文连续
```

```python
class SemanticChunker:
    """基于语义的切片器"""

    # 中英文句子结束符
    SENTENCE_ENDINGS = re.compile(r'[。！？.!?\n]')

    def chunk(self, text: str, chunk_size: int = 500, overlap: int = 50):
        # 1. 按段落分割
        paragraphs = text.split('\n\n')
        chunks = []

        for para in paragraphs:
            if len(para) <= chunk_size:
                chunks.append(para)
            else:
                # 2. 段落过长，按句子分割
                sentences = self._split_sentences(para)
                chunks.extend(self._merge_sentences(sentences, chunk_size, overlap))

        # 3. 合并过短切片（< 100 字符）
        chunks = self._merge_short_chunks(chunks, min_length=100, max_length=chunk_size)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """按句子结束符分割"""
        parts = self.SENTENCE_ENDINGS.split(text)
        # 保留标点
        sentences = []
        for i, part in enumerate(parts):
            if part:
                sentences.append(part)
        return sentences

    def _merge_sentences(self, sentences, max_len, overlap):
        """合并句子到目标长度，保留 overlap"""
        chunks = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > max_len and current:
                chunks.append(current.strip())
                # 保留最后 overlap 个字符作为下一段开头
                current = current[-overlap:] + sent if overlap else sent
            else:
                current += sent
        if current.strip():
            chunks.append(current.strip())
        return chunks
```

---

### 4.2 标题层级补全（P1）

**现状问题**：切片丢失了所属章节信息，检索到的片段缺乏上下文。

**优化方案**：在切片元数据中记录标题层级

```python
def chunk_with_heading(self, text: str, headings: list[dict]):
    """
    headings 格式: [{"level": 1, "title": "第一章", "offset": 0}, ...]
    """
    chunks = self.chunk(text)
    enriched = []
    for chunk_text, start_offset in chunks:
        # 找到当前切片所属的最近标题
        parent_heading = self._find_parent_heading(start_offset, headings)
        enriched_text = f"[{parent_heading}]\n{chunk_text}" if parent_heading else chunk_text
        enriched.append(enriched_text)
    return enriched
```

---

### 4.3 切片去重（P2）

**现状问题**：重复内容（如模板文字、页眉页脚）会占用向量空间并干扰检索。

```python
def deduplicate_chunks(chunks: list[str], threshold: float = 0.95) -> list[str]:
    """基于 SimHash 或编辑距离的切片去重"""
    seen_hashes = set()
    unique = []
    for chunk in chunks:
        h = simhash(chunk)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(chunk)
    return unique
```

---

## 5. 上下文构建优化

### 5.1 Token 级长度控制（P0）

> **✅ 已实现**：`_build_context()` 支持 `max_context_tokens` 参数，默认 3000。

**实际实现**：

```python
def _build_context(self, docs: list[dict], max_context_tokens: int = 3000) -> str:
    """构建上下文，带 token 长度控制"""
    context_parts = []
    current_tokens = 0

    for i, doc in enumerate(docs, 1):
        text = doc.get("text", "")
        source = doc.get("source", "未知来源")
        page = doc.get("page_number")

        # 构建单条参考
        ref = f"【参考{i}】来源: {source}"
        if page:
            ref += f", 第{page}页"
        ref += f"\n{text}\n"

        # 估算 token 数（中文约 1.5 字/token，英文约 4 字符/token）
        ref_tokens = estimate_tokens(ref)

        if current_tokens + ref_tokens > max_context_tokens:
            # 截断当前条目以适应剩余空间
            remaining = max_context_tokens - current_tokens
            if remaining > 100:  # 至少保留 100 token 的空间
                ref = ref[:remaining * 2] + "...\n"  # 粗略截断
                context_parts.append(ref)
            break

        context_parts.append(ref)
        current_tokens += ref_tokens

    return "\n".join(context_parts)


def estimate_tokens(text: str) -> int:
    """估算 token 数"""
    # 中文字符数 * 1.5 + 英文单词数 * 1.3
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    en_words = len(re.findall(r'[a-zA-Z]+', text))
    return int(cn_chars * 1.5 + en_words * 1.3)
```

---

### 5.2 参考来源排序优化（P2）

> **✅ 已实现**：`_build_context()` 中按 score 降序排列，高相关性内容优先被 LLM 看到。

**现状问题**：参考来源按检索返回顺序排列，未考虑与 query 的相关性排序。

**实际实现**：

```python
# rag/chain.py — _build_context()
def _build_context(self, docs, max_context_tokens=3000):
    if not docs:
        return "（无相关参考资料）"
    # 按相似度降序排列，确保高相关性内容优先被 LLM 看到
    docs = sorted(docs, key=lambda d: d.get("score", 0), reverse=True)
    ...
```

---

## 6. Prompt 优化

### 6.1 统一 Prompt 管理（P0）

> **✅ 已实现**：`chain.py` 移除内联 prompt，统一使用 `prompt_template.py` 中的 `SYSTEM_PROMPT`。

**实际实现**：

```python
# rag/prompt_template.py — 唯一的 SYSTEM_PROMPT 来源
SYSTEM_PROMPT = """你是一个专业的知识问答助手...

参考资料：
{context}
"""

# rag/chain.py — 引用统一模板
from rag.prompt_template import SYSTEM_PROMPT, estimate_tokens

def _build_messages(self, query, context, session_id):
    system_msg = SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": system_msg}]
    history = self.memory.get_context(session_id)
    messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages
```

---

### 6.2 增加 Prompt 约束（P1）

> **✅ 已实现**：`SYSTEM_PROMPT` 新增规则 6（矛盾信息处理）、规则 7（关键信息准确性）、规则 3（标准化拒答模板）。

**实际实现**：

```python
# rag/prompt_template.py — SYSTEM_PROMPT 新增规则
回答规则：
...
6. **多参考综合**：如果多个参考资料之间存在**矛盾或不一致**，
   请明确指出矛盾之处并分别说明各参考的观点
7. **关键信息准确性**：对于数字、日期、版本号、人名等关键信息，
   请确保准确引用原文，不要猜测或推断
...
3. **信息不足时坦诚说明**：请明确告知用户"当前知识库中暂无相关信息"
```

---

## 7. 对话记忆优化

### 7.1 基于 Token 的上下文预算（P1）

> **✅ 已实现**：`_trim_by_tokens()` 双重截断，轮次 + token 控制。

**实际实现**：

```python
# rag/memory.py
import re

def estimate_tokens(text: str) -> int:
    """中英文混合 token 估算"""
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    en_words = len(re.findall(r'[a-zA-Z]+', text))
    other = len(text) - cn_chars - sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))
    return int(cn_chars * 1.5 + en_words * 1.3 + other * 0.5)

def _trim_by_tokens(self, messages, max_tokens):
    """按 token 预算截断，从旧到新保留"""
    total = sum(estimate_tokens(m['content']) for m in messages)
    while total > max_tokens and len(messages) > 1:
        removed = messages.pop(0)
        total -= estimate_tokens(removed['content'])
    return messages
```

---

### 7.2 历史摘要（P2）

当对话轮数超过阈值时，自动对早期对话生成摘要：

```python
async def summarize_history(self, messages: list[dict]) -> str:
    """使用 LLM 对历史对话生成摘要"""
    prompt = f"请将以下对话历史压缩为简洁的摘要，保留关键信息：\n\n{format_messages(messages)}"
    summary = await self.llm.chat(prompt, max_tokens=200)
    return summary
```

---

## 8. API 稳定性优化

### 8.1 Embedding 重试机制（P1）

> **✅ 已实现**：同步 `retry_with_backoff` 装饰器，指数退避。

**实际实现**：

```python
# embeddings/embedder.py — 实际实现（同步版本）
import time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=10.0):
    """指数退避重试装饰器（同步版）"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}, {delay}s 后重试")
                    time.sleep(delay)
        return wrapper
    return decorator

class EmbeddingClient:
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(...)
            ...
```

---

### 8.2 Embedding 批量分片（P2）

**现状问题**：大批量切片一次性发送，可能超时或超出 API 限制。

**优化方案**：

```python
async def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """分批生成 embedding"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = await self._embed_batch(batch)
        all_embeddings.extend(embeddings)
        logger.debug(f"Embedding 进度: {min(i + batch_size, len(texts))}/{len(texts)}")
    return all_embeddings
```

---

### 8.3 文件大小校验生效（P2）

> **✅ 已实现**：单文件和批量上传均校验，返回 413 状态码，失败自动清理。

**实际实现**：

```python
# api/routes_docs.py — 单文件上传
content = await file.read()
if len(content) / (1024 * 1024) > settings.max_file_size_mb:
    raise HTTPException(
        status_code=413,
        detail=f"文件大小超过限制: {settings.max_file_size_mb}MB"
    )

# 批量上传同样校验，且失败时自动清理已生成的向量数据
```

---

## 9. 多模态链路优化

### 9.1 提取检索器公共基类（P1）

**现状问题**：`VectorRetriever` 和 `MultimodalRetriever` 有大量重复代码。

**优化方案**：

```python
class BaseRetriever(ABC):
    """检索器基类"""
    def __init__(self, milvus_uri, collection_name, embedder, ...):
        ...

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        ...

    def _vector_search(self, query_vector, top_k, threshold):
        """通用向量检索逻辑"""
        ...

    def _bm25_search(self, query, top_k):
        """通用 BM25 检索逻辑"""
        ...

    def _rrf_fusion(self, vector_results, bm25_results, k=60):
        """通用 RRF 融合"""
        ...


class TextRetriever(BaseRetriever):
    """纯文本检索器"""
    async def search(self, query, top_k=5):
        # 使用 bge-large-zh embedding
        ...


class MultimodalRetriever(BaseRetriever):
    """多模态检索器"""
    async def search(self, query, top_k=5):
        # 使用 Qwen3-VL embedding
        ...
```

---

### 9.2 图片 MIME 类型支持（P2）

> **✅ 已实现**：`embed_image()` 支持 `mime_type` 参数，兼容多种图片格式。

**实际实现**：

```python
# embeddings/embedder.py
def embed_image(self, image_b64: str, description: str = "", mime_type: str = "png") -> list[float]:
    """支持 png/jpeg/gif/webp/bmp/tiff 等格式"""
    if mime_type == "jpg":
        mime_type = "jpeg"  # 标准化
    data_uri = f"data:image/{mime_type};base64,{image_b64}"
    ...
```

---

## 10. 文档解析优化

### 10.1 编码自动检测（P2）

> **✅ 已实现**：`_parse_text()` 使用 chardet 自动检测编码，GBK/GB2312/Big5 等均可正确解析。

**实际实现**：

```python
# utils/file_parser.py — _parse_text()
@staticmethod
def _parse_text(file_path: str) -> str:
    """解析纯文本和 Markdown 文件（自动检测编码）"""
    import chardet
    with open(file_path, "rb") as f:
        raw = f.read()
    detected = chardet.detect(raw)
    encoding = detected.get("encoding") or "utf-8"
    if encoding.lower() == "ascii":
        encoding = "utf-8"
    return raw.decode(encoding, errors="replace")
```

---

### 10.2 表格结构保留（P2）

> **✅ 已实现**：`_df_to_markdown()` 将 DataFrame 转为 Markdown 表格，`_parse_excel()` 和 `_parse_excel_with_pages()` 均使用。

**实际实现**：

```python
# utils/file_parser.py — _df_to_markdown()
@staticmethod
def _df_to_markdown(df) -> str:
    """将 DataFrame 转为 Markdown 表格格式"""
    headers = [str(h).strip() for h in df.columns.tolist()]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(v).strip() if v is not None else "" for v in row.tolist()])
    # 计算每列最大宽度
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    # 构建 Markdown 表格
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator = "| " + " | ".join("-" * max(3, col_widths[i]) for i in range(len(headers))) + " |"
    data_lines = []
    for row in rows:
        line = "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"
        data_lines.append(line)
    return "\n".join([header_line, separator] + data_lines)
```

---

## 11. 配置与治理优化

### 11.1 增加配置校验（P2）

> **✅ 已实现**：`config/settings.py` 添加 pydantic `@field_validator` 和 `@model_validator`，覆盖温度、阈值、切片、token 等参数。

**实际实现**：

```python
# config/settings.py
from pydantic import field_validator, model_validator

class Settings(BaseSettings):
    ...

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError(f"llm_temperature 必须在 0~2 之间，当前值: {v}")
        return v

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"similarity_threshold 必须在 0~1 之间，当前值: {v}")
        return v

    @field_validator("chunk_size", "retriever_top_k", "max_context_tokens",
                     "max_history_tokens", "rrf_k", "max_file_size_mb", "llm_max_tokens")
    @classmethod
    def validate_positive_int(cls, v, info):
        if v <= 0:
            raise ValueError(f"{info.field_name} 必须大于 0，当前值: {v}")
        return v

    @model_validator(mode="after")
    def validate_model(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) 必须小于 chunk_size ({self.chunk_size})")
        return self
```

---

## 12. 查询路由优化

> **核心问题**：当前系统对所有用户提问都走 RAG 检索流程，包括问候语、闲聊、通用知识问题等。
> 这不仅浪费检索资源，还可能因为检索到不相关内容而降低回答质量。

> **✅ 实现状态**：已实现混合路由方案，创建 `rag/router.py` 模块，`rag/chain.py` 已集成。
> 实际实现使用同步 httpx（匹配项目现有模式），而非文档示例中的 async。

### 12.1 当前流程的问题

```
用户: "你好"        → 检索知识库 → 找到无关文档 → LLM 勉强回答
用户: "什么是AI"    → 检索知识库 → 知识库中没有 → LLM 回答"无法找到相关资料"
用户: "RAG是什么"   → 检索知识库 → 找到相关文档 → LLM 正确回答  ✅ 只有这种情况合理
```

### 12.2 目标路由架构

```
用户提问
    │
    ▼
┌─────────────────────────────────────┐
│           查询路由器 (Router)        │
│                                     │
│  规则匹配 → 命中? → 直接 LLM 回答   │
│      │                              │
│      ▼ 未命中                       │
│  LLM 分类 → 闲聊? → 直接 LLM 回答   │
│      │                              │
│      ▼ 知识类                       │
│  进入 RAG 检索流程                   │
└─────────────────────────────────────┘
```

### 12.3 实现方案：混合路由

推荐**规则 + LLM 分类**的混合方案，兼顾成本和准确率：

> **实际实现说明**：
> - 使用同步 `httpx.Client`（匹配项目现有同步模式），非 async
> - 使用 `utils.logger` 而非 `loguru`
> - `llm_classify()` 内部自行发起 HTTP 请求，无需外部传入 client
> - `route_query()` 为同步函数，直接返回 `QueryType`

```python
# rag/router.py — 实际实现

import re
from enum import Enum
from typing import Optional
import httpx
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("router")


class QueryType(str, Enum):
    RAG = "rag"           # 需要知识库检索
    CHITCHAT = "chitchat" # 闲聊/问候
    GENERAL = "general"   # 通用知识（不需要知识库）


# ========== 第一层：规则匹配（零成本，极速） ==========

GREETING_PATTERNS = [
    r'^(你好|您好|hi|hello|hey|嗨|哈喽|早上好|下午好|晚上好)[\s!！。.？?]*$',
    r'^(再见|拜拜|bye|goodbye|see you|晚安|回见)[\s!！。.？?]*$',
    r'^(谢谢|感谢|thanks|thank you|thx|辛苦了|多谢)[\s!！。.？?]*$',
    r'^(你是谁|你叫什么|你能做什么|介绍一下你自己|你是什么)[\s?？。.]*$',
    r'^(帮助|help|怎么用|使用说明|功能介绍)[\s?？。.]*$',
]

SMALL_TALK_PATTERNS = [
    r'^(今天天气|现在几点|几点了|吃了吗|在吗|在不在)(怎么样|如何|好不好|吗|了)?[\s?？]*$',
    r'^(无聊|开心|难过|哈哈|呵呵|嗯嗯|哦哦|好的|ok|okay)[\s!！。.？?]*$',
    r'^(你能理解我吗|你有感情吗|你有意识吗|你喜欢什么)[\s?？]*$',
]


def rule_based_route(query: str) -> Optional[QueryType]:
    """第一层：规则匹配，返回 None 表示未命中"""
    query_clean = query.strip().lower()
    if len(query_clean) < 2:
        return QueryType.CHITCHAT
    for pattern in GREETING_PATTERNS + SMALL_TALK_PATTERNS:
        if re.match(pattern, query_clean):
            logger.debug(f"规则匹配命中: {query} -> chitchat")
            return QueryType.CHITCHAT
    return None


# ========== 第二层：LLM 分类（准确，有少量成本） ==========

CLASSIFY_PROMPT = """你是一个查询分类器。请判断用户的问题属于以下哪一类：

1. rag — 需要从特定知识库中检索信息才能准确回答的问题。例如：关于特定产品、文档、项目、技术细节、课程内容的问题。
2. chitchat — 闲聊、问候、情感表达、与知识库无关的日常对话。
3. general — 通用知识问题，不需要特定知识库，用通用知识就能回答。例如：什么是Python、太阳系有几颗行星、如何学习编程。

只回答一个类别名（rag / chitchat / general），不要解释。

用户问题：{query}
类别："""


def llm_classify(query: str) -> QueryType:
    """第二层：LLM 分类（同步 httpx）"""
    try:
        url = settings.llm_api_url
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"

        prompt = CLASSIFY_PROMPT.format(query=query)
        payload = {
            "model": settings.llm_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 10,
            "stream": False
        }

        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip().lower()

        if content in ("rag", "chitchat", "general"):
            return QueryType(content)
        # 模糊匹配兜底
        if "chitchat" in content or "闲聊" in content:
            return QueryType.CHITCHAT
        elif "general" in content or "通用" in content:
            return QueryType.GENERAL
    except Exception as e:
        logger.warning(f"LLM 分类失败，默认走 RAG: {e}")

    return QueryType.RAG  # 分类失败时默认走 RAG


# ========== 路由器入口 ==========

def route_query(query: str) -> QueryType:
    """混合路由：先规则，再 LLM（同步）"""
    result = rule_based_route(query)
    if result is not None:
        logger.info(f"查询路由 [规则]: {query[:50]}... -> {result.value}")
        return result

    result = llm_classify(query)
    logger.info(f"查询路由 [LLM]: {query[:50]}... -> {result.value}")
    return result
```

### 12.4 在 RAG 链中集成路由

> **✅ 已实现**：`rag/chain.py` 的 `chat()` 方法已集成混合路由。

```python
# rag/chain.py — 实际集成方式

from rag.router import route_query, QueryType

class RAGChain:
    def chat(self, query, session_id="default", stream=False):
        # 1. 查询路由
        query_type = route_query(query)

        # 闲聊/通用知识：直接走 LLM，不走 RAG 检索
        if query_type in (QueryType.CHITCHAT, QueryType.GENERAL):
            direct_context = f"（当前为{query_type.value}模式，无需参考资料）"
            messages = self._build_messages(query, direct_context, session_id)
            self.memory.add(session_id, "user", query)
            if stream:
                return self._chat_stream(query, messages, session_id, [], query_type=query_type)
            else:
                answer = self._call_llm(messages, stream=False)
                self.memory.add(session_id, "assistant", answer)
                return answer, [], query_type

        # RAG：走完整检索 + 生成流程
        docs = self.retriever.search(query)
        # ... 检索 → 上下文构建 → LLM 生成
        return answer, docs, query_type  # 返回 3 元组
```

> **注意**：`chat()` 返回值从 `(answer, docs)` 变为 `(answer, docs, query_type)` 三元组，
> `api/routes_chat.py` 已同步适配。

### 12.5 路由结果的前端展示

> **✅ 已实现**：`ChatResponse` 模型已添加 `query_type` 字段，流式输出 metadata 中包含 `query_type`。

```python
# api/routes_chat.py — 实际响应格式

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    references: List[Dict]
    query_type: str = "rag"  # rag / chitchat / general
```

流式输出 metadata 格式：
```json
{
    "query_type": "rag",
    "sources": [
        {"source": "xxx.pdf", "chunk_index": 1, "content_snippet": "...", "score": 0.85}
    ]
}
```

### 12.6 成本与收益

| 项目 | 说明 |
|------|------|
| **规则匹配** | 零成本，正则匹配耗时 < 1ms |
| **LLM 分类** | 使用 mimo-v2.5，单次分类约 50 token，成本极低 |
| **收益** | 闲聊响应速度提升（跳过检索）、检索资源节省、回答质量提升 |
| **兜底策略** | LLM 分类失败时默认走 RAG，确保不丢失知识类问题 |

### 12.7 已修复：路由后 Prompt 冲突问题

> **问题**：路由正确识别了 chitchat/general 类型，但 `_build_messages()` 仍使用 RAG 的 `SYSTEM_PROMPT`（含"严格基于参考资料"指令），
> 导致 LLM 看到"无参考资料"就回答"抱歉，没有找到相关资料"。

> **修复**：在 `prompt_template.py` 中新增 `CHITCHAT_SYSTEM_PROMPT` 和 `GENERAL_SYSTEM_PROMPT`，
> `chain.py` 的 `_build_messages()` 根据 `query_type` 选择对应的系统提示词。

```python
# rag/prompt_template.py
CHITCHAT_SYSTEM_PROMPT = """你是一个友好、有趣的 AI 助手..."""  # 不含"参考资料"指令
GENERAL_SYSTEM_PROMPT = """你是一个知识丰富的 AI 助手..."""   # 不含"参考资料"指令

# rag/chain.py
def _build_messages(self, query, context, session_id, query_type=QueryType.RAG):
    if query_type == QueryType.CHITCHAT:
        system_msg = CHITCHAT_SYSTEM_PROMPT
    elif query_type == QueryType.GENERAL:
        system_msg = GENERAL_SYSTEM_PROMPT
    else:
        system_msg = SYSTEM_PROMPT.format(context=context)
```

---

## 13. 多知识库架构

> **核心问题**：当前所有文档都存入同一个 Milvus Collection，不同类型的数据混在一起。
> 当知识库规模增大后，检索精度会下降，且无法按领域/类型隔离管理。

### 13.1 当前架构 vs 目标架构

**当前：单一知识库**

```
所有文档 → knowledge_base (1024d) + knowledge_base_mm (4096d)
           ├── 技术文档
           ├── 产品手册
           ├── 会议记录
           └── ...混在一起
```

**目标：多知识库空间**

```
┌─────────────────────────────────────────────────┐
│                知识库管理器                       │
├──────────────┬──────────────┬───────────────────┤
│  技术文档库   │  产品手册库   │  会议记录库        │
│  kb_tech     │  kb_product  │  kb_meeting       │
│              │              │                   │
│  ├── 向量索引 │  ├── 向量索引 │  ├── 向量索引      │
│  ├── BM25    │  ├── BM25    │  ├── BM25         │
│  └── 文档列表 │  └── 文档列表 │  └── 文档列表      │
└──────────────┴──────────────┴───────────────────┘
                        │
                        ▼
              查询时指定检索哪个库（或联合检索）
```

### 13.2 数据模型设计

```python
# models/knowledge_base.py

from pydantic import BaseModel
from datetime import datetime

class KnowledgeBase(BaseModel):
    """知识库元数据"""
    id: str                          # 唯一 ID
    name: str                        # 知识库名称（如 "技术文档库"）
    description: str = ""            # 描述
    collection_name: str             # Milvus Collection 名称
    mm_collection_name: str          # 多模态 Collection 名称
    document_count: int = 0          # 文档数量
    chunk_count: int = 0             # 切片数量
    embedding_model: str = "bge-large-zh-v1.5"
    embedding_dim: int = 1024
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

    class Config:
        json_schema_extra = {
            "example": {
                "id": "kb_tech_001",
                "name": "技术文档库",
                "description": "存放项目技术文档和架构设计",
                "collection_name": "kb_tech",
                "mm_collection_name": "kb_tech_mm"
            }
        }
```

### 13.3 Milvus Collection 管理

```python
# rag/collection_manager.py

from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility

class CollectionManager:
    """Milvus Collection 生命周期管理"""

    def __init__(self, milvus_uri: str):
        connections.connect(uri=milvus_uri)

    def create_kb_collections(self, kb_id: str, embedding_dim: int = 1024):
        """为知识库创建文本 + 多模态两个 Collection"""
        collection_name = f"kb_{kb_id}"
        mm_collection_name = f"kb_{kb_id}_mm"

        # 文本 Collection
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("source", DataType.VARCHAR, max_length=512),
                FieldSchema("page_number", DataType.INT64),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=embedding_dim),
            ]
            schema = CollectionSchema(fields, description=f"知识库 {kb_id} 文本向量")
            collection = Collection(collection_name, schema)
            collection.create_index("vector", {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            })

        # 多模态 Collection（4096 维）
        if not utility.has_collection(mm_collection_name):
            fields = [
                FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema("text", DataType.VARCHAR, max_length=65535),
                FieldSchema("source", DataType.VARCHAR, max_length=512),
                FieldSchema("page_number", DataType.INT64),
                FieldSchema("image_path", DataType.VARCHAR, max_length=1024),
                FieldSchema("vector", DataType.FLOAT_VECTOR, dim=4096),
            ]
            schema = CollectionSchema(fields, description=f"知识库 {kb_id} 多模态向量")
            collection = Collection(mm_collection_name, schema)
            collection.create_index("vector", {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            })

        return collection_name, mm_collection_name

    def delete_kb_collections(self, kb_id: str):
        """删除知识库的所有 Collection"""
        for suffix in ["", "_mm"]:
            name = f"kb_{kb_id}{suffix}"
            if utility.has_collection(name):
                utility.drop_collection(name)

    def list_collections(self) -> list[str]:
        """列出所有知识库 Collection"""
        return [c for c in utility.list_collections()
                if c.startswith("kb_") and not c.endswith("_mm")]
```

### 13.4 API 接口设计

```
POST   /api/kb                        创建知识库
GET    /api/kb                        列出所有知识库
GET    /api/kb/{kb_id}                获取知识库详情
DELETE /api/kb/{kb_id}                删除知识库（含所有文档和向量）
PUT    /api/kb/{kb_id}                更新知识库信息

POST   /api/kb/{kb_id}/docs/upload    上传文档到指定知识库
GET    /api/kb/{kb_id}/docs           列出知识库中的文档
DELETE /api/kb/{kb_id}/docs/{name}    删除知识库中的文档

POST   /api/chat                      查询时新增 kb_id 参数（可选，支持多库联合）
```

```python
# 查询接口扩展
class ChatRequest(BaseModel):
    query: str
    session_id: str = ""
    stream: bool = True
    mode: str = "text"           # text / multimodal
    kb_ids: list[str] = []       # 新增：指定检索的知识库，空则检索全部
```

### 13.5 多库联合检索

```python
# rag/retriever.py — 扩展检索器

async def search_multi_kb(
    self,
    query: str,
    kb_ids: list[str] = [],
    top_k: int = 5
) -> list[dict]:
    """跨多个知识库检索并合并结果"""
    if not kb_ids:
        # 检索所有知识库
        kb_ids = self.collection_manager.list_collections()

    all_results = []
    for kb_id in kb_ids:
        retriever = self._get_retriever_for_kb(kb_id)
        results = await retriever.search(query, top_k=top_k)
        # 标记来源知识库
        for r in results:
            r["kb_id"] = kb_id
        all_results.extend(results)

    # 跨库 RRF 融合
    return self._cross_kb_rrf(all_results, top_k)
```

### 13.6 知识库元数据持久化

```python
# utils/kb_store.py

import json
from pathlib import Path

KB_STORE_PATH = Path("data/knowledge_bases.json")

def load_knowledge_bases() -> dict[str, KnowledgeBase]:
    """加载所有知识库元数据"""
    if not KB_STORE_PATH.exists():
        return {}
    with open(KB_STORE_PATH) as f:
        data = json.load(f)
    return {k: KnowledgeBase(**v) for k, v in data.items()}

def save_knowledge_bases(kbs: dict[str, KnowledgeBase]):
    """保存知识库元数据"""
    KB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(KB_STORE_PATH, "w") as f:
        json.dump({k: v.model_dump() for k, v in kbs.items()}, f,
                  ensure_ascii=False, indent=2, default=str)
```

### 13.7 前端交互设计

```
┌──────────────────────────────────────────────────┐
│  知识库管理                                [+新建] │
├──────────────────────────────────────────────────┤
│                                                  │
│  📚 技术文档库                         3 篇文档   │
│     存放项目技术文档和架构设计            [管理]   │
│                                                  │
│  📦 产品手册库                         8 篇文档   │
│     产品使用手册和 API 文档              [管理]   │
│                                                  │
│  📋 会议记录库                         12 篇文档  │
│     2026年会议纪要                      [管理]   │
│                                                  │
│  🗑️ 默认知识库                         5 篇文档   │
│     系统默认知识库（不可删除）            [管理]   │
│                                                  │
└──────────────────────────────────────────────────┘
```

对话页面新增知识库选择器：

```
对话页面侧边栏新增：
┌──────────────────┐
│ 检索范围          │
│ ☑ 技术文档库     │
│ ☑ 产品手册库     │
│ ☐ 会议记录库     │
│ ☑ 默认知识库     │
│                  │
│ [全选] [全不选]   │
└──────────────────┘
```

### 13.8 与查询路由的协同

多知识库和查询路由可以协同工作：

```
用户提问
    │
    ▼
查询路由判断 → 闲聊/通用 → 直接 LLM 回答
    │
    ▼ RAG 类型
根据用户选择的 kb_ids → 多库联合检索 → LLM 生成
```

---

## 14. LangGraph 多 Agent 架构升级规划

> **战略方向**：将当前单链式 RAG 架构升级为基于 LangGraph 的多 Agent 协作平台，
> 实现更灵活的检索策略编排、多步推理和工具调用能力。

### 14.1 架构对比

**当前：单链式 RAG**

```
用户提问 → 检索 → 拼接上下文 → LLM 生成 → 返回答案
```

问题：
- 检索策略固定，无法根据问题类型动态选择
- 不支持多步推理（如先检索、再追问、再检索）
- 无法调用外部工具（如数据库查询、API 调用）

**目标：LangGraph 多 Agent 协作**

```
用户提问
    │
    ▼
┌─────────────────┐
│   路由 Agent     │  判断问题类型，选择处理路径
└────────┬────────┘
         │
    ┌────┴─────┬──────────────┐
    ▼          ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐
│检索Agent│ │工具Agent  │ │闲聊Agent  │
│文本+MM │ │SQL/API等 │ │直接回复  │
└───┬────┘ └────┬─────┘ └────┬─────┘
    │           │             │
    └─────┬─────┘             │
          ▼                   │
   ┌──────────────┐           │
   │  生成 Agent   │◀──────────┘
   │  组织回答     │
   └──────┬───────┘
          ▼
   ┌──────────────┐
   │  质量 Agent   │  检查答案质量，决定是否重试
   └──────┬───────┘
          ▼
       返回答案
```

### 14.2 LangGraph 核心概念

| 概念 | 说明 | 映射到本系统 |
|------|------|-------------|
| **State** | 全局状态对象，在节点间传递 | 包含 query、docs、history、answer 等 |
| **Node** | 处理单元，执行具体逻辑 | 检索、生成、路由、重排等 |
| **Edge** | 节点间的连接 | 条件路由（按问题类型） |
| **Conditional Edge** | 根据状态决定走向 | 检索无结果 → 换策略重试 |
| **Checkpoint** | 状态快照，支持中断恢复 | 对话持久化 |

### 14.3 Agent 设计

#### 路由 Agent（Router）

```python
from typing import Literal
from langgraph.graph import StateGraph

def router_node(state: dict) -> Literal["retrieval", "tool", "chitchat"]:
    """根据问题类型路由到不同 Agent"""
    query = state["query"]
    # 使用 LLM 判断问题类型
    prompt = f"""判断以下问题属于哪一类：
1. retrieval - 需要从知识库中检索信息回答
2. tool - 需要调用外部工具（数据库查询、API等）
3. chitchat - 闲聊/问候/与知识库无关

问题：{query}
只回答类别名："""
    category = llm.invoke(prompt).strip().lower()
    return category
```

#### 检索 Agent（Retriever）

```python
async def retrieval_node(state: dict) -> dict:
    """多策略检索，根据 query 特征选择策略"""
    query = state["query"]
    mode = state.get("mode", "text")

    # 并行执行多路检索
    text_results = await text_retriever.search(query, top_k=10)
    if mode == "multimodal":
        mm_results = await mm_retriever.search(query, top_k=10)
        docs = rrf_fusion(text_results, mm_results)
    else:
        docs = text_results

    # 重排
    docs = await reranker.rerank(query, docs, top_k=5)

    return {"retrieved_docs": docs}
```

#### 生成 Agent（Generator）

```python
async def generator_node(state: dict) -> dict:
    """基于检索结果生成回答"""
    context = build_context(state["retrieved_docs"])
    messages = build_messages(context, state["history"], state["query"])
    answer = await llm.stream(messages)
    return {"answer": answer, "references": state["retrieved_docs"]}
```

#### 质量 Agent（Quality Checker）

```python
async def quality_check_node(state: dict) -> Literal["pass", "retry"]:
    """检查生成答案的质量"""
    answer = state["answer"]
    query = state["query"]
    docs = state["retrieved_docs"]

    # 检查：答案是否引用了参考资料、是否过于简短、是否包含“不确定”等
    if len(answer) < 20 or "不确定" in answer or "无法回答" in answer:
        if state.get("retry_count", 0) < 2:
            return "retry"  # 重试：扩大检索范围
    return "pass"
```

### 14.4 LangGraph 工作流定义

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class RAGState(TypedDict):
    query: str
    mode: str
    history: list[dict]
    retrieved_docs: list[dict]
    answer: str
    references: list[dict]
    retry_count: int

# 构建工作流
workflow = StateGraph(RAGState)

# 添加节点
workflow.add_node("router", router_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("generator", generator_node)
workflow.add_node("quality_check", quality_check_node)

# 添加边
workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda x: x["route"],
    {
        "retrieval": "retrieval",
        "tool": "tool_agent",
        "chitchat": "chitchat_agent",
    }
)
workflow.add_edge("retrieval", "generator")
workflow.add_edge("generator", "quality_check")
workflow.add_conditional_edges(
    "quality_check",
    lambda x: x["quality"],
    {"pass": END, "retry": "retrieval"}
)

# 编译
app = workflow.compile()
```

### 14.5 迁移收益

| 收益 | 说明 |
|------|------|
| **灵活路由** | 根据问题类型自动选择检索/工具/闲聊 |
| **多步推理** | 支持“检索→判断不足→扩大范围再检索”的循环 |
| **质量兜底** | 生成后检查质量，不合格自动重试 |
| **工具扩展** | 未来可接入 SQL 查询、API 调用、计算器等工具 |
| **可观测性** | LangGraph 内置状态追踪，方便调试和日志 |
| **切片统一** | 迁移时用 LangChain 切片器替代当前实现 |

### 14.6 迁移注意事项

| 事项 | 说明 |
|------|------|
| **依赖体积** | LangChain + LangGraph 包较大，确认部署环境支持 |
| **API 兼容** | 保持现有 `/api/chat` 接口不变，内部实现替换 |
| **流式输出** | LangGraph 支持 `stream` 模式，需适配前端 SSE 解析 |
| **渐进迁移** | 先迁移检索+生成链路，再逐步添加路由和质量检查 |
| **向后兼容** | 保留当前 `rag/` 模块作为 fallback |

### 14.7 推荐引入的 LangChain 生态组件

| 组件 | 用途 | 引入时机 |
|------|------|----------|
| `langchain.text_splitter` | 文档切片 | 迁移第一步 |
| `langchain.embeddings` | Embedding 封装 | 迁移时替换 embedder.py |
| `langchain.vectorstores` | 向量库封装 | 可选，Milvus 有 LangChain 集成 |
| `langchain.callbacks` | 回调和日志 | 迁移时引入 |
| `langsmith` | 追踪和调试 | 上线后引入 |

---

## 15. 优化路线图

### 阶段一：正确性修复（1-2 天）

| 优先级 | 优化项 | 预期收益 |
|--------|--------|----------|
| P0 | 确认相似度语义，修正阈值过滤 | 检索结果正确性 |
| P0 | 上下文 Token 长度控制 | 防止 Prompt 超长 |
| P0 | 统一 Prompt 管理 | 降低维护成本 |

### 阶段二：功能增强（3-5 天）

| 优先级 | 优化项 | 预期收益 |
|--------|--------|----------|
| P1 | 查询预处理 | 减少无效检索 |
| P1 | Embedding 重试机制 | 入库成功率提升 |
| P1 | **查询路由（规则 + LLM 分类）** | 闲聊跳过检索，响应更快更准 |
| P1 | 检索结果重排（Cross-Encoder） | 答案准确率 +10%~20% |
| P1 | Token 级对话记忆 | 长对话稳定性 |
| P1 | 提取检索器公共基类 | 为 LangGraph 迁移做准备 |

### 阶段三：多知识库架构（3-5 天）

| 步骤 | 内容 | 说明 |
|------|------|------|
| 3.1 | 设计知识库元数据模型 | KnowledgeBase Pydantic 模型 |
| 3.2 | 实现 CollectionManager | 创建/删除/列举 Milvus Collection |
| 3.3 | 实现知识库 CRUD API | `/api/kb` 系列接口 |
| 3.4 | 扩展检索器支持多库联合检索 | 跨库 RRF 融合 |
| 3.5 | 前端知识库管理页面 | 创建/选择/管理知识库 |
| 3.6 | 对话页面集成知识库选择器 | 用户指定检索范围 |

### 阶段四：LangGraph 架构迁移（1-2 周）

| 步骤 | 内容 | 说明 |
|------|------|------|
| 4.1 | 引入 LangChain 切片器 | 替代 `embeddings/chunker.py`，解决语义切片问题 |
| 4.2 | 封装 Embedding 为 LangChain 接口 | 统一 embedder 调用方式 |
| 4.3 | 构建 LangGraph 工作流 | 路由 → 检索 → 生成 → 质量检查 |
| 4.4 | 添加重排节点 | 集成 bge-reranker |
| 4.5 | 适配流式输出 | LangGraph stream → 前端 SSE |
| 4.6 | 保留旧 `rag/` 作为 fallback | 平滑过渡，可随时回退 |

### 阶段五：健壮性与体验（持续迭代）

| 优先级 | 优化项 | 预期收益 |
|--------|--------|----------|
| P2 | Embedding 批量分片 | 大文档入库稳定性 |
| P2 | 文件大小校验 | 防止资源滥用 |
| P2 | 编码自动检测 | 支持更多文件格式 |
| P2 | 表格结构保留 | 表格类文档问答质量 |
| P2 | 历史摘要机制 | 超长对话体验 |
| P2 | 工具 Agent 扩展 | 数据库查询、API 调用等 |

---

## 附录 A：关键参数建议值

| 参数 | 当前值 | 建议值 | 说明 |
|------|--------|--------|------|
| `chunk_size` | 500 字符 | 300~500 字符 | 语义切片后可适当放宽 |
| `chunk_overlap` | 50 字符 | 50~80 字符 | 保持上下文连续 |
| `retriever_top_k` | 5 | 5~10 | 精排后取 top 5 |
| `hybrid_alpha` | 0.7 | 0.6~0.8 | 向量检索权重 |
| `rrf_k` | 60 (硬编码) | 60 (可配置) | RRF 融合参数 |
| `memory_window` | 5 轮 | 按 token 预算 | 建议 2000~3000 token |
| `embedding_batch_size` | 无限 | 32 | 防止 API 超时 |
| `embedding_timeout` | 60s | 30s + 3次重试 | 更灵活的超时策略 |
| `max_context_tokens` | 无限制 | 3000 | 预留空间给历史和生成 |
| `max_file_size` | 100MB (未校验) | 100MB (校验生效) | 防止超大文件 |

---

## 附录 B：推荐外部工具/模型

| 用途 | 推荐方案 | 说明 |
|------|----------|------|
| **多 Agent 编排** | `langgraph` | LangGraph 工作流编排，迁移目标 |
| **LLM/Embedding 封装** | `langchain-core` | 统一接口，支持多种模型 |
| **文档切片** | `langchain.text_splitter.RecursiveCharacterTextSplitter` | 迁移时替代当前 chunker |
| **重排序** | `bge-reranker-base` | 开源轻量，API 调用方便 |
| **追踪调试** | `langsmith` | LangChain 官方追踪平台 |
| **编码检测** | `chardet` / `charset-normalizer` | Python 标准库级别 |
| **Token 估算** | `tiktoken` (OpenAI) 或字符比例法 | 精确或轻量 |
| **BM25 持久化** | `whoosh` / `elasticsearch` | 替代内存 BM25，支持增量更新 |

---

## 附录 C：版本变更记录

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v1.0 | 2026-06-19 | 初始版本，包含完整 RAG 优化方案 |
| v2.0 | 2026-06-19 | 新增 LangGraph 多 Agent 架构升级规划（第 14 章）；切片优化标记为暂缓，迁移时用 LangChain 方案统一解决；路线图由三阶段调整为四阶段，新增 LangGraph 迁移阶段；附录 B 补充 LangGraph 生态工具 |
| v3.0 | 2026-06-19 | 新增查询路由优化（第 12 章）和多知识库架构（第 13 章）；路线图由四阶段扩展为五阶段，新增功能增强和多知识库阶段；问题总览新增查询路由和单一知识库问题 |
