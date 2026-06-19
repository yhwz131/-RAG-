"""
RAG 链模块
实现完整的检索增强生成流程：查询 -> 检索 -> 构建 Prompt -> LLM 生成
支持查询路由：规则 + LLM 混合路由，区分 rag / chitchat / general 三类查询
"""
import httpx
from typing import List, Dict, Optional
from config.settings import settings
from utils.logger import get_logger
from rag.retriever import VectorRetriever
from rag.memory import ConversationMemory
from rag.prompt_template import SYSTEM_PROMPT, CHITCHAT_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT, MULTIMODAL_SYSTEM_PROMPT, estimate_tokens
from rag.router import route_query, QueryType, preprocess_query

logger = get_logger("chain")


class RAGChain:
    """RAG 检索增强生成链"""
    
    def __init__(
        self,
        retriever: Optional[VectorRetriever] = None,
        memory: Optional[ConversationMemory] = None,
        is_multimodal: bool = False
    ):
        self.retriever = retriever or VectorRetriever()
        self.memory = memory or ConversationMemory()
        self.is_multimodal = is_multimodal
        self.llm_url = settings.llm_api_url
        self.llm_key = settings.llm_api_key
        self.llm_model = settings.llm_model
        self.timeout = settings.llm_timeout
        logger.info(f"RAG Chain 初始化完成 (multimodal={is_multimodal})")
    
    def _build_context(self, docs: List[Dict], max_context_tokens: int = 3000) -> str:
        """将检索到的文档构建为上下文字符串（带 token 长度控制）
        
        Args:
            docs: 检索到的文档列表
            max_context_tokens: 上下文最大 token 数（默认 3000）
        """
        if not docs:
            return "（无相关参考资料）"
        
        # 按相似度降序排列，确保高相关性内容优先被 LLM 看到
        docs = sorted(docs, key=lambda d: d.get("score", 0), reverse=True)
        
        parts = []
        current_tokens = 0
        
        for i, doc in enumerate(docs, 1):
            source = doc.get("source", "未知来源")
            chunk_index = doc.get("chunk_index", "N/A")
            score = doc.get("score", 0)
            content = doc.get("content", "")
            page = doc.get("page_number", 0)
            # 构建来源描述
            loc_parts = [f"来源: {source}"]
            if page and page > 0:
                loc_parts.append(f"第{page}页")
            loc_parts.append(f"段落: {chunk_index}")
            loc_parts.append(f"相似度: {score:.4f}")
            location = ", ".join(loc_parts)
            ref = f"[参考{i}] ({location})\n{content}"
            
            # 估算当前参考的 token 数
            ref_tokens = estimate_tokens(ref)
            
            if current_tokens + ref_tokens > max_context_tokens:
                # 剩余空间不足，尝试截断当前条目
                remaining = max_context_tokens - current_tokens
                if remaining > 100:  # 至少保留 100 token 的空间
                    # 按比例截断内容
                    truncate_chars = int(remaining * 0.7)  # 留 30% 给格式开销
                    truncated_content = content[:truncate_chars] + "..."
                    ref = f"[参考{i}] ({location})\n{truncated_content}"
                    parts.append(ref)
                    logger.info(f"上下文截断: 第{i}条参考被截断，剩余空间={remaining} tokens")
                else:
                    logger.info(f"上下文已满: 已包含{i-1}条参考，跳过剩余{len(docs)-i+1}条")
                break
            
            parts.append(ref)
            current_tokens += ref_tokens
        
        context = "\n\n".join(parts)
        logger.info(f"上下文构建完成: {len(parts)}条参考, 估算{current_tokens} tokens")
        return context
    
    def _build_messages(self, query: str, context: str, session_id: str, query_type: QueryType = QueryType.RAG) -> List[Dict]:
        """构建发送给 LLM 的消息列表"""
        if query_type == QueryType.CHITCHAT:
            system_msg = CHITCHAT_SYSTEM_PROMPT
        elif query_type == QueryType.GENERAL:
            system_msg = GENERAL_SYSTEM_PROMPT
        elif self.is_multimodal:
            system_msg = MULTIMODAL_SYSTEM_PROMPT.format(context=context)
        else:
            system_msg = SYSTEM_PROMPT.format(context=context)
        messages = [{"role": "system", "content": system_msg}]
        # 加入历史对话
        history = self.memory.get_context(session_id)
        messages.extend(history)
        # 加入当前查询
        messages.append({"role": "user", "content": query})
        return messages
    
    def _call_llm(self, messages: List[Dict], stream: bool = False):
        """调用 LLM API（兼容 OpenAI 格式）"""
        headers = {"Content-Type": "application/json"}
        if self.llm_key:
            headers["Authorization"] = f"Bearer {self.llm_key}"
        
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": stream,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens
        }
        
        if stream:
            return self._stream_llm(headers, payload)
        else:
            return self._sync_llm(headers, payload)
    
    def _sync_llm(self, headers: dict, payload: dict) -> str:
        """同步调用 LLM"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(self.llm_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise
    
    def _stream_llm(self, headers: dict, payload: dict):
        """流式调用 LLM"""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", self.llm_url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                import json
                                data = json.loads(data_str)
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except Exception:
                                continue
        except Exception as e:
            logger.error(f"LLM 流式调用失败: {e}")
            raise
    
    def chat(self, query: str, session_id: str = "default", stream: bool = False):
        """
        执行 RAG 对话（支持查询路由）
        
        Args:
            query: 用户查询
            session_id: 会话ID
            stream: 是否流式输出
        
        Returns:
            非流式返回 (answer, docs)，流式返回生成器
        """
        logger.info(f"收到查询: session={session_id}, query={query[:100]}")
        
        # 0. 查询路由：混合路由（规则 + LLM 分类）
        query_type = route_query(query)
        logger.info(f"查询路由结果: {query_type.value}")
        
        # 闲聊/通用知识：直接走 LLM，不走 RAG 检索
        if query_type in (QueryType.CHITCHAT, QueryType.GENERAL):
            logger.info(f"查询路由: {query_type.value} 类型，跳过 RAG 检索")
            messages = self._build_messages(query, "", session_id, query_type=query_type)
            self.memory.add(session_id, "user", query)
            if stream:
                return self._chat_stream(query, messages, session_id, [], query_type=query_type)
            else:
                answer = self._call_llm(messages, stream=False)
                self.memory.add(session_id, "assistant", answer)
                return answer, [], query_type
        
        # 1. 检索相关文档（预处理查询，仅用于检索；LLM 仍看原始 query）
        retrieval_query = preprocess_query(query)
        docs = self.retriever.search(retrieval_query)
        
        # 1.5 硬校验：无相关文档时直接返回，不调用 LLM
        if not docs:
            no_ref_answer = (
                "抱歉，知识库中没有找到与您问题相关的内容。"
                "请尝试换个问法，或上传相关文档后再提问。"
            )
            self.memory.add(session_id, "user", query)
            self.memory.add(session_id, "assistant", no_ref_answer)
            logger.info("知识校验: 未检索到相关文档，返回拒答")
            if stream:
                def _empty_gen():
                    yield no_ref_answer
                return _empty_gen(), [], query_type
            return no_ref_answer, [], query_type
        
        # 2. 构建上下文（带 token 长度控制）
        context = self._build_context(docs, max_context_tokens=settings.max_context_tokens)
        
        # 3. 构建消息
        messages = self._build_messages(query, context, session_id)
        
        # 4. 记录用户消息
        self.memory.add(session_id, "user", query)
        
        # 5. 调用 LLM
        if stream:
            return self._chat_stream(query, messages, session_id, docs, query_type=query_type)
        else:
            answer = self._call_llm(messages, stream=False)
            # 记录助手回答
            self.memory.add(session_id, "assistant", answer)
            logger.info(f"回答生成完成: {answer[:100]}")
            return answer, docs, query_type
    
    def _chat_stream(self, query: str, messages: List[Dict], session_id: str, docs: List[Dict], query_type: QueryType = QueryType.RAG):
        """流式对话，先返回引用信息，再流式返回回答"""
        import json as _json
        # 先 yield 查询类型和引用信息
        meta = {
            "query_type": query_type.value,
            "sources": [
                {
                    "source": d.get("source", "未知"),
                    "chunk_index": d.get("chunk_index", 0),
                    "content_snippet": d.get("content", "")[:200],
                    "score": d.get("score", 0),
                    "page_number": d.get("page_number", 0),
                    "has_image": d.get("has_image", False),
                    "image_url": d.get("image_url", ""),
                }
                for d in docs
            ]
        }
        yield f"[SOURCES]{_json.dumps(meta, ensure_ascii=False)}[/SOURCES]\n\n"
        
        # 流式返回回答
        full_answer = ""
        for chunk in self._call_llm(messages, stream=True):
            full_answer += chunk
            yield chunk
        
        # 记录完整回答
        self.memory.add(session_id, "assistant", full_answer)