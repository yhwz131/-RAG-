"""
RAG 链模块
实现完整的检索增强生成流程：查询 -> 检索 -> 构建 Prompt -> LLM 生成
支持查询路由：规则 + LLM 混合路由，区分 rag / chitchat / general / database 四类查询
database 类型走 Text-to-SQL 路径：表结构 -> LLM 生成 SQL -> 执行 -> LLM 总结
"""
import json
import httpx
from typing import List, Dict, Optional
from config.settings import settings
from utils.logger import get_logger
from rag.retriever import VectorRetriever
from rag.memory import ConversationMemory
from rag.prompt_template import (
    SYSTEM_PROMPT, CHITCHAT_SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT,
    MULTIMODAL_SYSTEM_PROMPT, SQL_GENERATION_PROMPT, SQL_RESULT_PROMPT, estimate_tokens,
)
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
        self.llm_url = settings.mm_llm_api_url if is_multimodal else settings.llm_api_url
        self.llm_key = settings.llm_api_key
        self.llm_model = settings.mm_llm_model if is_multimodal else settings.llm_model
        self.timeout = settings.llm_timeout
        logger.info(f"RAG Chain 初始化完成 (multimodal={is_multimodal}, model={self.llm_model})")
    
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
    
    def _build_messages(self, query: str, context: str, session_id: str, query_type: QueryType = QueryType.RAG, images: Optional[List[str]] = None, file_context: Optional[str] = None) -> List[Dict]:
        """构建发送给 LLM 的消息列表
        
        Args:
            query: 用户查询
            context: 检索上下文
            session_id: 会话ID
            query_type: 查询类型
            images: 用户上传的图片列表（base64 编码）
            file_context: 用户上传的文档提取的文本内容
        """
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
        
        # 如果有文档上下文，拼接到查询前面
        user_query = query
        if file_context:
            user_query = f"以下是用户上传的文件内容，请结合这些内容回答问题：\n\n{file_context}\n\n---\n问题：{query}"
        
        # 加入当前查询（有图片时使用 vision API 格式）
        if images and self.is_multimodal:
            content = []
            for img_b64 in images:
                # 自动检测图片格式（默认 png）
                mime = "image/png"
                if img_b64.startswith("/9j/"):
                    mime = "image/jpeg"
                elif img_b64.startswith("iVBOR"):
                    mime = "image/png"
                elif img_b64.startswith("R0lGOD"):
                    mime = "image/gif"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"}
                })
            content.append({"type": "text", "text": user_query})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_query})
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
    
    # ========== Text-to-SQL 路径 ==========
    
    def _get_db_source(self):
        """获取数据库数据源（延迟导入，避免循环依赖）"""
        from api.pipeline.engines.database import create_database_source
        return create_database_source()
    
    def _handle_database_query(self, query: str, session_id: str, stream: bool = False):
        """Text-to-SQL 处理流程：表结构 -> LLM 生成 SQL -> 安全执行 -> LLM 总结
        
        Args:
            query: 用户自然语言查询
            session_id: 会话 ID
            stream: 是否流式输出
        """
        logger.info(f"[Text-to-SQL] 开始处理: {query[:80]}")
        
        # 1. 获取数据库连接和表结构
        db_source = self._get_db_source()
        if not db_source:
            err_msg = "未配置数据库连接，请先在管理页面配置数据库。"
            self.memory.add(session_id, "user", query)
            self.memory.add(session_id, "assistant", err_msg)
            if stream:
                def _err_gen():
                    yield f'[SOURCES]{{"query_type":"database","sources":[]}}[/SOURCES]\n\n'
                    yield err_msg
                return _err_gen(), [], QueryType.DATABASE
            return err_msg, [], QueryType.DATABASE
        
        try:
            # 2. 获取表结构
            schema_text = db_source.get_schema_for_llm()
            logger.info(f"[Text-to-SQL] 表结构:\n{schema_text}")
            
            # 3. LLM 生成 SQL
            sql_prompt = SQL_GENERATION_PROMPT.format(
                db_type=settings.db_type or "mysql",
                db_name=settings.db_name,
                schema_info=schema_text,
                query=query,
            )
            sql_messages = [
                {"role": "system", "content": "你是 SQL 专家，只输出可执行的 SQL 语句，不要输出任何解释。"},
                {"role": "user", "content": sql_prompt},
            ]
            raw_sql = self._call_llm(sql_messages, stream=False).strip()
            
            # 清理 SQL（去掉可能的 markdown 代码块标记）
            if raw_sql.startswith("```"):
                raw_sql = raw_sql.split("\n", 1)[-1]
            if raw_sql.endswith("```"):
                raw_sql = raw_sql.rsplit("```", 1)[0]
            raw_sql = raw_sql.strip().rstrip(";")
            
            # 检查 LLM 是否拒绝生成
            if raw_sql.upper().startswith("ERROR"):
                err_msg = f"无法为您的问题生成 SQL 查询。请尝试换个问法。"
                logger.warning(f"[Text-to-SQL] LLM 拒绝生成 SQL: {raw_sql}")
                self.memory.add(session_id, "user", query)
                self.memory.add(session_id, "assistant", err_msg)
                if stream:
                    def _err_gen():
                        yield f'[SOURCES]{{"query_type":"database","sources":[]}}[/SOURCES]\n\n'
                        yield err_msg
                    return _err_gen(), [], QueryType.DATABASE
                return err_msg, [], QueryType.DATABASE
            
            logger.info(f"[Text-to-SQL] 生成的 SQL: {raw_sql}")
            
            # 4. 安全执行 SQL
            result = db_source.execute_sql(raw_sql)
            logger.info(f"[Text-to-SQL] 查询结果: {result['row_count']} 行")
            
            # 5. 格式化结果
            result_text = self._format_sql_result(result)
            
            # 6. LLM 总结回答
            answer_prompt = SQL_RESULT_PROMPT.format(
                query=query,
                sql=raw_sql,
                result=result_text,
            )
            answer_messages = [
                {"role": "system", "content": "你是数据分析助手，根据 SQL 查询结果用自然语言回答用户问题。"},
                *self.memory.get_context(session_id),
                {"role": "user", "content": answer_prompt},
            ]
            
            # 记录用户消息
            self.memory.add(session_id, "user", query)
            
            if stream:
                return self._chat_stream_database(query, raw_sql, result, answer_messages, session_id)
            else:
                answer = self._call_llm(answer_messages, stream=False)
                self.memory.add(session_id, "assistant", answer)
                # 构造一个伪 docs 用于前端展示 SQL 信息
                sql_doc = {
                    "source": f"SQL: {raw_sql}",
                    "content": result_text[:500],
                    "score": 1.0,
                    "chunk_index": 0,
                    "page_number": 0,
                    "has_image": False,
                    "image_url": "",
                }
                logger.info(f"[Text-to-SQL] 回答生成完成")
                return answer, [sql_doc], QueryType.DATABASE
                
        except Exception as e:
            logger.error(f"[Text-to-SQL] 执行失败: {e}")
            err_msg = f"数据库查询失败：{str(e)}"
            self.memory.add(session_id, "user", query)
            self.memory.add(session_id, "assistant", err_msg)
            if stream:
                def _err_gen():
                    yield f'[SOURCES]{{"query_type":"database","sources":[]}}[/SOURCES]\n\n'
                    yield err_msg
                return _err_gen(), [], QueryType.DATABASE
            return err_msg, [], QueryType.DATABASE
        
        finally:
            try:
                db_source.close()
            except Exception:
                pass
    
    def _format_sql_result(self, result: Dict, max_rows: int = 20) -> str:
        """将 SQL 查询结果格式化为 LLM 可读的文本表格"""
        columns = result.get("columns", [])
        rows = result.get("rows", [])
        row_count = result.get("row_count", 0)
        
        if not rows:
            return "（查询结果为空）"
        
        # 限制行数
        display_rows = rows[:max_rows]
        
        # 构建 Markdown 表格
        header = " | ".join(str(c) for c in columns)
        separator = " | ".join("---" for _ in columns)
        body_lines = []
        for row in display_rows:
            line = " | ".join(str(row.get(c, "")) for c in columns)
            body_lines.append(line)
        
        table = f"| {header} |\n| {separator} |\n" + "\n".join(f"| {l} |" for l in body_lines)
        
        if row_count > max_rows:
            table += f"\n\n（共 {row_count} 行，仅显示前 {max_rows} 行）"
        else:
            table += f"\n\n（共 {row_count} 行）"
        
        return table
    
    def _chat_stream_database(self, query: str, sql: str, result: Dict, messages: List[Dict], session_id: str):
        """Text-to-SQL 流式输出"""
        import json as _json
        
        # 先 yield 元信息（SQL + 引用）
        result_preview = self._format_sql_result(result, max_rows=5)
        meta = {
            "query_type": "database",
            "sources": [{
                "source": f"SQL: {sql}",
                "content_snippet": result_preview[:300],
                "score": 1.0,
                "chunk_index": 0,
                "page_number": 0,
                "has_image": False,
                "image_url": "",
            }]
        }
        yield f"[SOURCES]{_json.dumps(meta, ensure_ascii=False)}[/SOURCES]\n\n"
        
        # 流式返回 LLM 回答
        full_answer = ""
        for chunk in self._call_llm(messages, stream=True):
            full_answer += chunk
            yield chunk
        
        self.memory.add(session_id, "assistant", full_answer)
    
    def chat(self, query: str, session_id: str = "default", stream: bool = False, images: Optional[List[str]] = None, file_context: Optional[str] = None):
        """
        执行 RAG 对话（支持查询路由）
        
        Args:
            query: 用户查询
            session_id: 会话ID
            stream: 是否流式输出
            images: 用户上传的图片列表（base64 编码，可选）
            file_context: 用户上传的文档提取的文本内容（可选）
        
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
            messages = self._build_messages(query, "", session_id, query_type=query_type, images=images, file_context=file_context)
            self.memory.add(session_id, "user", query)
            if stream:
                return self._chat_stream(query, messages, session_id, [], query_type=query_type)
            else:
                answer = self._call_llm(messages, stream=False)
                self.memory.add(session_id, "assistant", answer)
                return answer, [], query_type
        
        # 数据库查询：走 Text-to-SQL 路径
        if query_type == QueryType.DATABASE:
            logger.info(f"查询路由: database 类型，走 Text-to-SQL 路径")
            return self._handle_database_query(query, session_id, stream=stream)
        
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
        messages = self._build_messages(query, context, session_id, images=images, file_context=file_context)
        
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