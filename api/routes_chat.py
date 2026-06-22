"""
聊天相关 API 路由
"""
import os
import uuid
import base64
import tempfile
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from rag.chain import RAGChain
from utils.file_parser import FileParser
from utils.logger import get_logger

logger = get_logger("routes_chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])

# 全局 RAG 链实例（由 main.py 初始化注入）
rag_chain: Optional[RAGChain] = None
mm_rag_chain: Optional[RAGChain] = None


def set_rag_chain(chain: RAGChain):
    global rag_chain
    rag_chain = chain


def set_mm_rag_chain(chain: RAGChain):
    global mm_rag_chain
    mm_rag_chain = chain


class ChatFile(BaseModel):
    """用户上传的文件"""
    name: str       # 文件名
    content: str    # base64 编码内容


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    stream: bool = False
    mode: str = "text"  # "text" 或 "multimodal"
    images: List[str] = []       # 用户上传的图片列表（base64 编码，不含 data:image 前缀）
    files: List[ChatFile] = []   # 用户上传的文档列表（base64 编码）


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    references: List[Dict]
    query_type: str = "rag"  # rag / chitchat / general


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """对话接口"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG 链未初始化")
    
    session_id = req.session_id or str(uuid.uuid4())

    # 根据模式选择链路
    if req.mode == "multimodal" and mm_rag_chain:
        active_chain = mm_rag_chain
        logger.info(f"使用多模态链路 (mode={req.mode})")
    else:
        active_chain = rag_chain
        logger.info(f"使用纯文本链路 (mode={req.mode}, mm_chain={'有' if mm_rag_chain else '无'})")

    # 处理上传的文档文件 → 提取文本作为上下文
    file_context = None
    if req.files:
        file_texts = []
        for f in req.files:
            try:
                suffix = Path(f.name).suffix.lower()
                if suffix not in FileParser.SUPPORTED_FORMATS:
                    logger.warning(f"不支持的文件格式: {f.name}")
                    continue
                # base64 → 临时文件 → 解析
                raw = base64.b64decode(f.content)
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(raw)
                    tmp_path = tmp.name
                try:
                    text = FileParser.parse(tmp_path)
                    # 限制单个文件长度，防止上下文爆炸
                    if len(text) > 5000:
                        text = text[:5000] + "\n...(内容过长已截断)"
                    file_texts.append(f"[文件: {f.name}]\n{text}")
                    logger.info(f"解析上传文件成功: {f.name}, {len(text)} 字符")
                finally:
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"解析文件 {f.name} 失败: {e}")
        if file_texts:
            file_context = "\n\n".join(file_texts)

    if req.stream:
        # 流式返回 — chat(stream=True) 返回 (generator, docs, query_type)
        result = active_chain.chat(req.query, session_id, stream=True, images=req.images, file_context=file_context)
        stream_gen = result[0] if isinstance(result, tuple) else result

        async def generate():
            for chunk in stream_gen:
                yield chunk

        return StreamingResponse(generate(), media_type="text/plain")
    
    # 非流式
    answer, docs, query_type = active_chain.chat(req.query, session_id, stream=False, images=req.images, file_context=file_context)
    refs = []
    for d in docs:
        ref = {
            "content": d["content"][:200],
            "source": d["source"],
            "score": d["score"],
            "page_number": d.get("page_number", 0),
        }
        # 多模态链路：附带图片信息
        if d.get("has_image") and d.get("image_url"):
            ref["has_image"] = True
            ref["image_url"] = d["image_url"]
        refs.append(ref)
    return ChatResponse(
        answer=answer,
        session_id=session_id,
        references=refs,
        query_type=query_type.value
    )


@router.post("/clear")
async def clear_history(session_id: str):
    """清空指定会话的历史"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG 链未初始化")
    rag_chain.memory.clear(session_id)
    return {"message": f"会话 {session_id} 历史已清空"}


# ========== 会话管理 API ==========

@router.get("/sessions")
async def list_sessions():
    """列出所有历史会话"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG 链未初始化")
    sessions = rag_chain.memory.list_sessions()
    return {"sessions": sessions}


@router.get("/{session_id}/history")
async def get_session_history(session_id: str):
    """获取指定会话的对话历史"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG 链未初始化")
    history = rag_chain.memory.get_full_history(session_id)
    if not history and session_id not in rag_chain.memory.meta:
        raise HTTPException(status_code=404, detail="会话不存在")
    meta = rag_chain.memory.meta.get(session_id, {})
    return {"session_id": session_id, "title": meta.get("title", "新对话"), "messages": history}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG 链未初始化")
    rag_chain.memory.clear(session_id)
    return {"message": f"会话 {session_id} 已删除"}