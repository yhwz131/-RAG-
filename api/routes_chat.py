"""
聊天相关 API 路由
"""
import os
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from rag.chain import RAGChain

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


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    stream: bool = False
    mode: str = "text"  # "text" 或 "multimodal"


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
    else:
        active_chain = rag_chain

    if req.stream:
        # 流式返回
        async def generate():
            for chunk in active_chain.chat(req.query, session_id, stream=True):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    # 非流式
    answer, docs, query_type = active_chain.chat(req.query, session_id, stream=False)
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