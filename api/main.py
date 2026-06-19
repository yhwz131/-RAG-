"""
FastAPI 主入口
"""
import os
import sys
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from utils.logger import get_logger
from embeddings.chunker import TextChunker
from embeddings.embedder import EmbeddingClient, MultimodalEmbedder
from rag.retriever import VectorRetriever, MultimodalRetriever
from rag.chain import RAGChain
from rag.memory import ConversationMemory
from api.routes_chat import router as chat_router, set_rag_chain, set_mm_rag_chain
from api.routes_docs import (
    router as docs_router,
    set_retriever, set_chunker,
    set_mm_retriever,
)
from api.routes_health import router as health_router

logger = get_logger("api")

# 创建 FastAPI 应用
app = FastAPI(
    title="知识问答系统 API",
    description="基于 RAG 的知识库问答系统",
    version="2.0.0"
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含路由
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(docs_router)

# 挂载静态文件目录（用于前端访问图片）
_images_dir = os.path.join(settings.upload_dir, "images")
os.makedirs(_images_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=settings.upload_dir), name="static")


@app.on_event("startup")
async def startup_event():
    """服务启动时初始化组件"""
    logger.info("正在初始化服务组件...")
    
    try:
        # 初始化各组件
        chunker = TextChunker()
        memory = ConversationMemory()

        # 纯文本链路（bge-large-zh, 1024d）
        text_retriever = VectorRetriever()
        text_chain = RAGChain(retriever=text_retriever, memory=memory)

        # 多模态链路（Qwen3-VL-Embedding, 4096d）
        mm_retriever = MultimodalRetriever()
        mm_chain = RAGChain(retriever=mm_retriever, memory=memory, is_multimodal=True)

        # 注入到路由模块
        set_rag_chain(text_chain)
        set_mm_rag_chain(mm_chain)
        set_retriever(text_retriever)
        set_mm_retriever(mm_retriever)
        set_chunker(chunker)
        
        logger.info("服务组件初始化完成（纯文本 + 多模态双链路）")
    except Exception as e:
        logger.error(f"服务组件初始化失败: {e}")
        raise


# ========== 前端静态文件服务 ==========
_frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")


@app.get("/")
async def root():
    """根路径 - 返回前端页面或 API 信息"""
    index_html = os.path.join(_frontend_dist, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return {
        "service": "知识问答系统",
        "version": "2.0.0",
        "docs": "/docs"
    }


# 挂载前端静态资源（JS/CSS/图片等）
if os.path.exists(_frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(_frontend_dist, "assets")), name="frontend-assets")

    # SPA 路由兜底：非 API/静态路径全部返回 index.html
    @app.get("/{full_path:path}")
    async def spa_fallback(request: Request, full_path: str):
        """SPA 路由兜底"""
        # 如果是 API 路径或静态文件路径，不处理
        if full_path.startswith(("api/", "static/", "docs", "health", "assets/")):
            return FileResponse(os.path.join(_frontend_dist, "index.html"))
        # 检查是否请求的是前端静态文件
        file_path = os.path.join(_frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        # 其他路径返回 index.html（SPA 路由）
        return FileResponse(os.path.join(_frontend_dist, "index.html"))


def start_server():
    """启动服务"""
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )


if __name__ == "__main__":
    start_server()