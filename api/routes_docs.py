"""
文档管理 API 路由
"""
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from config.settings import settings
from utils.logger import get_logger
from utils.file_parser import FileParser
from embeddings.chunker import TextChunker
from embeddings.embedder import EmbeddingClient
from rag.retriever import VectorRetriever, MultimodalRetriever

logger = get_logger("routes_docs")

router = APIRouter(prefix="/api/docs", tags=["documents"])

# 全局实例（由 main.py 初始化注入）
retriever: Optional[VectorRetriever] = None
mm_retriever: Optional[MultimodalRetriever] = None
chunker: Optional[TextChunker] = None


def set_retriever(r: VectorRetriever):
    global retriever
    retriever = r


def set_mm_retriever(r: MultimodalRetriever):
    global mm_retriever
    mm_retriever = r


def set_chunker(c: TextChunker):
    global chunker
    chunker = c


class UploadResponse(BaseModel):
    filename: str
    chunks: int
    message: str


class BatchUploadResponse(BaseModel):
    total_files: int
    success_count: int
    fail_count: int
    results: List[UploadResponse]
    errors: List[str]


class DocStatsResponse(BaseModel):
    total_docs: int
    collection_name: str


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档并入库"""
    if not retriever or not chunker:
        raise HTTPException(status_code=503, detail="服务未初始化")

    # 验证文件类型
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {settings.allowed_extensions}"
        )

    # 保存文件
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())[:8]
    save_name = f"{file_id}_{file.filename}"
    save_path = os.path.join(settings.upload_dir, save_name)

    try:
        content = await file.read()
        
        # P2-9: 文件大小校验
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > settings.max_file_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"文件大小 ({file_size_mb:.1f}MB) 超过限制 ({settings.max_file_size_mb}MB)"
            )
        
        with open(save_path, "wb") as f:
            f.write(content)

        # 解析文件
        parser = FileParser()
        pages = parser.parse_with_pages(save_path)

        if not pages or not any(p["text"].strip() for p in pages):
            # P2-10: 内容为空时清理文件
            if os.path.exists(save_path):
                os.remove(save_path)
            raise HTTPException(status_code=400, detail="文件内容为空")

        # 切片（保留页码信息）
        chunks_list = chunker.chunk_with_pages(pages, source=file.filename or "unknown")

        # 入库纯文本 Collection
        count = retriever.insert_documents(chunks_list)

        # 双写：多模态 Collection（文本切片 + 文档图片）
        mm_count = 0
        if mm_retriever:
            try:
                mm_count = mm_retriever.insert_documents(chunks_list)
                # 提取文档中的图片入库
                img_dir = os.path.join(settings.upload_dir, "images", file_id)
                images = []
                if ext == ".pdf":
                    images = FileParser.extract_images_from_pdf(save_path, img_dir)
                elif ext in (".docx", ".doc"):
                    images = FileParser.extract_images_from_docx(save_path, img_dir)
                elif ext in (".pptx", ".ppt"):
                    images = FileParser.extract_images_from_pptx(save_path, img_dir)
                if images:
                    mm_retriever.insert_images(images, source=file.filename or "unknown")
            except Exception as e:
                logger.warning(f"多模态入库失败（不影响纯文本链路）: {e}")

        logger.info(f"文档上传成功: {file.filename}, {count} 个切片")
        return UploadResponse(
            filename=file.filename or "unknown",
            chunks=count,
            message=f"文档已成功上传并切分为 {count} 个片段"
        )

    except HTTPException:
        # P2-10: 业务异常时清理残留文件
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
                logger.info(f"已清理上传失败的文件: {save_path}")
            except Exception as cleanup_err:
                logger.warning(f"清理残留文件失败: {cleanup_err}")
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_documents_batch(files: List[UploadFile] = File(...)):
    """批量上传文档并入库"""
    # P2-12: 批量上传文件数量限制
    if len(files) > 20:
        raise HTTPException(status_code=400, detail=f"单次最多上传 20 个文件，当前 {len(files)} 个")
    if not retriever or not chunker:
        raise HTTPException(status_code=503, detail="服务未初始化")

    results: List[UploadResponse] = []
    errors: List[str] = []

    for file in files:
        save_path = None
        try:
            # 验证文件类型
            ext = os.path.splitext(file.filename or "")[1].lower()
            if ext not in settings.allowed_extensions:
                errors.append(f"{file.filename}: 不支持的文件类型 {ext}")
                continue

            # 保存文件
            os.makedirs(settings.upload_dir, exist_ok=True)
            file_id = str(uuid.uuid4())[:8]
            save_name = f"{file_id}_{file.filename}"
            save_path = os.path.join(settings.upload_dir, save_name)

            content = await file.read()
            
            # P2-9: 文件大小校验
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > settings.max_file_size_mb:
                errors.append(f"{file.filename}: 文件大小 ({file_size_mb:.1f}MB) 超过限制 ({settings.max_file_size_mb}MB)")
                continue
            
            with open(save_path, "wb") as f:
                f.write(content)

            # 解析文件
            parser = FileParser()
            pages = parser.parse_with_pages(save_path)

            if not pages or not any(p["text"].strip() for p in pages):
                errors.append(f"{file.filename}: 文件内容为空")
                # P2-10: 清理空文件
                if os.path.exists(save_path):
                    os.remove(save_path)
                continue

            # 切片入库
            chunks_list = chunker.chunk_with_pages(pages, source=file.filename or "unknown")
            count = retriever.insert_documents(chunks_list)

            # 双写多模态
            if mm_retriever:
                try:
                    mm_retriever.insert_documents(chunks_list)
                    # 提取文档中的图片入库
                    img_dir = os.path.join(settings.upload_dir, "images", file_id)
                    images = []
                    if ext == ".pdf":
                        images = FileParser.extract_images_from_pdf(save_path, img_dir)
                    elif ext in (".docx", ".doc"):
                        images = FileParser.extract_images_from_docx(save_path, img_dir)
                    elif ext in (".pptx", ".ppt"):
                        images = FileParser.extract_images_from_pptx(save_path, img_dir)
                    if images:
                        mm_retriever.insert_images(images, source=file.filename or "unknown")
                except Exception as e:
                    logger.warning(f"多模态入库失败（不影响纯文本链路）: {e}")

            logger.info(f"文档上传成功: {file.filename}, {count} 个切片")
            results.append(UploadResponse(
                filename=file.filename or "unknown",
                chunks=count,
                message=f"上传成功，{count} 个片段"
            ))

        except Exception as e:
            logger.error(f"文档上传失败 {file.filename}: {e}")
            errors.append(f"{file.filename}: {str(e)}")
            # P2-10: 失败时清理残留文件
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已清理上传失败的文件: {save_path}")
                except Exception as cleanup_err:
                    logger.warning(f"清理残留文件失败: {cleanup_err}")

    return BatchUploadResponse(
        total_files=len(files),
        success_count=len(results),
        fail_count=len(errors),
        results=results,
        errors=errors,
    )


@router.get("/stats", response_model=DocStatsResponse)
async def get_stats():
    """获取文档库统计信息"""
    if not retriever:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        stats = retriever._client.get_collection_stats(settings.collection_name)
        total = int(stats.get("row_count", 0))
    except Exception:
        total = 0

    return DocStatsResponse(
        total_docs=total,
        collection_name=settings.collection_name
    )


@router.get("/list")
async def list_documents():
    """列出知识库中的所有文档（按文件名分组）"""
    if not retriever:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        docs = retriever.list_documents()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        logger.error(f"列出文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{filename}")
async def delete_document(filename: str):
    """按文件名删除文档（同时删除向量数据和原始文件）"""
    if not retriever:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        # 1. 删除向量数据
        count = retriever.delete_by_filename(filename)

        # 1b. 删除多模态向量数据
        mm_count = 0
        if mm_retriever:
            try:
                mm_count = mm_retriever.delete_by_filename(filename)
            except Exception as e:
                logger.warning(f"多模态删除失败: {e}")

        # 2. 删除原始文件（精确匹配 uuid 前缀后的原始文件名）
        deleted_files = []
        if os.path.isdir(settings.upload_dir):
            for f in os.listdir(settings.upload_dir):
                # 文件名格式: {uuid}_{原始文件名}，去掉 uuid 前缀后精确比较
                stripped = f
                if "_" in f:
                    # uuid 格式: 8-4-4-4-12，前缀是 uuid + "_"
                    parts = f.split("_", 1)
                    if len(parts) == 2 and len(parts[0]) >= 8:
                        stripped = parts[1]
                if stripped == filename or f == filename:
                    fpath = os.path.join(settings.upload_dir, f)
                    os.remove(fpath)
                    deleted_files.append(f)

        logger.info(f"已删除文档 [{filename}]: {count} 个向量切片, {len(deleted_files)} 个原始文件")
        return {
            "message": f"已删除文档 [{filename}]",
            "deleted_chunks": count,
            "deleted_files": deleted_files,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_documents():
    """清空所有文档"""
    if not retriever:
        raise HTTPException(status_code=503, detail="服务未初始化")

    try:
        retriever._client.drop_collection(settings.collection_name)
        # 重新初始化
        retriever._ensure_collection()

        # 同时清空多模态 Collection
        if mm_retriever:
            try:
                mm_retriever._client.drop_collection(settings.multimodal_collection_name)
                mm_retriever._ensure_collection()
            except Exception as e:
                logger.warning(f"清空多模态 Collection 失败: {e}")
        logger.info("文档库已清空")
        return {"message": "文档库已清空"}
    except Exception as e:
        logger.error(f"清空文档库失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh-descriptions")
async def refresh_image_descriptions(source: Optional[str] = None):
    """刷新图片描述：为没有有效描述的图片重新生成描述

    Args:
        source: 指定来源文件名（可选），为空则刷新所有图片
    """
    if not mm_retriever:
        raise HTTPException(status_code=503, detail="多模态检索器未初始化")

    try:
        refreshed = mm_retriever.refresh_image_descriptions(source=source)
        return {
            "message": f"图片描述刷新完成",
            "refreshed": refreshed,
        }
    except Exception as e:
        logger.error(f"刷新图片描述失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))