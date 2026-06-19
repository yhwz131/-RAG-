"""
查询路由模块
实现规则 + LLM 分类的混合路由方案

路由类型：
- rag: 需要从知识库检索信息才能准确回答的问题
- chitchat: 闲聊、问候、情感表达、与知识库无关的对话
- general: 通用知识问题，不需要特定知识库，用通用知识就能回答

路由策略：
1. 第一层：规则匹配（零成本，极速） - 命中则直接返回
2. 第二层：LLM 分类（准确，有少量成本） - 未命中规则时使用
3. 兜底：分类失败默认走 RAG，确保不丢失知识类问题
"""
import re
from enum import Enum
from typing import Optional
import httpx
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("router")


class QueryType(str, Enum):
    """查询类型枚举"""
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
    """第一层：规则匹配，返回 None 表示未命中
    
    Args:
        query: 用户查询文本
        
    Returns:
        QueryType.CHITCHAT 如果命中规则，否则 None
    """
    query_clean = query.strip().lower()
    
    # 空查询或过短查询
    if len(query_clean) < 2:
        return QueryType.CHITCHAT
    
    # 匹配问候和闲聊模式
    for pattern in GREETING_PATTERNS + SMALL_TALK_PATTERNS:
        if re.match(pattern, query_clean):
            logger.debug(f"规则匹配命中: {query} -> chitchat")
            return QueryType.CHITCHAT
    
    return None  # 未命中，进入下一层


# ========== 第二层：LLM 分类（准确，有少量成本） ==========

CLASSIFY_PROMPT = """你是一个查询分类器。请判断用户的问题属于以下哪一类：

1. rag — 需要从特定知识库中检索信息才能准确回答的问题。例如：关于特定产品、文档、项目、技术细节、课程内容的问题。
2. chitchat — 闲聊、问候、情感表达、与知识库无关的日常对话。
3. general — 通用知识问题，不需要特定知识库，用通用知识就能回答。例如：什么是Python、太阳系有几颗行星、如何学习编程。

只回答一个类别名（rag / chitchat / general），不要解释。

用户问题：{query}
类别："""


def llm_classify(query: str) -> QueryType:
    """第二层：LLM 分类
    
    使用轻量级 LLM 调用进行分类，失败时默认走 RAG。
    
    Args:
        query: 用户查询文本
        
    Returns:
        QueryType: 分类结果
    """
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
        
        # 解析分类结果
        if content in ("rag", "chitchat", "general"):
            logger.debug(f"LLM 分类: {query} -> {content}")
            return QueryType(content)
        
        # 尝试从响应中提取关键词
        if "chitchat" in content or "闲聊" in content:
            return QueryType.CHITCHAT
        elif "general" in content or "通用" in content:
            return QueryType.GENERAL
        elif "rag" in content or "知识" in content:
            return QueryType.RAG
            
    except Exception as e:
        logger.warning(f"LLM 分类失败，默认走 RAG: {e}")
    
    return QueryType.RAG  # 分类失败时默认走 RAG


def route_query(query: str) -> QueryType:
    """混合路由入口：先规则，再 LLM
    
    路由策略：
    1. 先进行规则匹配（零成本）
    2. 规则未命中时使用 LLM 分类
    3. LLM 分类失败时默认走 RAG
    
    Args:
        query: 用户查询文本
        
    Returns:
        QueryType: 路由结果
    """
    # 第一层：规则匹配
    result = rule_based_route(query)
    if result is not None:
        logger.info(f"查询路由 [规则]: {query[:50]}... -> {result.value}")
        return result
    
    # 第二层：LLM 分类
    result = llm_classify(query)
    logger.info(f"查询路由 [LLM]: {query[:50]}... -> {result.value}")
    return result


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
    # 合并连续空格
    text = re.sub(r'\s+', ' ', text)
    # 全角英数字 → 半角（纯 ASCII 映射，长度精确对齐）
    _full = (
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
        'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        '０１２３４５６７８９'
    )
    _half = (
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        'abcdefghijklmnopqrstuvwxyz'
        '0123456789'
    )
    text = text.translate(str.maketrans(_full, _half))
    # 全角标点 → 半角标点
    _pun_full = '！？，；：、'
    _pun_half = '!?,;:,'
    text = text.translate(str.maketrans(_pun_full, _pun_half))
    # 连续重复标点压缩为单个
    text = re.sub(r'([!！?？.。,，;；:：])\1+', r'\1', text)
    if text != query.strip():
        logger.debug(f"查询预处理: [{query.strip()}] -> [{text}]")
    return text
