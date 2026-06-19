"""
查询路由模块
实现规则 + LLM 分类的混合路由方案

路由类型：
- rag: 需要从知识库检索信息才能准确回答的问题
- chitchat: 闲聊、问候、情感表达、与知识库无关的对话
- general: 通用知识问题，不需要特定知识库，用通用知识就能回答
- database: 数据库结构化查询，需要 SQL 查询才能回答（Text-to-SQL）

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
    DATABASE = "database" # 数据库结构化查询（Text-to-SQL）


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
    r'^(测试|test|123|ping)[\s!！。.？?]*$',
    # 情感/状态表达
    r'^(今天)?(心情|感觉|状态)(今天)?(不错|很好|挺好|糟糕|不好|一般|还行|美滋滋|开心|郁闷)',
    r'^(今天)?(挺|真|太|好)(开心|高兴|无聊|烦|累|困|饿|热|冷|爽)',
    r'^(最近|今天|这几天)(过得|日子|生活)(怎么样|如何|还行|不错)',
    r'^(晚安|早安|午安|周末愉快|节日快乐|新年好)',
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


# 通用常识模式 — 未提及项目/文档/知识库时默认 general
GENERAL_PATTERNS = [
    r'^(如何|怎么|怎样)(学习|学|入门|精通|提高|练习)',
    r'^(什么是|什么叫|何为|何谓)\s*\w+[\s?？]*$',
    r'^(怎么做|如何做|怎样做)\s*\w+[\s?？]*$',
]
# 项目/知识库关键词 — 出现这些词时倾向 rag
KB_KEYWORDS = ['知识库', '文档', '项目', '系统', '架构', '数据库', '论文', '毕设', '毕业设计', '课程']

# 数据库查询关键词 — 出现这些词时倾向 database（Text-to-SQL）
DB_QUERY_KEYWORDS = [
    '查询', '统计', '排名', '排行', '列表', '多少', '几个', '哪些',
    '最多', '最少', '最高', '最低', '平均', '总计', '总数', '数量',
    'top', '前', '后', '大于', '小于', '等于', '超过', '低于',
    '播放量', '弹幕', '时长', '发布', '视频',
]


def _check_database_query(query: str) -> Optional[QueryType]:
    """检查是否为数据库查询类问题（Text-to-SQL）
    
    通过关键词匹配判断用户问题是否需要查询数据库。
    要求：至少命中 2 个数据库查询关键词，或命中特定强模式。
    """
    query_lower = query.strip().lower()
    hit_count = sum(1 for kw in DB_QUERY_KEYWORDS if kw in query_lower)
    
    # 强模式：包含明确的数据库查询意图
    strong_patterns = [
        r'(查询|统计|列出|显示).*(数据|信息|记录|表)',
        r'(播放量|弹幕|时长|视频).*(最多|最少|最高|最低|排名|排行|top)',
        r'(多少|几个).*(视频|数据)',
        r'(前|top)\s*\d+',
        r'(平均|总计|总共|一共)',
    ]
    for pattern in strong_patterns:
        if re.search(pattern, query_lower):
            logger.debug(f"数据库查询强模式命中: {query} -> database")
            return QueryType.DATABASE
    
    # 柔性匹配：至少 2 个关键词命中
    if hit_count >= 2:
        logger.debug(f"数据库查询关键词命中({hit_count}个): {query} -> database")
        return QueryType.DATABASE
    
    return None


def _check_general_or_rag(query: str) -> Optional[QueryType]:
    """启发式：通用常识 vs 知识库检索"""
    query_lower = query.strip().lower()
    has_kb_kw = any(kw in query_lower for kw in KB_KEYWORDS)
    
    if has_kb_kw:
        return None  # 有知识库关键词，交给 LLM 判断
    
    for pattern in GENERAL_PATTERNS:
        if re.match(pattern, query_lower):
            logger.debug(f"通用常识模式命中: {query} -> general")
            return QueryType.GENERAL
    
    return None


# ========== 第二层：LLM 分类（准确，有少量成本） ==========

CLASSIFY_PROMPT = """你是查询分类器。判断用户问题属于哪一类，只回答分类名：

rag — 问题涉及特定文档、项目、产品、内部资料，不检索就无法准确回答
chitchat — 闲聊、问候、情感、日常对话、寒暄
general — 通用常识，不涉及任何特定文档即可回答
database — 需要查询数据库中的结构化数据才能回答，涉及统计、排序、筛选、排名等操作

重要：如果问题没有提到具体的文档、项目、产品、知识库，就不要分类为 rag！
如果问题是关于数据统计、排名、查询具体数据（如播放量、数量、列表等），分类为 database！

示例：
"系统架构是什么样的" → rag
"这个项目的数据库用的什么" → rag
"RAG 是什么" → general
"你好" → chitchat
"今天心情不错" → chitchat
"如何学习编程" → general
"知识库里有哪些文档" → rag
"谢谢" → chitchat
"Python怎么学" → general
"项目用了什么技术栈" → rag
"播放量最多的视频是什么" → database
"统计一下弹幕数量前10的视频" → database
"有哪些视频时长超过10分钟" → database
"视频平均播放量是多少" → database

用户问题：{query}
分类："""


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
            "max_tokens": 15,
            "stream": False
        }
        
        with httpx.Client(timeout=10) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip().lower()
        
        # 解析分类结果（多策略匹配，兼容模型输出格式不一致的情况）
        # 1. 精确匹配
        if content in ("rag", "chitchat", "general", "database"):
            return QueryType(content)
        
        # 2. 去除常见前缀后匹配（如 "类别：rag"、"分类: chitchat"）
        cleaned = re.sub(r'^(类别|分类|答案|结果|类型|回答)[：:]\s*', '', content).strip()
        if cleaned in ("rag", "chitchat", "general", "database"):
            return QueryType(cleaned)
        
        # 3. 从响应中提取第一个匹配的关键词
        if "chitchat" in content or "闲聊" in content or "聊天" in content or "问候" in content:
            return QueryType.CHITCHAT
        if "general" in content or "通用" in content or "常识" in content:
            return QueryType.GENERAL
        if "database" in content or "数据库" in content or "sql" in content or "查询" in content:
            return QueryType.DATABASE
        if "rag" in content or "检索" in content or "知识库" in content:
            return QueryType.RAG
            
    except Exception as e:
        logger.warning(f"LLM 分类失败，默认走 RAG: {e}")
    
    return QueryType.RAG  # 分类失败时默认走 RAG


def route_query(query: str) -> QueryType:
    """混合路由入口：先规则，再数据库检测，再 LLM
    
    路由策略：
    1. 先进行规则匹配（零成本）
    2. 数据库查询检测（关键词匹配，零成本）
    3. 通用常识启发式
    4. LLM 分类
    5. 分类失败默认走 RAG
    
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
    
    # 第二层：数据库查询检测（Text-to-SQL）
    result = _check_database_query(query)
    if result is not None:
        logger.info(f"查询路由 [数据库]: {query[:50]}... -> {result.value}")
        return result
    
    # 第三层：通用常识启发式
    result = _check_general_or_rag(query)
    if result is not None:
        logger.info(f"查询路由 [启发式]: {query[:50]}... -> {result.value}")
        return result
    
    # 第四层：LLM 分类
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
