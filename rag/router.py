"""
查询路由模块（LLM-first 架构 + Function Calling 结构化输出）

路由策略：
1. 规则匹配（零成本，拦截明确的问候/闲聊）
2. Follow-up 感知（上下文延续）
3. LLM 智能分类（Function Calling 强制结构化输出，Few-shot 引导判断标准）

路由类型：
- rag: 需要从知识库检索信息才能准确回答
- chitchat: 闲聊/问候/情感
- general: 通用知识，不需要特定知识库
- database: 数据库结构化查询（Text-to-SQL）
- clarification: 查询意图不明确，需要向用户确认
"""
import re
import time
from enum import Enum
from typing import List, Optional, Tuple
import httpx
from pydantic import BaseModel, Field
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("router")

# 模块级 httpx 客户端（连接池复用，避免每次分类都重新握手）
# 推理模型分类需要较长时间（reasoning tokens），超时设为 60s
_http_client = httpx.Client(
    timeout=60,
    limits=httpx.Limits(
        max_connections=10,
        max_keepalive_connections=5,
        keepalive_expiry=30,
    ),
)


def close_http_client():
    """关闭模块级 httpx 客户端（供 shutdown hook 调用）"""
    global _http_client
    if _http_client and not _http_client.is_closed:
        _http_client.close()
        logger.info("Router httpx 连接池已关闭")


class QueryType(str, Enum):
    """查询类型枚举"""
    RAG = "rag"               # 需要知识库检索
    CHITCHAT = "chitchat"     # 闲聊/问候
    GENERAL = "general"       # 通用知识（不需要知识库）
    DATABASE = "database"     # 数据库结构化查询（Text-to-SQL）
    CLARIFICATION = "clarification"  # 需要用户澄清意图


class RouteResult(BaseModel):
    """路由分类结果 schema（用于 Function Calling）"""
    query_type: str = Field(
        description="查询类型: rag/chitchat/general/database/clarification"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="置信度 0.0-1.0"
    )
    reason: str = Field(
        description="一句话判断理由"
    )


# Function Calling 工具定义（mimo-v2.5 支持）
_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_query",
        "description": "对用户查询进行意图分类，判断应走知识库检索、通用回答、闲聊、数据库查询还是需要澄清",
        "parameters": RouteResult.model_json_schema(),
    }
}

_TYPE_MAP = {
    "rag": QueryType.RAG,
    "chitchat": QueryType.CHITCHAT,
    "general": QueryType.GENERAL,
    "database": QueryType.DATABASE,
    "clarification": QueryType.CLARIFICATION,
}


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
    """第一层：规则匹配（仅拦截明确的问候/闲聊/帮助，零成本）"""
    query_clean = query.strip().lower()
    if len(query_clean) < 2:
        return QueryType.CHITCHAT
    for pattern in GREETING_PATTERNS + SMALL_TALK_PATTERNS:
        if re.match(pattern, query_clean):
            logger.debug(f"规则匹配命中: {query} -> chitchat")
            return QueryType.CHITCHAT

    # 系统元问题 → general（不走 RAG，知识库里没有系统自身的信息）
    _META_PATTERNS = [
        r'(本项目|本系统|这个系统|你的系统|该系统).*(架构|功能|介绍|说明|描述)',
        r'(你是干嘛的|你是做什么的|你是什么|你的功能|你能做什么)',
        r'(系统架构|项目架构|技术架构|系统设计).*(介绍|说明|描述|是什么)',
    ]
    for pattern in _META_PATTERNS:
        if re.search(pattern, query_clean):
            logger.debug(f"规则匹配命中: {query} -> general (系统元问题)")
            return QueryType.GENERAL

    return None


# Follow-up 指代词 — 追问时沿用上一轮路由
FOLLOWUP_PATTERNS = [
    r'^(那|这个|那个|它|它们|上面|前面|刚才|之前)(提到|说|讲)?(的)?',
    r'^(有没有|还有|有没有其他|还有没有|别的|另外|除此之外|除此之外呢)',
    r'^(详细|具体|深入|展开|补充|解释|说明)(说|讲|描述)?(一下|说说)?',
    r'^(为什么|为啥|原因是|怎么理解|什么意思|何为)',
    r'^(能(不能|否)|可以|可不可以)(给我|帮|再)(详细|具体)?(说|讲|解释)',
    r'^(呢|那么|然后|所以|因此|接着|接下来)',
]


def _is_followup(query: str) -> bool:
    """判断是否是 follow-up 追问"""
    query_clean = query.strip().lower()
    for pattern in FOLLOWUP_PATTERNS:
        if re.match(pattern, query_clean):
            return True
    if len(query_clean) <= 6 and re.search(r'[？?]|呢$|吗$|么$', query_clean):
        return True
    return False


# ========== 知识库文档清单（方向3：动态匹配） ==========

_retriever = None  # 由 startup 注入


def set_retriever(r):
    """注入 retriever 引用（startup 时调用）"""
    global _retriever
    _retriever = r


# TTL 缓存：避免每次路由都查 Milvus
_doc_titles_cache: List[str] = []
_doc_titles_ts: float = 0
_DOC_TITLES_TTL = 60  # 60 秒


def get_kb_doc_titles() -> List[str]:
    """获取知识库文档清单（带 TTL 缓存）"""
    global _doc_titles_cache, _doc_titles_ts
    now = time.time()
    if _doc_titles_cache and (now - _doc_titles_ts) < _DOC_TITLES_TTL:
        return _doc_titles_cache
    if _retriever is None:
        return []
    try:
        docs = _retriever.list_documents()
        _doc_titles_cache = [d["filename"] for d in docs]
        _doc_titles_ts = now
    except Exception as e:
        logger.debug(f"获取文档清单失败: {e}")
    return _doc_titles_cache


def invalidate_doc_titles_cache():
    """文档变更时主动失效缓存（入库/删除时调用）"""
    global _doc_titles_cache, _doc_titles_ts
    _doc_titles_cache = []
    _doc_titles_ts = 0
    logger.debug("文档清单缓存已失效")


# ========== 第二层：LLM 智能分类（Function Calling 结构化输出） ==========


def llm_classify(query: str, history: Optional[list] = None) -> Tuple[QueryType, float, str]:
    """LLM 智能分类（Function Calling 结构化输出）

    使用 tools 参数强制 LLM 返回合法 JSON，消除兜底解析逻辑。

    Args:
        query: 用户查询文本
        history: 会话历史（可选，辅助判断）

    Returns:
        (QueryType, confidence, reason) 三元组
    """
    try:
        url = settings.llm_api_url
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"

        # 动态构建提示词（含文档清单 + Few-shot）
        db_available = _get_db_available()
        kb_titles = get_kb_doc_titles()
        classify_prompt = _build_classify_prompt(
            db_available=db_available,
            kb_titles=kb_titles,
        )

        # 构造消息（带历史上下文辅助判断）
        messages = []
        if history:
            for msg in history[-8:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content[:200]})
        messages.append({"role": "user", "content": classify_prompt.format(query=query)})

        payload = {
            "model": settings.llm_model_name,
            "messages": messages,
            "tools": [_CLASSIFY_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "classify_query"}},
            "temperature": 0,
            "max_tokens": 500,
            "stream": False,
        }

        # 重试（FC 有时返回 tool_calls=None，mimo 模型非确定性）
        for attempt in range(5):
            resp = _http_client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]

            # Function Calling 返回 —— 解析 tool_calls
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                args_str = tool_calls[0]["function"]["arguments"]
                result = RouteResult.model_validate_json(args_str)
                qt = _TYPE_MAP.get(result.query_type, QueryType.RAG)
                logger.debug(
                    f"FC 分类: {qt.value} ({result.confidence:.2f}) | {result.reason}"
                )
                return qt, result.confidence, result.reason
            if attempt == 0:
                logger.debug("FC 分类首次无结果，重试中...")

        # 兜底：FC 未返回 tool_calls（极少），从 content 解析
        content = msg.get("content", "").strip()
        if content:
            try:
                result = RouteResult.model_validate_json(content)
                qt = _TYPE_MAP.get(result.query_type, QueryType.RAG)
                return qt, result.confidence, result.reason
            except Exception:
                pass

        logger.warning("LLM 分类无有效结果，回退 RAG")
        return QueryType.RAG, 0.5, "FC 无有效结果"

    except Exception as e:
        logger.warning(f"LLM 分类异常，回退 RAG: {e}")
        return QueryType.RAG, 0.5, f"分类异常: {e}"


# ========== 多查询分解（Multi-Query Retrieval） ==========


class DecomposeResult(BaseModel):
    """查询分解结果 schema"""
    sub_queries: List[str] = Field(
        min_length=2, max_length=4,
        description="拆分后的子查询列表，每个子查询聚焦一个独立话题"
    )


_DECOMPOSE_TOOL = {
    "type": "function",
    "function": {
        "name": "decompose_query",
        "description": "将涉及多个话题的复合查询拆分为聚焦的子查询",
        "parameters": DecomposeResult.model_json_schema(),
    }
}


def is_cross_doc_query(query: str) -> bool:
    """启发式判断是否为跨文档查询（轻量，无 LLM 调用）"""
    q = query.strip()
    # 含连接词且长度足够
    cross_patterns = ["和", "与", "及", "对比", "重叠", "区别", "异同", "比较"]
    has_conj = any(p in q for p in cross_patterns)
    has_multi_topic = len(q) > 15 and has_conj
    if has_multi_topic:
        return True

    # 多个问号 → 多个子问题，倾向跨文档
    # 如："小优教育助手的设计理念能否应用到 LangChain 框架中？需要用到哪些组件？"
    q_mark_count = q.count("？") + q.count("?")
    if q_mark_count >= 2 and len(q) > 20:
        return True

    # "能否/是否可以" + "框架/应用到" → 跨领域综合
    synthesis_patterns = ["能否应用", "是否可以应用", "应用到", "结合"]
    has_synth = any(p in q for p in synthesis_patterns)
    if has_synth and len(q) > 20:
        return True

    return False


def _rule_based_decompose(query: str) -> List[str] | None:
    """基于规则的查询拆解（FC 失败时的兜底方案）

    识别常见跨文档模式，拆分为 2 个子查询。

    Returns:
        子查询列表，无法拆解时返回 None
    """
    import re
    q = query.strip()

    # 模式1: 连接词拆分："A和B"、"A与B"、"A及B"
    for conj in ["和", "与", "及"]:
        if conj in q:
            parts = q.split(conj, 1)
            left, right = parts[0].strip(), parts[1].strip()
            # 至少每边 2 个字
            if len(left) >= 2 and len(right) >= 2:
                # 去掉末尾的问号和"分别"等修饰
                left = re.sub(r'[？?]+$', '', left).strip()
                right = re.sub(r'[？?]+$', '', right).strip()
                # 去掉末尾的"分别考什么"等
                right = re.sub(r'分别.*$', '', right).strip()
                if left and right:
                    return [left, right]

    # 模式2: "能否/是否可以" + "应用到/结合" → 拆为 "A" + "B"
    synth_match = re.search(r'(.+?)(能否|是否可以|是否能)(应用到?|结合)\s*(.+?)[？?]?', q)
    if synth_match:
        topic_a = synth_match.group(1).strip()
        topic_b = synth_match.group(4).strip()
        if len(topic_a) >= 2 and len(topic_b) >= 2:
            return [topic_a, f"{topic_b}核心组件与架构"]

    # 模式3: 多个问号 → 按问号拆分
    q_marks = [m.start() for m in re.finditer(r'[？?]', q)]
    if len(q_marks) >= 2:
        parts = re.split(r'[？?]', q)
        subs = [p.strip() for p in parts if len(p.strip()) >= 4]
        if len(subs) >= 2:
            return subs[:3]

    # 模式4: 对比/比较/区别
    for keyword in ["对比", "比较", "区别", "异同", "重叠"]:
        if keyword in q:
            parts = q.split(keyword, 1)
            left = re.sub(r'[？?]+$', '', parts[0].strip())
            right = re.sub(r'[？?]+$', '', parts[1].strip())
            if len(left) >= 2 and len(right) >= 2:
                return [left, right]

    return None


def query_decompose(query: str, max_sub: int = 3) -> List[str]:
    """将复合查询拆分为多个聚焦子查询（FC 结构化输出）

    Args:
        query: 原始查询
        max_sub: 最多子查询数

    Returns:
        子查询列表，至少 2 个。失败时返回 [query]（降级为单查询）
    """
    try:
        url = settings.llm_api_url
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"

        prompt = f"""将以下复合查询拆分为 2-{max_sub} 个聚焦子查询，使每个子查询只涉及一个独立话题。
拆分后应覆盖原始查询的所有方面。

示例:
输入: "信息安全和网络技术考试中哪些知识点重叠？"
→ ["信息安全考试大纲知识点", "网络技术考试大纲知识点"]

输入: "小优教育助手的设计理念能否应用到 LangChain 中？"
→ ["小优教育助手的设计理念", "LangChain 框架核心组件"]

输入: "Python二级和三级网络技术分别考什么？"
→ ["Python二级考试大纲", "三级网络技术考试大纲"]

请拆分: "{query}" """

        payload = {
            "model": settings.llm_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [_DECOMPOSE_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "decompose_query"}},
            "temperature": 0,
            "max_tokens": 300,
            "stream": False,
        }

        # 重试（FC 有时返回 tool_calls=None，mimo 模型非确定性）
        for attempt in range(5):
            resp = _http_client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]

            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                args_str = tool_calls[0]["function"]["arguments"]
                result = DecomposeResult.model_validate_json(args_str)
                subs = result.sub_queries[:max_sub]
                logger.info(f"查询分解: [{query[:30]}...] → {subs}")
                return subs
            if attempt == 0:
                logger.debug("查询分解首次 FC 无结果，重试中...")

        # FC 兜底：尝试规则拆解
        subs = _rule_based_decompose(query)
        if subs:
            logger.info(f"查询分解(规则兜底): [{query[:30]}...] → {subs}")
            return subs
        logger.warning("查询分解 FC 无结果，降级为单查询")
        return [query]

    except Exception as e:
        logger.warning(f"查询分解异常，降级为单查询: {e}")
        return [query]


def _build_classify_prompt(
    db_available: bool = False,
    kb_titles: Optional[List[str]] = None,
) -> str:
    """构建 LLM 分类提示词（方向2: Few-shot + 方向3: 动态文档清单）

    Args:
        db_available: 数据库是否可用
        kb_titles: 知识库文档文件名列表（动态注入）
    """
    type_table = """| 类型 | 说明 |
|------|------|
| rag | 查询涉及知识库中已有文档的具体内容，需要检索才能准确回答 |
| general | 通用知识/概念性问题，不需要特定资料即可回答 |
| chitchat | 闲聊/问候/情感表达，与知识无关 |
| clarification | 查询太模糊/太短/有多种理解方式，无法确定用户意图 |"""

    db_row = ""
    db_hint = ""
    if db_available:
        db_row = "\n| database | 需要对数据库做统计/排名/筛选/SQL查询 |"
        db_hint = "\n- 如果需要统计/排名/数量等结构化数据 → database"
    type_table = type_table.replace("\n| clarification |", db_row + "\n| clarification |")

    # 知识库文档清单（方向3）
    kb_section = ""
    if kb_titles:
        kb_section = (
            f"\n\n### 知识库当前包含的文档\n"
            + "\n".join(f"- {t}" for t in kb_titles)
            + "\n\n判断 rag vs general 的关键：查询是否涉及上述文档的具体内容。"
        )

    return f"""你是一个查询意图分类器。请判断用户查询的类型，返回分类结果。

## 分类定义
{type_table}

## 判断规则
1. 查询涉及上述知识库文档的具体内容 → rag
2. 通用技术概念（如"什么是Python"）、问系统本身的功能 → general
3. 查询极短（<4字）、指代不明、多种理解方式 → clarification
4. 打招呼/闲聊/情感 → chitchat{db_hint}

## 示例
query: "LangChain 1.0 的核心架构风格是什么"
→ rag (0.9) — 查询知识库中 LangChain 文档的具体内容

query: "什么是 Python"
→ general (0.85) — 通用技术概念，非知识库特有

query: "小优教育助手的核心目标是什么"
→ rag (0.9) — 查询知识库中教育服务文档的具体内容

query: "三级信息安全考试大纲考什么"
→ rag (0.95) — 查询知识库中考试大纲文档

query: "你好"
→ chitchat (0.95) — 问候

query: "那个"
→ clarification (0.7) — 查询过短，指代不明

query: "帮我看看"
→ clarification (0.6) — 缺乏具体指向

query: "Python怎么学"
→ general (0.8) — 通用学习建议，知识库中无此内容
{kb_section}

## 用户查询
{{query}}"""


# ========== 数据库连接检查 ==========

def is_database_available() -> bool:
    """检查数据库是否可用（db_type 已配置且能建立连接）
    
    Returns:
        True 表示 database 路由可启用，False 表示应跳过
    """
    if not settings.db_type:
        return False
    try:
        from api.pipeline.engines.database import create_database_source
        source = create_database_source()
        if source is None:
            return False
        ok = source.connect()
        if ok:
            source.close()
        return ok
    except Exception as e:
        logger.debug(f"数据库连接检查失败: {e}")
        return False


# 模块级缓存数据库可用性（避免每次路由都尝试连接）
_db_available_cache: Optional[bool] = None
_db_cache_ts: float = 0
_DB_CACHE_TTL = 60  # 缓存有效期（秒）


def _get_db_available() -> bool:
    """获取数据库可用性（带缓存，60 秒刷新一次）"""
    global _db_available_cache, _db_cache_ts
    import time
    now = time.time()
    if _db_available_cache is None or (now - _db_cache_ts) > _DB_CACHE_TTL:
        _db_available_cache = is_database_available()
        _db_cache_ts = now
        logger.debug(f"数据库可用性刷新: {_db_available_cache}")
    return _db_available_cache


def invalidate_db_cache():
    """清除数据库可用性缓存（配置变更时调用）"""
    global _db_available_cache, _db_cache_ts
    _db_available_cache = None
    _db_cache_ts = 0
    logger.info("数据库可用性缓存已清除")


# ========== 路由入口 ==========

CONFIDENCE_THRESHOLD = 0.7  # 低于此值时建议向用户确认


def route_query(query: str, history: Optional[list] = None) -> Tuple[QueryType, float, str]:
    """路由入口：规则匹配 → Follow-up 感知 → LLM 智能分类

    Args:
        query: 用户查询文本
        history: 会话历史消息列表（可选）

    Returns:
        (QueryType, confidence, reason) 三元组
    """
    # 第一层：规则匹配（零成本）
    result = rule_based_route(query)
    if result is not None:
        logger.info(f"路由 [规则]: {query[:50]}... -> {result.value}")
        return result, 1.0, "规则匹配"

    # 第二层：Follow-up 感知
    if _is_followup(query) and history:
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                logger.info(f"路由 [follow-up]: {query[:50]}... -> rag (沿用上下文)")
                return QueryType.RAG, 0.85, "追问沿用上下文"

    # 第三层：LLM 智能分类（Function Calling）
    qt, confidence, reason = llm_classify(query, history=history)
    logger.info(f"路由 [LLM]: {query[:50]}... -> {qt.value} (置信度={confidence:.2f}, {reason})")
    return qt, confidence, reason


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
