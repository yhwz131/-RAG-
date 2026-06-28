"""
查询路由模块（LLM-first 架构）

路由策略：
1. 规则匹配（零成本，拦截明确的问候/闲聊）
2. Follow-up 感知（上下文延续）
3. LLM 智能分类（主力，返回置信度，不确定时主动提问）

路由类型：
- rag: 需要从知识库检索信息才能准确回答
- chitchat: 闲聊/问候/情感
- general: 通用知识，不需要特定知识库
- database: 数据库结构化查询（Text-to-SQL）
- clarification: 查询意图不明确，需要向用户确认
"""
import json
import re
from enum import Enum
from typing import Optional, Tuple
import httpx
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


# ========== 第二层：LLM 智能分类（主力路由） ==========

# 提示词由 _build_classify_prompt() 动态生成（根据数据库可用性调整可选类型）


def llm_classify(query: str, history: Optional[list] = None) -> Tuple[QueryType, float, str]:
    """LLM 智能分类（主力路由）

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

        # 构造消息（带历史上下文辅助判断）
        # 使用动态提示词（根据数据库可用性调整可选类型）
        db_available = _get_db_available()
        classify_prompt = _build_classify_prompt(db_available=db_available)
        
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
            "temperature": 0,
            "max_tokens": 1000,
            "stream": False,
            "reasoning_effort": "low",
        }

        resp = _http_client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content", "").strip()
        raw_reasoning = msg.get("reasoning_content", "").strip()

        logger.debug(f"LLM content={content[:200] if content else '(空)'}, reasoning长度={len(raw_reasoning)}")

        # 推理模型可能把输出放在 reasoning_content 中
        if not content:
            if raw_reasoning:
                logger.debug(f"LLM content 为空，从 reasoning 提取")
                content = _extract_from_reasoning(raw_reasoning)

        if not content:
            logger.warning("LLM 分类返回空内容，回退 RAG")
            return QueryType.RAG, 0.5, "LLM 返回空"

        return _parse_classify_result(content, query)

    except Exception as e:
        logger.warning(f"LLM 分类异常，回退 RAG: {e}")
        return QueryType.RAG, 0.5, f"分类异常: {e}"


def _extract_from_reasoning(reasoning: str) -> str:
    """从推理模型的 reasoning_content 中提取分类结果"""
    tail = reasoning[-500:] if len(reasoning) > 500 else reasoning

    # 1. 尝试提取 JSON
    json_match = re.search(r'\{[^}]*"query_type"[^}]*\}', tail)
    if json_match:
        return json_match.group(0)

    # 2. 匹配结论性表述
    conclusion_patterns = [
        r'分类[为是]?\s*(rag|chitchat|general|database|clarification)',
        r'属于\s*(rag|chitchat|general|database|clarification)',
        r'(应该|应|可以)\s*(?:分类为|属于|归类为?)?\s*(rag|chitchat|general|database|clarification)',
        r'→\s*(rag|chitchat|general|database|clarification)',
    ]
    for pat in conclusion_patterns:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            label = m.group(m.lastindex) if m.lastindex else m.group(1)
            return json.dumps({"query_type": label, "confidence": 0.7, "reason": "从推理中提取"})

    # 3. 关键词兜底
    kw_map = [
        ("database", "database"), ("chitchat", "chitchat"),
        ("clarification", "clarification"), ("general", "general"), ("rag", "rag"),
        ("知识库", "rag"), ("闲聊", "chitchat"), ("通用", "general"),
    ]
    for kw, label in kw_map:
        if kw in tail.lower():
            return json.dumps({"query_type": label, "confidence": 0.6, "reason": f"推理关键词={kw}"})

    return ""


def _parse_classify_result(content: str, query: str) -> Tuple[QueryType, float, str]:
    """解析 LLM 分类结果（兼容多种输出格式）"""
    # 1. 尝试 JSON 解析
    try:
        json_str = content
        if "```" in json_str:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)

        result = json.loads(json_str)
        qt_str = result.get("query_type", "").strip().lower()
        confidence = float(result.get("confidence", 0.7))
        reason = result.get("reason", "")

        type_map = {
            "rag": QueryType.RAG, "chitchat": QueryType.CHITCHAT,
            "general": QueryType.GENERAL, "database": QueryType.DATABASE,
            "clarification": QueryType.CLARIFICATION,
        }
        if qt_str in type_map:
            return type_map[qt_str], confidence, reason
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    # 2. 非 JSON 格式：从文本中提取
    content_lower = content.lower().strip()
    cleaned = re.sub(r'^(类别|分类|答案|结果|类型|回答)[：:]\s*', '', content_lower).strip()

    type_keywords = [
        ("clarification", ["clarification", "澄清", "不确定", "模糊"]),
        ("chitchat", ["chitchat", "闲聊", "聊天", "问候"]),
        ("general", ["general", "通用", "常识"]),
        ("database", ["database", "数据库", "sql", "查询"]),
        ("rag", ["rag", "检索", "知识库"]),
    ]
    for label, keywords in type_keywords:
        for kw in keywords:
            if kw in cleaned:
                return QueryType(label), 0.6, f"文本提取: {kw}"

    logger.warning(f"LLM 分类结果无法解析: {content[:80]}")
    return QueryType.RAG, 0.4, "无法解析，回退 RAG"


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


def _build_classify_prompt(db_available: bool = False) -> str:
    """构建 LLM 分类提示词（根据系统能力动态调整可选类型）
    
    Args:
        db_available: 数据库是否可用，为 False 时排除 database 类型
    """
    type_table = """| 类型 | 说明 | 典型例子 |
|------|------|----------|
| rag | 需要检索知识库文档才能回答。涉及具体文档、项目、产品、技术细节 | "LangChain有哪些核心组件？"、"Python大纲考什么？" |
| general | 通用常识问题，不需要特定资料即可回答 | "Python怎么学？"、"什么是机器学习？"、"系统有哪些功能？" |
| chitchat | 闲聊/问候/情感表达，与知识无关 | "你好"、"今天心情不错"、"谢谢" |
| clarification | 查询太模糊/太短/有多种理解方式，无法确定用户意图 | "那个"、"帮我看看"、"怎么样？"（无上下文） |"""

    if db_available:
        db_row = "\n| database | 需要对数据库做统计/排名/筛选/SQL查询 | \"上传了多少文件？\"、\"哪个文档最长？\" |"
        type_table = type_table.replace(
            "| clarification |",
            db_row + "\n| clarification |"
        )
        db_rule = "\n4. 如果需要统计/排名/数量等结构化数据 → database"
    else:
        db_rule = ""

    rules = f"""1. 如果查询涉及知识库中已有文档的具体内容（如 LangChain、Python大纲、教育服务、信息安全、计算机网络等） → rag
2. 如果查询是通用知识、或问"本项目/本系统"的架构/功能（知识库中没有项目自身的文档） → general
3. 如果查询极短（<4字）、指代不明、或有多种合理解读 → clarification{db_rule}
5. 如果是打招呼/闲聊/情感 → chitchat
6. 关键判断：只有当查询能从知识库文档中找到答案时才走 rag，否则走 general"""

    return f"""你是一个查询意图分类器。请判断用户查询的类型，并给出置信度。

## 分类定义

{type_table}

## 判断规则
{rules}

## 输出格式
严格返回 JSON，不要其他文字：
{{{{"query_type": "类型", "confidence": 0.0~1.0, "reason": "一句话理由"}}}}

## 用户查询
{{query}}"""


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

    # 第三层：LLM 智能分类（主力）
    qt, confidence, reason = llm_classify(query, history=history)
    logger.info(f"路由 [LLM]: {query[:50]}... -> {qt.value} (置信度={confidence:.2f}, {reason})")
    return qt, confidence, reason
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
