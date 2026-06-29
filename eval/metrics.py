"""
评估指标模块
基于关键词匹配 + 来源召回 + 拒答检测的轻量评估方案
（不依赖 LLM 调用，纯规则匹配，适合快速迭代）
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    """单条测试用例的评估结果"""
    id: int
    query: str
    test_type: str
    difficulty: str
    
    # 检索指标
    source_recall: float = 0.0      # 期望来源召回率 (0-1)
    sources_found: List[str] = field(default_factory=list)
    sources_missing: List[str] = field(default_factory=list)
    
    # 答案指标
    keyword_recall: float = 0.0     # 关键词命中率 (0-1)
    keywords_found: List[str] = field(default_factory=list)
    keywords_missing: List[str] = field(default_factory=list)
    
    # 拒答指标 (仅 reject 类型)
    reject_correct: Optional[bool] = None  # 是否正确拒答
    
    # 原始数据
    answer: str = ""
    retrieved_docs: List[Dict] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """综合判定是否通过"""
        if self.test_type == "reject":
            return self.reject_correct is True
        # 非拒答类型：关键词召回率 >= 50% 且 来源召回率 >= 50%
        return self.keyword_recall >= 0.5 and self.source_recall >= 0.5


def keyword_match(answer: str, expected_keywords: List[str]) -> tuple:
    """
    计算关键词命中率
    
    使用宽松匹配：忽略大小写，支持部分匹配（关键词是答案中某词的子串）
    
    Returns:
        (recall, found_list, missing_list)
    """
    if not expected_keywords:
        return 1.0, [], []
    
    answer_lower = answer.lower()
    found = []
    missing = []
    
    for kw in expected_keywords:
        kw_lower = kw.lower()
        if kw_lower in answer_lower:
            found.append(kw)
        else:
            missing.append(kw)
    
    recall = len(found) / len(expected_keywords)
    return recall, found, missing


def source_match(retrieved_docs: List[Dict], expected_sources: List[str]) -> tuple:
    """
    计算来源召回率
    
    检查期望的文档来源是否出现在检索结果中
    
    Returns:
        (recall, found_list, missing_list)
    """
    if not expected_sources:
        return 1.0, [], []
    
    retrieved_sources = set()
    for doc in retrieved_docs:
        source = doc.get("source", "")
        if source:
            retrieved_sources.add(source)
    
    found = []
    missing = []
    
    for expected in expected_sources:
        # 宽松匹配：期望来源是检索来源的子串，或反过来
        matched = False
        for retrieved in retrieved_sources:
            if expected in retrieved or retrieved in expected:
                matched = True
                break
        if matched:
            found.append(expected)
        else:
            missing.append(expected)
    
    recall = len(found) / len(expected_sources)
    return recall, found, missing


def reject_detect(answer: str, retrieved_docs: List[Dict], 
                  similarity_threshold: float = 0.35) -> bool:
    """
    检测系统是否正确拒答
    
    判定拒答的条件（满足任一即可）：
    1. 检索结果为空或最高分数很低
    2. 答案中包含明确的"不知道""无法回答"等表述
    
    Args:
        answer: LLM 生成的回答
        retrieved_docs: 检索到的文档列表
        similarity_threshold: 最高相似度低于此值视为"无相关结果"
    
    Returns:
        True 表示正确拒答
    """
    # 条件 1：检索结果为空或质量极低
    if not retrieved_docs:
        return True
    
    max_score = max(d.get("rerank_score") or d.get("score", 0) for d in retrieved_docs)
    if max_score < similarity_threshold:
        return True
    
    # 条件 2：答案中包含拒答表述
    reject_patterns = [
        r"没有(找到|检索到|查到|相关|提供).{0,10}(信息|内容|资料|文档|答案)",
        r"(无法|不能|难以)(回答|确定|判断|回答这个问题)",
        r"(知识库|资料库|文档).{0,6}(中没有|里没有|未包含|不包含|没有找到)",
        r"(不(太|确定|清楚)|不太了解|无法确认)",
        r"(抱歉|sorry).{0,15}(无法|不能|没有|找不到)",
        r"根据.{0,10}(已有的|提供的|检索到的).{0,10}(信息|内容|资料).{0,10}(无法|不能|没有)",
    ]
    
    answer_lower = answer.lower()
    for pattern in reject_patterns:
        if re.search(pattern, answer_lower):
            return True
    
    return False


def evaluate_case(test_case: Dict, answer: str, retrieved_docs: List[Dict]) -> EvalResult:
    """
    评估单条测试用例
    
    Args:
        test_case: 测试用例 (来自 eval_testset.json)
        answer: LLM 生成的回答
        retrieved_docs: 检索到的文档列表
    
    Returns:
        EvalResult 评估结果
    """
    test_type = test_case.get("type", "factual")
    
    result = EvalResult(
        id=test_case["id"],
        query=test_case["query"],
        test_type=test_type,
        difficulty=test_case.get("difficulty", "medium"),
        answer=answer,
        retrieved_docs=retrieved_docs,
    )
    
    if test_type == "reject":
        # 拒答类型：只检测是否正确拒答
        result.reject_correct = reject_detect(answer, retrieved_docs)
        result.keyword_recall = 1.0 if result.reject_correct else 0.0
        result.source_recall = 1.0 if result.reject_correct else 0.0
    else:
        # 非拒答类型：计算关键词召回和来源召回
        expected_kw = test_case.get("expected_answer_keywords", [])
        result.keyword_recall, result.keywords_found, result.keywords_missing = \
            keyword_match(answer, expected_kw)
        
        expected_src = test_case.get("expected_sources", [])
        result.source_recall, result.sources_found, result.sources_missing = \
            source_match(retrieved_docs, expected_src)
    
    return result
