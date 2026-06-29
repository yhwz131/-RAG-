"""Token 估算工具（统一定义，供 rag/ 和 utils/ 共用）"""
import re


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数

    粗略估算规则：
    - 中文字符约 2.0 token/字
    - 英文单词约 1.3 token/词
    - 标点符号和空白按 0.5 token/字符
    """
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    en_words = len(re.findall(r'[a-zA-Z]+', text))
    other_chars = len(re.findall(r'[^\u4e00-\u9fffa-zA-Z]+', text))
    return int(cn_chars * 2.0 + en_words * 1.3 + other_chars * 0.5)
