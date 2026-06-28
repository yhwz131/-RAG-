#!/usr/bin/env python3
"""
多模态图片分析脚本 — 调用 mimo-v2.5 模型理解图片内容

用法:
    python3 image_describe.py <图片路径> [用户问题]
    python3 image_describe.py --url <图片URL> [用户问题]

环境变量（可从 .env 自动加载）:
    LLM_API_KEY   — API 密钥（必需）
    LLM_BASE_URL   — API 端点（默认 https://token-plan-cn.xiaomimimo.com/v1）
    MM_LLM_MODEL   — 模型名（默认 mimo-v2.5）
"""

import sys
import os
import base64
import json
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

# ---------- 加载 .env ----------
def load_dotenv(env_path: Path):
    """从 .env 文件加载环境变量（不覆盖已有的）"""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = value

# 自动定位项目根目录的 .env
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # .github/skills/multimodal-image -> 项目根
load_dotenv(PROJECT_ROOT / ".env")

# ---------- 配置 ----------
API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1").rstrip("/")
MODEL = os.environ.get("MM_LLM_MODEL", "mimo-v2.5")
API_ENDPOINT = f"{BASE_URL}/chat/completions"

# 支持的图片格式
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# MIME 类型映射
MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

DEFAULT_PROMPT = "请详细描述这张图片的内容，包括文字、图表、代码、错误信息等所有可见信息。"


def get_mime_type(file_path: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    ext = Path(file_path).suffix.lower()
    return MIME_MAP.get(ext, "image/jpeg")


def encode_image_base64(file_path: str) -> str:
    """将本地图片编码为 base64 data URL"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {file_path}")

    # 检查文件大小（限制 20MB）
    file_size = path.stat().st_size
    if file_size > 20 * 1024 * 1024:
        raise ValueError(f"图片文件过大 ({file_size / 1024 / 1024:.1f}MB > 20MB)，请压缩后重试")

    mime = get_mime_type(str(path))
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def build_image_url(source: str) -> str:
    """构建图片 URL（本地文件转 base64，远程 URL 直接使用）"""
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        # 远程 URL
        return source
    else:
        # 本地文件
        return encode_image_base64(source)


def call_mimo_omni(image_url: str, question: str) -> str:
    """调用 mimo-v2.5 API 分析图片"""
    import httpx

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(API_ENDPOINT, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # 提取回复内容
    return data["choices"][0]["message"]["content"]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="调用 mimo-v2.5 分析图片")
    parser.add_argument("image", help="图片文件路径或 --url 图片URL")
    parser.add_argument("--url", action="store_true", help="将 image 参数作为 URL 处理")
    parser.add_argument("question", nargs="?", default=DEFAULT_PROMPT, help="用户问题（可选）")
    args = parser.parse_args()

    # 检查 API Key
    if not API_KEY:
        print("错误: LLM_API_KEY 未设置", file=sys.stderr)
        print(f"请在 {PROJECT_ROOT / '.env'} 中配置 LLM_API_KEY", file=sys.stderr)
        sys.exit(2)

    # 检查图片格式（仅本地文件）
    if not args.url:
        ext = Path(args.image).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            print(f"错误: 不支持的图片格式 '{ext}'", file=sys.stderr)
            print(f"支持的格式: {', '.join(sorted(SUPPORTED_FORMATS))}", file=sys.stderr)
            sys.exit(3)

    try:
        # 构建图片 URL
        image_url = build_image_url(args.image) if args.url else encode_image_base64(args.image)

        # 调用 API
        result = call_mimo_omni(image_url, args.question)
        print(result)

    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(4)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(5)
    except httpx.TimeoutException:
        print("错误: API 调用超时，请稍后重试", file=sys.stderr)
        sys.exit(6)
    except httpx.HTTPStatusError as e:
        print(f"错误: API 调用失败 (HTTP {e.response.status_code})", file=sys.stderr)
        print(e.response.text[:500], file=sys.stderr)
        sys.exit(7)
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
