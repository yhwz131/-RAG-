#!/usr/bin/env bash
# ============================================================
# image_describe.sh — 调用 mimo-v2-omni 多模态模型分析图片
#
# 用法:
#   bash image_describe.sh <图片路径> [用户问题]
#   bash image_describe.sh --url <图片URL> [用户问题]
#
# 环境变量（需提前 source .env 或手动设置）:
#   LLM_API_KEY      — API 密钥（必需）
#   LLM_BASE_URL      — API 端点（可选，默认 https://token-plan-cn.xiaomimimo.com/v1）
#   MM_LLM_MODEL      — 模型名（可选，默认 mimo-v2-omni）
# ============================================================

set -euo pipefail

# ---------- 颜色 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# ---------- 参数解析 ----------
IMAGE_URL=""
IMAGE_PATH=""
USER_QUESTION="请详细描述这张图片的内容，包括文字、图表、代码、错误信息等所有可见信息。"

if [[ $# -lt 1 ]]; then
    echo -e "${RED}错误: 缺少参数${NC}" >&2
    echo "用法: bash image_describe.sh <图片路径> [用户问题]" >&2
    echo "      bash image_describe.sh --url <图片URL> [用户问题]" >&2
    exit 1
fi

if [[ "$1" == "--url" ]]; then
    shift
    IMAGE_URL="$1"
    shift
else
    IMAGE_PATH="$1"
    shift
fi

# 剩余参数作为用户问题
if [[ $# -gt 0 ]]; then
    USER_QUESTION="$1"
fi

# ---------- 环境变量 ----------
# 尝试从项目 .env 加载
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    # 只加载需要的变量，避免覆盖已有环境变量
    export LLM_API_KEY="${LLM_API_KEY:-$(grep -E '^LLM_API_KEY=' "$PROJECT_ROOT/.env" | cut -d'=' -f2- | tr -d '"' | tr -d "'")}"
    export LLM_BASE_URL="${LLM_BASE_URL:-$(grep -E '^LLM_BASE_URL=' "$PROJECT_ROOT/.env" | cut -d'=' -f2- | tr -d '"' | tr -d "'")}"
fi

# 默认值
LLM_BASE_URL="${LLM_BASE_URL:-https://token-plan-cn.xiaomimimo.com/v1}"
MM_LLM_MODEL="${MM_LLM_MODEL:-mimo-v2-omni}"

# 检查 API Key
if [[ -z "${LLM_API_KEY:-}" ]]; then
    echo -e "${RED}错误: LLM_API_KEY 未设置${NC}" >&2
    echo "请在 $PROJECT_ROOT/.env 中配置 LLM_API_KEY" >&2
    exit 2
fi

# ---------- 构建图片内容 ----------
build_image_content() {
    local image_data=""
    local mime_type=""

    if [[ -n "$IMAGE_URL" ]]; then
        # URL 模式
        # 根据扩展名猜 MIME
        case "${IMAGE_URL,,}" in
            *.png)  mime_type="image/png" ;;
            *.gif)  mime_type="image/gif" ;;
            *.webp) mime_type="image/webp" ;;
            *.bmp)  mime_type="image/bmp" ;;
            *)      mime_type="image/jpeg" ;;
        esac
        image_data="$IMAGE_URL"
    else
        # 本地文件模式
        if [[ ! -f "$IMAGE_PATH" ]]; then
            echo -e "${RED}错误: 图片文件不存在: $IMAGE_PATH${NC}" >&2
            exit 3
        fi

        # 检查文件大小（限制 20MB）
        local file_size
        file_size=$(stat -c%s "$IMAGE_PATH" 2>/dev/null || stat -f%z "$IMAGE_PATH" 2>/dev/null || echo 0)
        if [[ "$file_size" -gt 20971520 ]]; then
            echo -e "${RED}错误: 图片文件过大 (>${20}MB)，请压缩后重试${NC}" >&2
            exit 4
        fi

        # 猜测 MIME 类型
        local ext="${IMAGE_PATH##*.}"
        case "${ext,,}" in
            png)  mime_type="image/png" ;;
            gif)  mime_type="image/gif" ;;
            webp) mime_type="image/webp" ;;
            bmp)  mime_type="image/bmp" ;;
            *)    mime_type="image/jpeg" ;;
        esac

        # Base64 编码
        image_data="data:${mime_type};base64,$(base64 -w0 "$IMAGE_PATH" 2>/dev/null || base64 "$IMAGE_PATH" 2>/dev/null)"
    fi

    echo "$image_data"
}

# ---------- 构建请求体 ----------
IMAGE_DATA=$(build_image_content)

# 转义用户问题中的特殊字符
ESCAPED_QUESTION=$(echo "$USER_QUESTION" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))" 2>/dev/null || echo "\"$USER_QUESTION\"")

# 使用 Python 构建精确的 JSON 请求体
REQUEST_BODY=$(python3 -c "
import json, sys

image_data = '''$IMAGE_DATA'''
question = json.loads('$ESCAPED_QUESTION')

payload = {
    'model': '$MM_LLM_MODEL',
    'messages': [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': question
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': image_data
                    }
                }
            ]
        }
    ],
    'max_tokens': 2048,
    'temperature': 0.3
}

print(json.dumps(payload))
" 2>/dev/null)

if [[ -z "$REQUEST_BODY" ]]; then
    echo -e "${RED}错误: 构建请求体失败${NC}" >&2
    exit 5
fi

# ---------- 调用 API ----------
API_ENDPOINT="${LLM_BASE_URL}/chat/completions"

RESPONSE=$(curl -s -w "\n%{http_code}" \
    --max-time 60 \
    -X POST "$API_ENDPOINT" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${LLM_API_KEY}" \
    -d "$REQUEST_BODY" 2>/dev/null)

# 分离响应体和状态码
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

# ---------- 处理响应 ----------
if [[ "$HTTP_CODE" != "200" ]]; then
    echo -e "${RED}错误: API 调用失败 (HTTP $HTTP_CODE)${NC}" >&2
    echo "$BODY" >&2
    exit 6
fi

# 提取回复内容
RESULT=$(python3 -c "
import json, sys
try:
    data = json.loads('''$BODY''')
    content = data['choices'][0]['message']['content']
    print(content)
except Exception as e:
    print(f'解析响应失败: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

if [[ $? -ne 0 ]]; then
    echo -e "${RED}错误: 解析 API 响应失败${NC}" >&2
    echo "$BODY" >&2
    exit 7
fi

echo "$RESULT"
