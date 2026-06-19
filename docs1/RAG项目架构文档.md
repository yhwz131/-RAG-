# 知识问答系统 架构设计文档 v2.1

## 文档信息

| 项目 | 内容 |
|------|------|
| **文档名称** | 知识问答系统架构设计文档 |
| **版本** | v2.1 |
| **更新日期** | 2026-06-19 |
| **仓库地址** | [https://github.com/yhwz131/-RAG-](https://github.com/yhwz131/-RAG-) |
| **技术栈** | FastAPI + Vue 3 + Milvus + RAG |
| **描述** | 基于 RAG 的多模态知识问答系统整体架构设计 |

---

## 1. 系统概述

### 1.1 系统定位

本系统是一个基于 **RAG（检索增强生成）** 技术的智能知识问答平台，支持上传多种格式文档（PDF、Word、PPT、Excel、TXT、Markdown 等），自动切片入库后通过大语言模型进行精准问答。系统同时支持纯文本检索和多模态检索两条链路，能够处理包含图片的文档内容。

### 1.2 核心功能

| 功能 | 描述 |
|------|------|
| **智能问答** | 基于 RAG 的上下文增强问答，支持流式输出 |
| **查询路由** | 规则 + LLM 混合路由，自动区分 rag/chitchat/general 三类查询 |
| **文档管理** | 批量上传、解析、切片、向量化入库，文件大小校验，失败自动清理 |
| **多模态检索** | 纯文本链路 (bge-large-zh) + 多模态链路 (Qwen3-VL-Embedding) |
| **混合检索** | 向量检索 + BM25 关键词检索 + RRF 融合排序，相似度阈值过滤 |
| **对话记忆** | 多轮对话上下文管理，轮次 + token 双重截断，会话持久化 |
| **流式响应** | SSE 风格的流式 token 输出 |
| **主题切换** | 明亮 / 暗黑双主题 |

### 1.3 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                          前端层 (Frontend)                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  frontend/ — Vue 3 + TypeScript + Vite + Element Plus         │  │
│  │  ├── views/ChatView.vue    对话页面（流式问答 + 会话管理）     │  │
│  │  ├── views/DocsView.vue   文档管理（批量上传 + 统计）         │  │
│  │  ├── stores/chat.ts       Pinia 状态管理                      │  │
│  │  ├── stores/theme.ts      主题切换（暗黑/明亮）               │  │
│  │  └── api/index.ts         API 客户端封装                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                    │ HTTP / SSE
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       API 服务层 (FastAPI)                          │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │
│  │  main.py      │  │ routes_chat   │  │ routes_docs           │   │
│  │  应用入口     │  │ 对话接口      │  │ 文档管理接口          │   │
│  │  组件初始化   │  │ 流式/非流式   │  │ 上传/列表/删除        │   │
│  │  静态文件托管 │  │ 会话管理      │  │ 批量上传              │   │
│  └───────────────┘  └───────────────┘  └───────────────────────┘   │
│  ┌───────────────┐                                                  │
│  │ routes_health │                                                  │
│  │ 健康检查      │                                                  │
│  └───────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────┘
                    │               │               │
        ┌───────────┘               │               └───────────┐
        ▼                           ▼                           ▼
┌───────────────┐  ┌───────────────────────────┐  ┌──────────────────┐
│  RAG 核心层   │  │    Embedding 层           │  │  工具层           │
│               │  │                           │  │                  │
│  rag/chain    │  │  embeddings/embedder      │  │  utils/           │
│  rag/retriever│  │  embeddings/chunker       │  │  file_parser     │
│  rag/memory   │  │                           │  │  logger          │
│  rag/prompt   │  │                           │  │  spark/processor │
└───────┬───────┘  └───────────┬───────────────┘  └──────────────────┘
        │                      │
        ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         存储层 (Storage)                            │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────┐ │
│  │  Milvus Lite      │  │  文件系统         │  │  会话持久化     │ │
│  │  向量数据库       │  │  data/raw/        │  │  data/sessions/ │ │
│  │  knowledge_base   │  │  data/processed/  │  │  JSON 文件      │ │
│  │  knowledge_base_mm│  │  data/uploads/    │  │                 │ │
│  └───────────────────┘  └───────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       外部服务层 (External)                         │
│  ┌───────────────────────┐  ┌───────────────────────────────────┐  │
│  │  LLM API              │  │  Embedding API                    │  │
│  │  SiliconFlow / 自建   │  │  bge-large-zh-v1.5 (1024d)        │  │
│  │  mimo-v2.5            │  │  Qwen3-VL-Embedding-8B (4096d)    │  │
│  └───────────────────────┘  └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 技术栈

### 2.1 后端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.9+ | 主要编程语言 |
| **FastAPI** | 0.104+ | Web API 框架 |
| **Uvicorn** | 0.24+ | ASGI 服务器 |
| **Pydantic** | 2.0+ | 数据验证与配置管理 |
| **pymilvus** | 2.4+ | Milvus 向量数据库客户端 |
| **httpx** | 0.25+ | 异步 HTTP 客户端（调用 LLM API） |
| **openai** | 1.0+ | OpenAI 兼容 API 客户端 |
| **jieba** | 0.42+ | 中文分词（BM25 检索） |
| **rank_bm25** | 0.2+ | BM25 关键词检索 |
| **PyMuPDF** | 1.23+ | PDF 解析与图片提取 |
| **python-docx** | 1.1+ | Word 文档解析 |
| **openpyxl** | 3.1+ | Excel 文件解析 |
| **python-pptx** | 0.6+ | PPT 文件解析 |
| **loguru** | 0.7+ | 日志管理 |

### 2.2 前端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **Vue 3** | 3.5+ | 前端框架（Composition API） |
| **TypeScript** | 5.7+ | 类型系统 |
| **Vite** | 6.0+ | 构建工具与开发服务器 |
| **Vue Router** | 4.5+ | 路由管理 |
| **Pinia** | 2.3+ | 状态管理 |
| **Element Plus** | 2.9+ | UI 组件库 |
| **Axios** | 1.7+ | HTTP 客户端 |
| **marked** | 15.0+ | Markdown 渲染 |

### 2.3 AI 模型

| 模型 | 用途 | 部署方式 |
|------|------|----------|
| **bge-large-zh-v1.5** | 文本 Embedding (1024 维) | API 调用 (SiliconFlow) |
| **Qwen3-VL-Embedding-8B** | 多模态 Embedding (4096 维) | API 调用 / 本地部署 |
| **mimo-v2.5** | 大语言模型（问答生成） | API 调用 |

---

## 3. 目录结构

```
knowledge-qa-system/
├── api/                            # API 服务层
│   ├── main.py                     # FastAPI 入口，组件初始化，静态文件托管
│   ├── routes_chat.py              # 对话接口（流式/非流式，会话管理）
│   ├── routes_docs.py              # 文档管理接口（上传/批量上传/列表/删除）
│   └── routes_health.py            # 健康检查接口
│
├── config/                         # 配置管理
│   ├── __init__.py
│   └── settings.py                 # 全局配置（Pydantic Settings，从 .env 读取）
│
├── rag/                            # RAG 核心模块
│   ├── __init__.py
│   ├── chain.py                    # RAG 链：查询路由 → 检索 → 上下文构建 → LLM 生成
│   ├── router.py                   # 查询路由器（规则 + LLM 混合路由，区分 rag/chitchat/general）
│   ├── retriever.py                # 向量检索器（Milvus + BM25 + RRF 混合检索）
│   ├── memory.py                   # 对话记忆管理（轮次 + token 双重截断，会话持久化）
│   └── prompt_template.py          # Prompt 模板（纯文本/多模态/闲聊/通用，含 estimate_tokens）
│
├── embeddings/                     # Embedding 模块
│   ├── __init__.py
│   ├── embedder.py                 # Embedding 客户端（文本 + 多模态）
│   └── chunker.py                  # 文档切片器
│
├── utils/                          # 工具模块
│   ├── __init__.py
│   ├── file_parser.py              # 统一文件解析器（PDF/Word/PPT/Excel/TXT/MD）
│   └── logger.py                   # 日志工具
│
├── spark/                          # Spark 大数据处理（可选）
│   └── processor.py                # 批量数据处理
│
├── web/                            # 旧版 Gradio 前端（已废弃）
│   └── __init__.py
│
├── frontend/                       # Vue 3 前端项目
│   ├── src/
│   │   ├── api/index.ts            # API 客户端封装
│   │   ├── views/
│   │   │   ├── ChatView.vue        # 对话页面
│   │   │   └── DocsView.vue        # 文档管理页面
│   │   ├── stores/
│   │   │   ├── chat.ts             # 对话状态管理
│   │   │   └── theme.ts            # 主题切换
│   │   ├── router/index.ts         # 路由配置
│   │   ├── App.vue                 # 根组件（导航栏 + 主题切换）
│   │   ├── main.ts                 # 应用入口
│   │   └── style.css               # 全局样式（暗黑/明亮双主题）
│   ├── dist/                       # 构建产物（由后端托管）
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── data/                           # 数据目录
│   ├── raw/                        # 上传的原始文件
│   ├── uploads/                    # 处理后的文件
│   ├── processed/                  # 处理结果
│   ├── sessions/                   # 会话持久化（JSON）
│   └── milvus.db                   # Milvus Lite 本地数据库
│
├── logs/                           # 日志目录
│   ├── backend.log                 # 后端日志
│   └── frontend.log                # 前端开发日志
│
├── docs/                           # 项目文档
├── docs1/                          # 架构文档
├── .pids/                          # 进程 PID 文件（运行时生成）
├── embeddings/                     # Embedding 模块
├── start.sh                        # 启动脚本（start/stop/restart/status/logs）
├── requirements.txt                # Python 依赖
├── .env                            # 环境变量配置
└── .env.example                    # 环境变量模板
```

---

## 4. 模块架构详解

### 4.1 RAG 核心流程

```
用户提问
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. 接收问题 (routes_chat.py)                             │
│    - 解析 query、session_id、mode                        │
│    - 加载对话历史 (memory)                               │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 2. 查询路由 (router.py)                                  │
│    - 第一层：规则匹配（正则，零成本 <1ms）                │
│    - 第二层：LLM 分类（mimo-v2.5，~50 token）             │
│    - 返回 QueryType: rag / chitchat / general            │
│    - 兜底：LLM 分类失败默认走 rag                        │
└─────────────────────────┬───────────────────────────────┘
                          │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
        chitchat       general          rag
            │              │              │
            ▼              ▼              ▼
     直接 LLM 回答   直接 LLM 回答   继续步骤 3
     (跳过检索)      (跳过检索)
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 3. 检索相关文档 (retriever.py)                           │
│    - 查询预处理: preprocess_query() 轻量清洗（去全角、压缩标点）│
│    - Embedding: 将 query 转为向量                        │
│    - 向量检索: Milvus COSINE 相似度搜索 (top_k)         │
│    - 关键词检索: BM25 文本匹配（jieba 分词）             │
│    - RRF 融合: score(d) = Σ 1/(k+rank_i(d)+1)          │
│    - 相似度阈值过滤: score < 0.3 丢弃                   │
│    - 多模态链路: 同时检索 knowledge_base_mm              │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 4. 硬性验证                                              │
│    - 阈值过滤后无结果 → 直接返回拒答提示（不调用 LLM）   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 5. 构建上下文 (chain.py._build_context)                  │
│    - 拼接检索结果为参考资料（带来源标注）                 │
│    - Token 长度控制（默认上限 3000 token）                │
│    - 超出限制时截断内容                                   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 6. 构建 Prompt + LLM 生成 (chain.py)                     │
│    - system: 根据模式选择 Prompt 模板:                   │
│      · 纯文本: SYSTEM_PROMPT                             │
│      · 多模态: MULTIMODAL_SYSTEM_PROMPT（图文结合规则）  │
│      · 闲聊: CHITCHAT_SYSTEM_PROMPT                      │
│      · 通用: GENERAL_SYSTEM_PROMPT                       │
│    - 历史对话（轮次 + token 双重截断）+ 当前问题         │
│    - 调用 mimo-v2.5 API，支持同步/流式                   │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ 7. 返回结果                                              │
│    - 流式: [SOURCES]{json}[/SOURCES] + token 流          │
│    - 非流式: { answer, references, session_id, query_type }
│    - metadata 包含 query_type 标识回答来源               │
│    - 保存对话历史到 memory                               │
└─────────────────────────────────────────────────────────┘
```

### 4.2 双链路检索架构

```
                    用户 Query
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐
   │  纯文本链路      │    │  多模态链路      │
   │                 │    │                 │
   │  bge-large-zh   │    │  Qwen3-VL       │
   │  1024 维        │    │  4096 维        │
   │  knowledge_base │    │  knowledge_base_mm│
   └────────┬────────┘    └────────┬────────┘
            │                       │
            ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐
   │  向量检索 + BM25 │    │  向量检索       │
   │  混合检索        │    │  + BM25 检索    │
   │                 │    │  + 图片描述生成  │
   └────────┬────────┘    └────────┬────────┘
            │                       │
            └───────────┬───────────┘
                        ▼
               合并结果 → LLM 生成
```

### 4.3 文档处理流程

```
用户上传文件（支持批量）
        │
        ▼
┌─────────────────────────────────────────┐
│ 1. 文件验证                              │
│    - 检查文件类型（白名单）               │
│    - 检查文件大小                         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 2. 文件保存                              │
│    - 生成唯一 ID                         │
│    - 保存到 data/raw/                    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 3. 文件解析 (file_parser.py)             │
│    - PDF → PyMuPDF 提取文本 + 图片       │
│    - DOCX → python-docx 解析 + 图片提取 │
│    - PPTX → python-pptx 解析 + 图片提取 │
│    - XLSX → openpyxl 解析                │
│    - TXT/MD/CSV → 直接读取               │
│    - 输出: pages[] (每页文本 + 页码)      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 4. 文档切片 (chunker.py)                 │
│    - 按 chunk_size 切分                  │
│    - 保留 overlap 重叠                   │
│    - 记录来源文件名 + 页码               │
│    - 输出: chunks[] (文本 + 元数据)      │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 5. 向量化入库                            │
│    - 纯文本链路: embedder → knowledge_base│
│    - 多模态链路: embedder → knowledge_base_mm│
│    - 文档图片: extract_images → mm 链路  │
│    - 图片描述: mimo-v2.5 vision 生成     │
│    - chunk_id 去重: 防止重复入库         │
└─────────────────────────────────────────┘
```

---

## 5. API 接口汇总

### 5.1 对话接口 `/api/chat`

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/chat` | 发送消息（支持 stream/non-stream） |
| POST | `/api/chat/clear` | 清空指定会话历史 |
| GET | `/api/chat/sessions` | 获取所有会话列表 |
| GET | `/api/chat/{session_id}` | 获取指定会话历史 |
| DELETE | `/api/chat/{session_id}` | 删除指定会话 |

### 5.2 文档管理接口 `/api/docs`

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/docs/upload` | 上传单个文档 |
| POST | `/api/docs/upload/batch` | 批量上传文档 |
| GET | `/api/docs/stats` | 获取知识库统计 |
| GET | `/api/docs/list` | 列出所有文档 |
| DELETE | `/api/docs/delete/{filename}` | 删除指定文档 |
| DELETE | `/api/docs/clear` | 清空所有文档 |

### 5.3 系统接口

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/` | 前端页面（SPA） |

### 5.4 流式响应协议

流式对话采用自定义协议，先发送引用来源和查询类型，再逐 token 输出：

```
[SOURCES]{"query_type": "rag", "sources": [...]}[/SOURCES]
你
好
，
我
是
...
```

- `query_type` 标识查询路由结果：`rag`（知识库检索）、`chitchat`（闲聊）、`general`（通用知识）
- `sources` 仅在 `rag` 类型下有值，`chitchat`/`general` 类型为空数组
- 前端通过 `fetch` + `ReadableStream` 解析，先提取 `[SOURCES]` 中的 JSON，再逐 chunk 渲染 Markdown

---

## 6. 配置管理

### 6.1 环境变量 (.env)

```bash
# ========== 数据目录 ==========
UPLOAD_DIR=./data/raw
PROCESSED_DIR=./data/processed
SESSIONS_DIR=./data/sessions
MILVUS_DB=./data/milvus.db

# ========== Embedding ==========
EMBEDDING_MODEL_NAME=BAAI/bge-large-zh-v1.5
EMBEDDING_DIM=1024

# ========== 多模态 Embedding ==========
MULTIMODAL_EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-8B
MULTIMODAL_EMBEDDING_DIM=4096
MULTIMODAL_COLLECTION_NAME=knowledge_base_mm

# ========== Milvus ==========
COLLECTION_NAME=knowledge_base
RETRIEVER_TOP_K=5
SIMILARITY_THRESHOLD=0.3

# ========== RAG 参数 ==========
MAX_CONTEXT_TOKENS=3000     # 上下文最大 token 数
RRF_K=60                    # RRF 融合参数

# ========== LLM ==========
LLM_BASE_URL=https://token-plan-cn.xiaomimimo.com/v1
LLM_API_KEY=your-api-key-here
LLM_MODEL_NAME=mimo-v2.5
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048
LLM_TIMEOUT=120.0

# ========== 对话记忆 ==========
MAX_HISTORY_ROUNDS=10       # 最大历史轮次
MAX_HISTORY_TOKENS=2000     # 历史最大 token 数

# ========== 文件上传 ==========
MAX_FILE_SIZE_MB=100        # 单文件大小限制

# ========== 服务器 ==========
API_HOST=0.0.0.0
API_PORT=8000

# ========== 日志 ==========
LOG_LEVEL=INFO
```

### 6.2 配置优先级

```
环境变量 > .env 文件 > 代码默认值
```

配置通过 `pydantic-settings` 自动加载，支持类型验证和默认值。

---

## 7. 前端架构

### 7.1 路由结构

| 路径 | 组件 | 描述 |
|------|------|------|
| `/` | 重定向 | → `/chat` |
| `/chat` | ChatView | 对话页面（欢迎页 + 快捷提问） |
| `/chat/:sessionId` | ChatView | 指定会话 |
| `/docs` | DocsView | 文档管理 |

### 7.2 状态管理 (Pinia)

**chat store**：
- `sessionId` — 当前会话 ID
- `sessions[]` — 会话列表
- `messages[]` — 当前对话消息
- `currentReferences[]` — 当前引用来源
- `loading` — 是否等待回复
- `mode` — 检索模式 (text / multimodal)
- `sidebarCollapsed` — 侧边栏折叠状态

**theme store**：
- `theme` — 当前主题 (dark / light)
- `toggleTheme()` — 切换主题
- 自动持久化到 localStorage

### 7.3 API 客户端

前端通过 `axios` (非流式) 和 `fetch` (流式) 两种方式调用后端：

```typescript
// 流式对话
chatStream(query, sessionId, mode, onSources, onToken, onDone, onError)

// 批量上传
uploadDocuments(files: File[]): Promise<BatchUploadResult>
```

### 7.4 主题系统

通过 CSS 变量实现双主题，`data-theme` 属性控制：

```css
:root           { --bg-dark: #1e1e2e; --text-primary: #e0e0e0; ... }
[data-theme="light"] { --bg-dark: #f5f7fa; --text-primary: #303133; ... }
```

所有组件使用 `var(--xxx)` 引用颜色，切换主题时自动生效。

---

## 8. 启动与部署

### 8.1 一键启动脚本

```bash
# 首次使用：安装依赖 + 构建前端
./start.sh start --install --build-frontend

# 日常启动（后台运行）
./start.sh start

# 开发模式（前端热更新）
./start.sh start --dev

# 前台运行（调试用）
./start.sh start --foreground

# 停止 / 重启 / 状态 / 日志
./start.sh stop
./start.sh restart
./start.sh status
./start.sh logs
./start.sh logs backend
```

### 8.2 服务端口

| 服务 | 端口 | 描述 |
|------|------|------|
| **FastAPI** | 8000 | 后端 API + 前端静态文件托管 |
| **Vite Dev** | 5173 | 前端开发服务器（仅 --dev 模式） |

### 8.3 生产部署

生产环境只需启动 FastAPI，它会自动托管 `frontend/dist/` 下的前端资源：

```bash
# 构建前端
cd frontend && npm run build

# 启动后端（自动托管前端）
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000` 即可使用完整系统。

---

## 9. 数据存储

### 9.1 向量数据库 (Milvus Lite)

| Collection | 维度 | 用途 |
|------------|------|------|
| `knowledge_base` | 1024 | 纯文本链路 (bge-large-zh) |
| `knowledge_base_mm` | 4096 | 多模态链路 (Qwen3-VL) |

每个向量包含字段：
- `id` — 主键
- `text` — 原始文本
- `vector` — Embedding 向量
- `source` — 来源文件名
- `page_number` — 页码

### 9.2 文件存储

| 目录 | 用途 |
|------|------|
| `data/raw/` | 上传的原始文件 |
| `data/raw/images/{file_id}/` | PDF 提取的图片 |
| `data/uploads/` | 处理后的文件 |
| `data/sessions/` | 会话 JSON 文件 |
| `data/milvus.db` | Milvus Lite 数据库文件 |

---

## 10. 错误处理与日志

### 10.1 日志配置

- 日志框架：loguru
- 日志文件：`logs/backend.log`
- 日志级别：通过 `LOG_LEVEL` 环境变量控制
- 控制台 + 文件双输出

### 10.2 错误处理策略

| 场景 | 处理方式 |
|------|----------|
| 文件类型不支持 | 返回 400，提示支持的格式 |
| 文件内容为空 | 返回 400，跳过继续 |
| LLM API 超时 | 返回 500，记录日志 |
| Embedding 失败 | 降级到关键词检索 |
| 多模态入库失败 | 不影响纯文本链路 |
| 图片描述生成失败 | 降级为默认标签 `[图片] 来源: xxx, 第x页` |
| 批量上传部分失败 | 返回成功/失败详情 |

---

## 11. 安全考虑

| 措施 | 描述 |
|------|------|
| 文件类型白名单 | 只允许指定扩展名 |
| 文件大小限制 | 默认 100MB |
| 路径遍历防护 | 文件名加 UUID 前缀 |
| CORS | 生产环境建议配置 |
| API Key | 通过环境变量管理，不硬编码 |

---

## 12. 扩展性

| 方向 | 方案 |
|------|------|
| **LLM 切换** | 修改 `.env` 中的 `LLM_API_BASE` 和 `LLM_MODEL_NAME` |
| **Embedding 模型** | 修改 `EMBEDDING_MODEL_NAME` 和 `EMBEDDING_DIM` |
| **用户认证** | 添加 FastAPI 中间件 (JWT) |
| **分布式部署** | Milvus Lite → Milvus Standalone/Cluster |
| **更多文件格式** | 在 `file_parser.py` 中扩展解析器 |
| **知识图谱** | 在 retriever 中集成图数据库检索 |
