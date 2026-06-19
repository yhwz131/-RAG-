# 🧠 知识问答系统（RAG-KBQA）

基于 RAG（检索增强生成）的多模态智能知识问答平台。支持上传多种格式文档，自动切片入库后通过大语言模型进行精准问答。

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **智能问答** | 基于 RAG 的上下文增强问答，支持流式/非流式输出 |
| **混合检索** | 向量检索 + BM25 关键词检索 + RRF 融合排序 |
| **双链路检索** | 纯文本链路（bge-large-zh 1024 维）+ 多模态链路（Qwen3-VL 4096 维） |
| **查询路由** | 规则 + LLM 混合路由，自动区分 rag / chitchat / general |
| **多格式文档** | 支持 PDF、Word、PPT、Excel、TXT、Markdown |
| **图片理解** | 文档图片自动提取 + mimo-v2.5 视觉模型生成描述 |
| **多轮对话** | JSON 持久化 + 滑动窗口，支持历史上下文 |
| **前端界面** | Vue 3 + TypeScript + Element Plus，明暗双主题 |

## 🛠 技术栈

| 层级 | 技术 |
|------|------|
| **后端框架** | FastAPI |
| **向量数据库** | Milvus Lite（pymilvus MilvusClient） |
| **文本 Embedding** | BAAI/bge-large-zh-v1.5（1024 维） |
| **多模态 Embedding** | Qwen3-VL-Embedding-8B（4096 维） |
| **大语言模型** | mimo-v2.5（支持视觉理解） |
| **中文分词** | jieba |
| **前端** | Vue 3 + TypeScript + Vite + Element Plus |
| **状态管理** | Pinia |

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- Miniconda（推荐）

### 安装

```bash
# 克隆仓库
git clone git@github.com:yhwz131/-RAG-.git
cd -RAG-

# 创建 conda 环境
conda create -n kbqa python=3.13
conda activate kbqa

# 安装 Python 依赖（国内源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装前端依赖并构建
cd frontend
npm install
npm run build
cd ..
```

### 配置

复制 `.env.example` 为 `.env`，填入 API Key：

```bash
cp .env.example .env
# 编辑 .env 填写以下必填项:
# LLM_API_KEY       - mimo-v2.5 API Key
# EMBEDDING_API_KEY  - Embedding API Key（可选，本地部署可不填）
```

### 启动

```bash
# 启动服务（后台运行）
./start.sh start

# 重启服务
./start.sh restart

# 停止服务
./start.sh stop

# 查看状态
./start.sh status

# 查看日志
./start.sh logs
```

服务启动后访问 `http://localhost:8000`。

## 📁 项目结构

```
├── api/                    # FastAPI 路由
│   ├── main.py            # 应用入口
│   ├── routes_chat.py     # 对话接口（流式/非流式）
│   ├── routes_docs.py     # 文档管理接口
│   └── routes_health.py   # 健康检查
├── config/
│   └── settings.py        # 全局配置（Pydantic）
├── rag/                   # RAG 核心
│   ├── chain.py           # RAG 链（检索→Prompt→LLM）
│   ├── retriever.py       # 向量检索（Milvus + BM25 + RRF）
│   ├── memory.py          # 对话记忆
│   ├── prompt_template.py # Prompt 模板
│   └── router.py          # 查询路由
├── embeddings/
│   ├── chunker.py         # 文档切片
│   └── embedder.py        # Embedding 客户端
├── utils/
│   ├── file_parser.py     # 文档解析 + 图片提取
│   └── logger.py          # 日志工具
├── frontend/              # Vue 3 前端源码
├── web/                   # 前端构建产物
├── data/                  # 运行时数据（已 gitignore）
├── docs/                  # 项目文档
├── docs1/                 # 开发文档
├── start.sh               # 服务管理脚本
└── requirements.txt       # Python 依赖
```

## 🔍 RAG 流程

```
用户提问 → 查询路由(rag/chitchat/general)
                ↓
         Embedding → Milvus 向量检索 + BM25 关键词检索
                ↓
            RRF 融合排序 → 相似度阈值过滤
                ↓
         构建上下文 → Prompt 模板 → LLM 生成回答
                ↓
         流式/非流式返回 answer + references
```

## 📖 文档

- [架构设计文档](docs/知识问答系统架构设计文档.md)
- [RAG 优化建议](docs1/RAG优化建议文档.md)
- [多模态优化建议](docs1/多模态优化建议文档.md)
- [开发规范文档](docs1/开发规范文档.md)

## 📄 License

MIT
