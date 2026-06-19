# 🧠 知识问答系统（RAG-KBQA）

基于 RAG（检索增强生成）的多模态智能知识问答平台。支持上传多种格式文档，自动切片入库后通过大语言模型进行精准问答。集成 PySpark 大数据处理管线，支持批量数据清洗、去重、统计分析和自动入库。

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
| **大数据管线** | PySpark 批量处理：数据清洗、MD5 去重、词频统计、质量报告 |
| **双引擎架构** | Spark 引擎（分布式）+ Simple 引擎（单机），自动降级 |
| **三模式入库** | 快速入库（秒级）/ 批量入库（Spark）/ 数据库导入 |
| **管线统计** | 前端实时展示处理文件数、切片数、入库数、处理历史 |

## 🛠 技术栈

| 层级 | 技术 |
|------|------|
| **后端框架** | FastAPI |
| **向量数据库** | Milvus Lite（pymilvus MilvusClient） |
| **文本 Embedding** | BAAI/bge-large-zh-v1.5（1024 维） |
| **多模态 Embedding** | Qwen3-VL-Embedding-8B（4096 维） |
| **大语言模型** | mimo-v2.5（支持视觉理解） |
| **大数据处理** | PySpark 4.1.2 + JDK 17 |
| **中文分词** | jieba |
| **前端** | Vue 3 + TypeScript + Vite + Element Plus |
| **状态管理** | Pinia |

## 🚀 快速开始

### 环境要求

- Python 3.10+（推荐 3.12）
- Node.js 18+
- JDK 17（PySpark 大数据管线需要）
- Miniconda（推荐）

### 安装

```bash
# 克隆仓库
git clone git@github.com:yhwz131/-RAG-.git
cd -RAG-

# 创建 conda 环境
conda create -n kbqa python=3.12
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

# 重启服务（安全重启，含 Milvus 锁释放检测）
./start.sh restart

# 停止服务
./start.sh stop

# 查看状态
./start.sh status

# 查看日志
./start.sh logs
```

> **注意**：`start.sh` 会自动配置 PySpark 所需的 Java 环境（JDK 17 + WSL2 cgroupv2 兼容）。重启时会执行完整的锁释放流程：SIGTERM → 等待退出 → SIGKILL → Milvus 文件锁验证 → 缓冲等待。

服务启动后访问 `http://localhost:8000`。

## 📁 项目结构

```
├── api/                    # FastAPI 路由
│   ├── main.py            # 应用入口
│   ├── routes_chat.py     # 对话接口（流式/非流式）
│   ├── routes_docs.py     # 文档管理接口（上传/列表/删除）
│   ├── routes_health.py   # 健康检查
│   ├── routes_pipeline.py # 管线 API（状态/历史/任务/数据库导入）
│   └── pipeline/          # 大数据处理管线
│       ├── service.py     # 管线服务（调度 + Milvus 导入 + 字段映射）
│       ├── schema.py      # 数据模型定义
│       ├── adapter.py     # 引擎适配器
│       └── engines/       # 处理引擎
│           ├── spark_engine.py # PySpark 引擎（分布式，含文件级 MD5 去重）
│           ├── simple.py      # Simple 引擎（单机，含文件级 MD5 去重）
│           └── database.py    # 数据库导入引擎（SQL 表/查询导入）
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
├── spark/
│   └── processor.py       # PySpark 数据处理（清洗/去重/统计）
├── utils/
│   ├── file_parser.py     # 文档解析 + 图片提取
│   └── logger.py          # 日志工具
├── frontend/              # Vue 3 前端源码
│   └── src/
│       ├── views/
│       │   ├── ChatView.vue   # 对话页面
│       │   └── DocsView.vue   # 文档管理（含管线统计 + 三模式入库）
│       ├── api/index.ts       # API 客户端封装
│       └── router/index.ts    # 前端路由
├── web/                   # 前端构建产物
├── data/                  # 运行时数据（已 gitignore）
│   ├── milvus.db/         # Milvus Lite 本地向量库
│   ├── raw/               # 原始文件
│   ├── processed/         # 管线处理输出
│   └── sessions/          # 对话会话 JSON
├── docs/                  # 项目文档
├── docs1/                 # 开发文档
├── start.sh               # 服务管理脚本（含 Java/PySpark 环境 + 锁安全重启）
└── requirements.txt       # Python 依赖
```

## 📋 开发进展

### 已完成功能

| 模块 | 功能 | 状态 |
|------|------|------|
| **RAG 问答** | 向量检索 + BM25 + RRF 融合排序 | ✅ |
| **RAG 问答** | 查询路由（rag/chitchat/general） | ✅ |
| **RAG 问答** | 流式/非流式对话输出 | ✅ |
| **RAG 问答** | 多轮对话记忆（JSON 持久化 + 滑动窗口） | ✅ |
| **多模态** | 文档图片自动提取 + 视觉模型描述生成 | ✅ |
| **多模态** | 双链路 Embedding（文本 1024 维 + 视觉 4096 维） | ✅ |
| **文档管理** | 多格式上传（PDF/Word/PPT/Excel/TXT/MD） | ✅ |
| **文档管理** | 快速入库（秒级直接解析） | ✅ |
| **文档管理** | 批量入库（PySpark 管线处理） | ✅ |
| **文档管理** | 数据库导入（SQL 数据源接入） | ✅ |
| **文档管理** | 单文件删除 + 全量清空（含 Milvus 向量清理） | ✅ |
| **大数据管线** | PySpark 分布式处理引擎 | ✅ |
| **大数据管线** | 文件级 MD5 去重（防止重复处理） | ✅ |
| **大数据管线** | source→filename 字段映射修复 | ✅ |
| **大数据管线** | 多层 UUID 前缀自动剥离 | ✅ |
| **大数据管线** | Spark→Simple 自动降级 | ✅ |
| **管线统计** | 前端实时展示处理文件数、切片数、入库数 | ✅ |
| **管线统计** | 处理历史记录（最近 5 次） | ✅ |
| **管线统计** | 整合到文件管理页面（原 Admin 页合并） | ✅ |
| **运维** | start.sh 安全重启（SIGTERM→等待→SIGKILL→锁检测） | ✅ |
| **运维** | Milvus Lite 文件锁安全释放机制 | ✅ |
| **前端** | Vue 3 + TypeScript + Element Plus | ✅ |
| **前端** | 明暗双主题切换 | ✅ |

### 技术亮点

1. **大数据管线集成**：基于 PySpark 4.1.2 + JDK 17，支持分布式文档处理，自动降级到单机模式
2. **字段映射修复**：解决了管线 ChunkData 的 `source` 字段与 Milvus `insert_documents()` 期望的 `filename` 字段不一致问题
3. **多层 UUID 清理**：自动剥离多次处理累积的 UUID 前缀（如 `257dab77_2639779b_xxx`）
4. **文件锁安全重启**：`start.sh restart` 采用 SIGTERM→5s 等待→SIGKILL→进程退出检测→Milvus 锁验证→3s 缓冲的完整流程
5. **页面整合**：将 Admin 管线统计页面合并到文件管理页面，减少路由复杂度

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

## 📊 大数据管线流程

```
文档上传 → 快速入库（直接解析）
         → 批量入库（PySpark 管线）
                ↓
         文件扫描 → MD5 去重 → 格式解析
                ↓
         文本清洗 → 切片 → 词频统计
                ↓
         UUID 前缀剥离 → 字段映射(source→filename)
                ↓
         Milvus 向量入库 → 前端统计展示
```

## 📖 文档

- [架构设计文档](docs/知识问答系统架构设计文档.md)
- [RAG 优化建议](docs1/RAG优化建议文档.md)
- [多模态优化建议](docs1/多模态优化建议文档.md)
- [开发规范文档](docs1/开发规范文档.md)

## 📄 License

MIT
