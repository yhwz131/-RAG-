# Copilot 项目指令

## ⛔ 强制规则：图片处理
当用户消息中包含**图片附件**（attachment、粘贴的截图、image.png/jpg 等）时，**必须**在回复前先执行以下步骤：
1. 获取图片文件的本地路径（附件通常保存在 `/tmp/` 或 workspace 目录下）
2. 运行命令调用多模态 API 分析图片：
   ```bash
   python3 /home/yhwz/knowledge-qa-system/.github/skills/multimodal-image/image_describe.py "<图片路径>" "<用户的问题或默认描述请求>"
   ```
3. 将返回的图片描述整合到回答中
4. 如果脚本报错（API Key 缺失、超时等），告知用户具体原因

> **为什么**：当前基座模型（mimo-v2.5）不支持多模态输入，无法直接理解图片。
> 必须通过 mimo-v2-omni 模型（同 API 端点，模型名不同）来处理图片。
> 技能详情参见 `.github/skills/multimodal-image/SKILL.md`

## 通用规则
- 始终使用简体中文回复
- 项目使用 conda 环境 `kbqa`（Python 3.12）
- 遇到报错先分析原因再操作，不要盲目重试
- 提交代码只推送到 `graduation-project` 分支，不要动 `main`

## 技术栈
- 后端：FastAPI + Milvus Lite + PySpark 4.1.2
- 前端：Vue 3 + TypeScript + Element Plus + Vite
- LLM：mimo-v2.5（文本）+ mimo-v2-omni（多模态，同 API 端点，仅模型名不同）
- Embedding：bge-large-zh-v1.5（1024 维）+ Qwen3-VL（4096 维多模态）
- Java：OpenJDK 17.0.2（路径 /opt/jdk-17.0.2）

## 环境变量（启动时自动设置）
- JAVA_HOME=/opt/jdk-17.0.2
- JAVA_TOOL_OPTIONS=-XX:-UseContainerSupport（WSL2 cgroupv2 兼容）
- PYSPARK_PYTHON=/home/yhwz/miniconda3/envs/kbqa/bin/python3

## 代码规范
- Python：遵循 PEP 8，使用 type hints
- Vue：Composition API + `<script setup>` 语法
- 日志：使用 `utils/logger.py` 的 loguru
- 配置：使用 `config/settings.py` 的 Pydantic Settings

## 项目结构要点
- `api/pipeline/` — 数据管线模块（Simple/Spark/Database 三引擎）
- `rag/` — RAG 核心（chain/retriever/router/memory/prompt）
- `embeddings/` — 文档切片与向量化
- `frontend/src/views/DocsView.vue` — 文档管理页（含管线统计，Admin 页已合并）
- `data/milvus.db/` — Milvus Lite 本地数据库，使用 flock 文件锁
