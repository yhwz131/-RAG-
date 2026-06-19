# JNAAIPPT 系统架构设计文档 v2.0

## 文档信息

| 项目 | 内容 |
|------|------|
| **文档名称** | JNAAIPPT 系统架构设计文档 |
| **版本** | v2.0 |
| **更新日期** | 2026-04-19（修订：前端目录与 TTS 端口与仓库一致） |
| **作者** | AI 数字人项目组 |
| **描述** | AI 数字人 PPT 视频生成系统整体架构设计 |

---

## 1. 系统概述

### 1.1 系统定位

JNAAIPPT 是一个基于 AI 技术的数字人 PPT 视频生成系统，能够将 PPT 课件自动转换为带有数字人讲解、口播音频和字幕的教学视频。该系统面向教育、培训、企业演示等场景，实现了从 PPT 到视频的自动化转换流程。

### 1.2 核心功能

- **PPT 解析**：将 PPT/PDF 转换为结构化 Markdown 格式
- **口播文案生成**：基于 AI 大模型自动生成每页 PPT 的讲解文案
- **语音合成**：将口播文案转换为数字人的讲解音频
- **唇形同步**：基于 Wav2Lip 算法实现数字人唇形与音频同步
- **视频合成**：将数字人视频与 PPT 背景合成为完整的教学视频
- **字幕生成**：基于 Whisper 模型自动生成字幕
- **SSE 实时监控**：通过 Server-Sent Events 实时推送处理进度和日志
- **模板系统**：支持数字人模板和 PPT 模板的自定义配置
- **视频合并**：支持多页视频的自动合并和 HLS 切片
- **BGM 配置**：支持自定义背景音乐和音效系统
- **情感控制**：TTS 支持情感参数调节（sad/happy/angry/surprise/fearful/disgusted/default）
- **异步任务调度**：完整的异步任务管理机制

### 1.3 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             前端层 (Frontend)                            │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │  frontend-new/ — Vue 3 + Vite + TypeScript + vue-router       │       │
│  │  （生产 Nginx root → frontend-new/dist；可选副本 frontend/；另见 new_frontend/） │       │
│  └─────────────────────────────────────────────────────────────┘       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐       │
│  │ BGMConfig.vue   │  │ EmotionConfig   │  │ useSSE.ts          │       │
│  │ (BGM 配置)      │  │ (情感控制配置)   │  │ (SSE 实时通信)     │       │
│  └─────────────────┘  └─────────────────┘  └────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓ HTTP/SSE
┌─────────────────────────────────────────────────────────────────────────┐
│                        API 网关层 (Nginx)                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  反向代理 + 负载均衡 + 静态资源服务                                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     核心业务层 (digital_human_interface)                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐       │
│  │ FastAPI 主服务   │  │ 任务调度服务     │  │ SSE 推送服务       │       │
│  │ (main.py)       │  │ (scheduler)     │  │ (sse_monitor)      │       │
│  │ Port: 9088      │  │                 │  │                    │       │
│  └─────────────────┘  └─────────────────┘  └────────────────────┘       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐       │
│  │ 转换服务        │  │ 视频合并服务     │  │ 文件清理服务       │       │
│  │ (conversion)    │  │ (video_merge)   │  │ (cleanup)          │       │
│  │ PPT→PDF→Img     │  │ + HLS 切片      │  │ + 磁盘监控         │       │
│  └─────────────────┘  └─────────────────┘  └────────────────────┘       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────────┐       │
│  │ 路由层          │  │ 配置管理         │  │ 核心转换器         │       │
│  │ files/video/tts │  │ (settings.py)   │  │ (converter)        │       │
│  │ dh/sse_monitor  │  │                 │  │                    │       │
│  └─────────────────┘  └─────────────────┘  └────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                    ↓ HTTP              ↓ HTTP              ↓ HTTP
┌──────────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  OCR 识别层          │  │ 语音合成层       │  │ 数字人视频层         │  │
│                      │  │                  │  │                      │  │
│  paddleocr/          │  │ index-tts-vllm/  │  │ wav2lip_workspce/    │  │
│  api_paddleocr_vl    │  │ tts_server_v2    │  │ lx/测试/main.py      │  │
│  _ai_ppt.py          │  │ batch.py         │  │                      │  │
│  Port: 8802          │  │ Port: 6006       │  │ Port: 5000           │  │
│                      │  │                  │  │                      │  │
│  • PaddleOCR-VL      │  │ • IndexTTS-vLLM  │  │ • Wav2Lip 唇形同步   │  │
│  • PDF/图片解析       │  │ • 情感控制        │  │ • MediaPipe 抠像     │  │
│  • Markdown 输出     │  │ • 批量推理        │  │ • Whisper 字幕     │  │
│  • 口播文案生成       │  │ • CUDA 内存管理  │  │ • 并行处理         │  │
│  • style 参数支持    │  │ • 文件自动清理   │  │ • BGM/音效系统     │  │
│  • DeepSeek LLM     │  │                  │  │ • 无数字人模式       │  │
└──────────────────────┘  └──────────────────┘  └──────────────────────┘  │
```

---

## 2. 技术栈

### 2.1 后端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.8+ | 主要编程语言 |
| **FastAPI** | 0.104+ | API 框架 |
| **Uvicorn** | 0.24+ | ASGI 服务器 |
| **Pydantic** | 2.0+ | 数据验证 |
| **httpx** | 0.25+ | 异步 HTTP 客户端 |
| **OpenCV** | 4.8+ | 视频/图像处理 |
| **FFmpeg** | 5.0+ | 视频处理 |
| **PaddleOCR** | 2.7+ | OCR 识别 |
| **Wav2Lip** | - | 唇形同步 |
| **IndexTTS** | - | 语音合成 |
| **Whisper** | large-v3 | 语音识别 |
| **MediaPipe** | 0.10.9 | 人像分割 |

### 2.2 前端技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| **Vue 3** | 3.5+ | 前端框架（主工程 `frontend-new/`） |
| **Vite** | 6.x | 构建工具 |
| **TypeScript** | 5.9+ | 类型系统 |
| **Vue Router** | 4.x | 路由管理 |
| **hls.js** | 1.6+ | HLS 播放 |
| **Element Plus** | （可选） | 仅 **`new_frontend/`** 试验工程使用，非当前部署主线 |

### 2.3 AI 模型

| 模型 | 用途 | 部署方式 |
|------|------|----------|
| **PaddleOCR-VL** | PDF/图片解析 | 本地部署 (Port 8802) |
| **DeepSeek Chat** | 口播文案生成 | API 调用 |
| **IndexTTS** | 语音合成 | 本地部署（**Port 6006**，与 `start_all.sh` / `server/.env` 一致） |
| **Wav2Lip** | 唇形同步 | 本地推理 (Port 5000) |
| **Whisper (large-v3)** | 语音识别/字幕 | 本地推理 (Port 5000) |
| **MediaPipe Selfie** | 人像分割 | 本地推理 (Port 5000) |

---

## 3. 模块架构

### 3.1 模块关系图

```
                                    ┌─────────────┐
                                    │   前端层     │
                                    │ frontend-new│
                                    └──────┬──────┘
                                           │ HTTP/SSE
                                    ┌──────▼──────┐
                                    │ API 网关    │
                                    │   Nginx     │
                                    └──────┬──────┘
                                           │
                            ┌──────────────▼──────────────┐
                            │   digital_human_interface   │
                            │   (Port 9088)               │
                            │                             │
                            │  ┌─────────┐ ┌────────────┐ │
                            │  │ 路由器  │ │  SSE 监控  │ │
                            │  │         │ │            │ │
                            │  └─────────┘ └────────────┘ │
                            │  ┌─────────┐ ┌────────────┐ │
                            │  │ 服务层  │ │  转换器    │ │
                            │  │         │ │            │ │
                            │  └─────────┘ └────────────┘ │
                            └──────┬──────────┬───────────┘
                                   │          │
                        ┌──────────▼──┐  ┌────▼────────┐
                        │  OCR 服务   │  │ TTS 服务    │
                        │  Port 8802  │  │ Port 6006   │
                        │  paddleocr/ │  │ index-tts/  │
                        └──────┬──────┘  └────┬────────┘
                               │              │
                        ┌──────▼──────────────▼──────┐
                        │   Wav2Lip 服务 (Port 5000) │
                        │   wav2lip_workspce/        │
                        └────────────────────────────┘
```

### 3.2 目录结构

```
/home/ubuntu/workspace/PPTTalK/
├── docs/                              # 项目文档
│   ├── jnaaippt_architecture_v1.md    # 系统架构文档
│   ├── jnawav2lip_data_flow.md        # Wav2Lip 数据流文档
│   ├── jnadigital_human_data_flow.md  # 数字人数据流文档
│   ├── jnaindex_tts_data_flow.md      # TTS 数据流文档
│   ├── jnapaddleocr_service_detail.md # PaddleOCR 服务文档
│   ├── 各服务配置与常见问题说明.md    # 各服务 .env、脚本与部署风险
│   ├── 文档索引.md                    # docs 总目录与前端目录对照
│   └── 修缮与功能总结.md              # 功能变更记录
│
├── frontend-new/                    # 主前端 (Vue 3 + Vite + TS；vite build → dist/，Nginx 指向 dist/)
│   ├── src/
│   │   ├── pages/                   # IndexPage / UploadPage / VideoListPage
│   │   ├── components/              # 业务组件
│   │   ├── composables/useSSE.ts    # SSE
│   │   ├── api/                     # 后端 API 封装
│   │   └── router/
│   ├── package.json
│   └── vite.config.ts
│
├── new_frontend/                    # 另一套 Vue 前端（Element Plus 等），非 setup_env 部署主线
│   └── …
│
├── digital_human_interface/           # 核心业务服务 (Port 9088)
│   ├── main.py                        # FastAPI 入口
│   ├── config/
│   │   └── settings.py                # 配置管理
│   ├── routers/
│   │   ├── conversion.py              # 转换服务路由
│   │   ├── files.py                   # 文件管理路由
│   │   ├── my_digital_human.py        # 数字人路由
│   │   ├── tts.py                     # TTS 路由
│   │   ├── video.py                   # 视频管理路由
│   │   └── sse_monitor.py             # SSE 监控路由
│   ├── services/
│   │   ├── conversion_service.py      # 转换服务
│   │   ├── file_service.py            # 文件服务
│   │   ├── video_merge_service.py     # 视频合并服务
│   │   ├── video_his_service.py       # HLS 切片服务
│   │   ├── cleanup_service.py         # 清理服务
│   │   └── scheduler_service.py       # 任务调度服务
│   ├── core/
│   │   ├── converter.py               # 转换器基类
│   │   └── libreoffice_converter.py   # LibreOffice 转换器
│   ├── templates/                     # HTML 模板
│   │   └── index.html                 # 首页模板
│   └── static/                        # 静态资源
│       └── Digital_human/             # 数字人模板目录
│
├── paddleocr/                         # PaddleOCR 服务 (Port 8802)
│   ├── api_paddleocr_vl_ai_ppt.py     # API 入口
│   ├── until/
│   │   └── get_voice_content.py       # 口播文案生成
│   ├── .env                           # 环境变量
│   ├── start.sh                       # 启动脚本
│   └── stop.sh                        # 停止脚本
│
├── index-tts-vllm/                    # TTS 服务 (Port 6006)
│   ├── server/
│   │   ├── tts_server_v2_batch.py     # 批处理 API
│   │   ├── api_server_v2.py           # API 服务器
│   │   └── .env                       # 环境变量
│   ├── indextts/
│   │   └── infer_vllm_v2.py           # VLLM 推理
│   ├── webui_v2.py                    # Web UI
│   └── requirements.txt               # 依赖列表
│
├── wav2lip_workspce/                  # Wav2Lip 服务 (Port 5000)
│   ├── lx/
│   │   ├── 测试/                       # 主用项目
│   │   │   ├── main.py                # FastAPI 入口
│   │   │   ├── config.py              # 配置管理
│   │   │   ├── models.py              # 数据模型
│   │   │   ├── api/
│   │   │   │   └── routes.py          # API 路由
│   │   │   └── services/
│   │   │       ├── video_generator.py # 视频生成
│   │   │       ├── audio_processor.py # 音频处理
│   │   │       ├── subtitle_service.py# 字幕服务
│   │   │       ├── file_service.py    # 文件服务
│   │   │       └── wav2lip_model.py   # Wav2Lip 模型
│   │   └── digital_human_project/     # 备用项目
│   └── Wav2Lip/                       # 官方 Wav2Lip 代码
│
├── frontend/                          # 可选：构建产物副本（rsync）；生产主线为 frontend-new/dist
│   ├── assets/                        # 静态资源
│   └── index.html                     # 入口页面
│
├── start_all.sh                       # 一键启动脚本
└── nginx/                             # Nginx 配置 (可选)
```

---

## 4. 核心业务流程

### 4.1 完整视频生成流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. 用户提交任务                                                          │
│    - 上传 PPT/PDF 文件                                                   │
│    - 选择数字人模板                                                      │
│    - 配置 BGM/情感等参数                                                 │
│    - 发起视频生成请求                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. digital_human_interface 接收请求                                      │
│    - 创建任务记录 (task_id)                                              │
│    - 初始化 SSE 推送通道                                                 │
│    - 启动异步任务                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. PPT 解析阶段 (conversion_service)                                     │
│    - PPT/PDF → LibreOffice → PDF                                        │
│    - PDF → 逐页图像 (pdftoppm)                                          │
│    - 图像存储到临时目录                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. OCR 识别阶段 (paddleocr, Port 8802)                                   │
│    - 逐页调用 /parse 接口                                                │
│    - PaddleOCR-VL 识别页面内容                                           │
│    - 调用 DeepSeek 生成口播文案 (style 参数)                             │
│    - 返回 Markdown + 口播文案                                            │
│    - 输出: page_content + voice_content (每页)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. TTS 语音合成阶段 (index-tts-vllm, Port 6006)                          │
│    - 逐页调用 /v2/infer 接口                                             │
│    - IndexTTS 生成音频文件                                               │
│    - 支持情感参数 (emotion: sad/happy/angry/surprise/fearful/default)   │
│    - 输出: audio.wav (每页)                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. 数字人视频生成阶段 (wav2lip, Port 5000)                               │
│    - 调用 /generate/upload 接口                                          │
│    - Wav2Lip 唇形同步                                                    │
│    - MediaPipe 人像分割                                                  │
│    - Whisper 字幕生成                                                    │
│    - 并行处理: 背景视频 + Wav2Lip + 字幕                                 │
│    - 输出: video.mp4 (每页)                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. 视频合并阶段 (video_merge_service)                                    │
│    - 合并所有页面视频                                                    │
│    - 添加 BGM (mix_bgm)                                                  │
│    - 生成完整视频                                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 8. HLS 切片阶段 (video_his_service)                                      │
│    - 视频切片为 .m3u8 + .ts 文件                                         │
│    - 生成 HLS 播放列表                                                   │
│    - 支持流媒体播放                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 9. 任务完成                                                              │
│    - 更新任务状态为 completed                                             │
│    - 返回 HLS 播放地址                                                   │
│    - 清理临时文件                                                        │
│    - SSE 推送完成消息                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 任务状态流转

```
pending → processing → completed
    ↓         ↓
   failed   retrying
```

| 状态 | 描述 | 触发条件 |
|------|------|----------|
| **pending** | 任务等待中 | 任务刚创建 |
| **processing** | 任务处理中 | 开始处理任务 |
| **completed** | 任务完成 | 所有步骤成功 |
| **failed** | 任务失败 | 任何步骤失败 |
| **retrying** | 重试中 | 失败后重试 |

---

## 5. API 端点汇总

### 5.1 digital_human_interface (Port 9088)

| 路径 | 方法 | 描述 | 新增功能 |
|------|------|------|----------|
| `/` | GET | 首页 | - |
| `/api/convert` | POST | PPT 转换 | - |
| `/api/upload` | POST | 文件上传 | - |
| `/api/digital-human/upload` | POST | 上传数字人 | - |
| `/api/digital-human/list` | GET | 数字人列表 | - |
| `/api/digital-human/generate` | POST | 生成视频 | - |
| `/api/digital-human/generate/stream` | POST | 生成视频 (SSE) | ✅ 新增 |
| `/api/digital-human/generate/task` | POST | 创建任务 | ✅ 新增 |
| `/api/digital-human/generate/task/{task_id}` | GET | 获取任务状态 | ✅ 新增 |
| `/api/digital-human/delete` | POST | 删除数字人 | - |
| `/api/tts` | POST | TTS 合成 | - |
| `/api/video/merge` | POST | 合并视频 | ✅ mix_bgm |
| `/api/video/hls` | POST | HLS 切片 | ✅ 新增 |
| `/api/sse/monitor/{task_id}` | GET | SSE 监控 | ✅ 新增 |
| `/api/files/list` | GET | 文件列表 | - |
| `/api/files/delete` | DELETE | 删除文件 | - |
| `/api/cleanup` | POST | 清理任务 | - |
| `/health` | GET | 健康检查 | - |

### 5.2 paddleocr (Port 8802)

| 路径 | 方法 | 描述 | 参数 |
|------|------|------|------|
| `/parse` | POST | 解析 PPT | file, style (brief/normal/professional) |
| `/health` | GET | 健康检查 | - |

### 5.3 index-tts-vllm (Port 6006)

| 路径 | 方法 | 描述 | 参数 |
|------|------|------|------|
| `/v2/infer` | POST | 语音合成 | text, emotion, speaker_id |
| `/health` | GET | 健康检查 | - |

### 5.4 wav2lip (Port 5000)

| 路径 | 方法 | 描述 | 参数 |
|------|------|------|------|
| `/` | GET | API 信息 | - |
| `/generate` | POST | 生成视频 (路径) | background_image, face_video, audio_path |
| `/generate/upload` | POST | 生成视频 (上传) | bgm_enabled, bgm_path |
| `/upload` | POST | 上传文件 | file |
| `/status/{task_id}` | GET | 任务状态 | - |
| `/download/{task_id}` | GET | 下载视频 | - |
| `/video_path/{task_id}` | GET | 获取路径 | - |
| `/open_folder/{task_id}` | GET | 打开文件夹 | - |
| `/cleanup` | POST | 清理任务 | - |

---

## 6. 服务管理

### 6.1 启动脚本 (start_all.sh)

```bash
# 启动所有服务
start_all.sh start

# 停止所有服务
start_all.sh stop

# 查看服务状态
start_all.sh status
```

### 6.2 服务端口分配

| 服务 | 端口 | 描述 |
|------|------|------|
| **digital_human_interface** | 9088 | 核心业务服务 |
| **paddleocr** | 8802 | OCR 识别服务 |
| **index-tts-vllm** | 6006 | TTS 语音合成（`start_all.sh` 与 `tts_server_v2_batch.py`） |
| **wav2lip** | 5000 | 数字人视频生成 |
| **nginx** | 80/443 | 反向代理 |
| **frontend-new** | 5173 (dev) | 前端开发服务器；生产 **`frontend-new/dist`**（Nginx root）；**frontend/** 为可选副本 |

---

## 7. 配置管理

### 7.1 digital_human_interface 配置

```python
# config/settings.py
class Settings:
    # 基础配置
    host: str = "0.0.0.0"
    port: int = 9088
    debug: bool = False
    
    # 上传配置
    upload_dir: str = "./uploads"
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    
    # FFmpeg 配置
    ffmpeg_path: str = "ffmpeg"
    
    # LibreOffice 配置
    libreoffice_path: str = "libreoffice"
    
    # 服务地址
    paddleocr_url: str = "http://127.0.0.1:8802"
    tts_url: str = "http://127.0.0.1:6006"  # 实际业务代码中多为写死 URL，以 main.py / routers 为准
    wav2lip_url: str = "http://127.0.0.1:5000"
```

### 7.2 paddleocr 配置

```bash
# .env
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
DEVICE=cuda
PORT=8802
HOST=0.0.0.0
```

### 7.3 index-tts-vllm 配置

```bash
# server/.env（示例）
PORT=6006
HOST=0.0.0.0
MODEL_DIR=/path/to/IndexTTS-2-vLLM
# 若用 start_all.sh 启动且带 --gpu_memory_utilization 等参数，CLI 会覆盖部分 .env 默认值
```

### 7.4 wav2lip 配置

```python
# config.py
class Config:
    host_ip: str = "0.0.0.0"
    port: int = 5000
    
    # Wav2Lip 模型路径
    wav2lip_dir: str = "/path/to/Wav2Lip"
    wav2lip_checkpoint: str = "/path/to/Wav2Lip/models/Wav2Lip-SD-GAN.pt"
    
    # 音效文件目录
    sfx_dir: str = "./assets/sfx"
    default_notification_sound: str = "./完成提示音.wav"
    default_transition_sound: str = "./转场音效.wav"
    default_bgm: str = "./商务背景音乐.mp3"
```

---

## 8. 性能优化

### 8.1 模型预加载

| 服务 | 预加载模型 | 效果 |
|------|-----------|------|
| wav2lip | Wav2Lip 模型 + 人脸检测器 + Whisper | 消除冷启动延迟 |
| index-tts | IndexTTS 模型 | 快速响应首个请求 |

### 8.2 并行处理

wav2lip 服务采用并行处理架构：
```
并行任务:
├─ 背景视频生成
├─ Wav2Lip 唇形同步
└─ Whisper 字幕生成

串行任务:
├─ 调整数字人大小
├─ MediaPipe 人像分割
├─ 叠加数字人到背景
├─ 添加音频
├─ 添加字幕
└─ 添加音效系统
```

### 8.3 线程安全

- 移除所有 `os.chdir()` 调用
- 使用 `subprocess.run(cwd=...)` 指定工作目录
- 避免多线程环境下的路径冲突
- 所有目录路径基于 `__file__` 的绝对路径

### 8.4 CUDA 内存管理

- `index-tts-vllm` 服务集成 `CudaCacheCleanupMiddleware`
- 自动清理 CUDA 缓存，防止 OOM
- 定时清理策略

### 8.5 文件自动清理

- `cleanup_service` 定时清理临时文件
- 磁盘空间监控和告警
- 24 小时自动清理旧任务

---

## 9. 部署架构

### 9.1 部署方式

```
┌─────────────────────────────────────────────────────────────────┐
│                           服务器                                 │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Nginx      │  │  digital_   │  │  paddleocr  │            │
│  │  (80/443)   │  │  human_     │  │  (8802)     │            │
│  │             │  │  interface  │  │             │            │
│  │  反向代理    │  │  (9088)     │  │  OCR 服务   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  index-tts  │  │  wav2lip    │  │  frontend   │            │
│  │  (6006)     │  │  (5000)     │  │  (Nginx)    │            │
│  │             │  │             │  │             │            │
│  │  TTS 服务   │  │  视频生成   │  │  静态资源   │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐                              │
│  │  LibreOffice│  │  FFmpeg     │                              │
│  │  (转换)     │  │  (视频处理)  │                              │
│  └─────────────┘  └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 GPU 分配

| 服务 | GPU 需求 | 备注 |
|------|---------|------|
| paddleocr | ✅ GPU (CUDA) | PaddleOCR-VL 模型推理 |
| index-tts-vllm | ✅ GPU (CUDA) | IndexTTS 语音合成 |
| wav2lip | ✅ GPU (CUDA) | Wav2Lip 唇形同步 |
| digital_human_interface | ❌ CPU | 业务逻辑和调度 |

### 9.3 环境要求

```bash
# 系统要求
Ubuntu 20.04+ / CentOS 8+
Python 3.8+
Node.js 18+

# GPU 要求
NVIDIA GPU (推荐 RTX 3090+ 或 A100)
CUDA 11.7+
cuDNN 8.5+

# 依赖软件
FFmpeg 5.0+
LibreOffice 7.0+
OpenCC (简繁转换)
```

---

## 10. 安全与运维

### 10.1 安全措施

- CORS 配置管理
- 文件上传类型和大小限制
- 文件名清理，防止路径遍历攻击
- 任务 ID UUID 生成
- API 密钥管理

### 10.2 监控与日志

- SSE 实时推送处理进度和日志
- 健康检查端点 (/health)
- 磁盘空间监控
- GPU 使用率监控

### 10.3 运维建议

- 定期清理临时文件
- 监控 GPU 内存使用
- 定期备份重要数据
- 使用 systemd 管理服务进程
- 配置日志轮转

---

## 11. 变更日志

### v2.0 (2026-04-17)

**新增功能**：
1. ✅ SSE 实时监控推送
2. ✅ 数字人模板系统
3. ✅ BGM 配置和音效系统
4. ✅ TTS 情感控制参数
5. ✅ 无数字人视频模式
6. ✅ 并行处理架构
7. ✅ 模型预加载机制
8. ✅ 线程安全优化（移除 os.chdir）
9. ✅ HLS 视频切片
10. ✅ 文件自动清理

**优化改进**：
1. 🔄 目录路径基于 __file__ 绝对路径
2. 🔄 subprocess.run(cwd=...) 替代 os.chdir()
3. 🔄 统一错误处理和重试机制
4. 🔄 完善 API 端点文档
5. 🔄 前端组件化重构

**Bug 修复**：
1. 🐛 修复并发路径冲突
2. 🐛 修复临时文件清理问题
3. 🐛 修复 CUDA 内存泄漏
4. 🐛 修复 MediaPipe 导入失败

---

**文档版本**: v2.0  
**更新日期**: 2026-04-17  
**维护者**: AI 数字人项目组
