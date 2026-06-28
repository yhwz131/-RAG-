---
name: multimodal-image
description: |
  多模态图片理解技能。当用户在 Copilot Chat 中粘贴/上传图片时，使用此技能调用 mimo-v2.5 模型来分析图片内容。
  触发条件：用户消息中包含图片附件（attachment），且当前基座模型不支持多模态输入时。
  典型场景：用户粘贴截图提问、上传图片要求分析、发送错误截图求助等。
metadata:
  openclaw:
    emoji: 🖼️
    requires:
      env:
        - LLM_API_KEY
    primaryEnv: LLM_API_KEY
  security:
    credentials_usage: |
      此技能仅将用户提供的图片和 API Key 发送至配置的 LLM API 端点（token-plan-cn.xiaomimimo.com），
      不会将凭据记录到日志或传输到其他目的地。
    allowed_domains:
      - token-plan-cn.xiaomimimo.com
      - '*.xiaomimimo.com'
---

# 多模态图片理解技能

> **触发机制**：主要通过 `copilot-instructions.md` 中的强制规则触发（每次对话自动加载）。
> 本文件作为详细参考文档，提供完整的执行流程和错误处理指南。

## 目的

当用户在 VS Code Copilot Chat 中粘贴或上传图片时，调用 `mimo-v2.5` 多模态大模型来理解图片内容，
然后将图片描述作为上下文传递给基座模型进行后续处理。

## ⛔ 触发条件

**当且仅当**满足以下条件时使用此技能：
1. 用户消息中包含图片附件（attachment / image file）
2. 当前基座模型不支持多模态输入

## 执行流程

### 步骤 1：检测图片

检查用户消息是否包含图片附件。VS Code Copilot Chat 中图片通常以以下形式出现：
- 附件列表中的 `image.png`、`image.jpg` 等
- 用户粘贴的截图
- 指向本地图片文件的路径

如果**没有图片**，跳过此技能，按正常流程处理。

### 步骤 2：调用多模态 API 分析图片

使用 Python 辅助脚本分析图片（**推荐，更健壮**）：

```bash
# 分析本地图片文件
python3 /home/yhwz/knowledge-qa-system/.github/skills/multimodal-image/image_describe.py "<图片文件路径>"
```

如果图片是通过 URL 提供的（如 `https://...`），使用：
```bash
python3 /home/yhwz/knowledge-qa-system/.github/skills/multimodal-image/image_describe.py --url "<图片URL>"
```

如果需要指定用户问题来引导图片分析（例如用户问"这个报错是什么意思"）：
```bash
python3 /home/yhwz/knowledge-qa-system/.github/skills/multimodal-image/image_describe.py "<图片文件路径>" "用户的具体问题"
```

> **备选方案**：如果 Python 不可用，可使用 Bash 脚本 `image_describe.sh`，用法相同。

### 步骤 3：整合结果

将脚本返回的图片描述整合到你的回复中：

1. **图片内容描述**：将 mimo-v2.5 返回的描述作为"图片分析"提供给用户
2. **结合用户问题**：根据图片描述和用户的问题，给出完整的回答
3. **如果图片包含代码/报错**：提取关键信息，给出解决方案

### 步骤 4：回复格式

回复时使用以下格式：

```
📸 **图片分析**：[mimo-v2.5 返回的图片描述]

[基于图片内容和用户问题的完整回答]
```

## ⚠️ 错误处理

| 错误场景 | 处理方式 |
|---------|---------|
| API Key 未配置 | 提示用户在 `.env` 中设置 `LLM_API_KEY` |
| API 调用超时 | 重试一次，仍失败则告知用户"图片分析服务暂时不可用" |
| 图片格式不支持 | 告知用户支持的格式：png, jpg, jpeg, gif, webp |
| 图片文件不存在 | 检查路径是否正确，提示用户重新上传 |

## 支持的图片格式

- PNG (.png)
- JPEG (.jpg, .jpeg)
- GIF (.gif)
- WebP (.webp)
- BMP (.bmp)

## API 配置

- **模型**: mimo-v2.5
- **API 端点**: https://token-plan-cn.xiaomimimo.com/v1/chat/completions
- **认证**: Bearer Token（从 `.env` 的 `LLM_API_KEY` 读取）
- **图片编码**: Base64（自动处理）
