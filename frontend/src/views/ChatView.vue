<template>
  <div class="chat-view">
    <!-- 左侧会话列表 -->
    <aside class="chat-sidebar" :class="{ collapsed: store.sidebarCollapsed }">
      <div class="sidebar-header">
        <el-button type="primary" class="new-chat-btn" @click="handleNewSession">
          <el-icon><Plus /></el-icon>
          <span v-if="!store.sidebarCollapsed">新建对话</span>
        </el-button>
        <el-button
          v-if="!store.sidebarCollapsed && store.sessions.length > 0"
          text
          size="small"
          @click="toggleBatchMode"
          :type="batchMode ? 'danger' : 'default'"
          class="batch-toggle-btn"
        >
          <el-icon><Edit /></el-icon>
          <span>{{ batchMode ? '取消' : '管理' }}</span>
        </el-button>
      </div>

      <!-- 批量操作栏 -->
      <div v-if="batchMode && !store.sidebarCollapsed" class="batch-bar">
        <el-checkbox
          v-model="allSelected"
          :indeterminate="isIndeterminate"
          @change="toggleSelectAll"
        >
          全选 ({{ selectedSessions.size }}/{{ store.sessions.length }})
        </el-checkbox>
        <el-button
          type="danger"
          size="small"
          :disabled="selectedSessions.size === 0"
          @click="handleBatchDelete"
        >
          删除 ({{ selectedSessions.size }})
        </el-button>
      </div>

      <div class="session-list" v-if="!store.sidebarCollapsed">
        <div
          v-for="s in store.sessions"
          :key="s.session_id"
          class="session-item"
          :class="{ active: s.session_id === store.sessionId }"
          @click="batchMode ? toggleSelect(s.session_id) : handleSwitchSession(s.session_id)"
        >
          <el-checkbox
            v-if="batchMode"
            :model-value="selectedSessions.has(s.session_id)"
            @click.stop
            @change="toggleSelect(s.session_id)"
            class="session-checkbox"
          />
          <el-icon class="session-icon"><ChatLineSquare /></el-icon>
          <span class="session-title">{{ s.title || '新对话' }}</span>
          <el-icon
            v-if="!batchMode"
            class="delete-icon"
            @click.stop="handleDeleteSession(s.session_id)"
          >
            <Delete />
          </el-icon>
        </div>
        <div v-if="store.sessions.length === 0" class="empty-sessions">
          <span>暂无历史会话</span>
        </div>
      </div>

      <div class="sidebar-toggle" @click="store.sidebarCollapsed = !store.sidebarCollapsed">
        <el-icon>
          <DArrowLeft v-if="!store.sidebarCollapsed" />
          <DArrowRight v-else />
        </el-icon>
      </div>
    </aside>

    <!-- 右侧聊天区 -->
    <div class="chat-main">
      <!-- 消息列表 -->
      <div class="messages-container" ref="messagesRef">
        <!-- 欢迎页 -->
        <div v-if="store.messages.length === 0 && !store.loading" class="welcome-screen">
          <div class="welcome-icon">🤖</div>
          <h2>欢迎使用知识问答系统</h2>
          <p>基于 RAG 检索增强生成，为您提供精准的知识问答服务</p>
          <div class="welcome-hints">
            <div class="hint-card" @click="sendQuickQuestion('系统有哪些功能？')">
              <el-icon><Promotion /></el-icon>
              <span>系统有哪些功能？</span>
            </div>
            <div class="hint-card" @click="sendQuickQuestion('请介绍一下本项目的系统架构')">
              <el-icon><Connection /></el-icon>
              <span>请介绍一下本项目的系统架构</span>
            </div>
          </div>
        </div>

        <!-- 消息气泡 -->
        <div
          v-for="(msg, idx) in store.messages"
          :key="idx"
          class="message-row"
          :class="msg.role"
        >
          <div class="message-avatar">
            <el-icon v-if="msg.role === 'user'" :size="20"><User /></el-icon>
            <span v-else class="bot-avatar">🤖</span>
          </div>
          <div class="message-content">
            <div class="message-bubble" :class="msg.role">
              <div
                v-if="msg.role === 'assistant'"
                class="markdown-body"
                v-html="renderMarkdown(msg.content)"
              ></div>
              <div v-else class="user-text">{{ msg.content }}</div>
            </div>

            <!-- 引用来源（仅最后一条 assistant 消息显示） -->
            <div
              v-if="msg.role === 'assistant' && idx === store.messages.length - 1 && store.currentReferences.length > 0"
              class="references"
            >
              <div class="references-header">
                <el-icon><Collection /></el-icon>
                <span>参考来源</span>
              </div>
              <div class="references-list">
                <div
                  v-for="(ref, ri) in store.currentReferences"
                  :key="ri"
                  class="ref-item"
                >
                  <div class="ref-header">
                    <span class="ref-source">{{ ref.source }}</span>
                    <el-tag v-if="ref.page_number" size="small" type="info">P{{ ref.page_number }}</el-tag>
                    <el-tag size="small" effect="plain">{{ (ref.score * 100).toFixed(1) }}%</el-tag>
                  </div>
                  <div class="ref-content">{{ ref.content?.slice(0, 120) }}...</div>
                  <img
                    v-if="ref.has_image && ref.image_url"
                    :src="toImageUrl(ref.image_url)"
                    class="ref-image"
                    alt="参考图片"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 加载状态 -->
        <div v-if="store.loading && !streamingContent" class="message-row assistant">
          <div class="message-avatar"><span class="bot-avatar">🤖</span></div>
          <div class="message-content">
            <div class="message-bubble assistant loading-bubble">
              <div class="typing-indicator">
                <span></span><span></span><span></span>
              </div>
              <span class="loading-text">正在检索相关文档...</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 底部输入区 -->
      <div
        class="input-area"
        @dragover.prevent="isDragging = true"
        @dragleave.prevent="isDragging = false"
        @drop.prevent="handleDrop"
      >
        <!-- 拖拽提示覆盖层 -->
        <Transition name="fade">
          <div v-if="isDragging" class="drop-overlay">
            <el-icon :size="40"><UploadFilled /></el-icon>
            <span>松开鼠标上传文件</span>
          </div>
        </Transition>
        <!-- 文件预览区 -->
        <div v-if="store.pendingFiles.length > 0" class="file-preview-area">
          <div v-for="f in store.pendingFiles" :key="f.id" class="file-preview-item" :class="{ 'is-image': f.isImage }">
            <img v-if="f.isImage && f.preview" :src="f.preview" class="preview-thumb" />
            <div v-else class="preview-doc">
              <el-icon :size="28"><Document /></el-icon>
              <span class="doc-name">{{ f.fileName }}</span>
            </div>
            <el-icon class="preview-remove" @click="store.removePendingFile(f.id)"><CloseBold /></el-icon>
          </div>
        </div>
        <div class="input-toolbar">
          <div class="toolbar-left">
            <el-segmented v-model="store.mode" :options="modeOptions" size="small" />
            <el-button text size="small" @click="triggerFileUpload" :disabled="store.loading" title="上传文件">
              <el-icon><UploadFilled /></el-icon>
            </el-button>
          </div>
          <el-button
            text
            size="small"
            @click="handleClearHistory"
            :disabled="store.messages.length === 0"
          >
            <el-icon><Delete /></el-icon>
            清空对话
          </el-button>
        </div>
        <div class="input-row">
          <el-input
            v-model="inputText"
            type="textarea"
            :autosize="{ minRows: 1, maxRows: 5 }"
            placeholder="输入你的问题... (Enter 发送, Shift+Enter 换行)"
            @keydown="handleKeydown"
            :disabled="store.loading"
            resize="none"
            class="chat-input"
          />
          <el-button
            type="primary"
            :icon="Promotion"
            circle
            size="large"
            :loading="store.loading"
            :disabled="(!inputText.trim() && store.pendingFiles.length === 0) || store.loading"
            @click="sendMessage"
            class="send-btn"
          />
        </div>
        <!-- 隐藏的文件输入 -->
        <input
          ref="fileInputRef"
          type="file"
          accept="image/*,.pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.txt,.md,.csv"
          multiple
          style="display: none"
          @change="handleFileSelect"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Delete, Promotion, Connection, User, Collection, ChatLineSquare, DArrowLeft, DArrowRight, Picture, CloseBold, Document, UploadFilled, Edit } from '@element-plus/icons-vue'
import { marked } from 'marked'
import { useChatStore } from '@/stores/chat'
import { chatStream } from '@/api'

const store = useChatStore()
const route = useRoute()
const router = useRouter()

const inputText = ref('')
const messagesRef = ref<HTMLElement>()
const streamingContent = ref('')
const fileInputRef = ref<HTMLInputElement>()
const isDragging = ref(false)

// 批量选择模式
const batchMode = ref(false)
const selectedSessions = ref<Set<string>>(new Set())
const allSelected = computed(() => store.sessions.length > 0 && selectedSessions.value.size === store.sessions.length)
const isIndeterminate = computed(() => selectedSessions.value.size > 0 && selectedSessions.value.size < store.sessions.length)

const modeOptions = [
  { label: '📄 纯文本', value: 'text' },
  { label: '🖼️ 多模态', value: 'multimodal' },
]

// Markdown 渲染
function renderMarkdown(text: string): string {
  return marked.parse(text, { breaks: true }) as string
}

// 图片 URL 转换
function toImageUrl(path: string): string {
  if (path.startsWith('http')) return path
  const relative = path.replace(/.*\/data\/raw\//, '')
  return `/static/${relative}`
}

// 滚动到底部
function scrollToBottom() {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  })
}

// 打开文件选择
function triggerFileUpload() {
  fileInputRef.value?.click()
}

// 处理文件选择
function handleFileSelect(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files) {
    store.addPendingFiles(Array.from(input.files))
    input.value = '' // 重置，允许重复选同一文件
  }
}

// 处理拖拽上传
function handleDrop(e: DragEvent) {
  isDragging.value = false
  if (e.dataTransfer?.files?.length) {
    store.addPendingFiles(Array.from(e.dataTransfer.files))
  }
}

// 发送消息
async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || store.loading) return

  // 确保有会话 ID
  if (!store.sessionId) {
    store.newSession()
  }

  // 分离图片和文档
  const imageFiles = store.pendingFiles.filter(f => f.isImage)
  const docFiles = store.pendingFiles.filter(f => !f.isImage)
  const imageBase64s = imageFiles.map(p => p.base64)
  const fileMetas = docFiles.map(f => ({
    name: f.fileName,
    content: f.base64,
  }))

  // 添加用户消息（附带文件数量标记）
  let displayText = text
  const parts: string[] = []
  if (imageBase64s.length > 0) parts.push(`${imageBase64s.length}张图片`)
  if (docFiles.length > 0) parts.push(`${docFiles.length}个文件`)
  if (parts.length > 0) {
    displayText = `${text} [${parts.join('，')}]`
  }
  store.messages.push({ role: 'user', content: displayText })
  inputText.value = ''
  store.loading = true
  store.currentReferences = []
  streamingContent.value = ''
  scrollToBottom()

  // 添加空的 assistant 消息用于流式填充
  store.messages.push({ role: 'assistant', content: '' })

  await chatStream(
    text,
    store.sessionId,
    store.mode,
    // onSources
    (refs) => {
      store.currentReferences = refs
    },
    // onToken
    (token) => {
      streamingContent.value += token
      store.messages[store.messages.length - 1].content = streamingContent.value
      scrollToBottom()
    },
    // onDone
    async () => {
      store.loading = false
      streamingContent.value = ''
      store.clearPendingFiles()
      await store.loadSessions()
      // 更新 URL
      router.replace(`/chat/${store.sessionId}`)
    },
    // onError
    (err) => {
      store.loading = false
      streamingContent.value = ''
      store.clearPendingFiles()
      store.messages[store.messages.length - 1].content = `❌ ${err}`
    },
    // images
    imageBase64s,
    // files
    fileMetas
  )
}

// 快捷提问
function sendQuickQuestion(q: string) {
  inputText.value = q
  sendMessage()
}

// 快捷键
function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

// 新建会话
function handleNewSession() {
  store.newSession()
  router.push('/chat')
}

// 切换会话
async function handleSwitchSession(sid: string) {
  await store.switchSession(sid)
  router.push(`/chat/${sid}`)
  scrollToBottom()
}

// 删除会话
async function handleDeleteSession(sid: string) {
  try {
    await ElMessageBox.confirm('确定要删除这个会话吗？', '提示', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning',
    })
    await store.removeSession(sid)
    ElMessage.success('会话已删除')
  } catch { /* 取消 */ }
}

// 批量选择相关
function toggleBatchMode() {
  batchMode.value = !batchMode.value
  if (!batchMode.value) {
    selectedSessions.value.clear()
  }
}

function toggleSelect(sid: string) {
  if (selectedSessions.value.has(sid)) {
    selectedSessions.value.delete(sid)
  } else {
    selectedSessions.value.add(sid)
  }
  // 触发响应式更新
  selectedSessions.value = new Set(selectedSessions.value)
}

function toggleSelectAll(val: boolean) {
  if (val) {
    selectedSessions.value = new Set(store.sessions.map(s => s.session_id))
  } else {
    selectedSessions.value.clear()
    selectedSessions.value = new Set()
  }
}

async function handleBatchDelete() {
  const count = selectedSessions.value.size
  if (count === 0) return
  try {
    await ElMessageBox.confirm(`确定要删除选中的 ${count} 个会话吗？`, '批量删除', {
      confirmButtonText: `删除 ${count} 个`,
      cancelButtonText: '取消',
      type: 'warning',
    })
    let deleted = 0
    for (const sid of selectedSessions.value) {
      try {
        await store.removeSession(sid)
        deleted++
      } catch { /* ignore */ }
    }
    selectedSessions.value.clear()
    selectedSessions.value = new Set()
    batchMode.value = false
    ElMessage.success(`已删除 ${deleted} 个会话`)
  } catch { /* 取消 */ }
}

// 清空历史
async function handleClearHistory() {
  try {
    await ElMessageBox.confirm('确定要清空当前对话历史吗？', '提示', {
      confirmButtonText: '清空',
      cancelButtonText: '取消',
      type: 'warning',
    })
    await store.clearCurrentHistory()
    ElMessage.success('对话已清空')
  } catch { /* 取消 */ }
}

// 初始化
onMounted(async () => {
  await store.loadSessions()

  // 从 URL 恢复会话
  const sid = route.params.sessionId as string
  if (sid) {
    await store.switchSession(sid)
  } else if (store.sessions.length > 0) {
    await store.switchSession(store.sessions[0].session_id)
    router.replace(`/chat/${store.sessionId}`)
  } else {
    store.newSession()
  }

  scrollToBottom()
})

// 监听路由变化
watch(() => route.params.sessionId, async (sid) => {
  if (sid && sid !== store.sessionId) {
    await store.switchSession(sid as string)
    scrollToBottom()
  }
})
</script>

<style scoped>
.chat-view {
  display: flex;
  height: 100%;
  overflow: hidden;
}

/* ========== 侧边栏 ========== */
.chat-sidebar {
  width: var(--sidebar-width);
  background: var(--bg-darker);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  transition: width 0.3s;
  flex-shrink: 0;
  position: relative;
}

.chat-sidebar.collapsed {
  width: 50px;
}

.sidebar-header {
  padding: 12px;
}

.new-chat-btn {
  width: 100%;
}

.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 0 8px;
}

.session-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
  margin-bottom: 2px;
  color: var(--text-secondary);
}

.session-item:hover {
  background: rgba(255, 255, 255, 0.06);
  color: var(--text-primary);
}

.session-item.active {
  background: rgba(64, 158, 255, 0.15);
  color: var(--primary-color);
}

.session-icon {
  flex-shrink: 0;
}

.session-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
}

.delete-icon {
  opacity: 0;
  transition: opacity 0.2s;
  flex-shrink: 0;
  color: #f56c6c;
}

.session-item:hover .delete-icon {
  opacity: 1;
}

/* 批量选择模式 */
.batch-toggle-btn {
  margin-left: 4px;
  font-size: 12px;
}

.batch-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-card);
}

.batch-bar .el-checkbox {
  font-size: 12px;
}

.session-checkbox {
  margin-right: 4px;
}

.session-item:has(.session-checkbox) {
  cursor: pointer;
}

.empty-sessions {
  text-align: center;
  color: var(--text-secondary);
  font-size: 13px;
  padding: 20px;
}

.sidebar-toggle {
  position: absolute;
  right: -14px;
  top: 50%;
  transform: translateY(-50%);
  width: 28px;
  height: 28px;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  z-index: 10;
  color: var(--text-secondary);
  transition: all 0.2s;
}

.sidebar-toggle:hover {
  color: var(--primary-color);
  border-color: var(--primary-color);
}

/* ========== 聊天主区 ========== */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  scroll-behavior: smooth;
}

/* ========== 欢迎页 ========== */
.welcome-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  color: var(--text-secondary);
}

.welcome-icon {
  font-size: 64px;
  margin-bottom: 16px;
}

.welcome-screen h2 {
  font-size: 22px;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.welcome-screen p {
  font-size: 14px;
  margin-bottom: 32px;
}

.welcome-hints {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.hint-card {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 20px;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s;
  color: var(--text-secondary);
  font-size: 14px;
}

.hint-card:hover {
  border-color: var(--primary-color);
  color: var(--primary-color);
  background: rgba(64, 158, 255, 0.08);
}

/* ========== 消息气泡 ========== */
.message-row {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  max-width: 900px;
  margin-left: auto;
  margin-right: auto;
}

.message-row.user {
  flex-direction: row-reverse;
}

.message-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.message-row.user .message-avatar {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

.message-row.assistant .message-avatar {
  background: linear-gradient(135deg, #409eff, #53a8ff);
}

.bot-avatar {
  font-size: 18px;
}

.message-content {
  flex: 1;
  min-width: 0;
  max-width: calc(100% - 60px);
}

.message-bubble {
  padding: 12px 16px;
  border-radius: 16px;
  line-height: 1.6;
  word-wrap: break-word;
}

.message-bubble.user {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  border-bottom-right-radius: 4px;
}

.message-bubble.assistant {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-bottom-left-radius: 4px;
}

.user-text {
  white-space: pre-wrap;
}

/* ========== 加载动画 ========== */
.loading-bubble {
  display: flex;
  align-items: center;
  gap: 12px;
}

.typing-indicator {
  display: flex;
  gap: 4px;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  background: var(--primary-color);
  border-radius: 50%;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
  40% { transform: scale(1); opacity: 1; }
}

.loading-text {
  color: var(--text-secondary);
  font-size: 13px;
}

/* ========== 引用来源 ========== */
.references {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border-color);
}

.references-header {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 8px;
}

.references-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.ref-item {
  padding: 10px 12px;
  background: rgba(0, 0, 0, 0.15);
  border-radius: 8px;
  font-size: 12px;
}

.ref-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.ref-source {
  font-weight: 600;
  color: var(--primary-color);
}

.ref-content {
  color: var(--text-secondary);
  line-height: 1.5;
}

.ref-image {
  max-width: 300px;
  margin-top: 8px;
  border-radius: 6px;
}

/* ========== 输入区 ========== */
.input-area {
  position: relative;
  padding: 12px 20px 16px;
  border-top: 1px solid var(--border-color);
  background: var(--bg-darker);
  max-width: 900px;
  margin: 0 auto;
  width: 100%;
}

/* ========== 文件预览 ========== */
.file-preview-area {
  display: flex;
  gap: 8px;
  padding: 8px 0;
  flex-wrap: wrap;
}

.file-preview-item {
  position: relative;
  width: 64px;
  height: 64px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--border-color);
}

.file-preview-item.is-image {
  border-color: var(--el-color-primary-light-5);
}

.preview-thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.preview-doc {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: var(--el-fill-color-light);
  color: var(--el-text-color-regular);
  gap: 2px;
  padding: 4px;
}

.doc-name {
  font-size: 9px;
  text-align: center;
  line-height: 1.1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  width: 100%;
  max-width: 58px;
}

.preview-remove {
  position: absolute;
  top: 2px;
  right: 2px;
  width: 18px;
  height: 18px;
  background: rgba(0, 0, 0, 0.6);
  border-radius: 50%;
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  transition: background 0.2s;
}

.preview-remove:hover {
  background: #f56c6c;
}

/* ========== 拖拽上传覆盖层 ========== */
.drop-overlay {
  position: absolute;
  inset: 0;
  z-index: 10;
  background: rgba(var(--el-color-primary-rgb), 0.08);
  border: 2px dashed var(--el-color-primary);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--el-color-primary);
  font-size: 15px;
  pointer-events: none;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.toolbar-left {
  display: flex;
  align-items: center;
  gap: 4px;
}

.input-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}

.input-row {
  display: flex;
  gap: 12px;
  align-items: flex-end;
}

.chat-input {
  flex: 1;
}

.chat-input :deep(.el-textarea__inner) {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  border-radius: 12px;
  padding: 10px 16px;
  font-size: 14px;
}

.chat-input :deep(.el-textarea__inner):focus {
  border-color: var(--primary-color);
}

.send-btn {
  flex-shrink: 0;
  width: 44px;
  height: 44px;
}
</style>
