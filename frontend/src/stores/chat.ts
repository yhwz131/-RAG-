import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { SessionInfo, ChatMessage, Reference } from '@/api'
import { listSessions, getSessionHistory, deleteSession, clearHistory } from '@/api'

/** 待发送的文件 */
export interface PendingFile {
  id: string
  file: File
  preview: string | null  // blob URL for images, null for documents
  base64: string          // pure base64, no prefix
  isImage: boolean
  fileName: string
}

export const useChatStore = defineStore('chat', () => {
  // 当前会话 ID
  const sessionId = ref<string>('')
  // 会话列表
  const sessions = ref<SessionInfo[]>([])
  // 当前对话消息
  const messages = ref<ChatMessage[]>([])
  // 当前回复的引用来源
  const currentReferences = ref<Reference[]>([])
  // 是否正在等待回复
  const loading = ref(false)
  // 检索模式
  const mode = ref<'text' | 'multimodal'>('text')
  // 侧边栏折叠状态
  const sidebarCollapsed = ref(false)
  // 待发送文件列表
  const pendingFiles = ref<PendingFile[]>([])

  // 加载会话列表
  async function loadSessions() {
    try {
      sessions.value = await listSessions()
    } catch {
      sessions.value = []
    }
  }

  // 切换到指定会话
  async function switchSession(sid: string) {
    sessionId.value = sid
    currentReferences.value = []
    try {
      const data = await getSessionHistory(sid)
      messages.value = data.messages || []
    } catch {
      messages.value = []
    }
  }

  // 新建会话
  function newSession() {
    const sid = crypto.randomUUID()
    sessionId.value = sid
    messages.value = []
    currentReferences.value = []
    return sid
  }

  // 删除会话
  async function removeSession(sid: string) {
    try {
      await deleteSession(sid)
    } catch { /* ignore */ }
    await loadSessions()
    // 如果删除的是当前会话，切换到第一个
    if (sid === sessionId.value) {
      if (sessions.value.length > 0) {
        await switchSession(sessions.value[0].session_id)
      } else {
        newSession()
      }
    }
  }

  // 清空当前会话历史
  async function clearCurrentHistory() {
    if (sessionId.value) {
      try {
        await clearHistory(sessionId.value)
      } catch { /* ignore */ }
    }
    messages.value = []
    currentReferences.value = []
  }

  // 添加待发送文件（图片或文档）
  function addPendingFiles(files: File[]) {
    const imageExts = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
    for (const file of files) {
      const isImage = imageExts.includes(file.type) || file.type.startsWith('image/')
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        const base64 = result.split(',')[1] || ''
        pendingFiles.value.push({
          id: crypto.randomUUID(),
          file,
          preview: isImage ? URL.createObjectURL(file) : null,
          base64,
          isImage,
          fileName: file.name,
        })
      }
      reader.readAsDataURL(file)
    }
  }

  // 移除待发送文件
  function removePendingFile(id: string) {
    const idx = pendingFiles.value.findIndex(p => p.id === id)
    if (idx !== -1) {
      if (pendingFiles.value[idx].preview) {
        URL.revokeObjectURL(pendingFiles.value[idx].preview!)
      }
      pendingFiles.value.splice(idx, 1)
    }
  }

  // 清空待发送文件
  function clearPendingFiles() {
    for (const p of pendingFiles.value) {
      if (p.preview) URL.revokeObjectURL(p.preview)
    }
    pendingFiles.value = []
  }

  return {
    sessionId,
    sessions,
    messages,
    currentReferences,
    loading,
    mode,
    sidebarCollapsed,
    pendingFiles,
    loadSessions,
    switchSession,
    newSession,
    removeSession,
    clearCurrentHistory,
    addPendingFiles,
    removePendingFile,
    clearPendingFiles,
  }
})
