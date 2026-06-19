import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

// ========== 对话相关 ==========

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface Reference {
  content: string
  source: string
  score: number
  page_number?: number
  has_image?: boolean
  image_url?: string
}

export interface ChatResponse {
  answer: string
  session_id: string
  references: Reference[]
}

export interface SessionInfo {
  session_id: string
  title: string
  updated_at: number
}

export async function chatNonStream(
  query: string,
  sessionId: string,
  mode: string = 'text'
): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>('/chat', {
    query,
    session_id: sessionId,
    stream: false,
    mode,
  })
  return data
}

export async function chatStream(
  query: string,
  sessionId: string,
  mode: string = 'text',
  onSources: (refs: Reference[]) => void,
  onToken: (token: string) => void,
  onDone: () => void,
  onError: (err: string) => void,
  images: string[] = [],
  files: { name: string; content: string }[] = []
): Promise<void> {
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        session_id: sessionId,
        stream: true,
        mode,
        images,
        files,
      }),
    })

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: `HTTP ${resp.status}` }))
      onError(err.detail || '请求失败')
      return
    }

    const reader = resp.body?.getReader()
    if (!reader) {
      onError('无法读取响应流')
      return
    }

    const decoder = new TextDecoder()
    let sourcesDone = false
    let sourcesBuffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })

      if (!sourcesDone) {
        sourcesBuffer += chunk
        if (sourcesBuffer.includes('[/SOURCES]')) {
          sourcesDone = true
          const markerEnd = sourcesBuffer.indexOf('[/SOURCES]')
          const sourcesJson = sourcesBuffer.slice('[SOURCES]'.length, markerEnd)
          const remaining = sourcesBuffer.slice(markerEnd + '[/SOURCES]'.length)

          try {
            const parsed = JSON.parse(sourcesJson)
            // 将流式 sources 映射为 Reference 格式
            const refs: Reference[] = (parsed.sources || []).map((s: any) => ({
              content: s.content_snippet || s.content || '',
              source: s.source || '未知',
              score: s.score || 0,
              page_number: s.page_number || 0,
              has_image: s.has_image || false,
              image_url: s.image_url || '',
            }))
            onSources(refs)
          } catch { /* ignore */ }

          if (remaining.trim()) {
            onToken(remaining)
          }
        }
      } else {
        onToken(chunk)
      }
    }

    onDone()
  } catch (err: any) {
    onError(err.message || '网络错误')
  }
}

export async function clearHistory(sessionId: string): Promise<void> {
  await api.post('/chat/clear', null, { params: { session_id: sessionId } })
}

export async function listSessions(): Promise<SessionInfo[]> {
  const { data } = await api.get('/chat/sessions')
  return data.sessions || []
}

export async function getSessionHistory(sessionId: string): Promise<{
  session_id: string
  title: string
  messages: ChatMessage[]
}> {
  const { data } = await api.get(`/chat/${sessionId}/history`)
  return data
}

export async function deleteSession(sessionId: string): Promise<void> {
  await api.delete(`/chat/${sessionId}`)
}

// ========== 文档相关 ==========

export interface DocStats {
  total_docs: number
  collection_name: string
}

export interface DocItem {
  filename: string
  chunk_count: number
}

export interface UploadResult {
  filename: string
  chunks: number
  message: string
}

export interface BatchUploadResult {
  total_files: number
  success_count: number
  fail_count: number
  results: UploadResult[]
  errors: string[]
}

export async function uploadDocument(file: File): Promise<UploadResult> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post('/docs/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
  })
  return data
}

export async function uploadDocuments(files: File[]): Promise<BatchUploadResult> {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }
  const { data } = await api.post('/docs/upload/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 600000,
  })
  return data
}

export async function getDocStats(): Promise<DocStats> {
  const { data } = await api.get('/docs/stats')
  return data
}

export async function listDocuments(): Promise<DocItem[]> {
  const { data } = await api.get('/docs/list')
  return data.documents || []
}

export async function deleteDocument(filename: string): Promise<void> {
  await api.delete(`/docs/delete/${encodeURIComponent(filename)}`)
}

export async function clearDocuments(): Promise<void> {
  await api.delete('/docs/clear')
}

// ========== 健康检查 ==========

export async function healthCheck(): Promise<{ status: string; uptime_seconds: number }> {
  const { data } = await api.get('/health')
  return data
}

// ========== 数据管线 ==========

export interface PipelineEngine {
  name: string
  label: string
  description: string
}

export interface PipelineTask {
  task_id: string
  status: string
  created_at: string
  completed_at?: string
  files_count: number
  chunks_count: number
  error?: string
}

export interface PipelineStatus {
  last_run: Record<string, any>
  milvus_stats: Record<string, any>
  database_status: Record<string, any>
}

export interface DatabaseStatus {
  connected: boolean
  type?: string
  host?: string
  database?: string
  table?: string
  error?: string
}

export interface DatabaseTableInfo {
  database: string
  tables: Array<{
    name: string
    row_count: number
    columns: Array<{ name: string; type: string }>
  }>
}

export async function getPipelineEngines(): Promise<PipelineEngine[]> {
  const { data } = await api.get('/pipeline/engines')
  return data.engines || []
}

export async function processFiles(
  files: File[],
  engine: string = 'simple'
): Promise<{ task_id: string; status: string; message: string }> {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }
  const { data } = await api.post(`/pipeline/process?engine=${engine}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 600000,
  })
  return data
}

export async function getTaskStatus(taskId: string): Promise<PipelineTask> {
  const { data } = await api.get(`/pipeline/tasks/${taskId}`)
  return data
}

export async function getAllTasks(): Promise<PipelineTask[]> {
  const { data } = await api.get('/pipeline/tasks')
  return data.tasks || []
}

export async function getPipelineStatus(): Promise<PipelineStatus> {
  const { data } = await api.get('/pipeline/status')
  return data
}

export async function getPipelineHistory(): Promise<Record<string, any>[]> {
  const { data } = await api.get('/pipeline/history')
  return data.runs || []
}

export async function getQualityReport(): Promise<Record<string, any>> {
  const { data } = await api.get('/pipeline/quality')
  return data
}

export async function getDatabaseStatus(): Promise<DatabaseStatus> {
  const { data } = await api.get('/pipeline/database/status')
  return data
}

export async function getDatabaseTables(): Promise<DatabaseTableInfo> {
  const { data } = await api.get('/pipeline/database/tables')
  return data
}

export async function testDatabaseConnection(): Promise<{ status: string; type: string }> {
  const { data } = await api.post('/pipeline/database/test')
  return data
}

export interface DatabaseConfig {
  db_type: string
  db_host: string
  db_port: number
  db_user: string
  db_password: string
  db_name: string
  db_table: string
  db_text_columns?: string[]
}

export async function getDatabaseConfig(): Promise<DatabaseConfig> {
  const { data } = await api.get('/pipeline/database/config')
  return data
}

export async function saveDatabaseConfig(config: DatabaseConfig): Promise<{ status: string; message: string }> {
  const { data } = await api.post('/pipeline/database/config', config)
  return data
}

export async function testDatabaseConfig(config: DatabaseConfig): Promise<{ status: string; type: string }> {
  const { data } = await api.post('/pipeline/database/test', config)
  return data
}

export async function importFromDatabase(
  query?: string
): Promise<{ status: string; total_rows: number; total_chunks: number; message: string }> {
  const { data } = await api.post('/pipeline/database/import', { query })
  return data
}
