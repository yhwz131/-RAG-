<template>
  <div class="docs-view">
    <!-- 左侧：文档统计与上传 -->
    <div class="docs-left">
      <div class="stats-card">
        <h3>📊 知识库统计</h3>
        <div class="stat-items">
          <div class="stat-item">
            <span class="stat-value">{{ stats.total_docs }}</span>
            <span class="stat-label">文档片段</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">{{ documents.length }}</span>
            <span class="stat-label">已上传文件</span>
          </div>
        </div>
        <div class="stat-collection">
          <el-tag type="info" size="small">{{ stats.collection_name || 'knowledge_base' }}</el-tag>
        </div>
      </div>

      <!-- 管线统计（可折叠） -->
      <div class="pipeline-card">
        <div class="pipeline-header" @click="showPipeline = !showPipeline">
          <h3>📈 处理管线</h3>
          <el-icon :class="{ 'rotate-icon': showPipeline }"><ArrowDown /></el-icon>
        </div>
        <div v-show="showPipeline" class="pipeline-body">
          <div class="pipeline-stats">
            <div class="pipeline-stat">
              <span class="pstat-value">{{ pipelineStatus.last_run?.files_scanned || 0 }}</span>
              <span class="pstat-label">处理文件</span>
            </div>
            <div class="pipeline-stat">
              <span class="pstat-value">{{ pipelineStatus.last_run?.total_chunks || 0 }}</span>
              <span class="pstat-label">生成切片</span>
            </div>
            <div class="pipeline-stat">
              <span class="pstat-value">{{ pipelineStatus.milvus_stats?.row_count || 0 }}</span>
              <span class="pstat-label">已入库</span>
            </div>
            <div class="pipeline-stat">
              <span class="pstat-value fail">{{ pipelineStatus.last_run?.files_failed || 0 }}</span>
              <span class="pstat-label">失败</span>
            </div>
          </div>
          <div class="pipeline-info">
            <div class="pinfo-row">
              <span class="pinfo-label">引擎</span>
              <el-tag size="small">{{ pipelineStatus.last_run?.engine || '-' }}</el-tag>
            </div>
            <div class="pinfo-row">
              <span class="pinfo-label">最近运行</span>
              <span class="pinfo-value">{{ formatTime(pipelineStatus.last_run?.timestamp) }}</span>
            </div>
            <div class="pinfo-row" v-if="dbStatus.connected">
              <span class="pinfo-label">数据库</span>
              <el-tag type="success" size="small">{{ dbStatus.type?.toUpperCase() }}</el-tag>
            </div>
          </div>
          <!-- 处理历史 -->
          <div v-if="pipelineHistory.length > 0" class="pipeline-history">
            <div class="phist-title">处理历史</div>
            <div v-for="run in pipelineHistory.slice(0, 5)" :key="run.run_id" class="phist-item">
              <el-tag :type="run.success ? 'success' : 'danger'" size="small">{{ run.engine }}</el-tag>
              <span class="phist-info">{{ run.total_files }}文件 / {{ run.total_chunks }}切片</span>
              <span class="phist-time">{{ formatTime(run.timestamp) }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="upload-card">
        <h3>📤 上传文档</h3>
        <el-upload
          ref="uploadRef"
          :auto-upload="false"
          multiple
          :on-change="handleFileChange"
          :file-list="fileList"
          drag
          class="upload-area"
        >
          <el-icon class="upload-icon"><UploadFilled /></el-icon>
          <div class="upload-text">拖拽文件到此处，或 <em>点击选择</em></div>
          <div class="upload-hint">支持批量选择，格式: PDF, DOCX, PPTX, TXT, MD, CSV, XLSX</div>
        </el-upload>

        <!-- 处理模式选择 -->
        <div class="mode-selector">
          <div class="mode-label">处理模式</div>
          <div class="mode-options">
            <label
              class="mode-option"
              :class="{ active: processingMode === 'quick' }"
              @click="processingMode = 'quick'"
            >
              <input type="radio" value="quick" v-model="processingMode" class="mode-radio-input" />
              <span class="mode-radio-dot"></span>
              <div class="mode-info">
                <span class="mode-name">快速入库（推荐）</span>
                <span class="mode-desc">直接解析入库，几秒完成。适合日常补充文档。</span>
              </div>
            </label>
            <label
              class="mode-option"
              :class="{ active: processingMode === 'batch' }"
              @click="processingMode = 'batch'"
            >
              <input type="radio" value="batch" v-model="processingMode" class="mode-radio-input" />
              <span class="mode-radio-dot"></span>
              <div class="mode-info">
                <span class="mode-name">批量入库</span>
                <span class="mode-desc">批量清洗、去重、统计后自动入库。处理时间较长。</span>
              </div>
            </label>
            <label
              v-if="dbStatus.connected"
              class="mode-option"
              :class="{ active: processingMode === 'database' }"
              @click="processingMode = 'database'"
            >
              <input type="radio" value="database" v-model="processingMode" class="mode-radio-input" />
              <span class="mode-radio-dot"></span>
              <div class="mode-info">
                <span class="mode-name">数据库导入</span>
                <span class="mode-desc">从 {{ dbStatus.type?.toUpperCase() }} 数据库导入数据。</span>
              </div>
            </label>
          </div>
        </div>

        <!-- 已选文件列表 -->
        <div v-if="fileList.length > 0" class="selected-files">
          <div class="selected-header">
            <span>已选 {{ fileList.length }} 个文件</span>
            <el-button text size="small" @click="clearAllFiles">清空</el-button>
          </div>
          <div class="file-chips">
            <el-tag
              v-for="(f, idx) in fileList"
              :key="idx"
              closable
              size="small"
              @close="removeFile(idx)"
              class="file-chip"
            >
              {{ f.name }}
            </el-tag>
          </div>
        </div>

        <!-- 上传进度 -->
        <div v-if="uploading" class="upload-progress">
          <div class="progress-text">
            上传中 {{ uploadDone }} / {{ uploadTotal }}...
          </div>
          <el-progress
            :percentage="Math.round((uploadDone / uploadTotal) * 100)"
            :stroke-width="8"
            :format="() => `${uploadDone}/${uploadTotal}`"
          />
        </div>

        <!-- 上传结果 -->
        <div v-if="uploadResult" class="upload-result" :class="uploadResult.fail_count > 0 ? 'has-error' : 'all-ok'">
          <div v-if="uploadResult.success_count > 0" class="result-ok">
            ✅ 成功 {{ uploadResult.success_count }} 个，共 {{ totalChunks }} 个片段
          </div>
          <div v-if="uploadResult.fail_count > 0" class="result-err">
            ❌ 失败 {{ uploadResult.fail_count }} 个
            <div v-for="err in uploadResult.errors" :key="err" class="err-detail">{{ err }}</div>
          </div>
        </div>

        <el-button
          type="primary"
          :loading="uploading"
          :disabled="fileList.length === 0 && processingMode !== 'database'"
          @click="handleUpload"
          style="width: 100%; margin-top: 12px;"
        >
          <el-icon><Upload /></el-icon>
          {{ getButtonText() }}
        </el-button>
      </div>
    </div>

    <!-- 右侧：文档列表 -->
    <div class="docs-right">
      <div class="docs-list-header">
        <h3>📁 文档列表</h3>
        <div class="docs-actions">
          <el-button size="small" @click="loadData">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
          <el-button size="small" type="danger" plain @click="handleClearAll" :disabled="documents.length === 0">
            <el-icon><Delete /></el-icon>
            清空全部
          </el-button>
        </div>
      </div>

      <div class="docs-list" v-loading="loading">
        <el-empty v-if="documents.length === 0" description="暂无文档，请上传文件" />
        <div
          v-for="doc in documents"
          :key="doc.filename"
          class="doc-item"
        >
          <div class="doc-icon">
            <el-icon :size="28" :style="{ color: getFileColor(doc.filename) }">
              <Document />
            </el-icon>
          </div>
          <div class="doc-info">
            <div class="doc-name">{{ doc.filename }}</div>
            <div class="doc-meta">{{ doc.chunk_count }} 个片段</div>
          </div>
          <el-button
            type="danger"
            text
            @click="handleDeleteDoc(doc.filename)"
            class="doc-delete"
          >
            <el-icon><Delete /></el-icon>
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { UploadFilled, Upload, Refresh, Delete, Document, ArrowDown } from '@element-plus/icons-vue'
import type { UploadFile } from 'element-plus'
import {
  getDocStats, listDocuments, uploadDocuments, deleteDocument, clearDocuments,
  processFiles, importFromDatabase, getDatabaseStatus,
  getPipelineStatus, getPipelineHistory,
} from '@/api'
import type { DocStats, DocItem, BatchUploadResult, DatabaseStatus, PipelineStatus } from '@/api'

const stats = ref<DocStats>({ total_docs: 0, collection_name: '' })
const documents = ref<DocItem[]>([])
const loading = ref(false)
const uploading = ref(false)
const fileList = ref<File[]>([])
const uploadRef = ref()
const uploadResult = ref<BatchUploadResult | null>(null)
const uploadDone = ref(0)
const uploadTotal = ref(0)
const processingMode = ref<'quick' | 'batch' | 'database'>('quick')
const dbStatus = ref<DatabaseStatus>({ connected: false })
const showPipeline = ref(false)
const pipelineStatus = ref<PipelineStatus>({ last_run: {}, milvus_stats: {}, database_status: {} })
const pipelineHistory = ref<Record<string, any>[]>([])

const totalChunks = computed(() =>
  uploadResult.value?.results.reduce((sum, r) => sum + r.chunks, 0) ?? 0
)

// 加载数据
async function loadData() {
  loading.value = true
  try {
    const [s, docs, db] = await Promise.all([getDocStats(), listDocuments(), getDatabaseStatus()])
    stats.value = s
    documents.value = docs
    dbStatus.value = db
  } catch (err: any) {
    ElMessage.error('加载失败: ' + (err.message || '未知错误'))
  } finally {
    loading.value = false
  }
}

// 加载管线数据
async function loadPipelineData() {
  try {
    const [s, h] = await Promise.all([getPipelineStatus(), getPipelineHistory()])
    pipelineStatus.value = s
    pipelineHistory.value = (h || []).reverse()
  } catch { /* 静默 */ }
}

// 格式化时间
function formatTime(timestamp: string | undefined): string {
  if (!timestamp) return '-'
  try {
    const date = new Date(timestamp)
    return `${date.getMonth() + 1}-${date.getDate()} ${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`
  } catch {
    return timestamp
  }
}

// 获取按钮文本
function getButtonText(): string {
  if (uploading.value) return '处理中...'
  if (processingMode.value === 'database') return '从数据库导入'
  if (processingMode.value === 'batch') return `批量入库 (${fileList.value.length} 个文件)`
  return `快速入库 (${fileList.value.length} 个文件)`
}

// 文件选择（每次 onChange 都把最新文件列表同步过来）
function handleFileChange(_file: UploadFile, files: UploadFile[]) {
  fileList.value = files.map(f => f.raw!).filter(Boolean)
  uploadResult.value = null
}

// 移除单个文件
function removeFile(index: number) {
  fileList.value.splice(index, 1)
  uploadRef.value?.handleRemove?.(index)
}

// 清空所有文件
function clearAllFiles() {
  fileList.value = []
  uploadRef.value?.clearFiles()
  uploadResult.value = null
}

// 批量上传
async function handleUpload() {
  if (processingMode.value === 'database') {
    await handleDatabaseImport()
    return
  }

  if (fileList.value.length === 0) return

  uploading.value = true
  uploadResult.value = null
  uploadTotal.value = fileList.value.length
  uploadDone.value = 0

  try {
    if (processingMode.value === 'batch') {
      // 大数据处理模式：调用管线 API
      const result = await processFiles(fileList.value, 'spark')
      ElMessage.success(`处理任务已提交（ID: ${result.task_id}）`)
      // 刷新管线统计
      await loadPipelineData()
      showPipeline.value = true
    } else {
      // 快速入库模式：调用现有批量上传 API
      const result = await uploadDocuments(fileList.value)
      uploadResult.value = result
      uploadDone.value = result.total_files

      if (result.fail_count === 0) {
        ElMessage.success(`全部 ${result.success_count} 个文件上传成功！`)
      } else {
        ElMessage.warning(`${result.success_count} 个成功，${result.fail_count} 个失败`)
      }
    }

    // 清空选择
    fileList.value = []
    uploadRef.value?.clearFiles()
    await loadData()
  } catch (err: any) {
    const detail = err.response?.data?.detail || err.message || '上传失败'
    ElMessage.error(detail)
  } finally {
    uploading.value = false
  }
}

// 数据库导入
async function handleDatabaseImport() {
  uploading.value = true
  try {
    const result = await importFromDatabase()
    ElMessage.success(result.message)
    await loadData()
  } catch (err: any) {
    const detail = err.response?.data?.detail || err.message || '数据库导入失败'
    ElMessage.error(detail)
  } finally {
    uploading.value = false
  }
}

// 删除单个文档
async function handleDeleteDoc(filename: string) {
  try {
    await ElMessageBox.confirm(`确定要删除文档 "${filename}" 吗？`, '提示', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning',
    })
    await deleteDocument(filename)
    ElMessage.success('文档已删除')
    await loadData()
  } catch { /* 取消 */ }
}

// 清空全部
async function handleClearAll() {
  try {
    await ElMessageBox.confirm('确定要清空所有文档吗？此操作不可恢复！', '警告', {
      confirmButtonText: '确定清空',
      cancelButtonText: '取消',
      type: 'warning',
    })
    await clearDocuments()
    ElMessage.success('文档库已清空')
    await loadData()
  } catch { /* 取消 */ }
}

// 文件图标颜色
function getFileColor(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase()
  const colors: Record<string, string> = {
    pdf: '#e74c3c',
    docx: '#2980b9',
    doc: '#2980b9',
    pptx: '#e67e22',
    ppt: '#e67e22',
    xlsx: '#27ae60',
    xls: '#27ae60',
    txt: '#95a5a6',
    md: '#3498db',
    csv: '#27ae60',
  }
  return colors[ext || ''] || '#909399'
}

onMounted(() => {
  loadData()
  loadPipelineData()
})
</script>

<style scoped>
.docs-view {
  display: flex;
  height: 100%;
  overflow: hidden;
  gap: 0;
}

.docs-left {
  width: 380px;
  padding: 20px;
  border-right: 1px solid var(--border-color);
  overflow-y: auto;
  background: var(--bg-darker);
  flex-shrink: 0;
}

.docs-right {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ========== 统计卡片 ========== */
.stats-card {
  background: var(--bg-card);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
  border: 1px solid var(--border-color);
}

.stats-card h3 {
  font-size: 15px;
  margin-bottom: 16px;
  color: var(--text-primary);
}

.stat-items {
  display: flex;
  gap: 20px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: var(--primary-color);
}

.stat-label {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 4px;
}

.stat-collection {
  margin-top: 12px;
}

/* ========== 上传卡片 ========== */
.upload-card {
  background: var(--bg-card);
  border-radius: 12px;
  padding: 20px;
  border: 1px solid var(--border-color);
}

.upload-card h3 {
  font-size: 15px;
  margin-bottom: 16px;
  color: var(--text-primary);
}

.upload-area {
  width: 100%;
}

.upload-area :deep(.el-upload-dragger) {
  background: var(--bg-dark);
  border-color: var(--border-color);
  border-radius: 10px;
  padding: 24px;
}

.upload-area :deep(.el-upload-dragger:hover) {
  border-color: var(--primary-color);
}

.upload-icon {
  font-size: 40px;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.upload-text {
  color: var(--text-secondary);
  font-size: 14px;
}

.upload-text em {
  color: var(--primary-color);
  font-style: normal;
}

.upload-hint {
  color: var(--text-secondary);
  font-size: 12px;
  margin-top: 8px;
  opacity: 0.7;
}

/* ========== 处理模式选择 ========== */
.mode-selector {
  margin-top: 16px;
  padding: 12px;
  background: var(--bg-dark);
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.mode-label {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 10px;
  font-weight: 500;
}

.mode-options {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.mode-option {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  background: var(--bg-card);
  cursor: pointer;
  transition: all 0.2s;
  user-select: none;
}

.mode-option:hover {
  border-color: var(--primary-color);
}

.mode-option.active {
  border-color: var(--primary-color);
  background: rgba(64, 158, 255, 0.05);
}

.mode-radio-input {
  display: none;
}

.mode-radio-dot {
  flex-shrink: 0;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  border: 2px solid var(--border-color);
  margin-top: 2px;
  transition: all 0.2s;
  position: relative;
}

.mode-option.active .mode-radio-dot {
  border-color: var(--primary-color);
}

.mode-option.active .mode-radio-dot::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--primary-color);
}

.mode-info {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}

.mode-name {
  font-size: 13px;
  color: var(--text-primary);
  font-weight: 500;
}

.mode-desc {
  font-size: 11px;
  color: var(--text-secondary);
  line-height: 1.4;
}

/* ========== 文档列表 ========== */
.docs-list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-darker);
}

.docs-list-header h3 {
  font-size: 15px;
  color: var(--text-primary);
}

.docs-actions {
  display: flex;
  gap: 8px;
}

.docs-list {
  flex: 1;
  overflow-y: auto;
  padding: 12px 20px;
}

.doc-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: var(--bg-card);
  border-radius: 10px;
  margin-bottom: 8px;
  border: 1px solid var(--border-color);
  transition: all 0.2s;
}

.doc-item:hover {
  border-color: rgba(64, 158, 255, 0.3);
  background: rgba(64, 158, 255, 0.04);
}

.doc-icon {
  flex-shrink: 0;
}

.doc-info {
  flex: 1;
  min-width: 0;
}

.doc-name {
  font-size: 14px;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.doc-meta {
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 2px;
}

.doc-delete {
  opacity: 0;
  transition: opacity 0.2s;
}

.doc-item:hover .doc-delete {
  opacity: 1;
}

/* ========== 批量上传 ========== */
.selected-files {
  margin-top: 12px;
  background: var(--bg-dark);
  border-radius: 8px;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
}

.selected-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 13px;
  color: var(--text-secondary);
}

.file-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-height: 120px;
  overflow-y: auto;
}

.file-chip {
  max-width: 200px;
}

.file-chip :deep(.el-tag__content) {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.upload-progress {
  margin-top: 12px;
}

.progress-text {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.upload-result {
  margin-top: 12px;
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 13px;
  line-height: 1.6;
}

.upload-result.all-ok {
  background: rgba(103, 194, 58, 0.1);
  border: 1px solid rgba(103, 194, 58, 0.3);
  color: #67c23a;
}

.upload-result.has-error {
  background: rgba(245, 108, 108, 0.1);
  border: 1px solid rgba(245, 108, 108, 0.3);
  color: #f56c6c;
}

.result-ok {
  color: #67c23a;
}

.result-err {
  color: #f56c6c;
  margin-top: 4px;
}

.err-detail {
  font-size: 12px;
  opacity: 0.85;
  margin-left: 16px;
}

/* ========== 管线统计 ========== */
.pipeline-card {
  background: var(--bg-card);
  border-radius: 12px;
  border: 1px solid var(--border-color);
  overflow: hidden;
}

.pipeline-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  cursor: pointer;
  user-select: none;
}

.pipeline-header:hover {
  background: rgba(64, 158, 255, 0.04);
}

.pipeline-header h3 {
  font-size: 15px;
  margin: 0;
  color: var(--text-primary);
}

.rotate-icon {
  transform: rotate(180deg);
  transition: transform 0.3s;
}

.pipeline-body {
  padding: 0 20px 16px;
}

.pipeline-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin-bottom: 12px;
}

.pipeline-stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 10px 4px;
  background: var(--bg-dark);
  border-radius: 8px;
}

.pstat-value {
  font-size: 20px;
  font-weight: 700;
  color: var(--primary-color);
}

.pstat-value.fail {
  color: #f56c6c;
}

.pstat-label {
  font-size: 11px;
  color: var(--text-secondary);
  margin-top: 2px;
}

.pipeline-info {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 12px;
}

.pinfo-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
}

.pinfo-label {
  color: var(--text-secondary);
}

.pinfo-value {
  color: var(--text-primary);
  font-size: 12px;
}

.pipeline-history {
  border-top: 1px solid var(--border-color);
  padding-top: 10px;
}

.phist-title {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 8px;
  font-weight: 500;
}

.phist-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
  font-size: 12px;
}

.phist-info {
  flex: 1;
  color: var(--text-secondary);
}

.phist-time {
  color: var(--text-secondary);
  font-size: 11px;
}
</style>
