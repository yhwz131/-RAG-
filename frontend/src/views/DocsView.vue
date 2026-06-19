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
          :disabled="fileList.length === 0"
          @click="handleUpload"
          style="width: 100%; margin-top: 12px;"
        >
          <el-icon><Upload /></el-icon>
          {{ uploading ? '上传中...' : `开始上传 (${fileList.length} 个文件)` }}
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
import { UploadFilled, Upload, Refresh, Delete, Document } from '@element-plus/icons-vue'
import type { UploadFile } from 'element-plus'
import { getDocStats, listDocuments, uploadDocuments, deleteDocument, clearDocuments } from '@/api'
import type { DocStats, DocItem, BatchUploadResult } from '@/api'

const stats = ref<DocStats>({ total_docs: 0, collection_name: '' })
const documents = ref<DocItem[]>([])
const loading = ref(false)
const uploading = ref(false)
const fileList = ref<File[]>([])
const uploadRef = ref()
const uploadResult = ref<BatchUploadResult | null>(null)
const uploadDone = ref(0)
const uploadTotal = ref(0)

const totalChunks = computed(() =>
  uploadResult.value?.results.reduce((sum, r) => sum + r.chunks, 0) ?? 0
)

// 加载数据
async function loadData() {
  loading.value = true
  try {
    const [s, docs] = await Promise.all([getDocStats(), listDocuments()])
    stats.value = s
    documents.value = docs
  } catch (err: any) {
    ElMessage.error('加载失败: ' + (err.message || '未知错误'))
  } finally {
    loading.value = false
  }
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
  if (fileList.value.length === 0) return

  uploading.value = true
  uploadResult.value = null
  uploadTotal.value = fileList.value.length
  uploadDone.value = 0

  try {
    const result = await uploadDocuments(fileList.value)
    uploadResult.value = result
    uploadDone.value = result.total_files

    if (result.fail_count === 0) {
      ElMessage.success(`全部 ${result.success_count} 个文件上传成功！`)
    } else {
      ElMessage.warning(`${result.success_count} 个成功，${result.fail_count} 个失败`)
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

onMounted(loadData)
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
</style>
