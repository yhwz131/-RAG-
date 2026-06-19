<template>
  <div class="admin-view">
    <div class="admin-header">
      <h2>📊 数据处理管线</h2>
      <el-button @click="loadData" :loading="loading">
        <el-icon><Refresh /></el-icon>
        刷新
      </el-button>
    </div>

    <!-- 统计卡片 -->
    <div class="stats-cards">
      <div class="stat-card">
        <div class="stat-icon" style="background: rgba(64, 158, 255, 0.1); color: #409eff;">
          <el-icon :size="24"><Document /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ status.last_run?.files_scanned || 0 }}</div>
          <div class="stat-label">处理文件</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon" style="background: rgba(103, 194, 58, 0.1); color: #67c23a;">
          <el-icon :size="24"><Files /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ status.last_run?.total_chunks || 0 }}</div>
          <div class="stat-label">生成切片</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon" style="background: rgba(230, 162, 60, 0.1); color: #e6a23c;">
          <el-icon :size="24"><Coin /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ status.milvus_stats?.row_count || 0 }}</div>
          <div class="stat-label">已入库</div>
        </div>
      </div>
      <div class="stat-card">
        <div class="stat-icon" style="background: rgba(245, 108, 108, 0.1); color: #f56c6c;">
          <el-icon :size="24"><Warning /></el-icon>
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ status.last_run?.files_failed || 0 }}</div>
          <div class="stat-label">失败文件</div>
        </div>
      </div>
    </div>

    <div class="admin-content">
      <!-- 左侧：详细信息 -->
      <div class="admin-left">
        <!-- 引擎信息 -->
        <div class="info-card">
          <h3>⚙️ 处理引擎</h3>
          <div class="info-item">
            <span class="info-label">当前引擎</span>
            <el-tag>{{ status.last_run?.engine || 'simple' }}</el-tag>
          </div>
          <div class="info-item">
            <span class="info-label">最近运行</span>
            <span>{{ status.last_run?.timestamp || '暂无' }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">运行ID</span>
            <span>{{ status.last_run?.run_id || '-' }}</span>
          </div>
        </div>

        <!-- 数据库状态 -->
        <div class="info-card">
          <h3>🗄️ 数据库连接</h3>
          <div v-if="status.database_status?.connected" class="db-connected">
            <el-tag type="success">已连接</el-tag>
            <div class="db-info">
              <span>{{ status.database_status.type?.toUpperCase() }}</span>
              <span>{{ status.database_status.host }}/{{ status.database_status.database }}</span>
            </div>
          </div>
          <div v-else class="db-disconnected">
            <el-tag type="info">未配置</el-tag>
            <span class="db-hint">在 .env 中配置 DB_TYPE 等参数</span>
          </div>
        </div>

        <!-- 文件格式分布 -->
        <div class="info-card" v-if="formatData.length > 0">
          <h3>📁 文件格式分布</h3>
          <div class="format-list">
            <div v-for="item in formatData" :key="item.name" class="format-item">
              <span class="format-name">{{ item.name }}</span>
              <el-progress
                :percentage="item.percent"
                :stroke-width="10"
                :format="() => item.count.toString()"
                :color="getFormatColor(item.name)"
              />
            </div>
          </div>
        </div>
      </div>

      <!-- 右侧：关键词和历史 -->
      <div class="admin-right">
        <!-- 高频关键词 -->
        <div class="info-card">
          <h3>🔤 高频关键词 Top 10</h3>
          <div v-if="keywords.length > 0" class="keyword-cloud">
            <el-tag
              v-for="(kw, idx) in keywords"
              :key="kw.word"
              :type="getKeywordType(idx)"
              class="keyword-tag"
              effect="plain"
            >
              {{ kw.word }} ({{ kw.count }})
            </el-tag>
          </div>
          <el-empty v-else description="暂无关键词数据" :image-size="60" />
        </div>

        <!-- 处理历史 -->
        <div class="info-card">
          <h3>📋 处理历史</h3>
          <div v-if="history.length > 0" class="history-list">
            <div v-for="run in history" :key="run.run_id" class="history-item">
              <div class="history-header">
                <el-tag :type="run.success ? 'success' : 'danger'" size="small">
                  {{ run.engine }}
                </el-tag>
                <span class="history-time">{{ formatTime(run.timestamp) }}</span>
              </div>
              <div class="history-stats">
                <span>{{ run.total_files }} 文件</span>
                <span>{{ run.total_chunks }} 切片</span>
              </div>
            </div>
          </div>
          <el-empty v-else description="暂无处理历史" :image-size="60" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh, Document, Files, Coin, Warning } from '@element-plus/icons-vue'
import { getPipelineStatus, getPipelineHistory } from '@/api'
import type { PipelineStatus } from '@/api'

const loading = ref(false)
const status = ref<PipelineStatus>({
  last_run: {},
  milvus_stats: {},
  database_status: {},
})
const history = ref<Record<string, any>[]>([])

// 格式分布数据
const formatData = computed(() => {
  const breakdown = status.value.last_run?.format_breakdown || {}
  const total = Object.values(breakdown).reduce((sum: number, cnt: any) => sum + cnt, 0) as number
  if (total === 0) return []
  return Object.entries(breakdown)
    .map(([name, count]) => ({
      name,
      count: count as number,
      percent: Math.round(((count as number) / total) * 100),
    }))
    .sort((a, b) => b.count - a.count)
})

// 关键词数据
const keywords = computed(() => {
  return status.value.last_run?.top_keywords || []
})

// 加载数据
async function loadData() {
  loading.value = true
  try {
    const [s, h] = await Promise.all([getPipelineStatus(), getPipelineHistory()])
    status.value = s
    history.value = h.reverse() // 最新的在前面
  } catch (err: any) {
    ElMessage.error('加载失败: ' + (err.message || '未知错误'))
  } finally {
    loading.value = false
  }
}

// 格式化时间
function formatTime(timestamp: string): string {
  if (!timestamp) return '-'
  try {
    const date = new Date(timestamp)
    return `${date.getMonth() + 1}-${date.getDate()} ${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`
  } catch {
    return timestamp
  }
}

// 格式颜色
function getFormatColor(ext: string): string {
  const colors: Record<string, string> = {
    '.pdf': '#e74c3c',
    '.docx': '#2980b9',
    '.doc': '#2980b9',
    '.pptx': '#e67e22',
    '.ppt': '#e67e22',
    '.xlsx': '#27ae60',
    '.xls': '#27ae60',
    '.txt': '#95a5a6',
    '.md': '#3498db',
    '.csv': '#27ae60',
  }
  return colors[ext] || '#909399'
}

// 关键词标签类型
function getKeywordType(idx: number): '' | 'success' | 'warning' | 'info' {
  if (idx < 3) return ''
  if (idx < 6) return 'success'
  if (idx < 8) return 'warning'
  return 'info'
}

onMounted(loadData)
</script>

<style scoped>
.admin-view {
  padding: 20px;
  height: 100%;
  overflow-y: auto;
  background: var(--bg-dark);
}

.admin-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}

.admin-header h2 {
  font-size: 20px;
  color: var(--text-primary);
  margin: 0;
}

/* ========== 统计卡片 ========== */
.stats-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 20px;
}

.stat-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  background: var(--bg-card);
  border-radius: 12px;
  border: 1px solid var(--border-color);
}

.stat-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1;
}

.stat-label {
  font-size: 13px;
  color: var(--text-secondary);
  margin-top: 4px;
}

/* ========== 内容区域 ========== */
.admin-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.admin-left,
.admin-right {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* ========== 信息卡片 ========== */
.info-card {
  background: var(--bg-card);
  border-radius: 12px;
  padding: 20px;
  border: 1px solid var(--border-color);
}

.info-card h3 {
  font-size: 15px;
  margin: 0 0 16px 0;
  color: var(--text-primary);
}

.info-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid var(--border-color);
  font-size: 13px;
}

.info-item:last-child {
  border-bottom: none;
}

.info-label {
  color: var(--text-secondary);
}

/* ========== 数据库状态 ========== */
.db-connected,
.db-disconnected {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.db-info {
  display: flex;
  gap: 8px;
  font-size: 13px;
  color: var(--text-secondary);
}

.db-hint {
  font-size: 12px;
  color: var(--text-secondary);
  opacity: 0.7;
}

/* ========== 格式分布 ========== */
.format-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.format-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.format-name {
  width: 50px;
  font-size: 13px;
  color: var(--text-secondary);
  text-align: right;
}

.format-item :deep(.el-progress) {
  flex: 1;
}

/* ========== 关键词 ========== */
.keyword-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.keyword-tag {
  cursor: default;
}

/* ========== 历史记录 ========== */
.history-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
  max-height: 300px;
  overflow-y: auto;
}

.history-item {
  padding: 12px;
  background: var(--bg-dark);
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.history-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.history-time {
  font-size: 12px;
  color: var(--text-secondary);
}

.history-stats {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: var(--text-secondary);
}

/* ========== 响应式 ========== */
@media (max-width: 1024px) {
  .stats-cards {
    grid-template-columns: repeat(2, 1fr);
  }

  .admin-content {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 640px) {
  .stats-cards {
    grid-template-columns: 1fr;
  }
}
</style>
