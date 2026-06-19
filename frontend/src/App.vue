<template>
  <div class="app-layout">
    <!-- 顶部导航栏 -->
    <header class="app-header">
      <div class="header-left">
        <el-icon class="logo-icon" :size="24"><ChatDotRound /></el-icon>
        <h1 class="app-title">知识问答系统</h1>
      </div>
      <nav class="header-nav">
        <router-link to="/chat" class="nav-link" active-class="active">
          <el-icon><ChatLineRound /></el-icon>
          <span>对话</span>
        </router-link>
        <router-link to="/files" class="nav-link" active-class="active">
          <el-icon><Document /></el-icon>
          <span>文档管理</span>
        </router-link>
      </nav>
      <div class="header-right">
        <el-switch
          v-model="isDark"
          inline-prompt
          :active-icon="Moon"
          :inactive-icon="Sunny"
          active-text=""
          inactive-text=""
          style="--el-switch-on-color: #2c2c3a; --el-switch-off-color: #e6e8eb;"
        />
        <el-tag type="success" size="small" effect="dark">RAG v2.0</el-tag>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="app-main">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Sunny, Moon } from '@element-plus/icons-vue'
import { useThemeStore } from '@/stores/theme'

const themeStore = useThemeStore()

const isDark = computed({
  get: () => themeStore.theme === 'dark',
  set: () => themeStore.toggleTheme(),
})
</script>

<style scoped>
.app-layout {
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.app-header {
  height: var(--header-height);
  background: var(--bg-darker);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  flex-shrink: 0;
  z-index: 100;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.logo-icon {
  color: var(--primary-color);
}

.app-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  white-space: nowrap;
}

.header-nav {
  display: flex;
  gap: 4px;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 20px;
  border-radius: 8px;
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 14px;
  transition: all 0.2s;
}

.nav-link:hover {
  color: var(--text-primary);
  background: rgba(255, 255, 255, 0.06);
}

.nav-link.active {
  color: var(--primary-color);
  background: rgba(64, 158, 255, 0.12);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.app-main {
  flex: 1;
  overflow: hidden;
}
</style>
