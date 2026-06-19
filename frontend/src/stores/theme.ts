import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export type ThemeMode = 'dark' | 'light'

export const useThemeStore = defineStore('theme', () => {
  const theme = ref<ThemeMode>(
    (localStorage.getItem('theme') as ThemeMode) || 'dark'
  )

  function applyTheme(t: ThemeMode) {
    document.documentElement.setAttribute('data-theme', t)
  }

  function toggleTheme() {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }

  // 持久化 + 应用
  watch(theme, (val) => {
    localStorage.setItem('theme', val)
    applyTheme(val)
  }, { immediate: true })

  return { theme, toggleTheme }
})
