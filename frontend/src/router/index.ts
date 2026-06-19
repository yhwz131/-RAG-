import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/chat',
    },
    {
      path: '/chat',
      name: 'Chat',
      component: () => import('@/views/ChatView.vue'),
    },
    {
      path: '/chat/:sessionId',
      name: 'ChatSession',
      component: () => import('@/views/ChatView.vue'),
    },
    {
      path: '/files',
      name: 'Files',
      component: () => import('@/views/DocsView.vue'),
    },
    {
      path: '/admin',
      redirect: '/files',
    },
  ],
})

export default router
