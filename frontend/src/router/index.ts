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
      path: '/docs',
      name: 'Docs',
      component: () => import('@/views/DocsView.vue'),
    },
  ],
})

export default router
