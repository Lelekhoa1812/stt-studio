import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({ resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } }, plugins: [react()], server: { host: '127.0.0.1', proxy: { '/api': 'http://127.0.0.1:8765' } }, preview: { host: '127.0.0.1' } })
