import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'node:path'

export default defineConfig({
  plugins: [react()],

  resolve: {
    alias: {
      '@': path.resolve(process.cwd(), './src'),
    },
  },

  server: {
    host: '0.0.0.0',
    port: 5173,

    // Allow LocalTunnel public hostnames
    allowedHosts: ['.loca.lt'],

    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },

      '/static': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
    },
  },
})