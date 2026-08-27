import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// The backend port is configurable (PORT in the root .env), so the dev proxy
// reads VITE_API_TARGET and falls back to the documented default.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'http://127.0.0.1:8000'

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/api': { target, changeOrigin: true, ws: true },
        '/files': { target, changeOrigin: true },
      },
    },
    build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 1600 },
  }
})
