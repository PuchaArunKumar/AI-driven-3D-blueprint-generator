import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Normalise VITE_BASE to "/" or "/name/".
 *
 * GitHub Pages serves a project site under /<repo-name>/, so the build takes
 * its public base path from the environment; the router reads the same value
 * back through import.meta.env.BASE_URL. Both expect the slashes.
 */
function normaliseBase(raw: string | undefined): string {
  const trimmed = (raw ?? '').trim().replace(/^\/+|\/+$/g, '')
  // Git Bash rewrites "/name/" in the environment into "C:/Program Files/Git/name/".
  if (/^[a-z]:|\\/i.test(trimmed)) {
    throw new Error(
      `VITE_BASE looks like a filesystem path ("${raw}"). Git Bash converts leading-slash values: ` +
        'pass it without slashes (VITE_BASE=AI-driven-3D-blueprint-generator) or set MSYS_NO_PATHCONV=1.',
    )
  }
  return trimmed ? `/${trimmed}/` : '/'
}

// The backend port is configurable (PORT in the root .env), so the dev proxy
// reads VITE_API_TARGET and falls back to the documented default.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_TARGET || 'http://127.0.0.1:8000'

  return {
    base: normaliseBase(env.VITE_BASE),
    // No copy plugin is needed for ONNX Runtime: lib/text2voxel imports its wasm
    // with `?url`, so Vite emits it once, same-origin, under `${base}assets/`
    // like any other asset (ONNX Runtime's own reference points at that file).
    plugins: [react()],
    // The Text2Voxel worker is a module worker (`type: 'module'`). The default
    // iife format also builds and runs; 'es' keeps the worker's chunks as
    // modules, the same as the page's.
    worker: { format: 'es' },
    // The worker only loads when someone presses Generate, too late for the dev
    // server's dependency scan - it would re-optimise and reload the page
    // mid-generation. Pre-bundling its dependencies up front avoids that.
    optimizeDeps: {
      include: ['onnxruntime-web/wasm', 'three/examples/jsm/exporters/GLTFExporter.js'],
    },
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
