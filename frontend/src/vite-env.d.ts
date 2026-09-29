/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "true" in the GitHub Pages build: no backend, read-only snapshot + in-browser model. */
  readonly VITE_STATIC?: string
  /** Public base path the site is served from, e.g. "/AI-driven-3D-blueprint-generator/". */
  readonly VITE_BASE?: string
  /** Dev proxy target for /api and /files (vite.config.ts only). */
  readonly VITE_API_TARGET?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
