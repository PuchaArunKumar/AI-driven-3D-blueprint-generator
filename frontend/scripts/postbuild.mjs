#!/usr/bin/env node
/**
 * Post-build step for the static (GitHub Pages) deployment.
 *
 * GitHub Pages has no rewrite rules: a deep link such as /<repo>/generate is a
 * 404 on the server. It does serve 404.html for any missing path, so a copy of
 * the SPA shell there lets the client-side router take over. Every asset URL in
 * the shell is absolute (Vite's base), so the copy works from any depth.
 *
 * Also writes .nojekyll (harmless under the Actions deployment, and it keeps a
 * branch deployment from hiding files that start with "_"), then reports what
 * the in-browser model will fetch so a missing file shows up here, not in a
 * visitor's console.
 *
 * Usage: node frontend/scripts/postbuild.mjs [--dist <dir>] [--strict]
 *   --dist    build directory (default frontend/dist)
 *   --strict  exit 1 when a model file is missing (the Pages workflow uses it)
 */
import { copyFileSync, existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const args = process.argv.slice(2)
const distFlag = args.indexOf('--dist')
const dist = resolve(distFlag >= 0 && args[distFlag + 1] ? args[distFlag + 1] : join(here, '..', 'dist'))
const strict = args.includes('--strict')

function fail(message) {
  console.error(`postbuild: ERROR - ${message}`)
  process.exit(1)
}

function sizeOf(path) {
  if (!existsSync(path)) return null
  const stat = statSync(path)
  if (!stat.isDirectory()) return stat.size
  return readdirSync(path).reduce((total, name) => total + (sizeOf(join(path, name)) ?? 0), 0)
}

const mb = (bytes) => `${(bytes / (1024 * 1024)).toFixed(1)} MB`
const shown = (path) => relative(process.cwd(), path) || '.'

// ------------------------------------------------------------ SPA fallback
const index = join(dist, 'index.html')
if (!existsSync(index)) fail(`${index} not found - run the Vite build first.`)

copyFileSync(index, join(dist, '404.html'))
writeFileSync(join(dist, '.nojekyll'), '')
console.log(`postbuild: SPA fallback written to ${shown(join(dist, '404.html'))}`)

// ------------------------------------------------------------ model files
// The build itself does not need these; the Text → 3D page fetches them at runtime.
const modelFiles = [
  'models/text2voxel/decoder.onnx',
  'models/text2voxel/prior.onnx',
  'models/text2voxel/meta.json',
  'models/minilm/model_quantized.onnx',
  'models/minilm/tokenizer.json',
]
const missing = []
for (const file of modelFiles) {
  const size = sizeOf(join(dist, file))
  if (size === null) missing.push(file)
  else console.log(`postbuild: ${file.padEnd(36)} ${mb(size)}`)
}

if (missing.length) {
  const message =
    `Text → 3D will not work, missing from the build: ${missing.join(', ')}. ` +
    'Fetch them before building: node frontend/scripts/fetch-models.mjs'
  if (strict) fail(message)
  console.warn(`postbuild: WARNING - ${message}`)
} else {
  let meta
  try {
    meta = JSON.parse(readFileSync(join(dist, 'models/text2voxel/meta.json'), 'utf8'))
  } catch (error) {
    fail(`models/text2voxel/meta.json is not valid JSON: ${error.message}`)
  }
  console.log(`postbuild: model ${meta.name} v${meta.version} (${meta.created})`)
  if (meta.smoke) {
    console.warn('postbuild: WARNING - this is the untrained placeholder model (meta.smoke = true)')
  }
}

// ---------------------------------------------------------------- summary
const snapshot = sizeOf(join(dist, 'static-api'))
console.log(
  snapshot === null
    ? 'postbuild: no static-api snapshot (fine for a backend build, empty Gallery on Pages)'
    : `postbuild: static-api snapshot ${mb(snapshot)}`,
)
console.log(`postbuild: total ${mb(sizeOf(dist) ?? 0)} in ${shown(dist)}`)
