/// <reference types="vite/client" />
// In-browser Text2Voxel-64: text -> coloured 3D mesh + blueprint, with no backend.
//
// Everything heavy runs in a module worker (worker.ts): model download and
// caching, MiniLM, the diffusion prior, the decoder, meshing, GLB export and
// the blueprint projection. This module is the page-side API: it starts the
// worker, relays progress, and turns the worker's output into object URLs and
// download blobs. Model files are served from `${BASE_URL}models/...`; nothing
// comes from a CDN.

// The ONNX Runtime wasm, as a URL on this site. In a build Vite emits it as a
// hashed asset under `${BASE_URL}assets/` - ONNX Runtime's own bundle
// references the same file, so it is emitted exactly once and needs no
// vite.config.ts change; in dev it is served from node_modules.
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import { parseDimensions as parseDimensionsImpl } from './dimensions.ts'
import { downloadItems, fetchMeta, modelInfo, probeTotalBytes } from './loader.ts'
import type { MeshData } from './mesher.ts'
import type {
  ModelUrls,
  T2VModelInfo,
  T2VOptions,
  T2VProgress,
  T2VResult,
  WorkerRequest,
  WorkerResponse,
} from './protocol.ts'
import { randomSeed } from './rng.ts'
import { renderBlueprintSheet } from './sheet.ts'

export type { T2VModelInfo, T2VOptions, T2VProgress, T2VResult, T2VStage } from './protocol.ts'

/** Where the files live, resolved against the page so any Vite `base` works. */
function modelUrls(): ModelUrls {
  const root = new URL(import.meta.env.BASE_URL, document.baseURI)
  const at = (path: string) => new URL(path, root).href
  return {
    meta: at('models/text2voxel/meta.json'),
    decoder: at('models/text2voxel/decoder.onnx'),
    prior: at('models/text2voxel/prior.onnx'),
    minilm: at('models/minilm/model_quantized.onnx'),
    tokenizer: at('models/minilm/tokenizer.json'),
    ortWasm: new URL(ortWasmUrl, document.baseURI).href,
  }
}

// (module (func (result v128) i32.const 0 i8x16.splat i8x16.popcnt)) - the
// wasm-feature-detect SIMD probe. ONNX Runtime Web requires SIMD.
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11,
])

/** WebAssembly (with SIMD), module-capable workers and BigInt64Array are all present. */
export function isText2VoxelSupported(): boolean {
  try {
    return (
      typeof Worker !== 'undefined' &&
      typeof WebAssembly === 'object' &&
      typeof BigInt64Array !== 'undefined' &&
      WebAssembly.validate(SIMD_PROBE)
    )
  } catch {
    return false
  }
}

// ---- worker plumbing ------------------------------------------------------

interface Pending {
  resolve: (message: WorkerResponse) => void
  reject: (error: Error) => void
  onProgress?: (progress: T2VProgress) => void
}

let worker: Worker | null = null
let nextId = 1
const pending = new Map<number, Pending>()

function failAll(error: Error): void {
  for (const entry of pending.values()) entry.reject(error)
  pending.clear()
}

function getWorker(): Worker {
  if (worker) return worker
  const created = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'text2voxel' })
  created.onmessage = (event: MessageEvent<WorkerResponse>) => {
    const message = event.data
    const entry = pending.get(message.id)
    if (!entry) return
    if (message.type === 'progress') {
      entry.onProgress?.(message.progress)
      return
    }
    pending.delete(message.id)
    if (message.type === 'error') entry.reject(new Error(message.message))
    else entry.resolve(message)
  }
  created.onerror = (event: ErrorEvent) => {
    event.preventDefault()
    // A worker that fails to start leaves nothing to recover: start afresh next time.
    created.terminate()
    if (worker === created) worker = null
    failAll(new Error(`The Text2Voxel worker stopped: ${event.message || 'it could not be started'}.`))
  }
  worker = created
  return created
}

/** A worker request before `call` assigns its id. */
type WithoutId<T> = T extends unknown ? Omit<T, 'id'> : never
type CallRequest = WithoutId<WorkerRequest>

function call(request: CallRequest, onProgress?: (progress: T2VProgress) => void): Promise<WorkerResponse> {
  const id = nextId++
  const target = getWorker()
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress })
    target.postMessage({ ...request, id } as WorkerRequest)
  })
}

// ---- public API ------------------------------------------------------------

/** meta.json plus the total download size (HEAD requests / cache); downloads no model. */
export async function loadModelInfo(): Promise<T2VModelInfo> {
  const urls = modelUrls()
  const { meta, tag } = await fetchMeta(urls.meta)
  let totalBytes = await probeTotalBytes(downloadItems(urls, meta, tag, 'probe'))
  if (!totalBytes) {
    // Offline and uncached: the exported sizes in meta.json are better than nothing.
    const parity = meta.parity ?? {}
    totalBytes = Number(parity.decoder_bytes ?? 0) + Number(parity.prior_bytes ?? 0)
  }
  return modelInfo(meta, totalBytes)
}

/** Download (or read from cache) and initialise the model in the worker. */
export async function preloadModel(onProgress?: (p: T2VProgress) => void): Promise<T2VModelInfo> {
  const response = await call({ type: 'preload', urls: modelUrls() }, onProgress)
  if (response.type !== 'preloaded') throw new Error('Unexpected worker response')
  onProgress?.({ stage: 'done', fraction: 1, message: 'Model ready' })
  return response.info
}

/** Mesh arrays per result, for STL/OBJ/PLY export without re-parsing the GLB. */
const meshes = new WeakMap<T2VResult, MeshData>()

function positiveOrNull(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null
}

/** Prompt -> mesh, stats and blueprint. Loads the model first if needed. */
export async function generate(
  prompt: string,
  options: T2VOptions = {},
  onProgress?: (p: T2VProgress) => void,
): Promise<T2VResult> {
  const text = prompt.trim()
  if (!text) throw new Error('Describe the object to generate, for example "a red office chair with wheels".')
  if (options.seed !== undefined && !(Number.isInteger(options.seed) && options.seed >= 0 && options.seed <= 0xffffffff)) {
    throw new Error('The seed must be a whole number from 0 to 4294967295.')
  }
  if (options.steps !== undefined && !(Number.isInteger(options.steps) && options.steps >= 1 && options.steps <= 1000)) {
    throw new Error('Steps must be a whole number from 1 to 1000.')
  }
  if (options.guidance !== undefined && !(Number.isFinite(options.guidance) && options.guidance >= 0 && options.guidance <= 30)) {
    throw new Error('Guidance must be a number from 0 to 30.')
  }
  const dims = options.dimensionsMm
  const response = await call({
    type: 'generate',
    urls: modelUrls(),
    prompt: text,
    seed: options.seed ?? randomSeed(),
    guidance: options.guidance ?? null,
    steps: options.steps ?? null,
    dimensionsMm: dims
      ? { length: positiveOrNull(dims.length), width: positiveOrNull(dims.width), height: positiveOrNull(dims.height) }
      : null,
  }, onProgress)
  if (response.type !== 'result') throw new Error('Unexpected worker response')

  const { glb: bytes, mesh, ...rest } = response.result
  const glb = new Blob([bytes], { type: 'model/gltf-binary' })
  const result: T2VResult = { ...rest, glb, glbUrl: URL.createObjectURL(glb) }
  meshes.set(result, mesh)
  return result
}

/** The result as a download: GLB as generated, or STL / OBJ / PLY (with colours). */
export async function exportMesh(result: T2VResult, format: 'glb' | 'stl' | 'obj' | 'ply'): Promise<Blob> {
  if (format === 'glb') return result.glb
  // Loaded on demand: the exporters are only needed when someone downloads.
  const { exportMeshData, meshDataFromGlb } = await import('./exporters.ts')
  const mesh = meshes.get(result) ?? (await meshDataFromGlb(await result.glb.arrayBuffer()))
  return exportMeshData(mesh, format)
}

/** The combined A3 blueprint sheet (same layout as the backend's), as SVG text. */
export function blueprintSheetSvg(result: T2VResult, name: string, theme: 'white' | 'blueprint'): string {
  return renderBlueprintSheet(result.blueprint.views, result.stats, result.blueprint.spec, name, theme)
}

/** Target dimensions in mm, parsed exactly as the backend's prompt parser does. */
export function parseDimensions(prompt: string): { length: number | null; width: number | null; height: number | null } {
  return parseDimensionsImpl(prompt)
}

/** Release the result's object URL and cached mesh arrays. */
export function disposeResult(result: T2VResult): void {
  URL.revokeObjectURL(result.glbUrl)
  meshes.delete(result)
}
