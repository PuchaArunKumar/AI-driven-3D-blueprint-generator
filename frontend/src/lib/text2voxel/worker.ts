// Module worker hosting the Text2Voxel engine, so the ~1-3 s of inference and
// meshing never blocks the page. Started by index.ts with
// `new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })`.
//
// ONNX Runtime runs on its WebAssembly backend ('onnxruntime-web/wasm': a
// 14 MB wasm instead of the default entry's 28 MB WebGPU/JSEP build),
// single-threaded: threads need cross-origin isolation (COOP/COEP headers),
// which GitHub Pages cannot send. The wasm binary is fetched by our own loader
// (progress + Cache Storage) from this site - see ortWasmUrl in index.ts - and
// handed to ONNX Runtime directly, so no CDN is involved.

import * as ort from 'onnxruntime-web/wasm'
import { Text2VoxelEngine } from './engine.ts'
import { downloadAll, downloadItems, fetchMeta, modelInfo } from './loader.ts'
import type {
  ModelUrls,
  T2VModelInfo,
  T2VProgress,
  T2VStage,
  WorkerRequest,
  WorkerResponse,
} from './protocol.ts'

interface Loaded {
  engine: Text2VoxelEngine
  info: T2VModelInfo
}

let loading: Promise<Loaded> | null = null
let ready: Loaded | null = null
/** Everyone waiting on the current load hears its byte progress. */
const loadListeners = new Set<(progress: T2VProgress) => void>()
/** Generations run one at a time: a session cannot run two inferences at once. */
let queue: Promise<unknown> = Promise.resolve()

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  self.postMessage(message, { transfer })
}

async function load(urls: ModelUrls): Promise<Loaded> {
  const emit = (progress: T2VProgress) => loadListeners.forEach((listener) => listener(progress))
  emit({ stage: 'loading', fraction: 0, message: 'Reading the model description' })
  const { meta, tag } = await fetchMeta(urls.meta)
  const version = ort.env.versions.web ?? 'unknown'
  const items = downloadItems(urls, meta, tag, version)
  // The total grows as each response's headers arrive; never let the bar go backwards.
  let fraction = 0
  const [wasm, minilm, tokenizer, prior, decoder] = await downloadAll(items, (loadedBytes, totalBytes, name) => {
    const mb = (bytes: number) => (bytes / 1048576).toFixed(1)
    fraction = Math.max(fraction, totalBytes > 0 ? Math.min(loadedBytes / totalBytes, 1) * 0.95 : 0)
    emit({
      stage: 'loading',
      fraction,
      message: `Loading the model: ${mb(loadedBytes)} / ${mb(totalBytes)} MB (${name})`,
      loadedBytes,
      totalBytes,
    })
  })
  const totalBytes = [wasm, minilm, tokenizer, prior, decoder].reduce((sum, part) => sum + part!.byteLength, 0)

  emit({ stage: 'loading', fraction: 0.96, message: 'Starting ONNX Runtime', loadedBytes: totalBytes, totalBytes })
  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  ort.env.wasm.wasmPaths = { wasm: urls.ortWasm }
  ort.env.wasm.wasmBinary = wasm!
  let tokenizerJson: unknown
  try {
    tokenizerJson = JSON.parse(new TextDecoder().decode(tokenizer))
  } catch {
    throw new Error(`${urls.tokenizer} is not valid JSON`)
  }
  const engine = await Text2VoxelEngine.create(ort, {
    meta, tokenizer: tokenizerJson, minilm: minilm!, prior: prior!, decoder: decoder!,
  })
  emit({ stage: 'loading', fraction: 1, message: 'Model ready', loadedBytes: totalBytes, totalBytes })
  return { engine, info: modelInfo(meta, totalBytes) }
}

/** Load once; concurrent callers share the load and its progress. A failed load can be retried. */
async function ensureLoaded(urls: ModelUrls, onProgress: (progress: T2VProgress) => void): Promise<Loaded> {
  if (ready) return ready
  loadListeners.add(onProgress)
  try {
    if (!loading) {
      loading = load(urls).then(
        (loaded) => (ready = loaded),
        (error) => {
          loading = null
          throw error
        },
      )
    }
    return await loading
  } finally {
    loadListeners.delete(onProgress)
  }
}

/** Where each stage sits in the overall 0..1, without and with a model download first. */
const STAGE_SPAN: Record<Exclude<T2VStage, 'loading'>, [number, number]> = {
  encoding: [0, 0.04],
  sampling: [0.04, 0.75],
  decoding: [0.75, 0.85],
  meshing: [0.85, 1],
  done: [1, 1],
}
const LOAD_SHARE = 0.6

async function handle(request: WorkerRequest): Promise<void> {
  const { id } = request
  if (request.type === 'preload') {
    const { info } = await ensureLoaded(request.urls, (progress) => post({ id, type: 'progress', progress }))
    post({ id, type: 'preloaded', info })
    return
  }

  const needsLoad = ready === null
  const { engine, info } = await ensureLoaded(request.urls, (progress) =>
    post({ id, type: 'progress', progress: { ...progress, fraction: progress.fraction * LOAD_SHARE } }))
  const offset = needsLoad ? LOAD_SHARE : 0
  const share = 1 - offset

  const result = await engine.generate({
    prompt: request.prompt,
    seed: request.seed,
    guidance: request.guidance ?? info.guidance,
    steps: request.steps ?? info.sampleSteps,
    dimensionsMm: request.dimensionsMm,
  }, (stage, fraction, message) => {
    if (stage === 'loading') return
    const [from, to] = STAGE_SPAN[stage]
    post({ id, type: 'progress', progress: { stage, fraction: offset + share * (from + (to - from) * fraction), message } })
  })

  const { mesh, glb } = result
  post({ id, type: 'result', result }, [
    glb, mesh.positions.buffer, mesh.normals.buffer, mesh.colors.buffer, mesh.indices.buffer,
  ])
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  const run = () =>
    handle(request).catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error)
      post({ id: request.id, type: 'error', message })
    })
  // Preloads may overlap anything; generations are serialised.
  if (request.type === 'preload') void run()
  else queue = queue.then(run)
}
