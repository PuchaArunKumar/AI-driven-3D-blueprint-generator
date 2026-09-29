// Model downloads: byte-level progress and a Cache Storage copy.
//
// The first generation downloads ~70 MB (MiniLM, the prior, the decoder and
// the ONNX Runtime wasm). Each file is stored in Cache Storage under its URL
// plus a version tag - the model files by a digest of meta.json, MiniLM by its
// pinned revision, the wasm by the ONNX Runtime version - so a replaced model
// is fetched again and a stale copy is deleted. Where Cache Storage is missing
// or refuses (private windows, insecure origins, quota) everything still works;
// it just downloads each visit.

import type { ModelUrls, T2VMeta, T2VModelInfo } from './protocol.ts'

const CACHE_NAME = 'text2voxel-models-v1'
const TAG_PARAM = 't2v-cache'

async function openCache(): Promise<Cache | null> {
  try {
    if (typeof caches === 'undefined') return null
    return await caches.open(CACHE_NAME)
  } catch {
    return null
  }
}

function taggedKey(url: string, tag: string): string {
  const key = new URL(url)
  key.searchParams.set(TAG_PARAM, tag)
  return key.href
}

function untagged(url: string): string {
  const key = new URL(url)
  key.searchParams.delete(TAG_PARAM)
  return key.href
}

async function cachedBytes(cache: Cache | null, key: string): Promise<Uint8Array | null> {
  if (!cache) return null
  try {
    const hit = await cache.match(key)
    return hit ? new Uint8Array(await hit.arrayBuffer()) : null
  } catch {
    return null
  }
}

/** The bytes as a plain ArrayBuffer (no copy when the view spans its whole buffer). */
function arrayBufferOf(bytes: Uint8Array): ArrayBuffer {
  const { buffer, byteOffset, byteLength } = bytes
  if (byteOffset === 0 && byteLength === buffer.byteLength && buffer instanceof ArrayBuffer) return buffer
  return bytes.slice().buffer as ArrayBuffer
}

/** Store a copy and drop copies of the same URL under other tags. Best effort. */
async function remember(cache: Cache | null, url: string, tag: string, bytes: Uint8Array, type: string) {
  if (!cache) return
  try {
    const base = untagged(url)
    for (const request of await cache.keys()) {
      if (untagged(request.url) === base && request.url !== taggedKey(url, tag)) await cache.delete(request)
    }
    await cache.put(taggedKey(url, tag), new Response(arrayBufferOf(bytes), {
      headers: { 'content-type': type, 'content-length': String(bytes.byteLength) },
    }))
  } catch {
    // Quota or a refused origin: the download itself still succeeded.
  }
}

/** Short stable digest of a text (SHA-256 when available). */
export async function digest(text: string): Promise<string> {
  const subtle = globalThis.crypto?.subtle
  if (subtle) {
    try {
      const hash = await subtle.digest('SHA-256', new TextEncoder().encode(text))
      return Array.from(new Uint8Array(hash).slice(0, 8), (b) => b.toString(16).padStart(2, '0')).join('')
    } catch {
      // fall through (insecure context)
    }
  }
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i += 1) hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193) >>> 0
  return hash.toString(16).padStart(8, '0')
}

function missingHint(url: string): string {
  return url.includes('ort-wasm')
    ? ' The ONNX Runtime wasm ships with the onnxruntime-web package - reinstall the frontend ' +
      'dependencies (npm ci) and rebuild.'
    : ' The model files are fetched into frontend/public/models by ' +
      '`node frontend/scripts/fetch-models.mjs` (see the README).'
}

/**
 * meta.json from the network (revalidated, so a replaced model is noticed),
 * falling back to the cached copy when offline.
 */
export async function fetchMeta(url: string): Promise<{ meta: T2VMeta; tag: string }> {
  const cache = await openCache()
  let text: string | null = null
  let networkError: unknown = null
  try {
    const response = await fetch(url, { cache: 'no-cache' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    text = await response.text()
  } catch (error) {
    networkError = error
    const cached = await cachedBytes(cache, taggedKey(url, 'meta'))
    if (cached) text = new TextDecoder().decode(cached)
  }
  if (text === null) {
    throw new Error(`Could not load the model description (${url}): ${String(networkError)}.${missingHint(url)}`)
  }
  let meta: T2VMeta
  try {
    meta = JSON.parse(text) as T2VMeta
  } catch {
    throw new Error(`${url} is not valid JSON - is the model installed?${missingHint(url)}`)
  }
  if (!networkError) await remember(cache, url, 'meta', new TextEncoder().encode(text), 'application/json')
  return { meta, tag: `${meta.version}-${await digest(text)}` }
}

export interface DownloadItem {
  name: string
  url: string
  /** Version tag for the cache key. */
  tag: string
  type?: string
}

export type ByteProgress = (loadedBytes: number, totalBytes: number, message: string) => void

/** Download (or read from cache) every item concurrently, reporting combined bytes. */
export async function downloadAll(items: DownloadItem[], onProgress: ByteProgress = () => {}): Promise<Uint8Array[]> {
  const cache = await openCache()
  const loaded = items.map(() => 0)
  const totals: (number | null)[] = items.map(() => null)
  const report = (name: string) => {
    const done = loaded.reduce((a, b) => a + b, 0)
    // An unknown or compressed length counts as what has arrived so far.
    const total = totals.reduce<number>((sum, value, index) => sum + Math.max(value ?? 0, loaded[index]!), 0)
    onProgress(done, total, name)
  }

  return Promise.all(items.map(async (item, index): Promise<Uint8Array> => {
    const hit = await cachedBytes(cache, taggedKey(item.url, item.tag))
    if (hit) {
      loaded[index] = totals[index] = hit.byteLength
      report(`${item.name}, from cache`)
      return hit
    }

    let response: Response
    try {
      // Revalidate: a stale HTTP-cached file must never be stored under a new tag.
      response = await fetch(item.url, { cache: 'no-cache' })
    } catch (error) {
      throw new Error(`Could not download ${item.name} from ${item.url}: ${String(error)}`)
    }
    if (!response.ok) {
      throw new Error(`Could not download ${item.name} (HTTP ${response.status} from ${item.url}).${missingHint(item.url)}`)
    }
    const length = Number(response.headers.get('content-length'))
    // With content-encoding the header counts compressed bytes, not what we read.
    if (length > 0 && !response.headers.get('content-encoding')) totals[index] = length

    let bytes: Uint8Array
    if (response.body) {
      const reader = response.body.getReader()
      const chunks: Uint8Array[] = []
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        chunks.push(value)
        loaded[index] += value.byteLength
        report(item.name)
      }
      bytes = new Uint8Array(loaded[index]!)
      let offset = 0
      for (const chunk of chunks) {
        bytes.set(chunk, offset)
        offset += chunk.byteLength
      }
    } else {
      bytes = new Uint8Array(await response.arrayBuffer())
      loaded[index] = bytes.byteLength
    }
    totals[index] = bytes.byteLength
    report(item.name)
    await remember(cache, item.url, item.tag, bytes, item.type ?? 'application/octet-stream')
    return bytes
  }))
}

/** Size of one file without downloading it: any cached copy, else a HEAD request. */
async function probeSize(cache: Cache | null, item: DownloadItem): Promise<number> {
  try {
    const hit = cache ? await cache.match(item.url, { ignoreSearch: true }) : undefined
    if (hit) return Number(hit.headers.get('content-length')) || (await hit.arrayBuffer()).byteLength
  } catch {
    // ignore and ask the server
  }
  try {
    // no-store: a conditional HEAD revalidated against the HTTP cache can come
    // back without a length.
    const response = await fetch(item.url, { method: 'HEAD', cache: 'no-store' })
    const length = Number(response.headers.get('content-length'))
    if (response.ok && length > 0 && !response.headers.get('content-encoding')) return length
    // Some servers omit the length on HEAD; a one-byte range reports the total.
    const ranged = await fetch(item.url, { headers: { Range: 'bytes=0-0' }, cache: 'no-store' })
    const total = Number(ranged.headers.get('content-range')?.split('/')[1])
    await ranged.body?.cancel()
    if (ranged.status === 206 && total > 0) return total
  } catch {
    // offline: unknown
  }
  return 0
}

/**
 * What a first generation downloads, per file. `ortVersion` only affects the
 * cache tag of the wasm (size probes ignore tags).
 */
export function downloadItems(urls: ModelUrls, meta: T2VMeta, metaTag: string, ortVersion: string): DownloadItem[] {
  const revision = meta.text_encoder?.revision ?? 'unknown'
  return [
    { name: 'ONNX Runtime (wasm)', url: urls.ortWasm, tag: `ort-${ortVersion}`, type: 'application/wasm' },
    { name: 'MiniLM text encoder', url: urls.minilm, tag: `minilm-${revision}` },
    { name: 'MiniLM tokenizer', url: urls.tokenizer, tag: `minilm-${revision}`, type: 'application/json' },
    { name: 'diffusion prior', url: urls.prior, tag: metaTag },
    { name: 'shape decoder', url: urls.decoder, tag: metaTag },
  ]
}

/** Sum of the download sizes; files whose size cannot be learned count as 0. */
export async function probeTotalBytes(items: DownloadItem[]): Promise<number> {
  const cache = await openCache()
  const sizes = await Promise.all(items.map((item) => probeSize(cache, item)))
  return sizes.reduce((a, b) => a + b, 0)
}

export function modelInfo(meta: T2VMeta, totalBytes: number): T2VModelInfo {
  return {
    name: meta.name,
    version: meta.version,
    created: meta.created,
    smoke: meta.smoke === true,
    resolution: meta.resolution,
    sampleSteps: meta.diffusion.sample_steps,
    guidance: meta.diffusion.guidance,
    classes: meta.classes_modelnet ?? [],
    metrics: meta.metrics ?? {},
    training: meta.training ?? {},
    totalBytes,
  }
}
