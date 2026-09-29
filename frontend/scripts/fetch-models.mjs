#!/usr/bin/env node
/**
 * Fetch the Text2Voxel-64 + MiniLM model files for the in-browser Text → 3D page.
 *
 * Reads models/manifest.json at the repository root:
 *
 *   { "release": "text2voxel-v1",
 *     "files": { "text2voxel/decoder.onnx": { "url": "https://...", "sha256": null | "<hex>" }, ... } }
 *
 * and downloads every file to frontend/public/models/<key>, where Vite serves it
 * in development and copies it into the build. The Text2Voxel files come from
 * the GitHub release named in the manifest, the MiniLM encoder from Hugging
 * Face at a pinned revision.
 *
 * - A file that is already present is skipped when its sha256 matches the
 *   manifest. When the manifest has no checksum yet (sha256: null, before the
 *   trained model is published) an existing file is kept as-is - pass --force
 *   to replace it, e.g. to swap the dev placeholder for the released model.
 * - Downloads stream to "<file>.part", are hashed on the way, and only replace
 *   the target once complete and verified.
 * - GITHUB_TOKEN, when set, is sent to github.com hosts only (CI rate limits).
 *
 * Node 18+ built-ins only (fetch, crypto, fs) - no npm install needed.
 *
 * Usage: node frontend/scripts/fetch-models.mjs [--force] [--check]
 *                                               [--manifest <path>] [--out <dir>]
 *   --force     download every file, even when a matching copy exists
 *   --check     verify only (presence + checksum), never download
 *   --manifest  manifest path (default <repo>/models/manifest.json)
 *   --out       target directory (default frontend/public/models)
 *
 * Exit code 0 when every file is present (and verified where a checksum is
 * known), 1 otherwise - so CI fails instead of publishing a broken page.
 */
import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync } from 'node:fs'
import { dirname, isAbsolute, join, normalize, relative, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(here, '..', '..')

// ------------------------------------------------------------------- args
function parseArgs(argv) {
  const options = {
    force: false,
    check: false,
    manifest: join(repoRoot, 'models', 'manifest.json'),
    out: join(repoRoot, 'frontend', 'public', 'models'),
  }
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]
    if (arg === '--force') options.force = true
    else if (arg === '--check') options.check = true
    else if (arg === '--manifest' || arg === '--out') {
      const value = argv[index + 1]
      if (!value) usage(`${arg} needs a path`)
      options[arg.slice(2)] = resolve(value)
      index += 1
    } else if (arg === '--help' || arg === '-h') usage()
    else usage(`unknown argument: ${arg}`)
  }
  return options
}

function usage(problem) {
  if (problem) console.error(`fetch-models: ${problem}\n`)
  console.error(
    'Usage: node frontend/scripts/fetch-models.mjs [--force] [--check] [--manifest <path>] [--out <dir>]',
  )
  process.exit(problem ? 2 : 0)
}

// --------------------------------------------------------------- helpers
const mb = (bytes) => `${(bytes / (1024 * 1024)).toFixed(1)} MB`
/** Short relative path when under the working directory, absolute otherwise. */
const shown = (path) => {
  const rel = relative(process.cwd(), path)
  return rel && !rel.startsWith('..') && !isAbsolute(rel) ? rel : path
}
const sleep = (ms) => new Promise((done) => setTimeout(done, ms))

class FetchError extends Error {
  constructor(message, { retry = false } = {}) {
    super(message)
    this.retry = retry
  }
}

function loadManifest(path) {
  if (!existsSync(path)) {
    throw new Error(
      `manifest not found: ${shown(path)}\n` +
        '  It lists the model files and their release URLs; it is committed at models/manifest.json.',
    )
  }
  let manifest
  try {
    manifest = JSON.parse(readFileSync(path, 'utf8'))
  } catch (error) {
    throw new Error(`manifest is not valid JSON (${shown(path)}): ${error.message}`)
  }
  if (!manifest || typeof manifest.files !== 'object' || manifest.files === null) {
    throw new Error(`manifest has no "files" object: ${shown(path)}`)
  }
  return manifest
}

/** Resolve a manifest key under the output dir, refusing anything that escapes it. */
function targetFor(outDir, key) {
  const cleaned = normalize(key)
  if (isAbsolute(cleaned) || cleaned.split(sep).includes('..')) {
    throw new Error(`refusing manifest key outside the models directory: ${key}`)
  }
  return join(outDir, cleaned)
}

async function sha256Of(path) {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) hash.update(chunk)
  return hash.digest('hex')
}

function authHeaders(url) {
  const token = process.env.GITHUB_TOKEN
  if (!token) return {}
  const host = new URL(url).hostname
  // Only GitHub gets the token. fetch drops it on the cross-origin redirect
  // to the release CDN, which is what that CDN requires.
  return host === 'github.com' || host === 'api.github.com' ? { Authorization: `Bearer ${token}` } : {}
}

function explainStatus(status, url, release) {
  if (status === 404) {
    const releaseNote = url.includes('github.com')
      ? `\n  Has the "${release ?? 'text2voxel'}" release been published with this asset? See training/text2voxel/README.md.`
      : ''
    return `not found (HTTP 404): ${url}${releaseNote}`
  }
  if (status === 401 || status === 403) {
    return `access denied (HTTP ${status}): ${url}\n  Check GITHUB_TOKEN, or that the release is public.`
  }
  return `HTTP ${status} from ${url}`
}

/** Stream one URL to disk through a sha256, replacing `target` only on success. */
async function download(url, target, release) {
  const partial = `${target}.part`
  mkdirSync(dirname(target), { recursive: true })

  // Abort when the connection stalls, not after a fixed total time.
  const controller = new AbortController()
  let idle = setTimeout(() => controller.abort(), 60_000)
  const touch = () => {
    clearTimeout(idle)
    idle = setTimeout(() => controller.abort(), 60_000)
  }

  let response
  try {
    response = await fetch(url, {
      headers: { 'User-Agent': 'ai3d-blueprint-fetch-models', ...authHeaders(url) },
      redirect: 'follow',
      signal: controller.signal,
    })
  } catch (error) {
    clearTimeout(idle)
    throw new FetchError(`network error for ${url}: ${error.cause?.message ?? error.message}`, { retry: true })
  }
  if (!response.ok || !response.body) {
    clearTimeout(idle)
    throw new FetchError(explainStatus(response.status, url, release), {
      retry: response.status >= 500 || response.status === 429,
    })
  }

  const total = Number(response.headers.get('content-length')) || 0
  const hash = createHash('sha256')
  const file = createWriteStream(partial)
  const interactive = process.stdout.isTTY
  let received = 0
  let nextReport = 0.25

  try {
    for await (const chunk of response.body) {
      touch()
      hash.update(chunk)
      received += chunk.length
      if (!file.write(chunk)) await new Promise((done) => file.once('drain', done))
      if (interactive) {
        process.stdout.write(`\r  ${mb(received)}${total ? ` / ${mb(total)}` : ''}   `)
      } else if (total && received / total >= nextReport) {
        console.log(`  ${Math.round((received / total) * 100)}% (${mb(received)})`)
        nextReport += 0.25
      }
    }
    await new Promise((done, failed) => file.end((error) => (error ? failed(error) : done())))
  } catch (error) {
    file.destroy()
    rmSync(partial, { force: true })
    throw new FetchError(
      controller.signal.aborted
        ? `download stalled for 60 s: ${url}`
        : `download interrupted: ${url}: ${error.message}`,
      { retry: true },
    )
  } finally {
    clearTimeout(idle)
    if (interactive) process.stdout.write('\r\x1b[K')
  }

  if (total && received !== total) {
    rmSync(partial, { force: true })
    throw new FetchError(`truncated download (${received} of ${total} bytes): ${url}`, { retry: true })
  }
  return { partial, digest: hash.digest('hex'), bytes: received }
}

// ------------------------------------------------------------------- main
async function main() {
  const options = parseArgs(process.argv.slice(2))
  const manifest = loadManifest(options.manifest)
  const entries = Object.entries(manifest.files)
  if (entries.length === 0) throw new Error(`manifest lists no files: ${shown(options.manifest)}`)

  console.log(
    `fetch-models: ${entries.length} files from ${shown(options.manifest)}` +
      `${manifest.release ? ` (release ${manifest.release})` : ''} -> ${shown(options.out)}`,
  )

  const failures = []
  const unverified = []

  for (const [key, spec] of entries) {
    const url = spec?.url
    const expected = typeof spec?.sha256 === 'string' && spec.sha256 ? spec.sha256.toLowerCase() : null
    let target
    try {
      target = targetFor(options.out, key)
    } catch (error) {
      failures.push(error.message)
      continue
    }

    // ------------------------------------------------ existing copy?
    if (existsSync(target) && !options.force) {
      if (!expected) {
        unverified.push(key)
        console.log(`  keep     ${key} (${mb(statSync(target).size)}; manifest has no sha256 yet)`)
        continue
      }
      const actual = await sha256Of(target)
      if (actual === expected) {
        console.log(`  ok       ${key} (sha256 verified)`)
        continue
      }
      if (options.check) {
        failures.push(`${key}: sha256 mismatch (have ${actual.slice(0, 12)}..., manifest ${expected.slice(0, 12)}...)`)
        continue
      }
      console.log(`  stale    ${key} (sha256 differs from the manifest - downloading)`)
    } else if (options.check) {
      failures.push(`${key}: missing (${shown(target)})`)
      continue
    }

    if (typeof url !== 'string' || !/^https?:\/\//.test(url)) {
      failures.push(`${key}: manifest has no usable "url"`)
      continue
    }

    // ------------------------------------------------------ download
    console.log(`  fetch    ${key}\n           ${url}`)
    let result = null
    for (let attempt = 1; attempt <= 3 && !result; attempt += 1) {
      try {
        result = await download(url, target, manifest.release)
      } catch (error) {
        const last = attempt === 3 || !(error instanceof FetchError && error.retry)
        if (last) {
          failures.push(`${key}: ${error.message}`)
          break
        }
        console.log(`  retry    ${key} in ${attempt * 2} s (${error.message})`)
        await sleep(attempt * 2000)
      }
    }
    if (!result) continue

    if (expected && result.digest !== expected) {
      rmSync(result.partial, { force: true })
      failures.push(
        `${key}: sha256 mismatch after download (got ${result.digest}, manifest ${expected}) - ` +
          'the file at that URL is not the one the manifest pins',
      )
      continue
    }
    renameSync(result.partial, target)
    if (!expected) unverified.push(key)
    console.log(`  saved    ${key} (${mb(result.bytes)}${expected ? ', sha256 verified' : ''})`)
  }

  // A placeholder deploys and runs, but its shapes are meaningless - say so.
  const metaKey = entries.map(([key]) => key).find((key) => key.endsWith('text2voxel/meta.json'))
  if (metaKey) {
    const metaPath = targetFor(options.out, metaKey)
    if (existsSync(metaPath)) {
      try {
        const meta = JSON.parse(readFileSync(metaPath, 'utf8'))
        console.log(`fetch-models: model ${meta.name ?? '?'} v${meta.version ?? '?'} (${meta.created ?? '?'})`)
        if (meta.smoke) {
          console.warn('fetch-models: WARNING - meta.json has "smoke": true - this is the untrained placeholder model')
        }
      } catch (error) {
        failures.push(`${metaKey}: not valid JSON (${error.message})`)
      }
    }
  }

  if (unverified.length) {
    console.log(`fetch-models: not checksum-verified (manifest sha256 is null): ${unverified.join(', ')}`)
  }
  if (failures.length) {
    console.error(`\nfetch-models: FAILED - ${failures.length} problem(s):`)
    for (const failure of failures) console.error(`  - ${failure}`)
    process.exit(1)
  }
  console.log('fetch-models: all model files present')
}

main().catch((error) => {
  console.error(`fetch-models: ${error.message}`)
  process.exit(1)
})
