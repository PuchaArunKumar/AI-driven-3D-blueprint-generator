#!/usr/bin/env node
// Verify the in-browser Text2Voxel engine against the Python reference.
//
//   node frontend/scripts/verify-text2voxel.mjs [--skip-models] [--verbose]
//
// Imports the shipped TypeScript modules from src/lib/text2voxel directly (Node
// >= 22.18 strips the types natively), so what is tested is what the browser
// worker runs - not a copy. Python-side expectations come from
// scripts/fixtures/make_text2voxel_fixtures.py (run with the backend venv).
//
// Checks
//   1. tokenizer ids == HF `tokenizers` for every fixture string (>= 300, incl.
//      150+ captions, punctuation, accents, digits, emoji, CJK), and the
//      normaliser == Rust for all 1.1M code points (digest per 4096 block)
//   2. MiniLM embeddings == TextEncoder (cosine > 0.9999)
//   3. DDIM timesteps == numpy, and sampler parity with meta.reference
//   4. mesher closed + watertight + outward on an analytic sphere (and harder
//      grids), scale contract
//   5. blueprint extents on a box grid, feature lines on a chair-like grid
//   6. parseDimensions == backend prompt_parser, blueprint sheet SVG ==
//      backend blueprint_sheet byte for byte
//   7. an end-to-end generation with the installed model: GLB, exports,
//      determinism for a fixed seed
//
// The token ids (1) need frontend/public/models/minilm/tokenizer.json; steps 2,
// 3b and 7 need all of frontend/public/models/* (the placeholder or the real
// model). Missing files fail the run unless --skip-models is given, which runs
// everything else. Exits non-zero if any check fails.

import { createHash } from 'node:crypto'
import { existsSync, readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const FRONTEND = join(HERE, '..')
const ENGINE = join(FRONTEND, 'src', 'lib', 'text2voxel')
const MODELS = join(FRONTEND, 'public', 'models')
const args = new Set(process.argv.slice(2))
const skipModels = args.has('--skip-models')
const verbose = args.has('--verbose')

if (!process.features?.typescript) {
  console.error('This script imports the engine\'s .ts sources directly and needs Node >= 22.18 ' +
    `(native type stripping). Running ${process.version}.`)
  process.exit(2)
}

// GLTFExporter writes binary glTF through FileReader, which Node lacks.
if (typeof globalThis.FileReader === 'undefined') {
  globalThis.FileReader = class FileReader {
    constructor() {
      this.result = null
      this.onload = null
      this.onloadend = null
    }

    readAsArrayBuffer(blob) {
      blob.arrayBuffer().then((buffer) => {
        this.result = buffer
        this.onload?.({ target: this })
        this.onloadend?.({ target: this })
      })
    }
  }
}

const load = (name) => import(pathToFileURL(join(ENGINE, name)).href)
const [tokenizerMod, samplerMod, mesherMod, statsMod, blueprintMod, sheetMod, dimsMod, pyMod, engineMod,
  exportersMod] = await Promise.all([
  load('tokenizer.ts'), load('sampler.ts'), load('mesher.ts'), load('meshStats.ts'), load('blueprint.ts'),
  load('sheet.ts'), load('dimensions.ts'), load('pyformat.ts'), load('engine.ts'), load('exporters.ts'),
])

const fixtures = JSON.parse(readFileSync(join(HERE, 'fixtures', 'text2voxel_fixtures.json'), 'utf8'))
const probe = JSON.parse(readFileSync(join(HERE, 'fixtures', 'unicode_probe.json'), 'utf8'))

// ---- reporting -------------------------------------------------------------

const results = []
function report(status, name, detail) {
  results.push({ status, name, detail })
  const tag = status === 'pass' ? 'PASS' : status === 'fail' ? 'FAIL' : 'SKIP'
  console.log(`${tag}  ${name}${detail ? ` - ${detail}` : ''}`)
}
const check = (ok, name, detail) => report(ok ? 'pass' : 'fail', name, detail)
async function section(name, fn) {
  try {
    await fn()
  } catch (error) {
    report('fail', name, `threw: ${error?.stack ?? error}`)
  }
}
const fmt = (value, digits = 3) => (typeof value === 'number' ? value.toExponential(digits) : String(value))
const sameJson = (a, b) => JSON.stringify(a) === JSON.stringify(b)

// ---- 1. tokenizer -----------------------------------------------------------

const tokenizerJsonPath = join(MODELS, 'minilm', 'tokenizer.json')
const haveTokenizer = existsSync(tokenizerJsonPath)

await section('tokenizer ids', async () => {
  if (!haveTokenizer) {
    report(skipModels ? 'skip' : 'fail', 'tokenizer ids', `${tokenizerJsonPath} missing`)
    return
  }
  const tokenizer = new tokenizerMod.BertTokenizer(JSON.parse(readFileSync(tokenizerJsonPath, 'utf8')),
    fixtures.max_tokens)
  const cases = fixtures.tokens
  const bad = cases.filter((c) => !sameJson(tokenizer.encode(c.text), c.ids))
  for (const c of bad.slice(0, verbose ? 50 : 5)) {
    console.log(`      ${JSON.stringify(c.text).slice(0, 90)}\n        py ${c.ids.join(',')}\n        js ${tokenizer.encode(c.text).join(',')}`)
  }
  const truncated = cases.filter((c) => c.ids.length === fixtures.max_tokens).length
  check(cases.length >= 300 && fixtures.caption_count >= 150 && bad.length === 0, 'tokenizer ids',
    `${cases.length - bad.length}/${cases.length} strings identical to Python tokenizers ` +
    `${fixtures.tokenizers_version} (${fixtures.caption_count} captions, ${truncated} truncated at ` +
    `${fixtures.max_tokens})`)
})

await section('normaliser, every code point', async () => {
  const block = probe.block
  const bad = []
  let outputs = []
  for (let cp = 0; cp < 0x110000; cp += 1) {
    if (cp % block === 0 && cp) {
      const digest = createHash('sha256').update(outputs.join('\x01'), 'utf8').digest('hex').slice(0, 16)
      if (digest !== probe.normalized_digests[cp / block - 1]) bad.push(cp - block)
      outputs = []
    }
    outputs.push(cp >= 0xd800 && cp <= 0xdfff ? '' : tokenizerMod.normalizeText(String.fromCodePoint(cp)))
  }
  const last = createHash('sha256').update(outputs.join('\x01'), 'utf8').digest('hex').slice(0, 16)
  if (last !== probe.normalized_digests.at(-1)) bad.push(0x110000 - block)
  const hex = (cp) => `U+${cp.toString(16).toUpperCase().padStart(4, '0')}`
  check(bad.length === 0, 'normaliser, every code point',
    bad.length === 0
      ? `all ${probe.normalized_digests.length} blocks of ${block} code points match the Rust BertNormalizer`
      : `${bad.length} block(s) differ, first: ${bad.slice(0, 6).map((s) => `${hex(s)}-${hex(s + block - 1)}`).join(', ')} ` +
        '(browser Unicode tables newer/older than the Rust crate\'s?)')

  // Pre-tokenizer classes decode correctly.
  let wrong = 0
  const splitCases = [...probe.split_space.map((cp) => [cp, 'space'])]
  for (const [start, end] of probe.punct) for (let cp = start; cp <= end; cp += 1) splitCases.push([cp, 'punct'])
  for (const [cp, kind] of splitCases) {
    const char = String.fromCodePoint(cp)
    const words = tokenizerMod.preTokenize(`a${char}a`)
    if (!sameJson(words, kind === 'space' ? ['a', 'a'] : ['a', char, 'a'])) wrong += 1
  }
  check(wrong === 0, 'pre-tokenizer classes', `${splitCases.length - wrong}/${splitCases.length} ` +
    'whitespace/punctuation code points split as in Rust')
})

// ---- 3a. timesteps -----------------------------------------------------------

await section('DDIM timesteps', async () => {
  const bad = Object.entries(fixtures.timesteps).filter(([steps, expected]) =>
    !sameJson(samplerMod.ddimTimesteps(1000, Number(steps)), expected))
  check(bad.length === 0, 'DDIM timesteps',
    `${Object.keys(fixtures.timesteps).length - bad.length}/${Object.keys(fixtures.timesteps).length} step counts ` +
    `equal numpy linspace(999, 0, n).round()${bad.length ? `; wrong: ${bad.map(([s]) => s).join(', ')}` : ''}`)
})

// ---- 4. mesher -------------------------------------------------------------------

function grid(fn, colour = [0.8, 0.3, 0.1]) {
  const r = 64
  const n = r ** 3
  const occupancy = new Float32Array(n)
  const rgb = new Float32Array(3 * n)
  for (let x = 0; x < r; x += 1) {
    for (let y = 0; y < r; y += 1) {
      for (let z = 0; z < r; z += 1) {
        const i = (x * r + y) * r + z
        occupancy[i] = fn(x, y, z)
        rgb[i] = colour[0]
        rgb[n + i] = colour[1]
        rgb[2 * n + i] = colour[2]
      }
    }
  }
  return { resolution: r, occupancy, rgb }
}
const sigmoid = (v) => 1 / (1 + Math.exp(-v))

function mesh(g, dims = null) {
  const shape = mesherMod.prepareField(g, 0.5)
  const raw = mesherMod.extractSurface(shape, g)
  const placement = mesherMod.computePlacement(raw, dims)
  const data = mesherMod.placeMesh(raw, placement)
  const topology = statsMod.meshTopology(data)
  const stats = statsMod.meshStatistics(data, 0, null, topology)
  return { shape, raw, placement, data, topology, stats }
}

await section('mesher: sphere', async () => {
  // Radius 20 voxels, centre (32, 32, 32); a smooth field like the decoder's.
  const radius = 20
  const g = grid((x, y, z) => sigmoid(2 * (radius - Math.hypot(x - 32, y - 32, z - 32))))
  const m = mesh(g)
  const { topology, stats, data } = m
  // Default scale: largest extent -> 1 m, so the sphere's radius is 0.5 m.
  const r = stats.height / 2
  const volumeError = Math.abs(stats.volume / ((4 / 3) * Math.PI * r ** 3) - 1)
  const areaError = Math.abs(stats.surface_area / (4 * Math.PI * r ** 2) - 1)
  let outward = 0
  let unit = 0
  const centre = [(stats.bbox_min[0] + stats.bbox_max[0]) / 2, stats.height / 2, (stats.bbox_min[2] + stats.bbox_max[2]) / 2]
  const count = data.positions.length / 3
  for (let i = 0; i < count; i += 1) {
    const d = [0, 1, 2].map((k) => data.positions[i * 3 + k] - centre[k])
    const n = [0, 1, 2].map((k) => data.normals[i * 3 + k])
    if (d[0] * n[0] + d[1] * n[1] + d[2] * n[2] > 0) outward += 1
    if (Math.abs(Math.hypot(...n) - 1) < 1e-4) unit += 1
  }
  const signed = statsMod.areaAndVolume(data).signedVolume
  check(topology.watertight && topology.euler === 2 && stats.is_watertight, 'mesher: sphere closed',
    `${stats.vertices} vertices, ${stats.faces} faces, every edge in exactly 2 faces with opposite winding ` +
    `(${topology.boundaryEdges} open, ${topology.duplicateEdges} duplicate), Euler characteristic ${topology.euler}`)
  check(signed > 0 && outward === count && unit === count, 'mesher: sphere outward',
    `signed volume ${signed.toFixed(5)} > 0; ${outward}/${count} normals point outward, ${unit}/${count} unit length`)
  check(volumeError < 0.01 && areaError < 0.02, 'mesher: sphere measures',
    `volume ${stats.volume.toFixed(6)} m3 vs 4/3 pi r3 (error ${(volumeError * 100).toFixed(2)}%), ` +
    `area ${stats.surface_area.toFixed(6)} m2 (error ${(areaError * 100).toFixed(2)}%)`)
  check(Math.abs(stats.bbox_min[1]) < 1e-6 && Math.abs(stats.bbox_min[0] + stats.bbox_max[0]) < 1e-6 &&
    Math.abs(stats.bbox_min[2] + stats.bbox_max[2]) < 1e-6 && Math.abs(Math.max(stats.width, stats.height, stats.depth) - 1) < 1e-6,
  'mesher: placement', `base y = ${stats.bbox_min[1]}, centred on x/z, largest extent ${Math.max(stats.width, stats.height, stats.depth)} m`)

  // Scale contract.
  const byHeight = mesh(g, { height: 450, length: 9999 }).stats
  const byFootprint = mesh(g, { length: 1200, width: 1800 }).stats
  check(Math.abs(byHeight.height - 0.45) < 1e-6 && Math.abs(byFootprint.width - 1.8) < 1e-5 &&
    Math.abs(byFootprint.width / byFootprint.height - stats.width / stats.height) < 1e-5,
  'scale contract', `height 450 mm -> ${byHeight.height} m tall; length 1200 / width 1800 mm -> ` +
    `max(x, z) = ${Math.max(byFootprint.width, byFootprint.depth)} m; proportions kept`)
})

await section('mesher: hard grids', async () => {
  let seed = 12345
  const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647
  const cases = {
    'hollow sphere': grid((x, y, z) => {
      const d = Math.hypot(x - 32, y - 32, z - 32)
      return d < 24 && d > 14 ? 0.9 : 0.1
    }),
    'random noise': grid(() => random()),
    '3D checkerboard': grid((x, y, z) => ((x + y + z) % 2 === 0 && x > 4 && y > 4 && z > 4 && x < 59 && y < 59 && z < 59 ? 0.9 : 0.1)),
    'touches the grid walls': grid((x, y, z) => (Math.hypot(x - 32, y - 32, z - 32) < 40 ? 1 : 0)),
  }
  for (const [name, g] of Object.entries(cases)) {
    const { topology, stats } = mesh(g)
    const expectEuler = name === 'hollow sphere' ? 4 : null
    check(topology.watertight && (expectEuler === null || topology.euler === expectEuler), `mesher: ${name}`,
      `${stats.faces} faces, watertight=${topology.watertight} (${topology.boundaryEdges} open, ` +
      `${topology.duplicateEdges} duplicate edges), Euler ${topology.euler}`)
  }
  // Floater removal and the near-empty fallback.
  const floaters = mesh(grid((x, y, z) => (Math.hypot(x - 32, y - 32, z - 32) < 15 || (x === 3 && y === 3 && z === 3) ? 1 : 0)))
  check(floaters.shape.droppedComponents === 1 && floaters.shape.droppedVoxels === 1 && floaters.topology.euler === 2,
    'mesher: floaters', `a 1-voxel speck is dropped (${floaters.shape.droppedComponents} component, ` +
    `min ${mesherMod.MIN_COMPONENT_VOXELS} voxels), the body is kept`)
  const faint = mesh(grid((x, y, z) => 0.45 * sigmoid(2 * (10 - Math.hypot(x - 32, y - 32, z - 32)))))
  check(faint.shape.iso < 0.5 && faint.shape.warnings.length === 1 && faint.topology.watertight, 'mesher: near-empty output',
    `nothing reaches 0.5, surface taken at ${faint.shape.iso.toFixed(3)} with a warning`)
})

// ---- 5. blueprint ------------------------------------------------------------------

const inBox = (x, y, z, [x0, x1, y0, y1, z0, z1]) => x >= x0 && x <= x1 && y >= y0 && y <= y1 && z >= z0 && z <= z1

await section('blueprint: box', async () => {
  // 40 x 30 x 10 voxels -> 1.0 x 0.75 x 0.25 m by the default scale.
  const m = mesh(grid((x, y, z) => (inBox(x, y, z, [10, 49, 5, 34, 20, 29]) ? 1 : 0)))
  const drawing = blueprintMod.technicalDrawing(m.shape, m.placement, m.stats)
  const expected = { front: [1, 0.75], side: [0.25, 0.75], top: [1, 0.25] }
  for (const [view, [w, h]] of Object.entries(expected)) {
    const data = drawing.views[view]
    const points = data.outline.flat()
    const xs = points.map((p) => p[0])
    const ys = points.map((p) => p[1])
    const spanX = Math.max(...xs) - Math.min(...xs)
    const spanY = Math.max(...ys) - Math.min(...ys)
    const closed = data.outline.every((loop) => sameJson(loop[0], loop.at(-1)))
    check(Math.abs(data.width - w) < 1e-6 && Math.abs(data.height - h) < 1e-6 && Math.abs(spanX - w) < 1e-6 &&
      Math.abs(spanY - h) < 1e-6 && data.outline.length === 1 && closed && data.inline.length === 0 && data.hidden.length === 0,
    `blueprint: box ${view}`, `width ${data.width} x height ${data.height} m (expected ${w} x ${h}); outline ` +
      `${data.outline.length} closed loop spanning ${spanX.toFixed(6)} x ${spanY.toFixed(6)}; ` +
      `${data.inline.length} inline, ${data.hidden.length} hidden`)
  }
})

await section('blueprint: chair', async () => {
  const parts = [
    [16, 47, 28, 31, 16, 47], // seat
    [16, 47, 32, 60, 16, 19], // back, at the rear (-z)
    [16, 19, 0, 27, 16, 19], [44, 47, 0, 27, 16, 19], [16, 19, 0, 27, 44, 47], [44, 47, 0, 27, 44, 47],
  ]
  const m = mesh(grid((x, y, z) => (parts.some((box) => inBox(x, y, z, box)) ? 1 : 0)), { height: 900 })
  const { views } = blueprintMod.technicalDrawing(m.shape, m.placement, m.stats)
  // Front: the seat's front edge sits in front of the back rest -> one inline
  // line. Plan: the legs are under the seat -> hidden lines.
  check(views.front.inline.length >= 1 && views.top.hidden.length >= 1 && views.top.inline.length >= 1 &&
    Math.abs(views.front.height - 0.9) < 1e-6,
  'blueprint: chair', `front ${views.front.outline.length} outline / ${views.front.inline.length} inline / ` +
    `${views.front.hidden.length} hidden; plan ${views.top.outline.length} / ${views.top.inline.length} / ` +
    `${views.top.hidden.length}; height ${views.front.height} m`)
})

// ---- 6. ports of backend code ---------------------------------------------------------

await section('parseDimensions', async () => {
  const bad = fixtures.dimensions.filter((c) => !sameJson(dimsMod.parseDimensions(c.prompt),
    { length: c.length, width: c.width, height: c.height }))
  for (const c of bad.slice(0, 8)) {
    console.log(`      ${JSON.stringify(c.prompt)}: py ${JSON.stringify([c.length, c.width, c.height])} ` +
      `js ${JSON.stringify(dimsMod.parseDimensions(c.prompt))}`)
  }
  check(bad.length === 0, 'parseDimensions', `${fixtures.dimensions.length - bad.length}/${fixtures.dimensions.length} ` +
    'prompts parse exactly as backend prompt_parser._extract_dimensions')
})

await section('Python number formatting', async () => {
  const cases = [
    [pyMod.pyFixed(0.125, 2), '0.12'], [pyMod.pyFixed(0.375, 2), '0.38'], [pyMod.pyFixed(2.675, 2), '2.67'],
    [pyMod.pyFixed(2.5, 0), '2'], [pyMod.pyFixed(3.5, 0), '4'], [pyMod.pyFixed(-0.0, 2), '-0.00'],
    [pyMod.pyFixed(-0.001, 2), '-0.00'], [pyMod.pyFixed(1e21, 1), '1000000000000000000000.0'],
    [String(pyMod.pyRound(1.0625, 3)), '1.062'], [pyMod.pyFloatRepr(2), '2.0'], [pyMod.pyFloatRepr(1e-5), '1e-05'],
    [pyMod.pyThousands(1234567), '1,234,567'],
  ]
  const bad = cases.filter(([got, want]) => got !== want)
  check(bad.length === 0, 'Python number formatting', `${cases.length - bad.length}/${cases.length} ties-to-even / repr cases` +
    (bad.length ? `; got ${bad.map(([g, w]) => `${g} (want ${w})`).join(', ')}` : ''))
})

await section('blueprint sheet SVG', async () => {
  let firstDiff = null
  const bad = fixtures.sheets.filter((c) => {
    const svg = sheetMod.renderBlueprintSheet(c.drawing, c.stats, c.spec, c.name, c.theme)
    if (svg === c.svg) return false
    if (!firstDiff) {
      let i = 0
      while (i < svg.length && svg[i] === c.svg[i]) i += 1
      firstDiff = `at char ${i}: py ...${JSON.stringify(c.svg.slice(Math.max(0, i - 40), i + 40))} ` +
        `js ...${JSON.stringify(svg.slice(Math.max(0, i - 40), i + 40))}`
    }
    return true
  })
  if (firstDiff) console.log(`      ${firstDiff}`)
  check(bad.length === 0, 'blueprint sheet SVG', `${fixtures.sheets.length - bad.length}/${fixtures.sheets.length} ` +
    'sheets byte-identical to backend render_blueprint_sheet (both themes, specs, escaping, tie rounding)')
})

// ---- 2, 3b, 7. with the model files -----------------------------------------------

const modelFiles = {
  meta: join(MODELS, 'text2voxel', 'meta.json'),
  prior: join(MODELS, 'text2voxel', 'prior.onnx'),
  decoder: join(MODELS, 'text2voxel', 'decoder.onnx'),
  minilm: join(MODELS, 'minilm', 'model_quantized.onnx'),
  tokenizer: tokenizerJsonPath,
}
const missing = Object.values(modelFiles).filter((path) => !existsSync(path))

if (skipModels || missing.length) {
  const why = skipModels ? '--skip-models' : `missing ${missing.join(', ')}`
  for (const name of ['embeddings', 'sampler parity', 'end-to-end generation']) {
    report(skipModels ? 'skip' : 'fail', name, why)
  }
} else {
  const ort = await import('onnxruntime-web/wasm')
  const require = createRequire(import.meta.url)
  ort.env.wasm.numThreads = 1
  ort.env.wasm.wasmBinary = readFileSync(require.resolve('onnxruntime-web/ort-wasm-simd-threaded.wasm'))
  const meta = JSON.parse(readFileSync(modelFiles.meta, 'utf8'))
  let engine = null
  await section('engine init', async () => {
    const started = performance.now()
    engine = await engineMod.Text2VoxelEngine.create(ort, {
      meta,
      tokenizer: JSON.parse(readFileSync(modelFiles.tokenizer, 'utf8')),
      minilm: readFileSync(modelFiles.minilm),
      prior: readFileSync(modelFiles.prior),
      decoder: readFileSync(modelFiles.decoder),
    })
    report('pass', 'engine init', `onnxruntime-web ${ort.env.versions.web} (wasm, 1 thread), ${meta.name} ` +
      `v${meta.version}${meta.smoke ? ' [placeholder smoke model]' : ''}, ${Math.round(performance.now() - started)} ms`)
  })

  if (engine) {
    await section('embeddings', async () => {
      let worst = 1
      let worstText = ''
      let maxAbs = 0
      let exact = 0
      for (const c of fixtures.embeddings) {
        // Copy out of Node's shared Buffer pool before viewing as float32.
        const expected = new Float32Array(Uint8Array.from(Buffer.from(c.embedding, 'base64')).buffer)
        const got = await engine.embed(c.text)
        if (expected.length !== got.length) throw new Error(`embedding length ${got.length} != ${expected.length}`)
        let dot = 0
        let na = 0
        let nb = 0
        for (let k = 0; k < expected.length; k += 1) {
          dot += got[k] * expected[k]
          na += got[k] ** 2
          nb += expected[k] ** 2
          maxAbs = Math.max(maxAbs, Math.abs(got[k] - expected[k]))
        }
        const cosine = dot / Math.sqrt(na * nb)
        if (cosine > 0.99999) exact += 1
        if (!(cosine >= worst)) { // also catches NaN
          worst = cosine
          worstText = c.text
        }
      }
      check(worst > 0.9999 && Number.isFinite(maxAbs), 'embeddings', `${fixtures.embeddings.length} strings vs Python TextEncoder: min cosine ` +
        `${worst.toFixed(7)} (${JSON.stringify(worstText.slice(0, 40))}), ${exact}/${fixtures.embeddings.length} above ` +
        `0.99999, max |diff| ${fmt(maxAbs)} (int8 dynamic quantisation, WASM vs native kernels)`)
    })

    await section('sampler parity', async () => {
      const ref = meta.reference
      if (!ref) {
        report('fail', 'sampler parity', 'meta.json has no reference block')
        return
      }
      // meta.reference comes from the training machine. The int8 MiniLM gives
      // slightly different embeddings on different CPUs (~8e-3 per component
      // Kaggle vs a Windows laptop, native or WASM alike), which 50 guided
      // steps amplify to ~8e-2 in the latent. So the sampler is held to a
      // tight tolerance with the condition recorded on THIS machine by the
      // Python fixture script, and the drift from the training machine is
      // reported for information.
      const samplerFixture = join(HERE, 'fixtures', 'sampler_reference.json')
      if (!existsSync(samplerFixture)) {
        report('fail', 'sampler parity', 'fixtures/sampler_reference.json missing - run ' +
          'make_text2voxel_fixtures.py --sampler-only with the backend venv')
        return
      }
      const local = JSON.parse(readFileSync(samplerFixture, 'utf8'))
      const priorSha = createHash('sha256').update(readFileSync(join(MODELS, 'text2voxel', 'prior.onnx'))).digest('hex')
      if (local.prior_sha256 !== priorSha) {
        report('fail', 'sampler parity', `sampler_reference.json is for prior ${local.prior_sha256.slice(0, 12)}, ` +
          `installed prior is ${priorSha.slice(0, 12)} - rerun make_text2voxel_fixtures.py --sampler-only`)
        return
      }
      const floats = (b64) => { const buf = Buffer.from(b64, 'base64'); return Array.from(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)) }
      const localCond = floats(local.cond)
      const localLatent = floats(local.latent)
      const cond = await engine.embed(ref.prompt)
      let dot = 0
      for (let k = 0; k < cond.length; k++) dot += cond[k] * localCond[k]
      const headDiff = Math.max(...ref.cond_head.map((v, k) => Math.abs(v - cond[k])))
      check(dot > 0.9999, 'reference condition', `embed(${JSON.stringify(ref.prompt)}) vs Python on this machine: ` +
        `cosine ${dot.toFixed(7)} (> 0.9999); vs the training machine's reference.cond_head: max |diff| ` +
        `${fmt(headDiff)} (cross-CPU int8 drift, informational)`)
      const started = performance.now()
      const latent = await engine.sample(localCond, ref.noise, meta.diffusion.guidance, meta.diffusion.sample_steps)
      const ms = performance.now() - started
      const diff = Math.max(...localLatent.map((v, k) => Math.abs(v - latent[k])))
      const kaggleDiff = Math.max(...ref.latent.map((v, k) => Math.abs(v - latent[k])))
      check(diff < 2e-3, 'sampler parity', `${meta.diffusion.sample_steps} DDIM steps, guidance ` +
        `${meta.diffusion.guidance}, B=2 per prior call, identical condition: max |latent - python| = ${fmt(diff)} ` +
        `(< 2e-3); vs training-machine reference.latent ${fmt(kaggleDiff)} (informational); ${Math.round(ms)} ms, ` +
        `${(ms / meta.diffusion.sample_steps).toFixed(1)} ms/step`)
    })

    await section('end-to-end generation', async () => {
      const request = { prompt: 'a red office chair with armrests and wheels', seed: 42, guidance: meta.diffusion.guidance,
        steps: meta.diffusion.sample_steps, dimensionsMm: { height: 950 } }
      const stages = []
      const first = await engine.generate(request, (stage) => {
        if (stages.at(-1) !== stage) stages.push(stage)
      })
      const again = await engine.generate(request)
      const other = await engine.generate({ ...request, seed: 43 })
      const glb = new Uint8Array(first.glb)
      const magic = new TextDecoder().decode(glb.slice(0, 4))
      const version = new DataView(first.glb).getUint32(4, true)
      const identical = Buffer.compare(Buffer.from(first.glb), Buffer.from(again.glb)) === 0
      const differs = Buffer.compare(Buffer.from(first.glb), Buffer.from(other.glb)) !== 0
      const s = first.stats
      // Same request in a browser gives the same bytes (single-threaded wasm is deterministic).
      const sha = createHash('sha256').update(glb).digest('hex').slice(0, 16)
      check(magic === 'glTF' && version === 2 && s.file_size_bytes === glb.byteLength && s.is_watertight &&
        Math.abs(s.height - 0.95) < 1e-6 && identical && differs && sameJson(stages, ['encoding', 'sampling', 'decoding', 'meshing', 'done']),
      'end-to-end generation', `GLB ${glb.byteLength} bytes (glTF v${version}, sha256 ${sha}...), ${s.vertices} vertices / ${s.faces} faces, ` +
        `watertight ${s.is_watertight}, ${s.width} x ${s.height} x ${s.depth} m, ${first.occupiedVoxels} voxels at iso ` +
        `${first.isoLevel.toFixed(3)}; seed 42 twice -> identical GLB: ${identical}; seed 43 differs: ${differs}; ` +
        `timings ${JSON.stringify(first.timingsMs)}`)
      for (const warning of first.warnings) console.log(`      warning: ${warning}`)

      // GLB round trip through three's GLTFLoader, then the other formats.
      const back = await exportersMod.meshDataFromGlb(first.glb)
      const colourDiff = Math.max(...Array.from(back.colors.slice(0, 300), (v, k) => Math.abs(v - first.mesh.colors[k])))
      const stl = new Uint8Array(await (await exportersMod.exportMeshData(first.mesh, 'stl')).arrayBuffer())
      const stlTriangles = new DataView(stl.buffer).getUint32(80, true)
      const obj = await (await exportersMod.exportMeshData(first.mesh, 'obj')).text()
      const objVertices = (obj.match(/^v /gm) ?? []).length
      const objColoured = /^v (\S+ ){5}\S+$/m.test(obj)
      const ply = new Uint8Array(await (await exportersMod.exportMeshData(first.mesh, 'ply')).arrayBuffer())
      const plyHeader = new TextDecoder().decode(ply.slice(0, 400))
      check(back.positions.length === first.mesh.positions.length && colourDiff < 1e-6 && stlTriangles === s.faces &&
        stl.byteLength === 84 + 50 * s.faces && objVertices === s.vertices && objColoured &&
        plyHeader.includes('binary_little_endian') && plyHeader.includes('property uchar red') &&
        plyHeader.includes(`element vertex ${s.vertices}`),
      'exports', `GLB -> GLTFLoader: ${back.positions.length / 3} vertices, COLOR_0 max |diff| ${fmt(colourDiff)}; ` +
        `STL ${stlTriangles} triangles; OBJ ${objVertices} coloured vertices; PLY binary with RGB, ` +
        `${s.vertices} vertices`)

      const views = first.blueprint.views
      check(first.blueprint.units === 'm' && first.blueprint.spec === null &&
        Math.abs(views.front.width - s.width) < 1e-6 && Math.abs(views.side.width - s.depth) < 1e-6 &&
        Math.abs(views.top.height - s.depth) < 1e-6 && views.front.outline.length >= 1,
      'generated blueprint', ['front', 'side', 'top'].map((v) => `${v} ${views[v].width}x${views[v].height} m, ` +
        `${views[v].outline.length}/${views[v].inline.length}/${views[v].hidden.length} paths`).join('; '))
    })
    await engine.release()
  }
}

// ---- summary ---------------------------------------------------------------

const failed = results.filter((r) => r.status === 'fail')
const skipped = results.filter((r) => r.status === 'skip')
console.log(`\n${results.length - failed.length - skipped.length} passed, ${failed.length} failed, ${skipped.length} skipped`)
process.exit(failed.length ? 1 : 0)
