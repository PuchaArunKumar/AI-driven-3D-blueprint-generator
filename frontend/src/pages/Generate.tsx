import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'
import { Link } from 'react-router-dom'
import { useGLTF } from '@react-three/drei'
import {
  AlertTriangle,
  Cpu,
  Dices,
  Download,
  FileText,
  FlaskConical,
  Loader2,
  PlayCircle,
  RotateCcw,
  Shuffle,
  Sparkles,
  Wand2,
  X,
} from 'lucide-react'
import { BASE_URL, STATIC_MODE } from '../lib/api'
import {
  blueprintSheetSvg,
  disposeResult,
  exportMesh,
  generate,
  isText2VoxelSupported,
  loadModelInfo,
  parseDimensions,
  preloadModel,
} from '../lib/text2voxel'
import type { T2VModelInfo, T2VProgress, T2VResult } from '../lib/text2voxel'
import BlueprintView from '../components/BlueprintView'
import type { SheetTheme } from '../components/BlueprintView'
import GenerationProgress from '../components/GenerationProgress'
import ModelCard from '../components/ModelCard'
import SessionHistory from '../components/SessionHistory'
import Viewer3D, { formatLength } from '../components/Viewer3D'

/** Chairs and tables have human captions, so detail in the prompt is used. */
const DETAILED_EXAMPLES = [
  'a modern minimalist oak dining chair with tapered legs',
  'a red office chair with armrests and wheels',
  'a round glass coffee table with metal legs',
  'a long wooden dining table, 180 x 90 x 75 cm',
  'a black leather armchair',
]

/** Other ModelNet40 categories: the model knows the category and a plain colour. */
const CATEGORY_EXAMPLES = [
  'a blue sofa',
  'a silver sports car',
  'an airplane',
  'a floor lamp',
  'a guitar',
  'a toilet',
  'a green bathtub',
  'a computer monitor',
  'a park bench',
  'a wooden bookshelf',
  'a white vase',
  'a double bed',
]

const MESH_FORMATS: { id: 'glb' | 'stl' | 'obj' | 'ply'; label: string; note: string }[] = [
  { id: 'glb', label: 'GLB', note: 'Binary glTF with vertex colours - web, Blender, Unity' },
  { id: 'stl', label: 'STL', note: 'Triangles only, for slicers and 3D printing' },
  { id: 'obj', label: 'OBJ', note: 'Wavefront, opens almost anywhere' },
  { id: 'ply', label: 'PLY', note: 'Keeps per-vertex colour' },
]

const TIMING_LABEL: Record<string, string> = {
  load: 'Load model',
  loading: 'Load model',
  encode: 'Encode prompt',
  encoding: 'Encode prompt',
  sample: 'Sample latent',
  sampling: 'Sample latent',
  decode: 'Decode voxels',
  decoding: 'Decode voxels',
  mesh: 'Mesh',
  meshing: 'Mesh',
  blueprint: 'Blueprint',
  export: 'Export GLB',
  total: 'Total',
}

const HISTORY_LIMIT = 8
const MAX_SEED = 4294967295 // seeds are unsigned 32-bit (the sampler RNG is mulberry32)
const DEFAULT_GUIDANCE = 3
const DEFAULT_STEPS = 50

type Tab = '3d' | 'blueprint'
type DimensionKey = 'length' | 'width' | 'height'
type DimensionInputs = Record<DimensionKey, string>
type Dimensions = { length: number | null; width: number | null; height: number | null }

interface Entry {
  id: number
  result: T2VResult
  /** Title for the blueprint sheet and file names. */
  name: string
  dimensionsMm: Dimensions | null
}

const EMPTY_DIMENSIONS: DimensionInputs = { length: '', width: '', height: '' }

const MODEL_FILES_HINT = STATIC_MODE
  ? 'The model files could not be downloaded. Check your connection and retry; if it keeps failing, this deployment is missing them.'
  : `The model files are served from ${BASE_URL}models/. In a local checkout fetch them with: node frontend/scripts/fetch-models.mjs`

/** The hint for a failed model download, unless the engine's message already gives it. */
function modelFilesHint(message: string): string | undefined {
  return STATIC_MODE || !/fetch-models/.test(message) ? MODEL_FILES_HINT : undefined
}

function randomSeed(): number {
  return Math.floor(Math.random() * MAX_SEED)
}

function toInputs(dimensions: Dimensions): DimensionInputs {
  const show = (value: number | null) => (value === null ? '' : String(Math.round(value * 10) / 10))
  return { length: show(dimensions.length), width: show(dimensions.width), height: show(dimensions.height) }
}

function dimensionsFromPrompt(prompt: string): DimensionInputs {
  try {
    return toInputs(parseDimensions(prompt))
  } catch {
    // A parser hiccup should never block typing; the fields just stay blank.
    return EMPTY_DIMENSIONS
  }
}

/** Parse the size fields; `null` means "auto", an Error string means invalid. */
function readDimensions(inputs: DimensionInputs): Dimensions | null | string {
  const parsed: Dimensions = { length: null, width: null, height: null }
  for (const key of ['length', 'width', 'height'] as DimensionKey[]) {
    const raw = inputs[key].trim()
    if (!raw) continue
    const value = Number(raw)
    if (!Number.isFinite(value) || value <= 0) return `${key[0].toUpperCase()}${key.slice(1)} must be a positive number of millimetres.`
    parsed[key] = value
  }
  return parsed.length === null && parsed.width === null && parsed.height === null ? null : parsed
}

function titleFor(prompt: string): string {
  const text = prompt.trim().replace(/\s+/g, ' ')
  const short = text.length > 60 ? `${text.slice(0, 57)}...` : text
  return short.charAt(0).toUpperCase() + short.slice(1)
}

function fileStem(result: T2VResult): string {
  const slug = result.prompt
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48)
    .replace(/-+$/, '')
  return `${slug || 'text2voxel'}-s${result.seed}`
}

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  // Give the browser time to start the download before the URL is released.
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`
}

/** Turn a thrown value into a message plus, where we can tell, what to do about it. */
function describeFailure(cause: unknown): { message: string; hint?: string } {
  const message = cause instanceof Error ? cause.message : String(cause)
  if (/fetch|network|404|not found|failed to load|http/i.test(message)) {
    return { message, hint: modelFilesHint(message) }
  }
  if (/memory|allocat|out of bounds/i.test(message)) {
    return {
      message,
      hint: 'The browser ran out of memory. Close other tabs, or try a desktop browser.',
    }
  }
  if (/wasm|webassembly|worker/i.test(message)) {
    return {
      message,
      hint: 'The WebAssembly runtime could not start. Try a current Chrome, Edge, Firefox or Safari.',
    }
  }
  return { message }
}

export default function Generate() {
  const supported = useMemo(() => {
    try {
      return isText2VoxelSupported()
    } catch {
      return false
    }
  }, [])

  const [info, setInfo] = useState<T2VModelInfo | null>(null)
  const [infoError, setInfoError] = useState<string | null>(null)
  const [infoAttempt, setInfoAttempt] = useState(0)

  const [prompt, setPrompt] = useState('')
  const [seedInput, setSeedInput] = useState(() => String(randomSeed()))
  const [guidance, setGuidance] = useState<number | null>(null)
  const [steps, setSteps] = useState<number | null>(null)
  const [dimensions, setDimensions] = useState<DimensionInputs>(EMPTY_DIMENSIONS)
  const [dimensionsEdited, setDimensionsEdited] = useState(false)

  const [busy, setBusy] = useState<'preload' | 'generate' | null>(null)
  const [progress, setProgress] = useState<T2VProgress | null>(null)
  const [modelReady, setModelReady] = useState(false)
  const [error, setError] = useState<{ message: string; hint?: string } | null>(null)

  const [history, setHistory] = useState<Entry[]>([])
  const [currentId, setCurrentId] = useState<number | null>(null)
  const [tab, setTab] = useState<Tab>('3d')
  const [exporting, setExporting] = useState<string | null>(null)
  const [sheetPreview, setSheetPreview] = useState<{ url: string; theme: SheetTheme } | null>(null)
  // Snapshot of the 3D view per result, used as the blueprint's isometric panel.
  const [isoShots, setIsoShots] = useState<Record<number, string>>({})

  // The list lives in a ref as well, so disposal never runs inside a state updater.
  const historyRef = useRef<Entry[]>([])
  const nextId = useRef(1)
  const mounted = useRef(true)
  const promptRef = useRef<HTMLTextAreaElement>(null)
  const resultRef = useRef<HTMLDivElement>(null)
  const previewClose = useRef<HTMLButtonElement>(null)

  const effectiveGuidance = guidance ?? info?.guidance ?? DEFAULT_GUIDANCE
  const defaultSteps = Math.min(Math.max(info?.sampleSteps ?? DEFAULT_STEPS, 20), 100)
  const effectiveSteps = steps ?? defaultSteps
  const current = history.find((entry) => entry.id === currentId) ?? null
  const hasSurface = Boolean(current && current.result.stats.triangles > 0)

  // ------------------------------------------------------------ lifecycle
  useEffect(() => {
    let cancelled = false
    setInfoError(null)
    loadModelInfo()
      .then((loaded) => {
        if (!cancelled) setInfo(loaded)
      })
      .catch((cause) => {
        if (!cancelled) setInfoError(cause instanceof Error ? cause.message : String(cause))
      })
    return () => {
      cancelled = true
    }
  }, [infoAttempt])

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
      // Leaving the page drops the session: free every mesh and object URL.
      historyRef.current.forEach(release)
      historyRef.current = []
    }
  }, [])

  // The sheet preview owns an object URL; release it when the dialog closes.
  useEffect(() => {
    if (!sheetPreview) return
    previewClose.current?.focus()
    const onKey = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') setSheetPreview(null)
    }
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('keydown', onKey)
      URL.revokeObjectURL(sheetPreview.url)
    }
  }, [sheetPreview])

  // --------------------------------------------------------------- history
  function release(entry: Entry) {
    try {
      useGLTF.clear(entry.result.glbUrl)
    } catch {
      // Not cached (never shown) - nothing to clear.
    }
    disposeResult(entry.result)
  }

  const commitHistory = (next: Entry[]) => {
    historyRef.current = next
    setHistory(next)
    const keep = new Set(next.map((entry) => entry.id))
    setIsoShots((shots) =>
      Object.fromEntries(Object.entries(shots).filter(([id]) => keep.has(Number(id)))),
    )
  }

  const addResult = (result: T2VResult, dimensionsMm: Dimensions | null) => {
    const entry: Entry = { id: nextId.current++, result, name: titleFor(result.prompt), dimensionsMm }
    const next = [entry, ...historyRef.current]
    next.slice(HISTORY_LIMIT).forEach(release)
    commitHistory(next.slice(0, HISTORY_LIMIT))
    setCurrentId(entry.id)
  }

  const removeResult = (id: number) => {
    const entry = historyRef.current.find((item) => item.id === id)
    const next = historyRef.current.filter((item) => item.id !== id)
    if (currentId === id) setCurrentId(next[0]?.id ?? null)
    commitHistory(next)
    if (entry) release(entry)
  }

  const clearHistory = () => {
    const dropped = historyRef.current
    setCurrentId(null)
    commitHistory([])
    dropped.forEach(release)
  }

  // ---------------------------------------------------------------- inputs
  const updatePrompt = (text: string) => {
    setPrompt(text)
    // Sizes follow the prompt until the user types their own.
    if (!dimensionsEdited) setDimensions(dimensionsFromPrompt(text))
  }

  const resetDimensionsToPrompt = () => {
    setDimensionsEdited(false)
    setDimensions(dimensionsFromPrompt(prompt))
  }

  const pickExample = (text: string) => {
    updatePrompt(text)
    promptRef.current?.focus()
  }

  // ------------------------------------------------------------ generation
  const runGeneration = async (seedOverride?: number) => {
    if (busy || !supported) return
    const text = prompt.trim()
    if (text.length < 2) {
      setError({ message: 'Describe a shape to generate - for example "a red office chair".' })
      promptRef.current?.focus()
      return
    }

    const seed = seedOverride ?? Number(seedInput)
    if (!Number.isInteger(seed) || seed < 0 || seed > MAX_SEED) {
      setError({ message: `The seed must be a whole number from 0 to ${MAX_SEED}.` })
      return
    }

    const dimensionsMm = readDimensions(dimensions)
    if (typeof dimensionsMm === 'string') {
      setError({ message: dimensionsMm })
      return
    }

    setError(null)
    setBusy('generate')
    setProgress({ stage: 'loading', fraction: 0, message: 'Starting' })
    try {
      const result = await generate(
        text,
        { seed, guidance: effectiveGuidance, steps: effectiveSteps, dimensionsMm },
        (update) => {
          if (mounted.current) setProgress(update)
        },
      )
      if (!mounted.current) {
        disposeResult(result)
        return
      }
      setModelReady(true)
      addResult(result, dimensionsMm)
      // On a phone the result renders below the form - bring it into view.
      if (window.matchMedia('(max-width: 1023px)').matches) {
        const smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches
        resultRef.current?.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' })
      }
    } catch (cause) {
      if (mounted.current) setError(describeFailure(cause))
    } finally {
      if (mounted.current) {
        setBusy(null)
        setProgress(null)
      }
    }
  }

  const runVariation = () => {
    const base = Number(seedInput)
    const seed = Number.isInteger(base) && base >= 0 && base < MAX_SEED ? base + 1 : 0
    setSeedInput(String(seed))
    void runGeneration(seed)
  }

  const preload = async () => {
    if (busy || !supported) return
    setError(null)
    setBusy('preload')
    setProgress({ stage: 'loading', fraction: 0, message: 'Downloading the model' })
    try {
      const loaded = await preloadModel((update) => {
        if (mounted.current) setProgress(update)
      })
      if (mounted.current) {
        setInfo(loaded)
        setModelReady(true)
      }
    } catch (cause) {
      if (mounted.current) setError(describeFailure(cause))
    } finally {
      if (mounted.current) {
        setBusy(null)
        setProgress(null)
      }
    }
  }

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    void runGeneration()
  }

  const onPromptKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault()
      void runGeneration()
    }
  }

  // -------------------------------------------------------------- outputs
  const downloadMesh = async (format: 'glb' | 'stl' | 'obj' | 'ply') => {
    if (!current) return
    setExporting(format)
    try {
      const blob = format === 'glb' ? current.result.glb : await exportMesh(current.result, format)
      saveBlob(blob, `${fileStem(current.result)}.${format}`)
    } catch (cause) {
      setError({ message: `${format.toUpperCase()} export failed: ${describeFailure(cause).message}` })
    } finally {
      setExporting(null)
    }
  }

  const sheetBlob = (entry: Entry, theme: SheetTheme): Blob | null => {
    try {
      return new Blob([blueprintSheetSvg(entry.result, entry.name, theme)], { type: 'image/svg+xml' })
    } catch (cause) {
      setError({ message: `The blueprint sheet could not be drawn: ${describeFailure(cause).message}` })
      return null
    }
  }

  const previewSheet = (theme: SheetTheme) => {
    if (!current) return
    const blob = sheetBlob(current, theme)
    if (blob) setSheetPreview({ url: URL.createObjectURL(blob), theme })
  }

  const downloadSheet = (theme: SheetTheme) => {
    if (!current) return
    const blob = sheetBlob(current, theme)
    if (blob) saveBlob(blob, `${fileStem(current.result)}-blueprint-${theme}.svg`)
  }

  /**
   * Leaving the 3D view, keep a picture of it for the blueprint's isometric
   * panel - the rendered model as the user last framed it, not a redrawing.
   */
  const selectTab = (next: Tab) => {
    if (next === 'blueprint' && tab === '3d' && current && hasSurface) {
      const canvas = document.querySelector<HTMLCanvasElement>('#t2v-result-panel canvas')
      try {
        const shot = canvas?.toDataURL('image/png')
        if (shot && shot.length > 100) setIsoShots((shots) => ({ ...shots, [current.id]: shot }))
      } catch {
        // A tainted or lost context just leaves the panel without a picture.
      }
    }
    setTab(next)
  }

  const switchTab = (next: Tab) => {
    selectTab(next)
    document.getElementById(`t2v-tab-${next}`)?.focus()
  }

  const locked = busy !== null
  // Without meta.json the model cannot load, so Generate waits for a successful retry.
  const canRun = supported && !locked && !infoError
  const fromPrompt = !dimensionsEdited && Object.values(dimensions).some(Boolean)
  const resolution = info?.resolution ?? 64
  const timings = current ? Object.entries(current.result.timingsMs) : []
  const timingTotal = current
    ? current.result.timingsMs.total ??
      timings.reduce((sum, [, value]) => sum + (Number.isFinite(value) ? value : 0), 0)
    : 0

  return (
    <div className="mx-auto max-w-[1600px] px-4 sm:px-6 py-6">
      {/* ---------------------------------------------------------- header */}
      <div className="flex flex-wrap items-start justify-between gap-3 mb-5">
        <div className="min-w-0">
          <h1 className="text-xl font-bold text-white tracking-tight">Text → 3D</h1>
          <p className="text-[12px] text-slate-500 max-w-2xl">
            Text2Voxel-64 turns a sentence into a coloured 3D shape, a blueprint and mesh
            files - entirely in your browser, with no server involved.
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {info && (
            <span className="chip border-ink-600 text-slate-400 font-mono">
              {info.name} v{info.version}
            </span>
          )}
          <span className="chip border-blueprint-500/40 text-blueprint-400 bg-blueprint-500/10">
            <Cpu className="w-3 h-3" />
            On-device · WebAssembly
          </span>
        </div>
      </div>

      {/* --------------------------------------------------------- banners */}
      {info?.smoke && (
        <div
          className="panel border-signal-warn/40 bg-signal-warn/5 p-4 mb-5 flex items-start gap-3"
          role="note"
        >
          <FlaskConical className="w-4 h-4 text-signal-warn shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="text-slate-200 font-medium">Placeholder model - untrained.</p>
            <p className="text-slate-400 text-[13px] mt-0.5">
              The files deployed here are a smoke-test build trained for seconds on synthetic
              boxes. Everything runs for real - download, sampling, meshing, blueprint,
              exports - but the shapes mean nothing until the trained Text2Voxel-64 weights
              replace them.
            </p>
          </div>
        </div>
      )}

      {!supported && (
        <div
          className="panel border-signal-err/40 bg-signal-err/5 p-4 mb-5 flex items-start gap-3"
          role="alert"
        >
          <AlertTriangle className="w-4 h-4 text-signal-err shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="text-slate-200 font-medium">This browser cannot run the model.</p>
            <p className="text-slate-400 text-[13px] mt-0.5">
              Text → 3D needs WebAssembly with SIMD and module Web Workers. A current
              Chrome, Edge, Firefox or Safari has both. The{' '}
              <Link to="/gallery" className="text-blueprint-400 hover:underline">
                demo projects
              </Link>{' '}
              still work here.
            </p>
          </div>
        </div>
      )}

      {infoError && (
        <div
          className="panel border-signal-err/40 bg-signal-err/5 p-4 mb-5 flex items-start gap-3"
          role="alert"
        >
          <AlertTriangle className="w-4 h-4 text-signal-err shrink-0 mt-0.5" />
          <div className="text-sm flex-1 min-w-0">
            <p className="text-slate-200 font-medium">The model files could not be read.</p>
            <p className="text-slate-400 text-[13px] mt-0.5 break-words">{infoError}</p>
            {modelFilesHint(infoError) && (
              <p className="text-slate-400 text-[13px] mt-0.5 break-words">
                {modelFilesHint(infoError)}
              </p>
            )}
          </div>
          <button
            type="button"
            className="btn-ghost !py-1 !px-2 !text-xs shrink-0"
            onClick={() => setInfoAttempt((attempt) => attempt + 1)}
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Retry
          </button>
        </div>
      )}

      {error && (
        <div
          className="panel border-signal-err/40 bg-signal-err/5 p-4 mb-5 flex items-start gap-3"
          role="alert"
        >
          <AlertTriangle className="w-4 h-4 text-signal-err shrink-0 mt-0.5" />
          <div className="text-sm flex-1 min-w-0">
            <p className="text-slate-200 break-words">{error.message}</p>
            {error.hint && (
              <p className="text-slate-400 text-[13px] mt-0.5 break-words">{error.hint}</p>
            )}
          </div>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-slate-500 hover:text-white text-xs"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[380px_minmax(0,1fr)] gap-5 items-start">
        {/* ----------------------------------------------------- left column */}
        <div className="space-y-5 min-w-0">
          <form className="panel" onSubmit={onSubmit} aria-labelledby="t2v-form-title">
            <div className="panel-head">
              <h2 id="t2v-form-title" className="text-sm font-semibold text-white">
                Describe a shape
              </h2>
            </div>

            <fieldset disabled={locked} className="p-4 space-y-4 min-w-0">
              <div>
                <label className="label" htmlFor="t2v-prompt">
                  Prompt
                </label>
                <textarea
                  id="t2v-prompt"
                  ref={promptRef}
                  className="field resize-none"
                  rows={3}
                  value={prompt}
                  maxLength={500}
                  placeholder="a red office chair with armrests and wheels"
                  aria-describedby="t2v-prompt-hint"
                  onChange={(event) => updatePrompt(event.target.value)}
                  onKeyDown={onPromptKey}
                />
                <p id="t2v-prompt-hint" className="mt-1 text-[10px] text-slate-500">
                  Ctrl + Enter to generate. One object, plain words - see what it knows below.
                </p>
              </div>

              <div>
                <p className="label">Chairs and tables - detailed</p>
                <div className="space-y-1">
                  {DETAILED_EXAMPLES.map((example) => (
                    <button
                      key={example}
                      type="button"
                      className="w-full text-left text-[12px] text-slate-400 hover:text-blueprint-400 leading-snug transition-colors"
                      onClick={() => pickExample(example)}
                    >
                      <Sparkles className="w-3 h-3 inline mr-1.5 -mt-0.5" />
                      {example}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <p className="label">Other categories - coarse</p>
                <div className="flex flex-wrap gap-1.5">
                  {CATEGORY_EXAMPLES.map((example) => (
                    <button
                      key={example}
                      type="button"
                      className="chip border-ink-700 text-slate-400 hover:text-blueprint-400 hover:border-blueprint-500/50 transition-colors"
                      onClick={() => pickExample(example)}
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>

              {/* ---------------------------------------------- sampling */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <div className="col-span-2">
                  <label className="label" htmlFor="t2v-seed">
                    Seed
                  </label>
                  <div className="flex gap-2">
                    <input
                      id="t2v-seed"
                      className="field font-mono"
                      type="number"
                      inputMode="numeric"
                      min={0}
                      max={MAX_SEED}
                      step={1}
                      value={seedInput}
                      aria-describedby="t2v-seed-hint"
                      onChange={(event) => setSeedInput(event.target.value)}
                    />
                    <button
                      type="button"
                      className="btn-ghost !px-3 shrink-0"
                      onClick={() => setSeedInput(String(randomSeed()))}
                      aria-label="Random seed"
                      title="Random seed"
                    >
                      <Dices className="w-4 h-4" />
                    </button>
                  </div>
                  <p id="t2v-seed-hint" className="mt-1 text-[10px] text-slate-500">
                    The same prompt, seed and settings give the same shape.
                  </p>
                </div>

                <div className="col-span-2 sm:col-span-1 lg:col-span-2">
                  <div className="flex items-baseline justify-between">
                    <label className="label" htmlFor="t2v-guidance">
                      Guidance
                    </label>
                    <span className="font-mono text-[11px] text-slate-300">
                      {effectiveGuidance.toFixed(1)}
                    </span>
                  </div>
                  <input
                    id="t2v-guidance"
                    type="range"
                    min={1}
                    max={8}
                    step={0.5}
                    value={effectiveGuidance}
                    onChange={(event) => setGuidance(Number(event.target.value))}
                    className="w-full accent-blueprint-500"
                    aria-describedby="t2v-guidance-hint"
                  />
                  <p id="t2v-guidance-hint" className="text-[10px] text-slate-500">
                    Higher follows the text more literally, lower varies more. Model default{' '}
                    {info?.guidance ?? DEFAULT_GUIDANCE}.
                  </p>
                </div>

                <div className="col-span-2 sm:col-span-1 lg:col-span-2">
                  <div className="flex items-baseline justify-between">
                    <label className="label" htmlFor="t2v-steps">
                      Steps
                    </label>
                    <span className="font-mono text-[11px] text-slate-300">{effectiveSteps}</span>
                  </div>
                  <input
                    id="t2v-steps"
                    type="range"
                    min={20}
                    max={100}
                    step={5}
                    value={effectiveSteps}
                    onChange={(event) => setSteps(Number(event.target.value))}
                    className="w-full accent-blueprint-500"
                    aria-describedby="t2v-steps-hint"
                  />
                  <p id="t2v-steps-hint" className="text-[10px] text-slate-500">
                    Denoising steps. More take longer; the reference output uses {defaultSteps}.
                  </p>
                </div>

                {(guidance !== null || steps !== null) && (
                  <button
                    type="button"
                    className="col-span-2 justify-self-start text-[11px] text-slate-500 hover:text-white flex items-center gap-1"
                    onClick={() => {
                      setGuidance(null)
                      setSteps(null)
                    }}
                  >
                    <RotateCcw className="w-3 h-3" />
                    Model defaults
                  </button>
                )}
              </div>

              {/* -------------------------------------------- dimensions */}
              <div className="min-w-0" role="group" aria-labelledby="t2v-size-label">
                <div className="flex items-center justify-between gap-2 mb-1.5">
                  <p id="t2v-size-label" className="label !mb-0">
                    Size (mm, optional)
                  </p>
                  {fromPrompt ? (
                    <span className="chip !py-0.5 border-blueprint-500/40 text-blueprint-400">
                      from prompt
                    </span>
                  ) : dimensionsEdited ? (
                    <button
                      type="button"
                      className="text-[11px] text-slate-500 hover:text-white"
                      onClick={resetDimensionsToPrompt}
                    >
                      Use prompt sizes
                    </button>
                  ) : null}
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {(['length', 'width', 'height'] as DimensionKey[]).map((key) => (
                    <div key={key} className="min-w-0">
                      <label
                        className="block text-[10px] text-slate-500 mb-1 capitalize"
                        htmlFor={`t2v-${key}`}
                      >
                        {key}
                      </label>
                      <input
                        id={`t2v-${key}`}
                        className="field font-mono !px-2"
                        type="number"
                        inputMode="decimal"
                        min={1}
                        step="any"
                        placeholder="auto"
                        value={dimensions[key]}
                        aria-describedby="t2v-size-hint"
                        onChange={(event) => {
                          setDimensionsEdited(true)
                          setDimensions((currentInputs) => ({ ...currentInputs, [key]: event.target.value }))
                        }}
                      />
                    </div>
                  ))}
                </div>
                <p id="t2v-size-hint" className="mt-1.5 text-[10px] text-slate-500 leading-relaxed">
                  Scaling is uniform, so the shape is never stretched: a height sets the height,
                  otherwise the larger of length and width sets the footprint. Left blank, the
                  longest side is 1 m.
                </p>
              </div>

              <div className="flex gap-2">
                <button type="submit" className="btn-primary flex-1 !py-2.5" disabled={!canRun}>
                  {busy === 'generate' ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <PlayCircle className="w-4 h-4" />
                  )}
                  {busy === 'generate' ? 'Generating...' : 'Generate'}
                </button>
                {current && (
                  <button
                    type="button"
                    className="btn-ghost !py-2.5"
                    onClick={runVariation}
                    disabled={!canRun}
                    title="Run again with seed + 1 - a variation on the same prompt"
                  >
                    <Shuffle className="w-4 h-4" />
                    Variation
                  </button>
                )}
              </div>
            </fieldset>
          </form>

          <GenerationProgress
            progress={progress}
            busy={locked}
            modelReady={modelReady}
            totalBytes={info?.totalBytes ?? null}
            onPreload={supported && !infoError ? () => void preload() : undefined}
            preloadDisabled={locked}
          />
        </div>

        {/* ---------------------------------------------------- right column */}
        <div className="space-y-5 min-w-0" ref={resultRef}>
          <div className="panel overflow-hidden">
            <div className="panel-head">
              <div
                className="flex gap-1 bg-ink-950 rounded-lg p-0.5 shrink-0"
                role="tablist"
                aria-label="Result view"
              >
                {(
                  [
                    ['3d', '3D View'],
                    ['blueprint', 'Blueprint View'],
                  ] as [Tab, string][]
                ).map(([key, label]) => (
                  <button
                    key={key}
                    id={`t2v-tab-${key}`}
                    type="button"
                    role="tab"
                    aria-selected={tab === key}
                    aria-controls="t2v-result-panel"
                    tabIndex={tab === key ? 0 : -1}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold whitespace-nowrap transition-colors ${
                      tab === key ? 'bg-ink-800 text-white' : 'text-slate-500 hover:text-slate-300'
                    }`}
                    onClick={() => selectTab(key)}
                    onKeyDown={(event) => {
                      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
                        event.preventDefault()
                        switchTab(tab === '3d' ? 'blueprint' : '3d')
                      }
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {/* Mirrors the Progress panel, which can be scrolled out of view on a long form. */}
              {locked && progress ? (
                <span
                  className="chip bg-signal-run/15 text-signal-run border-signal-run/30 min-w-0"
                  aria-hidden="true"
                >
                  <Loader2 className="w-3 h-3 animate-spin shrink-0" />
                  <span className="truncate">
                    {Math.round(Math.min(Math.max(progress.fraction, 0), 1) * 100)}%
                    <span className="hidden sm:inline"> · {progress.message}</span>
                  </span>
                </span>
              ) : current ? (
                <span className="hidden sm:inline-flex chip border-ink-600 text-slate-400 font-mono">
                  seed {current.result.seed}
                </span>
              ) : null}
            </div>

            <div
              className="p-3"
              id="t2v-result-panel"
              role="tabpanel"
              aria-labelledby={`t2v-tab-${tab}`}
            >
              {tab === '3d' ? (
                <Viewer3D
                  url={hasSurface ? current!.result.glbUrl : null}
                  stats={current?.result.stats ?? null}
                  className="h-[340px] sm:h-[440px] lg:h-[520px]"
                  placeholder={
                    <div className="max-w-sm">
                      <Wand2 className="w-10 h-10 mx-auto mb-3 text-ink-600" strokeWidth={1.5} />
                      {current ? (
                        <p className="text-slate-400 text-sm">
                          This prompt and seed produced no surface. Try another seed, raise the
                          guidance, or pick a category the model knows.
                        </p>
                      ) : (
                        <p className="text-slate-400 text-sm">
                          Describe a shape and press Generate. The model runs in this tab - the
                          prompt never leaves your device.
                        </p>
                      )}
                    </div>
                  }
                />
              ) : (
                <BlueprintView
                  data={current && hasSurface ? current.result.blueprint : null}
                  projectName={current?.name}
                  isoUrl={current ? isoShots[current.id] ?? null : null}
                  sheetActions={
                    current && hasSurface
                      ? (theme) => (
                          <>
                            <button
                              type="button"
                              className="btn-ghost !py-1.5 !text-xs"
                              onClick={() => previewSheet(theme)}
                            >
                              Preview
                            </button>
                            <button
                              type="button"
                              className="btn-primary !py-1.5 !text-xs"
                              onClick={() => downloadSheet(theme)}
                            >
                              <Download className="w-3.5 h-3.5" />
                              Download SVG
                            </button>
                          </>
                        )
                      : undefined
                  }
                />
              )}
            </div>
          </div>

          {/* The engine's own caveats about this result (near-empty output, dropped fragments). */}
          {current && (current.result.warnings ?? []).length > 0 && (
            <div className="panel border-signal-warn/40 bg-signal-warn/5 p-4 flex items-start gap-3">
              <AlertTriangle className="w-4 h-4 text-signal-warn shrink-0 mt-0.5" />
              <ul className="text-[13px] text-slate-300 space-y-1 min-w-0">
                {(current.result.warnings ?? []).map((warning) => (
                  <li key={warning} className="break-words">
                    {warning}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {current && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 items-start">
              {/* ------------------------------------------------ downloads */}
              <section className="panel" aria-labelledby="t2v-downloads-title">
                <div className="panel-head">
                  <div>
                    <h2 id="t2v-downloads-title" className="text-sm font-semibold text-white">
                      Downloads
                    </h2>
                    <p className="text-[11px] text-slate-500">
                      Written in your browser from this result
                    </p>
                  </div>
                </div>
                <ul className="p-2 divide-y divide-ink-800">
                  {MESH_FORMATS.map((format) => (
                    <li key={format.id} className="flex items-center gap-3 px-2 py-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-slate-200">{format.label}</p>
                        <p className="text-[11px] text-slate-500 truncate">{format.note}</p>
                      </div>
                      <button
                        type="button"
                        className="btn-subtle !py-1 !px-2 !text-xs shrink-0"
                        disabled={!hasSurface || exporting !== null}
                        onClick={() => void downloadMesh(format.id)}
                        aria-label={`Download ${format.label}`}
                      >
                        {exporting === format.id ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Download className="w-3.5 h-3.5" />
                        )}
                        Download
                      </button>
                    </li>
                  ))}
                  <li className="flex flex-wrap items-center gap-3 px-2 py-2">
                    <div className="flex-1 min-w-[10rem]">
                      <p className="text-sm font-semibold text-slate-200 flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5 text-blueprint-400" />
                        Blueprint sheet
                      </p>
                      <p className="text-[11px] text-slate-500">A3 vector SVG, three views</p>
                    </div>
                    <div className="flex gap-1.5 shrink-0">
                      {(['white', 'blueprint'] as SheetTheme[]).map((theme) => (
                        <button
                          key={theme}
                          type="button"
                          className="btn-subtle !py-1 !px-2 !text-xs capitalize"
                          disabled={!hasSurface}
                          onClick={() => downloadSheet(theme)}
                          aria-label={`Download the ${theme} blueprint sheet`}
                        >
                          <Download className="w-3.5 h-3.5" />
                          {theme}
                        </button>
                      ))}
                    </div>
                  </li>
                </ul>
                <p className="px-4 py-3 border-t border-ink-800 text-[11px] text-slate-500 leading-relaxed">
                  BLEND and STEP need Blender and FreeCAD, so they come from the local backend:
                  there the same model is the <span className="font-mono">text2voxel</span> 3D
                  provider in the Design Studio.
                </p>
              </section>

              {/* -------------------------------------------------- details */}
              <section className="panel" aria-labelledby="t2v-details-title">
                <div className="panel-head">
                  <h2 id="t2v-details-title" className="text-sm font-semibold text-white">
                    Generation details
                  </h2>
                </div>
                <div className="p-4 space-y-4">
                  <p className="text-[13px] text-slate-300 break-words">{current.result.prompt}</p>
                  <dl className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                    {[
                      ['Seed', String(current.result.seed)],
                      ['Guidance', current.result.guidance.toFixed(1)],
                      ['Steps', String(current.result.steps)],
                      ['Width (X)', formatLength(current.result.stats.width)],
                      ['Height (Y)', formatLength(current.result.stats.height)],
                      ['Depth (Z)', formatLength(current.result.stats.depth)],
                      ['Triangles', current.result.stats.triangles.toLocaleString()],
                      ['Vertices', current.result.stats.vertices.toLocaleString()],
                      ['Watertight', current.result.stats.is_watertight ? 'yes' : 'no'],
                      [
                        'Voxels',
                        `${current.result.occupiedVoxels.toLocaleString()} (${(
                          (current.result.occupiedVoxels / resolution ** 3) *
                          100
                        ).toFixed(1)}%)`,
                      ],
                      [
                        'Requested',
                        current.dimensionsMm
                          ? (['length', 'width', 'height'] as DimensionKey[])
                              .map((key) => current.dimensionsMm?.[key] ?? '-')
                              .join(' x ') + ' mm'
                          : 'auto (1 m)',
                      ],
                    ].map(([key, value]) => (
                      <div key={key} className="min-w-0">
                        <dt className="stat-key">{key}</dt>
                        <dd className="stat-val break-words">{value}</dd>
                      </div>
                    ))}
                  </dl>

                  {timings.length > 0 && (
                    <div>
                      <p className="stat-key mb-1.5">Timings (this device)</p>
                      <dl className="font-mono text-[11px] space-y-0.5">
                        {timings
                          .filter(([key]) => key !== 'total')
                          .map(([key, value]) => (
                            <div key={key} className="flex justify-between gap-3">
                              <dt className="text-slate-500">{TIMING_LABEL[key] ?? key}</dt>
                              <dd className="text-slate-300">{formatMs(value)}</dd>
                            </div>
                          ))}
                        <div className="flex justify-between gap-3 border-t border-ink-800 pt-1 mt-1">
                          <dt className="text-slate-400">Total</dt>
                          <dd className="text-white">{formatMs(timingTotal)}</dd>
                        </div>
                      </dl>
                    </div>
                  )}
                </div>
              </section>
            </div>
          )}

          <SessionHistory
            items={history.map((entry) => ({
              id: entry.id,
              prompt: entry.result.prompt,
              seed: entry.result.seed,
              front: entry.result.stats.triangles > 0 ? entry.result.blueprint.views.front : null,
            }))}
            currentId={currentId}
            limit={HISTORY_LIMIT}
            disabled={locked}
            onSelect={setCurrentId}
            onRemove={removeResult}
            onClear={clearHistory}
          />
        </div>
      </div>

      <div className="mt-5">
        <ModelCard info={info} error={infoError} onPickCategory={pickExample} />
      </div>

      {/* ------------------------------------------------- sheet preview */}
      {sheetPreview && (
        <div
          className="fixed inset-0 z-50 bg-ink-950/95 flex items-center justify-center p-4 sm:p-6"
          role="dialog"
          aria-modal="true"
          aria-label={`Blueprint sheet preview (${sheetPreview.theme})`}
          onClick={() => setSheetPreview(null)}
        >
          <button
            ref={previewClose}
            type="button"
            className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white"
            aria-label="Close preview"
            onClick={() => setSheetPreview(null)}
          >
            <X className="w-6 h-6" />
          </button>
          <img
            src={sheetPreview.url}
            alt="Combined blueprint sheet"
            className="max-w-full max-h-full rounded-lg bg-white shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          />
        </div>
      )}
    </div>
  )
}
