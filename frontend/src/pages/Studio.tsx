import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import {
  AlertTriangle,
  Boxes,
  Layers,
  Loader2,
  PlayCircle,
  Ruler,
  Sparkles,
  Wand2,
  Wrench,
} from 'lucide-react'
import { ApiError, STATIC_MODE, api, watchJob } from '../lib/api'
import { CARVING_VIEWS } from '../lib/types'
import type {
  BlueprintData,
  CapabilityInfo,
  DesignSpec,
  ExportFormat,
  Job,
  Project,
  ProviderInfo,
  ViewName,
} from '../lib/types'
import BlueprintView from '../components/BlueprintView'
import ExportPanel from '../components/ExportPanel'
import PipelineStatus from '../components/PipelineStatus'
import SpecEditor from '../components/SpecEditor'
import StaticModeNotice from '../components/StaticModeNotice'
import Viewer3D from '../components/Viewer3D'
import ViewGallery from '../components/ViewGallery'

const EXAMPLES = [
  'A lightweight ergonomic wheelchair tray in matte black ABS, 45cm x 30cm x 2cm, 3D printable, must be waterproof',
  'A futuristic aerodynamic electric car, carbon fibre body, 4.5 m long',
  'A modern oak dining chair, 45 x 45 x 90 cm, for a small apartment',
  'A smart wearable fitness band with a silicone strap, 25 mm wide, for elderly users with arthritis',
]

type Tab = '3d' | 'blueprint'

export default function Studio() {
  // The hosted build has no backend; explain that instead of failing requests.
  if (STATIC_MODE) {
    return (
      <StaticModeNotice
        title="Design Studio"
        purpose="The Studio turns a description into a design specification, multi-view concept images, a reconstructed 3D model and CAD exports, stage by stage."
      />
    )
  }
  return <StudioWorkspace />
}

function StudioWorkspace() {
  const { projectId } = useParams()
  const navigate = useNavigate()

  const [prompt, setPrompt] = useState('')
  const [project, setProject] = useState<Project | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<{ message: string; hint?: string; code?: string } | null>(
    null,
  )
  const [tab, setTab] = useState<Tab>('3d')
  const [blueprint, setBlueprint] = useState<BlueprintData | null>(null)
  const [blueprintLoading, setBlueprintLoading] = useState(false)
  const [capabilities, setCapabilities] = useState<CapabilityInfo | null>(null)
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [selectedView, setSelectedView] = useState<ViewName | null>(null)
  const [views, setViews] = useState<ViewName[]>([...CARVING_VIEWS])
  const [threedProvider, setThreedProvider] = useState<string | null>(null)
  const stopWatch = useRef<(() => void) | null>(null)

  // --------------------------------------------------------------- loading
  useEffect(() => {
    api
      .systemStatus()
      .then((status) => {
        setCapabilities(status.capabilities)
        setProviders(status.providers)
      })
      .catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!projectId) {
      setProject(null)
      setBlueprint(null)
      return
    }
    api
      .getProject(projectId)
      .then((loaded) => {
        setProject(loaded)
        setPrompt(loaded.prompt)
      })
      .catch((cause: ApiError) => setError({ message: cause.message, hint: cause.hint }))
  }, [projectId])

  useEffect(() => () => stopWatch.current?.(), [])

  const refresh = useCallback(async () => {
    if (!projectId) return
    try {
      setProject(await api.getProject(projectId))
    } catch {
      // A refresh failure is not worth interrupting the user for.
    }
  }, [projectId])

  // ----------------------------------------------------------- job runner
  const follow = useCallback(
    (started: Job) => {
      setJob(started)
      setBusy(true)
      stopWatch.current?.()
      stopWatch.current = watchJob(
        started.id,
        (update) => setJob(update),
        (final) => {
          setJob(final)
          setBusy(false)
          if (final.status === 'failed' && final.error) {
            setError({ message: final.error, hint: final.hint || undefined })
          }
          void refresh()
          setBlueprint(null)
        },
      )
    },
    [refresh],
  )

  const run = useCallback(
    async (action: () => Promise<Job>) => {
      setError(null)
      try {
        follow(await action())
      } catch (cause) {
        const failure = cause as ApiError
        setError({ message: failure.message, hint: failure.hint })
        setBusy(false)
      }
    },
    [follow],
  )

  // ------------------------------------------------------------- actions
  const startDesign = async () => {
    const text = prompt.trim()
    if (text.length < 3) {
      setError({ message: 'Describe the product you want to design (at least a few words).' })
      return
    }
    setError(null)
    setBusy(true)
    try {
      const created = await api.createProject(text)
      setProject(created)
      navigate(`/studio/${created.id}`, { replace: true })
      follow(
        await api.runPipeline({
          project_id: created.id,
          views,
          threed_provider: threedProvider,
          export_formats: ['glb', 'stl'],
        }),
      )
    } catch (cause) {
      const failure = cause as ApiError
      setError({ message: failure.message, hint: failure.hint, code: failure.code })
      setBusy(false)
    }
  }

  const regenerateViews = (targets: ViewName[]) => {
    if (!project) return
    void run(() =>
      api.generateImages({ project_id: project.id, views: targets, replace: false }),
    )
  }

  const openBlueprint = async () => {
    setTab('blueprint')
    if (!project?.model_url || blueprint) return
    setBlueprintLoading(true)
    try {
      setBlueprint(await api.blueprint(project.id))
    } catch (cause) {
      setError({ message: (cause as ApiError).message })
    } finally {
      setBlueprintLoading(false)
    }
  }

  const saveSpec = async (spec: DesignSpec) => {
    if (!project) return
    setProject(await api.updateProject(project.id, { design_spec: spec }))
  }

  const hasImages = (project?.images.length ?? 0) > 0
  const hasModel = Boolean(project?.model_url)
  // Cache-bust so a regenerated model at the same path is refetched.
  const modelUrl = project?.model_url
    ? `${project.model_url}?v=${encodeURIComponent(project.updated_at)}`
    : null

  // The gallery selection picks which view illustrates the blueprint sheet.
  const isoUrl =
    (selectedView && project?.images.find((image) => image.view === selectedView)?.url) ??
    project?.images.find((image) => image.view === 'perspective')?.url ??
    project?.images[0]?.url ??
    null

  const threedProviders = providers.filter((p) => p.kind === 'threed' && p.available)

  const imageProvider = providers.find((p) => p.kind === 'image' && p.available)
  const noImageProvider = providers.length > 0 && !imageProvider
  // A text-conditioned 3D provider needs no views, so the pipeline still runs.
  const textProvider = threedProviders.find((p) => p.name === 'text2voxel')
  // Chosen explicitly, it builds the model from the prompt - "Generate 3D" needs no views.
  const fromText = Boolean(textProvider) && threedProvider === 'text2voxel'

  return (
    <div className="mx-auto max-w-[1600px] px-4 sm:px-6 py-6">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">Design Studio</h1>
          <p className="text-[12px] text-slate-500">
            {project ? project.name : 'Describe a product to begin'}
          </p>
        </div>
        {project && (
          <span className="chip border-ink-600 text-slate-400 font-mono">
            {project.status.replace('_', ' ')}
          </span>
        )}
      </div>

      {noImageProvider && (
        <div className="panel border-signal-warn/40 bg-signal-warn/5 p-4 mb-5 flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-signal-warn shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="text-slate-200 font-medium">No image provider is available.</p>
            <p className="text-slate-400 text-[13px] mt-0.5">
              Install the local Stable Diffusion stack or add an API key in Settings. You can
              still open and export the demo projects from the Gallery.
              {textProvider &&
                ' The pipeline can still build a model straight from the text with Text2Voxel-64 - pick it under 3D reconstruction.'}
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="panel border-signal-err/40 bg-signal-err/5 p-4 mb-5 flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-signal-err shrink-0 mt-0.5" />
          <div className="text-sm flex-1">
            <p className="text-slate-200">{error.message}</p>
            {error.hint && <p className="text-slate-400 text-[13px] mt-0.5">{error.hint}</p>}
            {error.code === 'network' && (
              <p className="text-slate-400 text-[13px] mt-0.5">
                <Link to="/generate" className="text-blueprint-400 hover:underline">
                  Text → 3D
                </Link>{' '}
                works without the backend - the model runs in your browser.
              </p>
            )}
          </div>
          <button
            onClick={() => setError(null)}
            className="text-slate-500 hover:text-white text-xs"
          >
            Dismiss
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[380px_minmax(0,1fr)] gap-5 items-start">
        {/* --------------------------------------------------- left column */}
        <div className="space-y-5">
          <div className="panel">
            <div className="panel-head">
              <h2 className="text-sm font-semibold text-white">Design Input</h2>
            </div>
            <div className="p-4 space-y-3">
              <div>
                <label className="label" htmlFor="prompt">
                  Product description
                </label>
                <textarea
                  id="prompt"
                  className="field resize-none"
                  rows={4}
                  value={prompt}
                  disabled={busy}
                  placeholder="Design a lightweight ergonomic wheelchair tray in matte black ABS, 45cm x 30cm..."
                  onChange={(event) => setPrompt(event.target.value)}
                />
              </div>

              <div>
                <p className="label">Views to generate</p>
                <div className="flex flex-wrap gap-1.5">
                  {CARVING_VIEWS.map((view) => {
                    const active = views.includes(view)
                    return (
                      <button
                        key={view}
                        disabled={busy}
                        onClick={() =>
                          setViews((current) =>
                            current.includes(view)
                              ? current.filter((item) => item !== view)
                              : [...current, view],
                          )
                        }
                        className={`chip transition-colors ${
                          active
                            ? 'border-blueprint-500/50 text-blueprint-400 bg-blueprint-500/10'
                            : 'border-ink-700 text-slate-500 hover:text-slate-300'
                        }`}
                      >
                        {view}
                      </button>
                    )
                  })}
                </div>
                <p className="mt-1.5 text-[10px] text-slate-500">
                  More views carve a tighter hull. Front plus one side is the practical minimum.
                </p>
              </div>

              {threedProviders.length > 1 && (
                <div>
                  <label className="label" htmlFor="threed-provider">
                    3D reconstruction
                  </label>
                  <select
                    id="threed-provider"
                    className="field"
                    disabled={busy}
                    value={threedProvider ?? ''}
                    onChange={(event) => setThreedProvider(event.target.value || null)}
                  >
                    <option value="">Backend default</option>
                    {threedProviders.map((provider) => (
                      <option key={provider.name} value={provider.name}>
                        {provider.label}
                        {provider.experimental ? ' (experimental)' : ''}
                      </option>
                    ))}
                  </select>
                  <p className="mt-1.5 text-[10px] text-slate-500">
                    Silhouette carving stays faithful to the views but cannot see
                    concavities. A learned model invents plausible structure the views
                    never showed.
                    {textProvider &&
                      ' Text2Voxel-64 skips the views and generates the shape from the text alone.'}
                  </p>
                </div>
              )}

              <button
                className="btn-primary w-full !py-2.5"
                onClick={() => void startDesign()}
                disabled={busy || views.length === 0}
              >
                {busy ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <PlayCircle className="w-4 h-4" />
                )}
                {busy ? 'Generating...' : 'Generate Design'}
              </button>

              {!project && (
                <div className="pt-1">
                  <p className="label">Try an example</p>
                  <div className="space-y-1.5">
                    {EXAMPLES.map((example) => (
                      <button
                        key={example}
                        className="w-full text-left text-[12px] text-slate-400 hover:text-blueprint-400 leading-snug transition-colors"
                        onClick={() => setPrompt(example)}
                      >
                        <Sparkles className="w-3 h-3 inline mr-1.5 -mt-0.5" />
                        {example}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <SpecEditor
            spec={project?.design_spec ?? null}
            disabled={busy}
            onSave={saveSpec}
            onReanalyse={
              project
                ? async () => {
                    await api.reanalyse(project.id)
                    await refresh()
                  }
                : undefined
            }
          />

          {/* Stage-by-stage controls, enabled only when their input exists. */}
          {project && (
            <div className="panel p-3 grid grid-cols-2 gap-2">
              <button
                className="btn-subtle !text-xs"
                disabled={busy}
                onClick={() => regenerateViews(views)}
              >
                <Layers className="w-3.5 h-3.5" />
                Generate Views
              </button>
              <button
                className="btn-subtle !text-xs"
                disabled={busy || !(hasImages || fromText)}
                title={
                  hasImages || fromText
                    ? undefined
                    : 'Generate views first, or pick Text2Voxel-64 under 3D reconstruction'
                }
                onClick={() =>
                  void run(() =>
                    api.generate3D({ project_id: project.id, provider: threedProvider }),
                  )
                }
              >
                <Boxes className="w-3.5 h-3.5" />
                Generate 3D
              </button>
              <button
                className="btn-subtle !text-xs"
                disabled={busy || !hasModel}
                title={hasModel ? undefined : 'Generate the 3D model first'}
                onClick={() => void run(() => api.processMesh({ project_id: project.id }))}
              >
                <Wrench className="w-3.5 h-3.5" />
                Process Mesh
              </button>
              <button
                className="btn-subtle !text-xs"
                disabled={busy || !hasModel}
                title={hasModel ? undefined : 'Generate the 3D model first'}
                onClick={() => void openBlueprint()}
              >
                <Ruler className="w-3.5 h-3.5" />
                Blueprint
              </button>
            </div>
          )}
        </div>

        {/* -------------------------------------------------- right column */}
        <div className="space-y-5">
          <div className="panel overflow-hidden">
            <div className="panel-head">
              <div className="flex gap-1 bg-ink-950 rounded-lg p-0.5">
                <button
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                    tab === '3d' ? 'bg-ink-800 text-white' : 'text-slate-500 hover:text-slate-300'
                  }`}
                  onClick={() => setTab('3d')}
                >
                  3D View
                </button>
                <button
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                    tab === 'blueprint'
                      ? 'bg-ink-800 text-white'
                      : 'text-slate-500 hover:text-slate-300'
                  }`}
                  onClick={() => void openBlueprint()}
                >
                  Blueprint View
                </button>
              </div>
              {project?.is_demo && (
                <span className="chip border-signal-warn/40 text-signal-warn">Demo Asset</span>
              )}
            </div>

            <div className="p-3">
              {tab === '3d' ? (
                <Viewer3D
                  url={modelUrl}
                  stats={project?.stats ?? null}
                  className="h-[420px] lg:h-[520px]"
                  placeholder={
                    <div className="max-w-sm">
                      <Wand2 className="w-10 h-10 mx-auto mb-3 text-ink-600" strokeWidth={1.5} />
                      <p className="text-slate-400 text-sm">
                        {hasImages
                          ? 'Views are ready. Run 3D reconstruction to build the model.'
                          : 'Describe a product and press Generate Design.'}
                      </p>
                    </div>
                  }
                />
              ) : (
                <BlueprintView
                  data={blueprint}
                  projectId={project?.id}
                  loading={blueprintLoading}
                  projectName={project?.name}
                  isoUrl={isoUrl}
                />
              )}
            </div>
          </div>

          {project && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
              <ViewGallery
                images={project.images}
                busy={busy}
                selected={selectedView}
                onSelect={setSelectedView}
                onRegenerate={regenerateViews}
                onDelete={(view) => {
                  void api
                    .deleteView(project.id, view)
                    .then(setProject)
                    .catch((cause: ApiError) => setError({ message: cause.message }))
                }}
                onUpload={(view, file) =>
                  api
                    .uploadReference(project.id, view, file)
                    .then(setProject)
                    .catch((cause: ApiError) => {
                      setError({ message: cause.message, hint: cause.hint })
                      throw cause
                    })
                }
              />
              <ExportPanel
                project={project}
                capabilities={capabilities}
                busy={busy}
                onExport={(formats: ExportFormat[]) =>
                  void run(() => api.exportProject(project.id, formats))
                }
              />
            </div>
          )}
        </div>
      </div>

      <div className="mt-5">
        <PipelineStatus
          job={job}
          onCancel={
            job ? () => void api.cancelJob(job.id).then(setJob).catch(() => undefined) : undefined
          }
        />
      </div>
    </div>
  )
}
