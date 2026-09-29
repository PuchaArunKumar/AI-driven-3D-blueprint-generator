import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, Clock, Loader2, PencilLine, Trash2 } from 'lucide-react'
import { ApiError, STATIC_MODE, api } from '../lib/api'
import type { BlueprintData, CapabilityInfo, ExportFormat, Job, Project } from '../lib/types'
import BlueprintView from '../components/BlueprintView'
import ExportPanel from '../components/ExportPanel'
import Viewer3D, { formatLength } from '../components/Viewer3D'

const EVENT_LABEL: Record<string, string> = {
  created: 'Project created',
  prompt_analysed: 'Prompt analysed',
  spec_edited: 'Specification edited',
  images_generated: 'Views generated',
  reference_uploaded: 'Reference uploaded',
  view_removed: 'View removed',
  model_generated: '3D model reconstructed',
  mesh_processed: 'Mesh processed',
  exported: 'Assets exported',
  demo_registered: 'Demo registered from committed files',
  static_snapshot: 'Exported for the hosted demo',
}

export default function ProjectDetail() {
  const { projectId } = useParams()
  const navigate = useNavigate()
  const [project, setProject] = useState<Project | null>(null)
  const [blueprint, setBlueprint] = useState<BlueprintData | null>(null)
  const [capabilities, setCapabilities] = useState<CapabilityInfo | null>(null)
  const [tab, setTab] = useState<'3d' | 'blueprint'>('3d')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [renaming, setRenaming] = useState(false)
  const [name, setName] = useState('')

  useEffect(() => {
    if (!projectId) return
    setLoading(true)
    api
      .getProject(projectId)
      .then((loaded) => {
        setProject(loaded)
        setName(loaded.name)
      })
      .catch((cause: ApiError) => setError(cause.message))
      .finally(() => setLoading(false))
    // No backend on the hosted demo: the export panel then offers downloads only.
    if (STATIC_MODE) return
    api
      .systemStatus()
      .then((status) => setCapabilities(status.capabilities))
      .catch(() => undefined)
  }, [projectId])

  const openBlueprint = async () => {
    setTab('blueprint')
    if (!project?.model_url || blueprint) return
    try {
      setBlueprint(await api.blueprint(project.id))
    } catch (cause) {
      setError((cause as ApiError).message)
    }
  }

  const exportFormats = async (formats: ExportFormat[]) => {
    if (!project) return
    setBusy(true)
    setError(null)
    try {
      const started: Job = await api.exportProject(project.id, formats)
      // Exports are quick; poll to completion then refresh the asset list.
      let job = started
      while (job.status === 'queued' || job.status === 'running') {
        await new Promise((resolve) => setTimeout(resolve, 900))
        job = await api.getJob(job.id)
      }
      if (job.status === 'failed' && job.error) setError(job.error)
      setProject(await api.getProject(project.id))
    } catch (cause) {
      setError((cause as ApiError).message)
    } finally {
      setBusy(false)
    }
  }

  const saveName = async () => {
    if (!project || !name.trim()) return
    setProject(await api.updateProject(project.id, { name: name.trim() }))
    setRenaming(false)
  }

  const remove = async () => {
    if (!project) return
    if (!window.confirm(`Delete "${project.name}" and all of its generated files?`)) return
    await api.deleteProject(project.id)
    navigate('/gallery')
  }

  if (loading) {
    return (
      <div className="py-24 text-center text-slate-500">
        <Loader2 className="w-6 h-6 mx-auto mb-3 animate-spin" />
        Loading project
      </div>
    )
  }

  if (!project) {
    return (
      <div className="mx-auto max-w-lg py-24 text-center">
        <p className="text-slate-300 mb-2">{error ?? 'That project could not be found.'}</p>
        <Link to="/gallery" className="btn-ghost mt-4">
          <ArrowLeft className="w-4 h-4" />
          Back to gallery
        </Link>
      </div>
    )
  }

  const modelUrl = project.model_url
    ? `${project.model_url}?v=${encodeURIComponent(project.updated_at)}`
    : null

  return (
    <div className="mx-auto max-w-[1600px] px-4 sm:px-6 py-6">
      <Link
        to="/gallery"
        className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-white mb-4 transition-colors"
      >
        <ArrowLeft className="w-3.5 h-3.5" />
        Gallery
      </Link>

      <div className="flex flex-wrap items-start justify-between gap-4 mb-6">
        <div className="min-w-0">
          {renaming ? (
            <div className="flex gap-2">
              <input
                className="field !py-1"
                value={name}
                autoFocus
                onChange={(event) => setName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') void saveName()
                  if (event.key === 'Escape') setRenaming(false)
                }}
              />
              <button className="btn-primary !py-1 !px-3 !text-xs" onClick={() => void saveName()}>
                Save
              </button>
            </div>
          ) : (
            <h1 className="text-2xl font-bold text-white tracking-tight flex flex-wrap items-center gap-2">
              {project.name}
              {!STATIC_MODE && (
                <button
                  className="text-slate-600 hover:text-white transition-colors"
                  onClick={() => setRenaming(true)}
                  aria-label="Rename project"
                >
                  <PencilLine className="w-4 h-4" />
                </button>
              )}
              {project.is_demo && (
                <span className="chip border-signal-warn/40 text-signal-warn">Demo Asset</span>
              )}
            </h1>
          )}
          <p className="mt-1.5 text-sm text-slate-400 max-w-3xl">{project.prompt}</p>
        </div>

        {/* Editing needs the backend; the hosted demo is read-only. */}
        {!STATIC_MODE && (
          <div className="flex gap-2 shrink-0">
            <Link to={`/studio/${project.id}`} className="btn-ghost !py-1.5 !text-xs">
              Open in Studio
            </Link>
            <button className="btn-danger !py-1.5 !text-xs" onClick={() => void remove()}>
              <Trash2 className="w-3.5 h-3.5" />
              Delete
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="panel border-signal-err/40 bg-signal-err/5 p-4 mb-5 text-sm text-slate-300">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_360px] gap-5 items-start">
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
            </div>
            <div className="p-3">
              {tab === '3d' ? (
                <Viewer3D url={modelUrl} stats={project.stats} className="h-[440px] lg:h-[560px]" />
              ) : (
                <BlueprintView
                  data={blueprint}
                  projectId={project.id}
                  projectName={project.name}
                  isoUrl={
                    project.images.find((image) => image.view === 'perspective')?.url ??
                    project.images[0]?.url ??
                    null
                  }
                />
              )}
            </div>
          </div>

          {project.images.length > 0 && (
            <div className="panel">
              <div className="panel-head">
                <h2 className="text-sm font-semibold text-white">Generated views</h2>
                <span className="text-[11px] text-slate-500">{project.images.length} images</span>
              </div>
              <div className="p-3 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
                {project.images.map((image) => (
                  <figure key={image.view} className="rounded-lg overflow-hidden border border-ink-700">
                    <img
                      src={image.url}
                      alt={`${image.view} view`}
                      className="w-full aspect-square object-cover bg-ink-950"
                      loading="lazy"
                    />
                    <figcaption className="px-2 py-1 text-[10px] text-slate-400 bg-ink-900 capitalize">
                      {image.view}
                    </figcaption>
                  </figure>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* ------------------------------------------------------- sidebar */}
        <div className="space-y-5">
          {project.stats && (
            <div className="panel">
              <div className="panel-head">
                <h2 className="text-sm font-semibold text-white">Model statistics</h2>
              </div>
              <dl className="p-4 grid grid-cols-2 gap-3">
                {[
                  ['Vertices', project.stats.vertices.toLocaleString()],
                  ['Triangles', project.stats.triangles.toLocaleString()],
                  ['Width', formatLength(project.stats.width)],
                  ['Height', formatLength(project.stats.height)],
                  ['Depth', formatLength(project.stats.depth)],
                  ['Watertight', project.stats.is_watertight ? 'yes' : 'no'],
                  ['Format', project.stats.file_format.toUpperCase()],
                  [
                    'File size',
                    `${(project.stats.file_size_bytes / 1024).toFixed(0)} KB`,
                  ],
                  [
                    'Generation',
                    project.stats.generation_time_s
                      ? `${project.stats.generation_time_s.toFixed(1)} s`
                      : '-',
                  ],
                ].map(([key, value]) => (
                  <div key={key}>
                    <dt className="stat-key">{key}</dt>
                    <dd className="stat-val">{value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}

          <ExportPanel
            project={project}
            capabilities={capabilities}
            busy={busy}
            onExport={(formats) => void exportFormats(formats)}
          />

          {project.design_spec && (
            <div className="panel">
              <div className="panel-head">
                <h2 className="text-sm font-semibold text-white">Design specification</h2>
              </div>
              <dl className="p-4 space-y-2.5 text-[13px]">
                {[
                  ['Object', project.design_spec.object],
                  ['Purpose', project.design_spec.purpose],
                  ['Style', project.design_spec.style],
                  ['Colour', project.design_spec.color],
                  ['Materials', project.design_spec.materials.join(', ')],
                  ['Constraints', project.design_spec.constraints.join(', ')],
                  ['Manufacturing', project.design_spec.manufacturing_requirements.join(', ')],
                  ['Accessibility', project.design_spec.accessibility_requirements.join(', ')],
                ]
                  .filter(([, value]) => value)
                  .map(([key, value]) => (
                    <div key={key}>
                      <dt className="stat-key">{key}</dt>
                      <dd className="text-slate-300">{value}</dd>
                    </div>
                  ))}
              </dl>
            </div>
          )}

          {project.history.length > 0 && (
            <div className="panel">
              <div className="panel-head">
                <h2 className="text-sm font-semibold text-white">Processing history</h2>
              </div>
              <ol className="p-4 space-y-3">
                {[...project.history].reverse().map((entry, index) => (
                  <li key={index} className="flex gap-3">
                    <span className="mt-1 shrink-0">
                      <Clock className="w-3 h-3 text-slate-600" />
                    </span>
                    <div className="min-w-0">
                      <p className="text-[13px] text-slate-300">
                        {EVENT_LABEL[entry.event] ?? entry.event}
                      </p>
                      {entry.detail && (
                        <p className="text-[11px] text-slate-500 break-words">{entry.detail}</p>
                      )}
                      <time className="text-[10px] font-mono text-slate-600">
                        {new Date(entry.at).toLocaleString()}
                      </time>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
