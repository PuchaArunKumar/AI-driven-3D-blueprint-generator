import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Boxes, Download, Loader2, Plus, Search, Trash2 } from 'lucide-react'
import { ApiError, api } from '../lib/api'
import type { ProjectSummary } from '../lib/types'

type Filter = 'all' | 'mine' | 'demo'

const STATUS_STYLE: Record<string, string> = {
  draft: 'border-slate-600 text-slate-400',
  specified: 'border-slate-600 text-slate-400',
  images_ready: 'border-signal-run/40 text-signal-run',
  model_ready: 'border-blueprint-500/40 text-blueprint-400',
  mesh_processed: 'border-signal-ok/40 text-signal-ok',
  failed: 'border-signal-err/40 text-signal-err',
}

export default function Gallery() {
  const [projects, setProjects] = useState<ProjectSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [deleting, setDeleting] = useState<string | null>(null)

  const load = () => {
    setLoading(true)
    api
      .listProjects(true)
      .then(setProjects)
      .catch((cause: ApiError) => setError(cause.message))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return projects.filter((project) => {
      if (filter === 'demo' && !project.is_demo) return false
      if (filter === 'mine' && project.is_demo) return false
      if (!needle) return true
      return (
        project.name.toLowerCase().includes(needle) ||
        project.prompt.toLowerCase().includes(needle)
      )
    })
  }, [projects, query, filter])

  const remove = async (project: ProjectSummary) => {
    if (!window.confirm(`Delete "${project.name}" and all of its generated files?`)) return
    setDeleting(project.id)
    try {
      await api.deleteProject(project.id)
      setProjects((current) => current.filter((item) => item.id !== project.id))
    } catch (cause) {
      setError((cause as ApiError).message)
    } finally {
      setDeleting(null)
    }
  }

  return (
    <div className="mx-auto max-w-[1600px] px-4 sm:px-6 py-8">
      <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Gallery</h1>
          <p className="text-sm text-slate-500 mt-1">
            {projects.length} project{projects.length === 1 ? '' : 's'} stored locally
          </p>
        </div>
        <Link to="/studio" className="btn-primary">
          <Plus className="w-4 h-4" />
          New design
        </Link>
      </div>

      <div className="flex flex-wrap gap-3 mb-6">
        <div className="relative flex-1 min-w-[220px]">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            className="field pl-9"
            placeholder="Search by name or prompt"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            aria-label="Search projects"
          />
        </div>
        <div className="flex gap-1 bg-ink-900 border border-ink-700 rounded-lg p-0.5">
          {(['all', 'mine', 'demo'] as Filter[]).map((option) => (
            <button
              key={option}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold capitalize transition-colors ${
                filter === option ? 'bg-ink-800 text-white' : 'text-slate-500 hover:text-slate-300'
              }`}
              onClick={() => setFilter(option)}
            >
              {option}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="panel border-signal-err/40 bg-signal-err/5 p-4 mb-5 text-sm text-slate-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="py-20 text-center text-slate-500">
          <Loader2 className="w-6 h-6 mx-auto mb-3 animate-spin" />
          Loading projects
        </div>
      ) : visible.length === 0 ? (
        <div className="panel py-20 text-center">
          <Boxes className="w-10 h-10 mx-auto mb-4 text-ink-600" strokeWidth={1.5} />
          <p className="text-slate-300 font-medium">
            {projects.length === 0 ? 'No projects yet' : 'Nothing matches that filter'}
          </p>
          <p className="text-sm text-slate-500 mt-1 mb-5">
            {projects.length === 0
              ? 'Create your first design in the Studio.'
              : 'Try a different search term.'}
          </p>
          {projects.length === 0 && (
            <Link to="/studio" className="btn-primary">
              <Plus className="w-4 h-4" />
              Start designing
            </Link>
          )}
        </div>
      ) : (
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {visible.map((project) => (
            <article
              key={project.id}
              className="panel overflow-hidden group hover:border-ink-600 transition-colors flex flex-col"
            >
              <Link to={`/projects/${project.id}`} className="block relative">
                <div className="aspect-video bg-ink-950 overflow-hidden">
                  {project.thumbnail_url ? (
                    <img
                      src={project.thumbnail_url}
                      alt={project.name}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                      loading="lazy"
                    />
                  ) : (
                    <div className="w-full h-full grid-bg flex items-center justify-center">
                      <Boxes className="w-8 h-8 text-ink-700" />
                    </div>
                  )}
                </div>
                {project.is_demo && (
                  <span className="absolute top-2 left-2 chip border-signal-warn/40 text-signal-warn bg-ink-950/85">
                    Demo Asset
                  </span>
                )}
              </Link>

              <div className="p-3 flex-1 flex flex-col">
                <div className="flex items-start justify-between gap-2">
                  <Link to={`/projects/${project.id}`} className="min-w-0">
                    <h2 className="font-semibold text-sm text-white truncate hover:text-blueprint-400 transition-colors">
                      {project.name}
                    </h2>
                  </Link>
                  <span
                    className={`chip shrink-0 !text-[9px] ${
                      STATUS_STYLE[project.status] ?? 'border-ink-700 text-slate-500'
                    }`}
                  >
                    {project.status.replace('_', ' ')}
                  </span>
                </div>

                <p className="mt-1 text-[11px] text-slate-500 line-clamp-2 flex-1">
                  {project.prompt}
                </p>

                <div className="mt-3 flex items-center justify-between gap-2">
                  <time className="text-[10px] font-mono text-slate-600">
                    {new Date(project.created_at).toLocaleDateString(undefined, {
                      year: 'numeric',
                      month: 'short',
                      day: 'numeric',
                    })}
                  </time>
                  <div className="flex items-center gap-1">
                    {project.model_format && (
                      <a
                        href={api.downloadUrl(project.id, 'glb')}
                        download
                        className="p-1.5 rounded text-slate-500 hover:text-blueprint-400 transition-colors"
                        title="Download GLB"
                        aria-label={`Download ${project.name} as GLB`}
                      >
                        <Download className="w-3.5 h-3.5" />
                      </a>
                    )}
                    <button
                      className="p-1.5 rounded text-slate-500 hover:text-signal-err transition-colors disabled:opacity-40"
                      title="Delete project"
                      aria-label={`Delete ${project.name}`}
                      disabled={deleting === project.id}
                      onClick={() => void remove(project)}
                    >
                      {deleting === project.id ? (
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <Trash2 className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  )
}
