import type {
  BlueprintData,
  DesignSpec,
  ExportFormat,
  Job,
  Project,
  ProjectSummary,
  StorageUsage,
  SystemStatus,
  ViewName,
} from './types'

/** Error carrying the backend's curated message and remedy hint. */
export class ApiError extends Error {
  readonly code: string
  readonly hint: string
  readonly status: number

  constructor(message: string, code = 'error', hint = '', status = 500) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.hint = hint
    this.status = status
  }
}

/**
 * True in the GitHub Pages build (`VITE_STATIC=true`).
 *
 * There is no backend there: reads come from a snapshot exported at build time
 * by backend/scripts/export_static_site.py, and every mutation fails with a
 * `static_demo` ApiError that says so instead of pretending to work.
 */
export const STATIC_MODE = import.meta.env.VITE_STATIC === 'true'

/** Where the site is served from, always with a trailing slash ("/" or "/repo-name/"). */
export const BASE_URL = import.meta.env.BASE_URL.endsWith('/')
  ? import.meta.env.BASE_URL
  : `${import.meta.env.BASE_URL}/`

/** The repository README, linked wherever the hosted demo points at running it locally. */
export const README_URL = 'https://github.com/PuchaArunKumar/AI-driven-3D-blueprint-generator#readme'

/**
 * Resolve a snapshot URL against the deployed base path.
 *
 * Snapshot JSON stores URLs relative to the site root with no leading slash
 * ("static-api/files/demo/<id>/model_processed.glb"). Left as-is they would
 * resolve against the current route (/projects/<id>/...), so they are
 * anchored to BASE_URL once, here, and components never see the difference.
 * Absolute paths and URLs with a scheme (http:, blob:, data:) pass through.
 */
export function resolveStaticUrl(url: string): string {
  if (!url || url.startsWith('/') || /^[a-z][a-z0-9+.-]*:/i.test(url)) return url
  return `${BASE_URL}${url.replace(/^\.\//, '')}`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      headers: init?.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
      ...init,
    })
  } catch {
    throw new ApiError(
      'Cannot reach the backend.',
      'network',
      'Start it with: uvicorn app.main:app --reload (from the backend directory).',
      0,
    )
  }

  if (response.status === 204) return undefined as T

  const text = await response.text()
  let payload: any = null
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = null
    }
  }

  if (!response.ok) {
    throw new ApiError(
      payload?.message ?? `Request failed (${response.status})`,
      payload?.code ?? 'error',
      payload?.hint ?? '',
      response.status,
    )
  }
  return payload as T
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  body: JSON.stringify(body),
})

const backendApi = {
  // ----------------------------------------------------------- projects
  interpretPrompt: (prompt: string) => request<DesignSpec>('/api/spec', json({ prompt })),

  createProject: (prompt: string, name?: string) =>
    request<Project>('/api/projects', json({ prompt, name: name ?? null })),

  listProjects: (includeDemo = true) =>
    request<ProjectSummary[]>(`/api/projects?include_demo=${includeDemo}`),

  getProject: (id: string) => request<Project>(`/api/projects/${id}`),

  updateProject: (id: string, body: { name?: string; design_spec?: DesignSpec }) =>
    request<Project>(`/api/projects/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),

  reanalyse: (id: string) => request<DesignSpec>(`/api/projects/${id}/reanalyse`, { method: 'POST' }),

  deleteProject: (id: string) => request<void>(`/api/projects/${id}`, { method: 'DELETE' }),

  blueprint: (id: string) => request<BlueprintData>(`/api/projects/${id}/blueprint`),

  blueprintSheetUrl: (id: string, theme: 'white' | 'blueprint', download = false) =>
    `/api/projects/${id}/blueprint.svg?theme=${theme}${download ? '&download=true' : ''}`,

  deleteView: (id: string, view: ViewName) =>
    request<Project>(`/api/projects/${id}/images/${view}`, { method: 'DELETE' }),

  uploadReference: (id: string, view: ViewName, file: File) => {
    const form = new FormData()
    form.append('file', file)
    return request<Project>(`/api/projects/${id}/images/${view}`, { method: 'POST', body: form })
  },

  // --------------------------------------------------------- generation
  generateImages: (body: {
    project_id: string
    views: ViewName[]
    provider?: string | null
    seed?: number | null
    replace?: boolean
  }) => request<Job>('/api/generate/images', json(body)),

  // `seed` is used by text-conditioned providers (text2voxel); others ignore it.
  generate3D: (body: {
    project_id: string
    provider?: string | null
    resolution?: number | null
    seed?: number | null
  }) => request<Job>('/api/generate/3d', json(body)),

  processMesh: (body: {
    project_id: string
    smooth_iterations?: number
    target_faces?: number | null
    fill_holes?: boolean
    remove_duplicates?: boolean
    recompute_normals?: boolean
  }) => request<Job>('/api/process/mesh', json(body)),

  runPipeline: (body: {
    project_id: string
    views: ViewName[]
    image_provider?: string | null
    threed_provider?: string | null
    seed?: number | null
    resolution?: number | null
    export_formats?: ExportFormat[]
  }) => request<Job>('/api/generate/pipeline', json(body)),

  exportProject: (id: string, formats: ExportFormat[]) =>
    request<Job>(`/api/export/${id}`, json({ formats })),

  downloadUrl: (id: string, format: string) => `/api/projects/${id}/download/${format}`,

  // --------------------------------------------------------------- jobs
  getJob: (id: string) => request<Job>(`/api/jobs/${id}`),
  cancelJob: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  projectJobs: (id: string) => request<Job[]>(`/api/projects/${id}/jobs`),

  // ------------------------------------------------------------- system
  systemStatus: () => request<SystemStatus>('/api/system/status'),
  storageUsage: () => request<StorageUsage>('/api/system/storage'),
}

// ------------------------------------------------------------------ static

const STATIC_HINT =
  'This hosted demo has no backend - run it locally (see the README), or use Text → 3D, which runs in your browser.'

/** Rejects the way a backend-only call must on the hosted demo. */
function needsBackend(action: string): Promise<never> {
  return Promise.reject(
    new ApiError(
      `${action} needs the Python backend, which this hosted demo does not have.`,
      'static_demo',
      STATIC_HINT,
      501,
    ),
  )
}

const SNAPSHOT_HINT =
  'Export it with: python backend/scripts/export_static_site.py --out frontend/public/static-api'

/**
 * GET one JSON file from the build-time snapshot under `${BASE_URL}static-api/`.
 *
 * `index` marks the file every snapshot has (projects.json): a 404 there means
 * the snapshot itself is missing, not that one project is.
 */
async function snapshot<T>(path: string, index = false): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE_URL}static-api/${path}`)
  } catch {
    throw new ApiError(
      'Could not load the demo data.',
      'network',
      'Check your connection and reload the page.',
      0,
    )
  }

  if (response.status === 404) {
    throw index
      ? new ApiError('This build has no demo snapshot.', 'static_snapshot', SNAPSHOT_HINT, 404)
      : new ApiError(
          'That is not part of this hosted demo.',
          'not_found',
          'Only the bundled demo projects are available here.',
          404,
        )
  }
  if (!response.ok) {
    throw new ApiError(`Could not load the demo data (${response.status}).`, 'error', '', response.status)
  }

  // A dev server answers a missing file with index.html and a 200, so a parse
  // failure means the snapshot was never exported rather than a server fault.
  const text = await response.text()
  try {
    return JSON.parse(text) as T
  } catch {
    throw new ApiError('The demo snapshot is missing or unreadable.', 'static_snapshot', SNAPSHOT_HINT, 500)
  }
}

function localiseSummary<T extends ProjectSummary>(project: T): T {
  return {
    ...project,
    thumbnail_url: project.thumbnail_url ? resolveStaticUrl(project.thumbnail_url) : null,
  }
}

function localiseProject(project: Project): Project {
  return {
    ...localiseSummary(project),
    model_url: project.model_url ? resolveStaticUrl(project.model_url) : null,
    images: project.images.map((image) => ({ ...image, url: resolveStaticUrl(image.url) })),
    exports: Object.fromEntries(
      Object.entries(project.exports ?? {}).map(([format, url]) => [format, resolveStaticUrl(url)]),
    ),
  }
}

/** Same surface as the backend client, served from the snapshot on the hosted demo. */
const staticApi: typeof backendApi = {
  interpretPrompt: () => needsBackend('Prompt analysis'),
  createProject: () => needsBackend('Creating a project'),

  // The snapshot holds demo projects only, so there is nothing to filter out.
  listProjects: () =>
    snapshot<ProjectSummary[]>('projects.json', true).then((projects) =>
      projects.map(localiseSummary),
    ),

  getProject: (id: string) =>
    snapshot<Project>(`projects/${encodeURIComponent(id)}.json`).then(localiseProject),

  updateProject: () => needsBackend('Editing a project'),
  reanalyse: () => needsBackend('Re-analysing the prompt'),
  deleteProject: () => needsBackend('Deleting a project'),

  blueprint: (id: string) =>
    snapshot<BlueprintData>(`projects/${encodeURIComponent(id)}/blueprint.json`),

  // Pre-rendered sheets; `download` is honoured by the <a download> that uses it.
  blueprintSheetUrl: (id: string, theme: 'white' | 'blueprint') =>
    `${BASE_URL}static-api/projects/${encodeURIComponent(id)}/blueprint-${theme}.svg`,

  deleteView: () => needsBackend('Removing a view'),
  uploadReference: () => needsBackend('Uploading a reference image'),

  generateImages: () => needsBackend('Image generation'),
  generate3D: () => needsBackend('3D reconstruction'),
  processMesh: () => needsBackend('Mesh processing'),
  runPipeline: () => needsBackend('The generation pipeline'),
  exportProject: () => needsBackend('Exporting new formats'),

  // Only formats that were exported when the snapshot was built exist on disk.
  downloadUrl: (id: string, format: string) =>
    `${BASE_URL}static-api/files/demo/${encodeURIComponent(id)}/exports/${encodeURIComponent(id)}.${format}`,

  getJob: () => needsBackend('Job tracking'),
  cancelJob: () => needsBackend('Job tracking'),
  // A static site has run no jobs - an empty list is the true answer.
  projectJobs: () => Promise.resolve([]),

  // Callers treat a rejected status as "no backend" and degrade gracefully.
  systemStatus: () => needsBackend('System status'),
  storageUsage: () => needsBackend('Storage usage'),
}

export const api = STATIC_MODE ? staticApi : backendApi

/**
 * Follow a job to completion.
 *
 * Prefers the WebSocket stream and falls back to polling if the socket cannot
 * be opened (proxy issues, corporate networks). Returns a cancel function.
 */
export function watchJob(
  jobId: string,
  onUpdate: (job: Job) => void,
  onDone: (job: Job) => void,
): () => void {
  let stopped = false
  let socket: WebSocket | null = null
  let pollTimer: number | null = null

  const finish = (job: Job) => {
    if (stopped) return
    stopped = true
    onUpdate(job)
    onDone(job)
    cleanup()
  }

  const cleanup = () => {
    if (socket) {
      socket.onclose = null
      socket.close()
      socket = null
    }
    if (pollTimer !== null) {
      window.clearTimeout(pollTimer)
      pollTimer = null
    }
  }

  const poll = async () => {
    if (stopped) return
    try {
      const job = await api.getJob(jobId)
      if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
        finish(job)
        return
      }
      onUpdate(job)
    } catch {
      // Transient failure - keep polling until the caller cancels.
    }
    if (!stopped) pollTimer = window.setTimeout(poll, 1200)
  }

  try {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    socket = new WebSocket(`${protocol}//${window.location.host}/api/ws/jobs/${jobId}`)

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data)
      if (payload.heartbeat || payload.error) return
      const job = payload as Job
      if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
        finish(job)
      } else {
        onUpdate(job)
      }
    }
    socket.onerror = () => {
      if (!stopped && pollTimer === null) void poll()
    }
    socket.onclose = () => {
      // Closed before a terminal status arrived - confirm by polling once.
      if (!stopped && pollTimer === null) void poll()
    }
  } catch {
    void poll()
  }

  return () => {
    stopped = true
    cleanup()
  }
}
