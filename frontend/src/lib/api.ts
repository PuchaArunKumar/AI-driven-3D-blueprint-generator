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

export const api = {
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

  generate3D: (body: { project_id: string; provider?: string | null; resolution?: number | null }) =>
    request<Job>('/api/generate/3d', json(body)),

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
