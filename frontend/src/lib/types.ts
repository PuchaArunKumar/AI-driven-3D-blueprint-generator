// Mirrors backend/app/schemas.py. Keep the two in step.

export type ViewName =
  | 'front'
  | 'rear'
  | 'left'
  | 'right'
  | 'top'
  | 'bottom'
  | 'perspective'

export const ALL_VIEWS: ViewName[] = [
  'front',
  'rear',
  'left',
  'right',
  'top',
  'bottom',
  'perspective',
]

/** Views the visual-hull reconstruction can actually carve with. */
export const CARVING_VIEWS: ViewName[] = ['front', 'rear', 'left', 'right', 'top', 'bottom']

export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
export type StageStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
export type ProjectStatus =
  | 'draft'
  | 'specified'
  | 'images_ready'
  | 'model_ready'
  | 'mesh_processed'
  | 'failed'

export type ExportFormat = 'glb' | 'gltf' | 'obj' | 'ply' | 'stl' | 'blend' | 'step'

export interface Dimensions {
  length_mm: number | null
  width_mm: number | null
  height_mm: number | null
  weight_g: number | null
}

export interface DesignSpec {
  object: string
  purpose: string
  dimensions: Dimensions
  materials: string[]
  style: string
  color: string
  constraints: string[]
  manufacturing_requirements: string[]
  accessibility_requirements: string[]
  intended_use: string
  tolerances_mm: number | null
}

export interface GeneratedImage {
  view: ViewName
  path: string
  url: string
  width: number
  height: number
  provider: string
  seed: number | null
  created_at: string
  is_reference: boolean
}

export interface ModelStats {
  vertices: number
  faces: number
  triangles: number
  bbox_min: number[]
  bbox_max: number[]
  width: number
  height: number
  depth: number
  volume: number | null
  surface_area: number | null
  is_watertight: boolean
  file_size_bytes: number
  file_format: string
  generation_time_s: number | null
}

export interface ProjectSummary {
  id: string
  name: string
  prompt: string
  status: ProjectStatus
  thumbnail_url: string | null
  model_format: string | null
  is_demo: boolean
  created_at: string
  updated_at: string
}

export interface HistoryEntry {
  event: string
  detail: string
  at: string
  payload?: Record<string, unknown>
}

export interface Project extends ProjectSummary {
  design_spec: DesignSpec | null
  images: GeneratedImage[]
  model_url: string | null
  model_path: string | null
  stats: ModelStats | null
  exports: Record<string, string>
  history: HistoryEntry[]
  metadata: Record<string, unknown>
}

export interface JobStage {
  key: string
  label: string
  status: StageStatus
  detail: string
  started_at: string | null
  finished_at: string | null
}

export interface Job {
  id: string
  project_id: string | null
  kind: string
  status: JobStatus
  progress: number
  message: string
  error: string | null
  hint: string
  stages: JobStage[]
  result: Record<string, any> | null
  created_at: string
  updated_at: string
}

export interface ProviderInfo {
  name: string
  kind: 'image' | 'threed' | 'mesh' | 'cad'
  label: string
  available: boolean
  reason: string
  experimental: boolean
  requires: string[]
}

export interface HardwareInfo {
  cpu: string
  cpu_cores: number
  ram_total_gb: number | null
  ram_available_gb: number | null
  gpu: string | null
  vram_total_gb: number | null
  cuda_available: boolean
  cuda_version: string | null
  torch_version: string | null
  platform: string
}

export interface CapabilityInfo {
  export_formats: string[]
  unavailable_export_formats: Record<string, string>
  blender: boolean
  blender_version: string | null
  freecad: boolean
  diffusers_installed: boolean
}

export interface SystemStatus {
  status: 'ok' | 'degraded'
  version: string
  hardware: HardwareInfo
  capabilities: CapabilityInfo
  providers: ProviderInfo[]
  settings: Record<string, any>
  project_count: number
  active_jobs: number
}

/**
 * One projected orthographic view. Paths are polylines of [x, y] in model units.
 *
 * `outline` closed silhouette loops, `inline` visible creases and interior
 * contours, `hidden` the same edges where the body occludes them.
 */
export interface TechnicalViewData {
  outline: number[][][]
  inline: number[][][]
  hidden: number[][][]
  width: number
  height: number
}

export interface BlueprintData {
  views: Record<'front' | 'side' | 'top', TechnicalViewData>
  stats: ModelStats
  spec: DesignSpec | null
  units: string
}

export interface StorageUsage {
  storage_dir: string
  disk_total_gb: number
  disk_free_gb: number
  images_bytes: number
  models_bytes: number
  exports_bytes: number
  demo_bytes: number
}
