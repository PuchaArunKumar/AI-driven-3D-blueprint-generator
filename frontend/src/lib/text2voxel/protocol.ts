// Types shared by the public API (index.ts), the worker and the engine.
// The T2V* names are re-exported from index.ts, which is the public contract.

import type { BlueprintData, ModelStats } from '../types.ts'
import type { MeshData } from './mesher.ts'

export type T2VStage = 'loading' | 'encoding' | 'sampling' | 'decoding' | 'meshing' | 'done'

export interface T2VProgress {
  stage: T2VStage
  /** 0..1 over the whole operation. */
  fraction: number
  message: string
  loadedBytes?: number
  totalBytes?: number
}

export interface T2VModelInfo {
  name: string
  version: number
  created: string
  smoke: boolean
  resolution: number
  sampleSteps: number
  guidance: number
  classes: string[]
  metrics: Record<string, unknown>
  training: Record<string, unknown>
  /** Everything a first generation downloads: the four model files plus the ONNX Runtime wasm. */
  totalBytes: number
}

export interface T2VOptions {
  /** Default: random. The result reports the seed used. */
  seed?: number
  guidance?: number
  steps?: number
  dimensionsMm?: { length?: number | null; width?: number | null; height?: number | null } | null
}

export interface T2VResult {
  prompt: string
  seed: number
  guidance: number
  steps: number
  glb: Blob
  /** Object URL for `glb`; release it with disposeResult(result). */
  glbUrl: string
  /** Metres, file_format 'glb'. */
  stats: ModelStats
  /** spec: null, units 'm'. */
  blueprint: BlueprintData
  timingsMs: Record<string, number>
  occupiedVoxels: number
  // Beyond the contract - optional for consumers:
  /** Occupancy level the surface was extracted at (meta.threshold unless the output was near-empty). */
  isoLevel: number
  /** Plain-language caveats about this result (empty output fallback, dropped fragments...). */
  warnings: string[]
}

/** meta.json, as written by training/text2voxel/k2_train.py main(). */
export interface T2VMeta {
  name: string
  version: number
  created: string
  resolution: number
  latent_dim: number
  cond_dim: number
  frame?: string
  diffusion: {
    parameterization: string
    train_steps: number
    schedule?: string
    alphas_cumprod: number[]
    sample_steps: number
    guidance: number
    x0_clip?: number
    null_cond?: string
  }
  text_encoder?: {
    repo?: string
    revision?: string
    file?: string
    pooling?: string
    normalize?: boolean
    max_tokens?: number
  }
  threshold?: number
  classes_modelnet?: string[]
  palette?: Record<string, number[]>
  metrics?: Record<string, unknown>
  training?: Record<string, unknown>
  parity?: Record<string, unknown>
  reference?: { prompt: string; noise: number[]; latent: number[]; cond_head: number[] }
  smoke?: boolean
}

/** Absolute URLs of everything the worker downloads. */
export interface ModelUrls {
  meta: string
  decoder: string
  prior: string
  minilm: string
  tokenizer: string
  ortWasm: string
}

// ---- worker messages ------------------------------------------------------

export type WorkerRequest =
  | { id: number; type: 'preload'; urls: ModelUrls }
  | {
      id: number
      type: 'generate'
      urls: ModelUrls
      prompt: string
      seed: number
      guidance: number | null
      steps: number | null
      dimensionsMm: T2VOptions['dimensionsMm']
    }

export interface WorkerGenerateResult {
  prompt: string
  seed: number
  guidance: number
  steps: number
  glb: ArrayBuffer
  mesh: MeshData
  stats: ModelStats
  blueprint: BlueprintData
  timingsMs: Record<string, number>
  occupiedVoxels: number
  isoLevel: number
  warnings: string[]
}

export type WorkerResponse =
  | { id: number; type: 'progress'; progress: T2VProgress }
  | { id: number; type: 'preloaded'; info: T2VModelInfo }
  | { id: number; type: 'result'; result: WorkerGenerateResult }
  | { id: number; type: 'error'; message: string }
