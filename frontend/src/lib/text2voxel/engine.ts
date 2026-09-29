// Text2Voxel-64 inference: prompt -> MiniLM embedding -> DDIM prior -> VAE
// decoder -> surface -> GLB + stats + blueprint.
//
// Runtime-agnostic: the onnxruntime module and the model bytes are passed in,
// so the browser worker (onnxruntime-web, WASM) and the Node verification
// script run this exact code. Nothing here touches the DOM, fetch or caches.

import type * as OrtModule from 'onnxruntime-web'
import type { BlueprintData, ModelStats } from '../types.ts'
import { technicalDrawing } from './blueprint.ts'
import { meshToGlb } from './glb.ts'
import {
  computePlacement,
  extractSurface,
  placeMesh,
  prepareField,
  type MeshData,
  type TargetDimensionsMm,
  type VoxelGrid,
} from './mesher.ts'
import { meshStatistics, meshTopology } from './meshStats.ts'
import type { T2VMeta, T2VStage } from './protocol.ts'
import { gaussianNoise } from './rng.ts'
import { ddimSample, type PriorFn } from './sampler.ts'
import { BertTokenizer } from './tokenizer.ts'

/** The parts of an onnxruntime module the engine uses. */
export interface OrtRuntime {
  Tensor: typeof OrtModule.Tensor
  InferenceSession: typeof OrtModule.InferenceSession
}

type Session = OrtModule.InferenceSession

export interface EngineFiles {
  meta: T2VMeta
  /** Parsed minilm/tokenizer.json. */
  tokenizer: unknown
  minilm: Uint8Array
  prior: Uint8Array
  decoder: Uint8Array
}

export interface EngineRequest {
  prompt: string
  seed: number
  guidance: number
  steps: number
  dimensionsMm?: TargetDimensionsMm | null
}

export interface EngineResult {
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
  /** Occupancy level the surface was taken at (meta.threshold unless the output was near-empty). */
  isoLevel: number
  warnings: string[]
}

export type StageCallback = (stage: T2VStage, fraction: number, message: string) => void

const now = () => performance.now()

export class Text2VoxelEngine {
  readonly meta: T2VMeta
  private readonly ort: OrtRuntime
  private readonly tokenizer: BertTokenizer
  private readonly minilm: Session
  private readonly prior: Session
  private readonly decoder: Session

  private constructor(ort: OrtRuntime, meta: T2VMeta, tokenizer: BertTokenizer,
    minilm: Session, prior: Session, decoder: Session) {
    this.ort = ort
    this.meta = meta
    this.tokenizer = tokenizer
    this.minilm = minilm
    this.prior = prior
    this.decoder = decoder
  }

  static async create(ort: OrtRuntime, files: EngineFiles): Promise<Text2VoxelEngine> {
    const { meta } = files
    const diffusion = meta.diffusion
    if (diffusion.parameterization !== 'v' || (diffusion.null_cond ?? 'zeros') !== 'zeros') {
      throw new Error(`Unsupported diffusion setup in meta.json (${diffusion.parameterization}, ` +
        `${diffusion.null_cond}); this engine implements v-prediction with a zero null condition.`)
    }
    if (diffusion.alphas_cumprod.length !== diffusion.train_steps) {
      throw new Error('meta.json: alphas_cumprod length does not match train_steps')
    }
    const tokenizer = new BertTokenizer(files.tokenizer as ConstructorParameters<typeof BertTokenizer>[0],
      meta.text_encoder?.max_tokens ?? 128)
    const options: OrtModule.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }
    // Sequential on purpose: three concurrent session inits triple peak memory.
    const minilm = await ort.InferenceSession.create(files.minilm, options)
    const prior = await ort.InferenceSession.create(files.prior, options)
    const decoder = await ort.InferenceSession.create(files.decoder, options)
    return new Text2VoxelEngine(ort, meta, tokenizer, minilm, prior, decoder)
  }

  /** Token ids exactly as the training TextEncoder produced them. */
  tokenize(text: string): number[] {
    return this.tokenizer.encode(text)
  }

  /** Mean-pooled, L2-normalised MiniLM embedding (TextEncoder.encode for one string). */
  async embed(text: string): Promise<Float32Array> {
    const ids = this.tokenizer.encode(text)
    const length = ids.length
    const shape = [1, length]
    const { Tensor } = this.ort
    const all: Record<string, OrtModule.Tensor> = {
      input_ids: new Tensor('int64', BigInt64Array.from(ids, (id) => BigInt(id)), shape),
      attention_mask: new Tensor('int64', new BigInt64Array(length).fill(1n), shape),
      token_type_ids: new Tensor('int64', new BigInt64Array(length), shape),
    }
    const feeds: Record<string, OrtModule.Tensor> = {}
    for (const name of this.minilm.inputNames) {
      if (!all[name]) throw new Error(`MiniLM model expects an unknown input "${name}"`)
      feeds[name] = all[name]
    }
    const output = await this.minilm.run(feeds)
    const hidden = output[this.minilm.outputNames[0]!]!
    const width = hidden.dims[2]!
    const data = hidden.data as Float32Array
    // Every token is attended (no padding), so the masked mean is a plain mean.
    const pooled = new Float64Array(width)
    for (let t = 0; t < length; t += 1) {
      for (let k = 0; k < width; k += 1) pooled[k] += data[t * width + k]!
    }
    let norm = 0
    for (let k = 0; k < width; k += 1) {
      pooled[k] /= Math.max(length, 1e-9)
      norm += pooled[k]! * pooled[k]!
    }
    norm = Math.max(Math.sqrt(norm), 1e-12)
    return Float32Array.from(pooled, (value) => value / norm)
  }

  /** DDIM + classifier-free guidance from `noise`; returns the standardised latent. */
  async sample(cond: Float32Array, noise: ArrayLike<number>, guidance: number, steps: number,
    onStep?: (index: number, total: number) => void): Promise<Float64Array> {
    const latentDim = this.meta.latent_dim
    const condDim = this.meta.cond_dim
    if (cond.length !== condDim) throw new Error(`condition has ${cond.length} dims, expected ${condDim}`)
    const { Tensor } = this.ort
    const prior: PriorFn = async (x, t, c) => {
      const output = await this.prior.run({
        x: new Tensor('float32', x, [2, latentDim]),
        t: new Tensor('float32', t, [2]),
        cond: new Tensor('float32', c, [2, condDim]),
      })
      return output[this.prior.outputNames[0]!]!.data as Float32Array
    }
    const diffusion = this.meta.diffusion
    return ddimSample(prior, cond, noise, {
      alphasCumprod: diffusion.alphas_cumprod,
      trainSteps: diffusion.train_steps,
      steps,
      guidance,
      x0Clip: diffusion.x0_clip ?? 6.0,
      onStep,
    })
  }

  /** Standardised latent -> 64^3 occupancy + RGB probabilities. */
  async decode(latent: ArrayLike<number>): Promise<VoxelGrid> {
    const z = Float32Array.from(latent)
    const output = await this.decoder.run({ z: new this.ort.Tensor('float32', z, [1, z.length]) })
    const occupancy = output.occupancy ?? output[this.decoder.outputNames[0]!]!
    const rgb = output.rgb ?? output[this.decoder.outputNames[1]!]!
    return {
      resolution: this.meta.resolution,
      occupancy: occupancy.data as Float32Array,
      rgb: rgb.data as Float32Array,
    }
  }

  async generate(request: EngineRequest, onStage: StageCallback = () => {}): Promise<EngineResult> {
    const { prompt, seed, guidance, steps } = request
    if (!Number.isInteger(steps) || steps < 1 || steps > this.meta.diffusion.train_steps) {
      throw new Error(`steps must be an integer from 1 to ${this.meta.diffusion.train_steps}`)
    }
    if (!Number.isFinite(guidance)) throw new Error('guidance must be a finite number')
    const timings: Record<string, number> = {}
    const started = now()
    let mark = started
    const lap = (name: string) => {
      const time = now()
      timings[name] = Math.round(time - mark)
      mark = time
    }

    onStage('encoding', 0, 'Encoding the prompt (MiniLM)')
    const cond = await this.embed(prompt)
    lap('encode')

    onStage('sampling', 0, `Sampling the shape latent (0/${steps})`)
    const noise = gaussianNoise(seed, this.meta.latent_dim)
    const latent = await this.sample(cond, noise, guidance, steps, (index, total) =>
      onStage('sampling', index / total, `Sampling the shape latent (${index}/${total})`))
    lap('sample')

    const r = this.meta.resolution
    onStage('decoding', 0, `Decoding ${r}³ occupancy and colour`)
    const grid = await this.decode(latent)
    lap('decode')

    onStage('meshing', 0, 'Extracting the surface')
    const shape = prepareField(grid, this.meta.threshold ?? 0.5)
    const raw = extractSurface(shape, grid)
    const placement = computePlacement(raw, request.dimensionsMm)
    const mesh = placeMesh(raw, placement)
    const topology = meshTopology(mesh)
    lap('mesh')

    onStage('meshing', 0.5, 'Writing the GLB')
    const glb = await meshToGlb(mesh, {
      generator: 'Text2Voxel-64 in-browser engine',
      model: `${this.meta.name} v${this.meta.version} (${this.meta.created})${this.meta.smoke ? ' placeholder' : ''}`,
      prompt, seed, guidance, steps,
    })
    lap('export')

    onStage('meshing', 0.75, 'Projecting the blueprint views')
    const stats = meshStatistics(mesh, glb.byteLength, null, topology)
    const blueprint = technicalDrawing(shape, placement, stats)
    lap('blueprint')

    timings.total = Math.round(now() - started)
    // The blueprint holds this same object, so both see the final time.
    stats.generation_time_s = Math.round(timings.total) / 1000
    const warnings = [...shape.warnings]
    if (!topology.watertight) {
      warnings.push(`The surface is not closed (${topology.boundaryEdges} open, ` +
        `${topology.duplicateEdges} duplicated edges); volume is not reported.`)
    }
    if (shape.droppedComponents > 0) {
      warnings.push(`Dropped ${shape.droppedComponents} floating fragment(s) ` +
        `(${shape.droppedVoxels} voxels) smaller than the minimum part size.`)
    }
    onStage('done', 1, 'Done')

    return {
      prompt, seed, guidance, steps,
      glb, mesh, stats, blueprint,
      timingsMs: timings,
      occupiedVoxels: shape.occupiedVoxels,
      isoLevel: shape.iso,
      warnings,
    }
  }

  async release(): Promise<void> {
    await Promise.all([this.minilm.release(), this.prior.release(), this.decoder.release()])
  }
}
