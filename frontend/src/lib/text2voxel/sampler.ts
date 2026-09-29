// Deterministic DDIM sampling with classifier-free guidance.
//
// Mirrors `ddim_sample` and the reference loop in training/text2voxel/k2_train.py
// exactly (v-parameterisation, x0 clamp, zero-embedding null condition). The
// prior network is passed in as a function, so the maths here is shared by the
// browser worker and the Node verification script.

/** numpy's round(): halves go to the even neighbour. */
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value)
  const diff = value - floor
  if (diff > 0.5) return floor + 1
  if (diff < 0.5) return floor
  return floor % 2 === 0 ? floor : floor + 1
}

/**
 * `np.linspace(train_steps - 1, 0, steps).round().astype(int)`, reproducing
 * numpy's float arithmetic (y = i * step + start, last element pinned to stop)
 * so the integer timesteps are identical.
 */
export function ddimTimesteps(trainSteps: number, steps: number): number[] {
  if (!Number.isInteger(steps) || steps < 1) throw new Error(`steps must be a positive integer, got ${steps}`)
  const start = trainSteps - 1
  const stop = 0
  if (steps === 1) return [roundHalfEven(start)]
  const step = (stop - start) / (steps - 1)
  const times: number[] = []
  for (let i = 0; i < steps; i += 1) {
    const value = i === steps - 1 ? stop : i * step + start
    times.push(roundHalfEven(value))
  }
  return times
}

/**
 * One batched prior evaluation. Called with B = 2 rows - [x, x] at the same
 * timestep with conditions [cond, zeros] - and returns v as [v_cond, v_uncond]
 * (2 x latentDim, row-major).
 */
export type PriorFn = (x: Float32Array, t: Float32Array, cond: Float32Array) => Promise<Float32Array>

export interface SampleOptions {
  alphasCumprod: ArrayLike<number>
  trainSteps: number
  steps: number
  guidance: number
  x0Clip: number
  onStep?: (index: number, total: number) => void
}

/**
 * Run DDIM from `noise` (the standardised latent's starting point) and return
 * the final latent. Arithmetic is float64; the network sees float32.
 */
export async function ddimSample(
  prior: PriorFn,
  cond: Float32Array,
  noise: ArrayLike<number>,
  options: SampleOptions,
): Promise<Float64Array> {
  const dim = noise.length
  const condDim = cond.length
  const times = ddimTimesteps(options.trainSteps, options.steps)
  const x = Float64Array.from(noise)

  // Both rows share x and t; the second row's condition stays all-zero.
  const xBatch = new Float32Array(2 * dim)
  const tBatch = new Float32Array(2)
  const condBatch = new Float32Array(2 * condDim)
  condBatch.set(cond, 0)

  for (let i = 0; i < times.length; i += 1) {
    const t = times[i]!
    const a = options.alphasCumprod[t]!
    const aPrev = i + 1 < times.length ? options.alphasCumprod[times[i + 1]!]! : 1.0
    for (let k = 0; k < dim; k += 1) {
      xBatch[k] = x[k]!
      xBatch[dim + k] = x[k]!
    }
    tBatch[0] = t
    tBatch[1] = t
    const v = await prior(xBatch, tBatch, condBatch)

    const sqrtA = Math.sqrt(a)
    const sqrtOneMinusA = Math.sqrt(1 - a)
    const sqrtAPrev = Math.sqrt(aPrev)
    const sqrtOneMinusAPrev = Math.sqrt(1 - aPrev)
    for (let k = 0; k < dim; k += 1) {
      const vCond = v[k]!
      const vUncond = v[dim + k]!
      const guided = vUncond + options.guidance * (vCond - vUncond)
      let x0 = sqrtA * x[k]! - sqrtOneMinusA * guided
      if (x0 > options.x0Clip) x0 = options.x0Clip
      else if (x0 < -options.x0Clip) x0 = -options.x0Clip
      const eps = sqrtOneMinusA * x[k]! + sqrtA * guided
      x[k] = sqrtAPrev * x0 + sqrtOneMinusAPrev * eps
    }
    options.onStep?.(i + 1, times.length)
  }
  return x
}
