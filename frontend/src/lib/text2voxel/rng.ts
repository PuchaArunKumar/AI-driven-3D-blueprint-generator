// Seeded Gaussian noise for the diffusion sampler.
//
// The same seed always gives the same starting latent, so a result can be
// reproduced from (prompt, seed, guidance, steps). The stream is NOT the one
// torch.randn produces - the Python reference sample stores its noise vector
// in meta.json instead, which is how the samplers are cross-checked.

/** mulberry32: a small, fast 32-bit PRNG with a full 2^32 period. */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** `count` independent standard-normal samples (Box-Muller) from `seed`. */
export function gaussianNoise(seed: number, count: number): Float64Array {
  const uniform = mulberry32(seed)
  const out = new Float64Array(count)
  for (let i = 0; i < count; i += 2) {
    // 1 - u keeps the logarithm's argument in (0, 1].
    const radius = Math.sqrt(-2 * Math.log(1 - uniform()))
    const angle = 2 * Math.PI * uniform()
    out[i] = radius * Math.cos(angle)
    if (i + 1 < count) out[i + 1] = radius * Math.sin(angle)
  }
  return out
}

/** A random 32-bit seed, from the platform CSPRNG when there is one. */
export function randomSeed(): number {
  const crypto = globalThis.crypto
  if (crypto?.getRandomValues) return crypto.getRandomValues(new Uint32Array(1))[0]!
  return Math.floor(Math.random() * 4294967296) >>> 0
}
