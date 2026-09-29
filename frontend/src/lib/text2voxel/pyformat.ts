// Python-compatible number formatting.
//
// The blueprint sheet and the dimension parser are ports of backend Python
// code, and their output must match it character for character. JavaScript's
// toFixed rounds an exact tie away from zero; Python rounds it to even
// (format(0.125, '.2f') == '0.12', (0.125).toFixed(2) == '0.13'). These
// helpers work on the exact binary value with BigInt, so they agree with
// Python for every double, not just the easy ones.

const SCRATCH = new DataView(new ArrayBuffer(8))

/** Exact decomposition of a finite, non-negative double: value = mantissa * 2^exponent. */
function decompose(value: number): [bigint, number] {
  SCRATCH.setFloat64(0, value)
  const high = SCRATCH.getUint32(0)
  const low = SCRATCH.getUint32(4)
  const biased = (high >>> 20) & 0x7ff
  let mantissa = (BigInt(high & 0xfffff) << 32n) | BigInt(low)
  if (biased === 0) return [mantissa, -1074] // subnormal
  mantissa |= 1n << 52n
  return [mantissa, biased - 1075]
}

/** Python's `format(value, f'.{digits}f')`: correctly rounded, ties to even. */
export function pyFixed(value: number, digits: number): string {
  if (Number.isNaN(value)) return 'nan'
  if (!Number.isFinite(value)) return value > 0 ? 'inf' : '-inf'
  const negative = value < 0 || Object.is(value, -0)
  const [mantissa, exponent] = decompose(Math.abs(value))

  let numerator = mantissa * 10n ** BigInt(digits)
  let denominator = 1n
  if (exponent >= 0) numerator <<= BigInt(exponent)
  else denominator <<= BigInt(-exponent)

  let quotient = numerator / denominator
  const twice = (numerator % denominator) * 2n
  if (twice > denominator || (twice === denominator && (quotient & 1n) === 1n)) quotient += 1n

  const text = quotient.toString().padStart(digits + 1, '0')
  const body = digits > 0 ? `${text.slice(0, -digits)}.${text.slice(-digits)}` : text
  return negative ? `-${body}` : body
}

/** Python's `round(value, digits)` for floats. */
export function pyRound(value: number, digits: number): number {
  if (!Number.isFinite(value)) return value
  return Number(pyFixed(value, digits))
}

/** Python's `repr(float)` / `str(float)`: shortest round-trip, always with a point or exponent. */
export function pyFloatRepr(value: number): string {
  if (Number.isNaN(value)) return 'nan'
  if (!Number.isFinite(value)) return value > 0 ? 'inf' : '-inf'
  if (value === 0) return Object.is(value, -0) ? '-0.0' : '0.0'
  const magnitude = Math.abs(value)
  if (magnitude < 1e-4 || magnitude >= 1e16) {
    const [mantissa, exponent] = value.toExponential().split('e')
    const power = Number(exponent)
    return `${mantissa}e${power < 0 ? '-' : '+'}${String(Math.abs(power)).padStart(2, '0')}`
  }
  const text = String(value)
  return Number.isInteger(value) ? `${text}.0` : text
}

/** Python's `f'{value:,}'` for integers. */
export function pyThousands(value: number): string {
  const text = String(Math.trunc(Math.abs(value)))
  const grouped = text.replace(/\B(?=(\d{3})+(?!\d))/g, ',')
  return value < 0 ? `-${grouped}` : grouped
}

/** Python's `html.escape(text)` (quote=True). */
export function htmlEscape(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;')
}

/** Python's `text[:count]` - slices by code point, not UTF-16 unit. */
export function pySlice(text: string, count: number): string {
  return Array.from(text).slice(0, count).join('')
}
