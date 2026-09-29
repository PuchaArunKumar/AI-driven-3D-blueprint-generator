// Target dimensions from a free-text prompt.
//
// A port of the dimension rules in backend/app/pipeline/prompt_parser.py
// (_extract_triple, _extract_named_axes and the _TO_MM table), so the browser
// pre-fills exactly what the backend would parse from the same prompt. Keep the
// two in step: change the Python first, then mirror it here.
//
// Python's `re` is Unicode-aware where JavaScript's is not: `\d` is any decimal
// digit, `\s` includes U+001C-001F and U+0085, and `\b` treats every letter or
// number as a word character. The patterns below spell those classes out so
// both engines agree on odd input too (verify-text2voxel.mjs checks the pair).

import { pyRound } from './pyformat.ts'

export interface ParsedDimensions {
  length: number | null
  width: number | null
  height: number | null
}

const TO_MM: Record<string, number> = {
  mm: 1.0, millimetre: 1.0, millimeter: 1.0, millimetres: 1.0, millimeters: 1.0,
  cm: 10.0, centimetre: 10.0, centimeter: 10.0, centimetres: 10.0, centimeters: 10.0,
  m: 1000.0, metre: 1000.0, meter: 1000.0, metres: 1000.0, meters: 1000.0,
  in: 25.4, inch: 25.4, inches: 25.4, '"': 25.4,
  ft: 304.8, foot: 304.8, feet: 304.8, "'": 304.8,
}

// Python: sorted((u for u in _TO_MM if u.isalpha()), key=len, reverse=True).
// Array.prototype.sort is stable, like Python's sorted, so ties keep table order.
const UNIT_ALT = Object.keys(TO_MM)
  .filter((unit) => /^\p{L}+$/u.test(unit))
  .sort((a, b) => b.length - a.length)
  .join('|')

/** Python `\d` (Unicode decimal digit). */
const D = '\\p{Nd}'
/** Python `\s` (str.isspace). */
const S = '[\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]'
/** Python `\b` after a word character: the next character is not a word character. */
const END = '(?![\\p{L}\\p{N}_])'
const NUMBER = `(${D}+(?:\\.${D}+)?)`

const TRIPLE = new RegExp(
  `${NUMBER}${S}*(${UNIT_ALT})?${S}*[x×]${S}*` +
  `${NUMBER}${S}*(${UNIT_ALT})?${S}*[x×]${S}*` +
  `${NUMBER}${S}*(${UNIT_ALT})`,
  'iu',
)

const AXIS_WORDS: [keyof ParsedDimensions, string[]][] = [
  ['length', ['long', 'length', 'deep', 'depth']],
  ['width', ['wide', 'width', 'across']],
  ['height', ['tall', 'high', 'height']],
]

const AXIS_PATTERNS = AXIS_WORDS.map(([field, words]) => {
  const wordAlt = words.join('|')
  return {
    field,
    // "<number><unit> tall"  /  "<number> <unit> in height"
    after: new RegExp(`${NUMBER}${S}*(${UNIT_ALT})${S}*(?:in${S}+)?(?:${wordAlt})${END}`, 'iu'),
    // "height of <number><unit>"  /  "a height of 40 cm"
    before: new RegExp(`(?:${wordAlt})${S}*(?:of|:|=)?${S}*${NUMBER}${S}*(${UNIT_ALT})${END}`, 'iu'),
  }
})

/** Python float() on a `\d+(\.\d+)?` match, which may use any script's digits. */
function parseNumber(text: string): number {
  let ascii = ''
  for (const char of text) {
    const code = char.codePointAt(0)!
    if (char === '.' || (code >= 0x30 && code <= 0x39)) {
      ascii += char
      continue
    }
    // Decimal digits are assigned in contiguous, aligned runs of ten (a
    // Unicode stability guarantee), so the value is the offset from the run's
    // zero: walk back to the start of the run of digits, then take mod 10.
    let start = code
    while (start > 0 && /\p{Nd}/u.test(String.fromCodePoint(start - 1))) start -= 1
    ascii += String((code - start) % 10)
  }
  return Number(ascii)
}

function toMm(value: number, unit: string | undefined): number | null {
  if (unit === undefined) return null
  const factor = TO_MM[unit.toLowerCase().trim()]
  return factor ? pyRound(value * factor, 3) : null
}

/** Normalise the way parse_prompt does before extracting: collapse whitespace. */
function normalise(prompt: string): string {
  const strip = new RegExp(`^${S}+|${S}+$`, 'gu')
  return prompt.replace(strip, '').replace(new RegExp(`${S}+`, 'gu'), ' ')
}

function extractTriple(text: string): [number, number, number] | null {
  const match = TRIPLE.exec(text)
  if (!match) return null
  const trailing = match[6]
  const values: number[] = []
  for (const [number, unit] of [[match[1], match[2]], [match[3], match[4]], [match[5], match[6]]]) {
    const millimetres = toMm(parseNumber(number!), unit || trailing)
    if (millimetres === null) return null
    values.push(millimetres)
  }
  return [values[0]!, values[1]!, values[2]!]
}

/**
 * Target dimensions in millimetres: "A x B x C unit" is length x width x
 * height, and named axes ("1.2 m tall", "height of 40 cm", "40cm wide")
 * override the triple, exactly as the backend's _extract_dimensions.
 */
export function parseDimensions(prompt: string): ParsedDimensions {
  const result: ParsedDimensions = { length: null, width: null, height: null }
  if (!prompt) return result
  const text = normalise(prompt)

  const triple = extractTriple(text)
  if (triple) [result.length, result.width, result.height] = triple

  for (const { field, after, before } of AXIS_PATTERNS) {
    const match = after.exec(text) ?? before.exec(text)
    if (!match) continue
    const millimetres = toMm(parseNumber(match[1]!), match[2])
    if (millimetres !== null) result[field] = millimetres
  }
  return result
}
