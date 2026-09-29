// BERT WordPiece tokenizer, read from a Hugging Face tokenizer.json.
//
// Reproduces what the `tokenizers` library does for all-MiniLM-L6-v2 - the
// tokenizer the training captions were embedded with - so the browser feeds
// MiniLM the same ids as TextEncoder in training/text2voxel/common.py:
//
//   added tokens   "[CLS]", "[SEP]", "[MASK]", ... matched in the raw text
//   BertNormalizer clean text (drop control/format chars, whitespace -> " "),
//                  pad CJK ideographs with spaces, NFD + drop non-spacing marks
//                  (strip_accents defaults to `lowercase`), lowercase per char
//   BertPreTokenizer split on whitespace, isolate each punctuation char
//   WordPiece      greedy longest match with "##" continuations; a word with an
//                  unmatched piece, or longer than 100 chars, becomes [UNK]
//   post-process   [CLS] ... [SEP], truncated to `maxTokens` in total
//
// The character classes come from bertTables.ts, probed from the Rust
// implementation (see make_text2voxel_fixtures.py). NFD and lowercasing use the
// JavaScript built-ins, but NFD only for the code points the Rust tables
// decompose: browsers carry newer Unicode data that also decomposes characters
// added since (Unicode 16's Tulu-Tigalari, Gurung Khema, ...), which would give
// MiniLM different input than training saw. verify-text2voxel.mjs checks every
// code point. Known remaining gap: characters are decomposed one at a time, so
// NFD's canonical reordering of combining marks is skipped. That can only
// change the order of combining marks that survive accent stripping (a handful
// of spacing marks), never the ids of ordinary text.

import { CHINESE, DECOMPOSE, MARKS, PUNCTUATION, REMOVED, SPLIT_SPACE, WHITESPACE } from './bertTables.ts'

const CLASS_REMOVED = 1
const CLASS_WHITESPACE = 2
const CLASS_CHINESE = 4
const CLASS_MARK = 8
const CLASS_PUNCT = 16
const CLASS_DECOMPOSE = 32
const CLASS_SPLIT = 64

const MAX_CODE_POINT = 0x110000

let classTable: Uint8Array | null = null

/** One byte of class flags per code point (1.1 MB), built on first use. */
function classes(): Uint8Array {
  if (classTable) return classTable
  const table = new Uint8Array(MAX_CODE_POINT)
  const mark = (encoded: string, flag: number) => {
    let previous = 0
    for (const part of encoded.split(',')) {
      const [gap, length] = part.split('.')
      const start = previous + parseInt(gap!, 36)
      const end = start + parseInt(length!, 36)
      for (let cp = start; cp <= end; cp += 1) table[cp] |= flag
      previous = end
    }
  }
  mark(REMOVED, CLASS_REMOVED)
  mark(WHITESPACE, CLASS_WHITESPACE)
  mark(CHINESE, CLASS_CHINESE)
  mark(MARKS, CLASS_MARK)
  mark(PUNCTUATION, CLASS_PUNCT)
  mark(DECOMPOSE, CLASS_DECOMPOSE)
  mark(SPLIT_SPACE, CLASS_SPLIT)
  table[0x20] |= CLASS_WHITESPACE
  // Lone surrogates cannot reach the Rust tokenizer at all (Python refuses to
  // encode them); drop them rather than feed MiniLM garbage.
  for (let cp = 0xd800; cp <= 0xdfff; cp += 1) table[cp] |= CLASS_REMOVED
  classTable = table
  return table
}

export interface NormalizerOptions {
  cleanText: boolean
  handleChineseChars: boolean
  stripAccents: boolean
  lowercase: boolean
}

const BERT_DEFAULTS: NormalizerOptions = {
  cleanText: true, handleChineseChars: true, stripAccents: true, lowercase: true,
}

/** BertNormalizer.normalize_str. */
export function normalizeText(text: string, options: NormalizerOptions = BERT_DEFAULTS): string {
  const table = classes()
  let cleaned = ''
  for (const char of text) {
    const flags = table[char.codePointAt(0)!]!
    if (options.cleanText) {
      if (flags & CLASS_REMOVED) continue
      if (flags & CLASS_WHITESPACE) {
        cleaned += ' '
        continue
      }
    }
    cleaned += options.handleChineseChars && flags & CLASS_CHINESE ? ` ${char} ` : char
  }
  if (!options.stripAccents && !options.lowercase) return cleaned

  let out = ''
  for (const char of cleaned) {
    const pieces = options.stripAccents && table[char.codePointAt(0)!]! & CLASS_DECOMPOSE
      ? char.normalize('NFD')
      : char
    for (const piece of pieces) {
      if (options.stripAccents && table[piece.codePointAt(0)!]! & CLASS_MARK) continue
      // Per character, like Rust's char::to_lowercase: no final-sigma context.
      out += options.lowercase ? piece.toLowerCase() : piece
    }
  }
  return out
}

/** BertPreTokenizer: split on whitespace, isolate every punctuation character. */
export function preTokenize(text: string): string[] {
  const table = classes()
  const words: string[] = []
  let current = ''
  for (const char of text) {
    const flags = table[char.codePointAt(0)!]!
    if (flags & CLASS_SPLIT) {
      if (current) words.push(current)
      current = ''
    } else if (flags & CLASS_PUNCT) {
      if (current) words.push(current)
      words.push(char)
      current = ''
    } else {
      current += char
    }
  }
  if (current) words.push(current)
  return words
}

interface TokenizerJson {
  added_tokens?: { id: number; content: string; normalized?: boolean }[]
  normalizer?: {
    type?: string
    clean_text?: boolean
    handle_chinese_chars?: boolean
    strip_accents?: boolean | null
    lowercase?: boolean
  } | null
  post_processor?: { special_tokens?: Record<string, { ids: number[] }> } | null
  model: {
    type?: string
    vocab: Record<string, number>
    unk_token?: string
    continuing_subword_prefix?: string
    max_input_chars_per_word?: number
  }
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

export class BertTokenizer {
  private readonly vocab: Map<string, number>
  private readonly unkId: number
  private readonly clsId: number
  private readonly sepId: number
  private readonly prefix: string
  private readonly maxWordChars: number
  private readonly normalizer: NormalizerOptions
  private readonly addedPattern: RegExp | null
  private readonly addedIds: Map<string, number>
  readonly maxTokens: number

  constructor(json: TokenizerJson, maxTokens = 128) {
    this.maxTokens = maxTokens
    if (json.model?.type && json.model.type !== 'WordPiece') {
      throw new Error(`tokenizer.json: expected a WordPiece model, got ${json.model.type}`)
    }
    this.vocab = new Map(Object.entries(json.model.vocab))
    const lookup = (token: string) => {
      const id = this.vocab.get(token)
      if (id === undefined) throw new Error(`tokenizer.json: vocabulary has no ${token}`)
      return id
    }
    this.unkId = lookup(json.model.unk_token ?? '[UNK]')
    const special = json.post_processor?.special_tokens ?? {}
    this.clsId = special['[CLS]']?.ids[0] ?? lookup('[CLS]')
    this.sepId = special['[SEP]']?.ids[0] ?? lookup('[SEP]')
    this.prefix = json.model.continuing_subword_prefix ?? '##'
    this.maxWordChars = json.model.max_input_chars_per_word ?? 100

    const norm = json.normalizer ?? {}
    const lowercase = norm.lowercase ?? true
    this.normalizer = {
      cleanText: norm.clean_text ?? true,
      handleChineseChars: norm.handle_chinese_chars ?? true,
      // HF: strip_accents = None means "follow lowercase".
      stripAccents: norm.strip_accents ?? lowercase,
      lowercase,
    }

    // Added tokens with normalized=false are matched in the raw input before
    // anything else; the longest wins where two could start at one position.
    const added = (json.added_tokens ?? []).filter((token) => token.normalized === false)
    this.addedIds = new Map(added.map((token) => [token.content, token.id]))
    const alternatives = added
      .map((token) => token.content)
      .sort((a, b) => b.length - a.length)
      .map(escapeRegExp)
    this.addedPattern = alternatives.length ? new RegExp(alternatives.join('|'), 'g') : null
  }

  /** WordPiece ids for one pre-tokenized word. */
  private wordPiece(word: string): number[] {
    const chars = Array.from(word)
    if (chars.length > this.maxWordChars) return [this.unkId]
    const ids: number[] = []
    let start = 0
    while (start < chars.length) {
      let end = chars.length
      let found = -1
      while (start < end) {
        const piece = chars.slice(start, end).join('')
        const id = this.vocab.get(start > 0 ? this.prefix + piece : piece)
        if (id !== undefined) {
          found = id
          break
        }
        end -= 1
      }
      if (found < 0) return [this.unkId]
      ids.push(found)
      start = end
    }
    return ids
  }

  private encodeSegment(text: string, into: number[]): void {
    for (const word of preTokenize(normalizeText(text, this.normalizer))) {
      into.push(...this.wordPiece(word))
    }
  }

  /** Token ids without [CLS]/[SEP] and without truncation. */
  tokenize(text: string): number[] {
    const ids: number[] = []
    if (!this.addedPattern) {
      this.encodeSegment(text, ids)
      return ids
    }
    let last = 0
    for (const match of text.matchAll(this.addedPattern)) {
      this.encodeSegment(text.slice(last, match.index), ids)
      ids.push(this.addedIds.get(match[0])!)
      last = match.index! + match[0].length
    }
    this.encodeSegment(text.slice(last), ids)
    return ids
  }

  /** [CLS] tokens [SEP], truncated from the right to `maxTokens` in total. */
  encode(text: string): number[] {
    const body = this.tokenize(text).slice(0, Math.max(0, this.maxTokens - 2))
    return [this.clsId, ...body, this.sepId]
  }
}
