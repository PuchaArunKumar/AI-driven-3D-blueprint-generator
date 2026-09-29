import { useMemo } from 'react'
import { History, X } from 'lucide-react'
import type { TechnicalViewData } from '../lib/types'

export interface HistoryItem {
  id: number
  prompt: string
  seed: number
  /** Front elevation of the result, drawn as the thumbnail. */
  front: TechnicalViewData | null
}

interface Props {
  items: HistoryItem[]
  currentId: number | null
  limit: number
  disabled?: boolean
  onSelect: (id: number) => void
  onRemove: (id: number) => void
  onClear: () => void
}

/**
 * A result's front silhouette as a tiny SVG.
 *
 * Drawn from the blueprint the result already carries, so a thumbnail costs
 * no extra render and no image memory.
 */
function MiniOutline({ view }: { view: TechnicalViewData | null }) {
  const path = useMemo(() => {
    if (!view) return ''
    const paths = [...view.outline, ...view.inline]
    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (const line of paths) {
      for (const [x, y] of line) {
        minX = Math.min(minX, x)
        maxX = Math.max(maxX, x)
        minY = Math.min(minY, y)
        maxY = Math.max(maxY, y)
      }
    }
    if (!Number.isFinite(minX)) return ''
    const size = 64
    const pad = 6
    const scale = Math.min((size - pad * 2) / Math.max(maxX - minX, 1e-9), (size - pad * 2) / Math.max(maxY - minY, 1e-9))
    const offsetX = (size - (maxX - minX) * scale) / 2
    const offsetY = (size - (maxY - minY) * scale) / 2
    return paths
      .filter((line) => line.length > 1)
      .map(
        (line) =>
          `M${line
            .map(([x, y]) => `${(offsetX + (x - minX) * scale).toFixed(1)},${(size - (offsetY + (y - minY) * scale)).toFixed(1)}`)
            .join(' L')}`,
      )
      .join(' ')
  }, [view])

  return (
    <svg viewBox="0 0 64 64" className="w-full h-full" aria-hidden="true">
      {path ? (
        <path d={path} fill="rgba(76,201,240,.08)" stroke="#4cc9f0" strokeWidth="1" strokeLinejoin="round" />
      ) : (
        <text x="32" y="35" textAnchor="middle" fill="#475569" fontSize="7">
          empty
        </text>
      )}
    </svg>
  )
}

/** This session's results, newest first. Kept in memory only. */
export default function SessionHistory({
  items,
  currentId,
  limit,
  disabled,
  onSelect,
  onRemove,
  onClear,
}: Props) {
  if (items.length === 0) return null

  return (
    <section className="panel" aria-labelledby="t2v-history-title">
      <div className="panel-head">
        <div>
          <h2 id="t2v-history-title" className="text-sm font-semibold text-white flex items-center gap-2">
            <History className="w-4 h-4 text-blueprint-400" />
            This session
          </h2>
          <p className="text-[11px] text-slate-500">
            In memory only (last {limit}) - download anything you want to keep.
          </p>
        </div>
        <button type="button" className="btn-ghost !py-1 !px-2 !text-xs" onClick={onClear} disabled={disabled}>
          Clear
        </button>
      </div>

      <ul className="p-3 flex gap-2.5 overflow-x-auto">
        {items.map((item) => {
          const active = item.id === currentId
          return (
            <li key={item.id} className="relative w-32 shrink-0">
              <button
                type="button"
                className={`w-full text-left rounded-lg border p-1.5 transition-colors ${
                  active
                    ? 'border-blueprint-500 ring-1 ring-blueprint-500/50 bg-blueprint-500/5'
                    : 'border-ink-700 hover:border-ink-600 bg-ink-950/60'
                }`}
                aria-pressed={active}
                aria-label={`Show result: ${item.prompt}, seed ${item.seed}`}
                disabled={disabled}
                onClick={() => onSelect(item.id)}
              >
                <div className="aspect-square rounded-md bg-ink-950 grid-bg-fine">
                  <MiniOutline view={item.front} />
                </div>
                <p className="mt-1 text-[11px] text-slate-300 truncate" title={item.prompt}>
                  {item.prompt}
                </p>
                <p className="font-mono text-[10px] text-slate-500">seed {item.seed}</p>
              </button>
              <button
                type="button"
                className="absolute top-2.5 right-2.5 p-0.5 rounded bg-ink-950/85 text-slate-500 hover:text-signal-err"
                aria-label={`Remove result: ${item.prompt}, seed ${item.seed}`}
                title="Remove"
                disabled={disabled}
                onClick={() => onRemove(item.id)}
              >
                <X className="w-3 h-3" />
              </button>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
