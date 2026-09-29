import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { AlertTriangle, Download, FileText } from 'lucide-react'
import { api } from '../lib/api'
import type { BlueprintData, TechnicalViewData } from '../lib/types'
import { formatLength } from './Viewer3D'

interface Props {
  data: BlueprintData | null
  projectId?: string
  loading?: boolean
  error?: string | null
  projectName?: string
  isoUrl?: string | null
  /**
   * Custom preview/download controls for the combined sheet, for results that
   * are not backend projects (the in-browser Text → 3D page builds its SVG
   * locally). Without it the sheet links go to the API for `projectId`.
   */
  sheetActions?: (theme: SheetTheme) => ReactNode
}

const SIZE = 300
const PAD = 46

export type SheetTheme = 'white' | 'blueprint'

/** Turn model-space polylines into one SVG path, fitted to the panel. */
function buildPath(
  paths: number[][][],
  minimum: [number, number],
  scale: number,
  offset: [number, number],
  closed: boolean,
): string {
  const commands: string[] = []
  for (const path of paths) {
    if (path.length < 2) continue
    const points = path.map(([x, y]) => {
      const px = offset[0] + (x - minimum[0]) * scale
      // SVG y grows downward; model y grows up.
      const py = SIZE - (offset[1] + (y - minimum[1]) * scale)
      return `${px.toFixed(2)},${py.toFixed(2)}`
    })
    commands.push(`M${points.join(' L')}${closed ? ' Z' : ''}`)
  }
  return commands.join(' ')
}

/**
 * One orthographic panel drawn from real projected geometry.
 *
 * Three line families, following drafting convention: a heavy continuous
 * outline, lighter continuous inline for creases and interior contours, and
 * dashed lines where the body occludes an edge.
 */
function OrthoPanel({
  title,
  code,
  view,
  horizontalLabel,
  verticalLabel,
  showHidden,
}: {
  title: string
  code: string
  view: TechnicalViewData
  horizontalLabel: string
  verticalLabel: string
  showHidden: boolean
}) {
  const geometry = useMemo(() => {
    const all = [...view.outline, ...view.inline, ...view.hidden]
    if (all.length === 0) return null

    let minX = Infinity
    let maxX = -Infinity
    let minY = Infinity
    let maxY = -Infinity
    for (const path of all) {
      for (const [x, y] of path) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    }
    const spanX = Math.max(maxX - minX, 1e-9)
    const spanY = Math.max(maxY - minY, 1e-9)
    const scale = Math.min((SIZE - PAD * 2) / spanX, (SIZE - PAD * 2) / spanY)
    const minimum: [number, number] = [minX, minY]
    const offset: [number, number] = [
      (SIZE - spanX * scale) / 2,
      (SIZE - spanY * scale) / 2,
    ]
    return {
      outline: buildPath(view.outline, minimum, scale, offset, true),
      inline: buildPath(view.inline, minimum, scale, offset, false),
      hidden: buildPath(view.hidden, minimum, scale, offset, false),
    }
  }, [view])

  return (
    <div className="relative bg-ink-950 border border-ink-700/70 rounded-lg overflow-hidden">
      <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-3 py-1.5 border-b border-ink-700/60 bg-ink-900/70 z-10">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-blueprint-400">
          {title}
        </span>
        <span className="font-mono text-[10px] text-slate-500">{code}</span>
      </div>

      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="w-full h-auto block"
        role="img"
        aria-label={`${title} orthographic projection`}
      >
        <defs>
          <pattern id={`bp-grid-${code}`} width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M20 0 L0 0 0 20" fill="none" stroke="rgba(76,201,240,.08)" strokeWidth="1" />
          </pattern>
          <marker id={`bp-arrow-${code}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L7,3.5 L0,7 z" fill="#4cc9f0" />
          </marker>
          <marker id={`bp-arrow-s-${code}`} markerWidth="7" markerHeight="7" refX="1" refY="3.5" orient="auto">
            <path d="M7,0 L0,3.5 L7,7 z" fill="#4cc9f0" />
          </marker>
        </defs>

        <rect width={SIZE} height={SIZE} fill={`url(#bp-grid-${code})`} />

        {geometry === null ? (
          <text x={SIZE / 2} y={SIZE / 2} textAnchor="middle" fill="#475569" fontSize="11">
            no geometry
          </text>
        ) : (
          <g>
            {showHidden && geometry.hidden && (
              <path
                d={geometry.hidden}
                fill="none"
                stroke="#3d5570"
                strokeWidth="0.7"
                strokeDasharray="4 3"
              />
            )}
            {geometry.inline && (
              <path
                d={geometry.inline}
                fill="none"
                stroke="#5f9ec9"
                strokeWidth="0.8"
                strokeLinecap="round"
                opacity="0.9"
              />
            )}
            {geometry.outline && (
              <path
                d={geometry.outline}
                fill="rgba(76,201,240,.07)"
                fillRule="evenodd"
                stroke="#4cc9f0"
                strokeWidth="1.7"
                strokeLinejoin="round"
              />
            )}
          </g>
        )}

        {/* Dimension lines, measured from the projected geometry. */}
        <g stroke="#4cc9f0" strokeWidth="0.9" opacity="0.85">
          <line
            x1={PAD * 0.55} y1={SIZE - 16} x2={SIZE - PAD * 0.55} y2={SIZE - 16}
            markerEnd={`url(#bp-arrow-${code})`} markerStart={`url(#bp-arrow-s-${code})`}
          />
          <line
            x1={16} y1={SIZE - PAD * 0.55} x2={16} y2={PAD * 0.55}
            markerEnd={`url(#bp-arrow-${code})`} markerStart={`url(#bp-arrow-s-${code})`}
          />
        </g>
        <text x={SIZE / 2} y={SIZE - 21} textAnchor="middle" fill="#93c5e8"
              fontSize="10" fontFamily="ui-monospace, monospace">
          {horizontalLabel} {formatLength(view.width)}
        </text>
        <text
          x={11} y={SIZE / 2} textAnchor="middle" fill="#93c5e8"
          fontSize="10" fontFamily="ui-monospace, monospace"
          transform={`rotate(-90 11 ${SIZE / 2})`}
        >
          {verticalLabel} {formatLength(view.height)}
        </text>
      </svg>
    </div>
  )
}

export default function BlueprintView({
  data,
  projectId,
  loading,
  error,
  projectName,
  isoUrl,
  sheetActions,
}: Props) {
  const [showHidden, setShowHidden] = useState(true)
  const [sheetTheme, setSheetTheme] = useState<SheetTheme>('white')

  if (loading) {
    return (
      <div className="panel p-10 text-center text-slate-400 text-sm">
        Projecting technical views from the mesh...
      </div>
    )
  }

  if (error) {
    return (
      <div className="panel p-8 text-center">
        <AlertTriangle className="w-8 h-8 mx-auto mb-3 text-signal-warn" />
        <p className="text-slate-300 text-sm">{error}</p>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="panel p-10 text-center text-slate-400 text-sm">
        Generate a 3D model to produce its blueprint.
      </div>
    )
  }

  const { stats, spec, views } = data
  const lineCount = (['front', 'side', 'top'] as const).reduce(
    (total, key) => total + views[key].outline.length + views[key].inline.length,
    0,
  )

  return (
    <div className="space-y-4">
      {/* Title block, as on a real drawing sheet. */}
      <div className="panel px-4 py-3 flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Technical drawing</p>
          <h3 className="text-lg font-bold text-white leading-tight">
            {projectName || spec?.object || 'Generated model'}
          </h3>
        </div>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-1 font-mono text-[11px]">
          <div>
            <span className="text-slate-500">UNITS </span>
            <span className="text-slate-200">metres</span>
          </div>
          <div>
            <span className="text-slate-500">PROJ </span>
            <span className="text-slate-200">third-angle</span>
          </div>
          <div>
            <span className="text-slate-500">PATHS </span>
            <span className="text-slate-200">{lineCount.toLocaleString()}</span>
          </div>
          <label className="flex items-center gap-1.5 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showHidden}
              onChange={(event) => setShowHidden(event.target.checked)}
              className="accent-blueprint-500"
            />
            <span className="text-slate-400">hidden lines</span>
          </label>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <OrthoPanel
          title="Front elevation" code="A-01" view={views.front}
          horizontalLabel="W" verticalLabel="H" showHidden={showHidden}
        />
        <OrthoPanel
          title="Side elevation" code="A-02" view={views.side}
          horizontalLabel="D" verticalLabel="H" showHidden={showHidden}
        />
        <OrthoPanel
          title="Plan (top)" code="A-03" view={views.top}
          horizontalLabel="W" verticalLabel="D" showHidden={showHidden}
        />

        {/* Isometric reference - the rendered model, not a re-drawn projection. */}
        <div className="relative bg-ink-950 border border-ink-700/70 rounded-lg overflow-hidden">
          <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-3 py-1.5 border-b border-ink-700/60 bg-ink-900/70 z-10">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-blueprint-400">
              Isometric
            </span>
            <span className="font-mono text-[10px] text-slate-500">A-04</span>
          </div>
          <div className="absolute inset-0 grid-bg-fine opacity-40" />
          {isoUrl ? (
            <img
              src={isoUrl}
              alt="Isometric view of the generated product"
              className="relative w-full h-full object-contain p-8"
              loading="lazy"
            />
          ) : (
            <div className="relative flex items-center justify-center h-full min-h-[240px] text-slate-600 text-xs">
              Switch to 3D View for the isometric model
            </div>
          )}
        </div>
      </div>

      {/* Combined sheet: all three views on one A3 drawing. */}
      {(projectId || sheetActions) && (
        <div className="panel p-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <FileText className="w-4 h-4 text-blueprint-400 mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-semibold text-white">Combined blueprint sheet</p>
              <p className="text-[12px] text-slate-400">
                All three views on one A3 sheet in third-angle projection, at a single
                shared scale, with dimensions and a title block. Vector SVG.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={sheetTheme}
              onChange={(event) => setSheetTheme(event.target.value as SheetTheme)}
              className="field !py-1.5 !w-auto text-xs"
              aria-label="Sheet style"
            >
              <option value="white">White sheet</option>
              <option value="blueprint">Blueprint</option>
            </select>
            {sheetActions ? (
              sheetActions(sheetTheme)
            ) : projectId ? (
              <>
                <a
                  href={api.blueprintSheetUrl(projectId, sheetTheme)}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="btn-ghost !py-1.5 !text-xs"
                >
                  Preview
                </a>
                <a
                  href={api.blueprintSheetUrl(projectId, sheetTheme, true)}
                  className="btn-primary !py-1.5 !text-xs"
                  download
                >
                  <Download className="w-3.5 h-3.5" />
                  Download SVG
                </a>
              </>
            ) : null}
          </div>
        </div>
      )}

      {/* Measured data, straight from the mesh. */}
      <div className="panel p-4">
        <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500 mb-3">
          Measured from geometry
        </p>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          {[
            ['Width (X)', formatLength(stats.width)],
            ['Height (Y)', formatLength(stats.height)],
            ['Depth (Z)', formatLength(stats.depth)],
            ['Triangles', stats.triangles.toLocaleString()],
            ['Vertices', stats.vertices.toLocaleString()],
            ['Surface area', stats.surface_area ? `${stats.surface_area.toFixed(4)} m²` : '-'],
            ['Volume', stats.volume ? `${(stats.volume * 1e6).toFixed(1)} cm³` : 'not watertight'],
            ['Watertight', stats.is_watertight ? 'yes' : 'no'],
            ['Format', stats.file_format.toUpperCase() || '-'],
            ['Tolerance', spec?.tolerances_mm ? `±${spec.tolerances_mm} mm` : 'not specified'],
          ].map(([key, value]) => (
            <div key={key}>
              <p className="stat-key">{key}</p>
              <p className="stat-val">{value}</p>
            </div>
          ))}
        </div>
      </div>

      <p className="text-[11px] text-slate-500 leading-relaxed border-l-2 border-signal-warn/50 pl-3">
        <strong className="text-slate-400">Design output, not a manufacturing drawing.</strong>{' '}
        The outline, inline and hidden lines are projected from the generated mesh to
        communicate intent and proportion. They carry no GD&amp;T, no tolerancing scheme
        and no material certification, and must be reviewed by a qualified engineer
        before any part is made from them.
      </p>
    </div>
  )
}
