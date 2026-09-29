// Compose the projected views into a single blueprint sheet as SVG.
//
// A line-for-line port of backend/app/pipeline/blueprint_sheet.py, so a sheet
// drawn in the browser is byte-identical to the one the backend serves for the
// same drawing (verify-text2voxel.mjs checks this against the Python output).
// Keep the two in step: change the Python first, then mirror it here.
//
// The three orthographic views are laid out in third-angle projection - plan
// above the front elevation, side elevation to its right - drawn at one shared
// scale, with dimension lines and a title block.

import type { DesignSpec, TechnicalViewData } from '../types.ts'
import { htmlEscape, pyFixed, pyFloatRepr, pySlice, pyThousands } from './pyformat.ts'

// The sheet is a real A3 landscape: 420 x 297 mm.
const SHEET_WIDTH = 1640
const SHEET_HEIGHT = 1160
const PAPER_WIDTH_MM = 420.0
const PAPER_HEIGHT_MM = 297.0
/** Millimetres of paper per viewBox unit. */
const MM_PER_UNIT = PAPER_WIDTH_MM / SHEET_WIDTH
const MARGIN = 26
const TITLE_WIDTH = 470
const TITLE_HEIGHT = 168
const PANEL_GAP = 18
const VIEW_PADDING = 58

export type SheetThemeName = 'white' | 'blueprint'

interface SheetTheme {
  background: string
  grid: string
  gridMajor: string
  outline: string
  inline: string
  hidden: string
  dimension: string
  text: string
  frame: string
}

const THEMES: Record<SheetThemeName, SheetTheme> = {
  white: {
    background: '#ffffff', grid: '#e9edf2', gridMajor: '#dbe2ea',
    outline: '#11151c', inline: '#5b6980', hidden: '#a9b4c4',
    dimension: '#2f6fb5', text: '#3b4657', frame: '#11151c',
  },
  blueprint: {
    background: '#0e4b8f', grid: '#2f6fb5', gridMajor: '#4a8ad0',
    outline: '#ffffff', inline: '#c9dcf5', hidden: '#7ea8d8',
    dimension: '#eaf2ff', text: '#dbe8fa', frame: '#ffffff',
  },
}

type ViewKey = 'front' | 'side' | 'top'
/** A view as the Python code sees it: every field optional (it uses dict.get). */
type SheetView = Partial<TechnicalViewData>
export type SheetDrawing = Partial<Record<ViewKey, SheetView | null>>

/** Python's `f'{v:.1f}'`, `f'{v:.2f}'`. */
const f1 = (value: number) => pyFixed(value, 1)
const f2 = (value: number) => pyFixed(value, 2)

/** An int or a float, printed the way Python's str() would print it. */
function pyNumber(value: number, isFloat: boolean): string {
  return isFloat ? pyFloatRepr(value) : String(value)
}

function formatLength(metres: number): string {
  const millimetres = metres * 1000.0
  if (millimetres < 10) return `${pyFixed(millimetres, 1)} mm`
  if (millimetres < 1000) return `${pyFixed(millimetres, 0)} mm`
  return `${pyFixed(metres, 3)} m`
}

/** Turn model-space polylines into a single SVG path string. */
function pathsToSvg(
  paths: number[][][],
  transform: (x: number, y: number) => [number, number],
  closed: boolean,
): string {
  const commands: string[] = []
  for (const path of paths) {
    if (path.length < 2) continue
    const points = path.map(([x, y]) => transform(x, y))
    const segment = 'M' + points.map(([x, y]) => `${f2(x)},${f2(y)}`).join(' L')
    commands.push(segment + (closed ? ' Z' : ''))
  }
  return commands.join(' ')
}

/** An arrowed dimension line with a centred label. */
function dimension(
  x1: number, y1: number, x2: number, y2: number, label: string,
  theme: SheetTheme, vertical = false,
): string {
  const midX = (x1 + x2) / 2
  const midY = (y1 + y2) / 2
  const rotate = vertical ? ` transform="rotate(-90 ${f1(midX)} ${f1(midY)})"` : ''
  return (
    `<line x1="${f1(x1)}" y1="${f1(y1)}" x2="${f1(x2)}" y2="${f1(y2)}" ` +
    `stroke="${theme.dimension}" stroke-width="0.9" ` +
    `marker-start="url(#dim-start)" marker-end="url(#dim-end)"/>` +
    `<text x="${f1(midX)}" y="${f1(midY - 5)}" fill="${theme.dimension}" ` +
    `font-size="13" font-family="ui-monospace,monospace" ` +
    `text-anchor="middle"${rotate}>${htmlEscape(label)}</text>`
  )
}

/**
 * Render one view in its panel at a given drawing origin.
 *
 * `origin` is the sheet position of the view's bottom-left corner. Passing it
 * explicitly (rather than centring in the panel) keeps the plan vertically
 * aligned with the front elevation and the side elevation horizontally aligned
 * with it - the defining property of a projection layout.
 */
function panel(
  view: SheetView, box: [number, number, number, number], scale: number,
  title: string, code: string, horizontalLabel: string, verticalLabel: string,
  theme: SheetTheme, origin: [number, number],
): string {
  const [left, top, width, height] = box
  const parts = [
    `<rect x="${f1(left)}" y="${f1(top)}" width="${f1(width)}" height="${f1(height)}" ` +
      `fill="none" stroke="${theme.frame}" stroke-width="0.8" opacity="0.45"/>`,
    `<text x="${f1(left + 12)}" y="${f1(top + 22)}" fill="${theme.text}" ` +
      `font-size="13" font-weight="600" font-family="Inter,system-ui,sans-serif" ` +
      `letter-spacing="1.4">${htmlEscape(title.toUpperCase())}</text>`,
    `<text x="${f1(left + width - 12)}" y="${f1(top + 22)}" fill="${theme.text}" ` +
      `font-size="12" font-family="ui-monospace,monospace" ` +
      `text-anchor="end" opacity="0.7">${htmlEscape(code)}</text>`,
  ]

  const viewWidth = view.width ?? 0.0
  const viewHeight = view.height ?? 0.0
  if (viewWidth <= 0 || viewHeight <= 0) {
    parts.push(
      `<text x="${f1(left + width / 2)}" y="${f1(top + height / 2)}" ` +
        `fill="${theme.text}" font-size="13" text-anchor="middle" ` +
        `opacity="0.6">no geometry</text>`,
    )
    return parts.join('')
  }

  const drawnWidth = viewWidth * scale
  const drawnHeight = viewHeight * scale
  const [originX, originY] = origin

  let minimumX = Infinity
  let minimumY = Infinity
  for (const group of ['outline', 'inline', 'hidden'] as const) {
    for (const path of view[group] ?? []) {
      for (const point of path) {
        if (point[0] < minimumX) minimumX = point[0]
        if (point[1] < minimumY) minimumY = point[1]
      }
    }
  }
  if (minimumX === Infinity) minimumX = 0.0
  if (minimumY === Infinity) minimumY = 0.0

  // SVG y grows downward; model y grows up.
  const transform = (x: number, y: number): [number, number] => [
    originX + (x - minimumX) * scale,
    originY - (y - minimumY) * scale,
  ]

  const hidden = pathsToSvg(view.hidden ?? [], transform, false)
  const inline = pathsToSvg(view.inline ?? [], transform, false)
  const outline = pathsToSvg(view.outline ?? [], transform, true)

  if (hidden) {
    parts.push(`<path d="${hidden}" fill="none" stroke="${theme.hidden}" ` +
      `stroke-width="0.7" stroke-dasharray="5 4" opacity="0.75"/>`)
  }
  if (inline) {
    parts.push(`<path d="${inline}" fill="none" stroke="${theme.inline}" ` +
      `stroke-width="0.85" stroke-linecap="round"/>`)
  }
  if (outline) {
    parts.push(`<path d="${outline}" fill="none" stroke="${theme.outline}" ` +
      `stroke-width="1.9" stroke-linejoin="round"/>`)
  }

  // Dimension lines below and to the left of the drawn extent.
  const baseline = originY + 26
  parts.push(dimension(originX, baseline, originX + drawnWidth, baseline,
    `${horizontalLabel} ${formatLength(viewWidth)}`, theme))
  const leftLine = originX - 26
  parts.push(dimension(leftLine, originY, leftLine, originY - drawnHeight,
    `${verticalLabel} ${formatLength(viewHeight)}`, theme, true))
  return parts.join('')
}

function titleBlock(
  name: string, stats: { triangles?: number } | null, spec: Partial<DesignSpec> | null,
  scaleLabel: string, theme: SheetTheme,
): string {
  const left = SHEET_WIDTH - MARGIN - TITLE_WIDTH
  const top = SHEET_HEIGHT - MARGIN - TITLE_HEIGHT
  const materials = (spec?.materials ?? []).join(', ')
  const tolerance = spec?.tolerances_mm
  const rows: [string, string][] = [
    ['SCALE', scaleLabel],
    ['SHEET', 'A3 landscape'],
    ['PROJECTION', 'third angle'],
    ['MATERIAL', materials || 'not specified'],
    ['TRIANGLES', pyThousands(stats?.triangles ?? 0)],
    // Python formats the float with str(): 0.5 -> "0.5", 2.0 -> "2.0".
    ['TOLERANCE', tolerance ? `±${pyFloatRepr(tolerance)} mm` : 'not specified'],
  ]

  const parts = [
    `<rect x="${left}" y="${top}" width="${TITLE_WIDTH}" height="${TITLE_HEIGHT}" ` +
      `fill="none" stroke="${theme.frame}" stroke-width="1.4"/>`,
    `<line x1="${left}" y1="${top + 42}" x2="${left + TITLE_WIDTH}" y2="${top + 42}" ` +
      `stroke="${theme.frame}" stroke-width="1"/>`,
    `<text x="${left + 14}" y="${top + 28}" fill="${theme.outline}" font-size="18" ` +
      `font-weight="700" font-family="Inter,system-ui,sans-serif">` +
      `${htmlEscape(pySlice(name, 38))}</text>`,
  ]
  rows.forEach(([key, value], index) => {
    const rowY = top + 66 + (index % 3) * 32
    const columnX = left + 14 + Math.floor(index / 3) * 232
    parts.push(
      `<text x="${columnX}" y="${rowY}" fill="${theme.text}" font-size="10" ` +
        `font-family="ui-monospace,monospace" letter-spacing="1.1" ` +
        `opacity="0.75">${key}</text>` +
        `<text x="${columnX}" y="${rowY + 15}" fill="${theme.outline}" ` +
        `font-size="13" font-family="ui-monospace,monospace">` +
        `${htmlEscape(value)}</text>`,
    )
  })
  return parts.join('')
}

/** Python truthiness of `drawing.get(view)`: present and a non-empty dict. */
function present(view: SheetView | null | undefined): view is SheetView {
  return !!view && Object.keys(view).length > 0
}

/** Compose the three views plus a title block into one SVG sheet (render_blueprint_sheet). */
export function renderBlueprintSheet(
  drawing: SheetDrawing,
  stats: { triangles?: number } | null,
  spec: Partial<DesignSpec> | null,
  name = 'Generated model',
  themeName: string = 'white',
): string {
  const theme = THEMES[themeName as SheetThemeName] ?? THEMES.white

  const contentWidth = SHEET_WIDTH - 2 * MARGIN
  const contentHeight = SHEET_HEIGHT - 2 * MARGIN
  const panelWidth = (contentWidth - PANEL_GAP) / 2
  const panelHeight = (contentHeight - PANEL_GAP) / 2

  // Third angle: plan above the front elevation, side elevation to its right.
  const boxes: [ViewKey, [number, number, number, number]][] = [
    ['top', [MARGIN, MARGIN, panelWidth, panelHeight]],
    ['front', [MARGIN, MARGIN + panelHeight + PANEL_GAP, panelWidth, panelHeight]],
    ['side', [MARGIN + panelWidth + PANEL_GAP, MARGIN + panelHeight + PANEL_GAP,
      panelWidth, panelHeight]],
  ]

  const extent = (view: ViewKey, key: 'width' | 'height'): number =>
    Math.max(Number(drawing[view]?.[key] ?? 0.0) || 0.0, 1e-9)

  // One scale across every view, so the sheet is internally measurable.
  const scales = boxes
    .filter(([view]) => present(drawing[view]))
    .map(([view]) => Math.min(
      (panelWidth - 2 * VIEW_PADDING) / extent(view, 'width'),
      (panelHeight - 2 * VIEW_PADDING) / extent(view, 'height'),
    ))
  if (scales.length === 0) throw new Error('min() arg is an empty sequence') // as Python would
  const scale = Math.min(...scales)

  // Front sits at the heart of the layout; the other two align to it.
  const frontWidth = extent('front', 'width') * scale
  const frontHeight = extent('front', 'height') * scale
  const frontX = MARGIN + (panelWidth - frontWidth) / 2
  const frontY = MARGIN + panelHeight + PANEL_GAP + (panelHeight + frontHeight) / 2

  const topHeight = extent('top', 'height') * scale
  const sideWidth = extent('side', 'width') * scale
  const sidePanelLeft = MARGIN + panelWidth + PANEL_GAP

  const origins: Record<ViewKey, [number, number]> = {
    // Shares the front's horizontal placement: both measure width across.
    top: [frontX, MARGIN + (panelHeight + topHeight) / 2],
    front: [frontX, frontY],
    // Shares the front's baseline: both measure height up.
    side: [sidePanelLeft + (panelWidth - sideWidth) / 2, frontY],
  }

  // With physical paper units the ratio is real: how many millimetres of
  // model one millimetre of paper represents.
  const paperMmPerMetre = scale * MM_PER_UNIT
  const ratio = paperMmPerMetre > 0 ? 1000.0 / paperMmPerMetre : 1.0
  const scaleLabel = ratio >= 10 ? `1:${pyFixed(ratio, 0)}` : `1:${pyFixed(ratio, 1)}`

  const labels: Record<ViewKey, [string, string, string, string]> = {
    front: ['Front elevation', 'A-01', 'W', 'H'],
    side: ['Side elevation', 'A-02', 'D', 'H'],
    top: ['Plan', 'A-03', 'W', 'D'],
  }

  const panels = boxes
    .filter(([view]) => present(drawing[view]))
    .map(([view, box]) => panel(drawing[view] as SheetView, box, scale, ...labels[view],
      theme, origins[view]))
    .join('')

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${pyNumber(PAPER_WIDTH_MM, true)}mm" \
height="${pyNumber(PAPER_HEIGHT_MM, true)}mm" viewBox="0 0 ${SHEET_WIDTH} ${SHEET_HEIGHT}">
  <defs>
    <pattern id="fine" width="16" height="16" patternUnits="userSpaceOnUse">
      <path d="M16 0 L0 0 0 16" fill="none" stroke="${theme.grid}"
            stroke-width="0.6" opacity="0.55"/>
    </pattern>
    <pattern id="coarse" width="96" height="96" patternUnits="userSpaceOnUse">
      <rect width="96" height="96" fill="url(#fine)"/>
      <path d="M96 0 L0 0 0 96" fill="none" stroke="${theme.gridMajor}" stroke-width="0.9"/>
    </pattern>
    <marker id="dim-end" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,1 L8,4 L0,7 z" fill="${theme.dimension}"/>
    </marker>
    <marker id="dim-start" markerWidth="8" markerHeight="8" refX="1" refY="4" orient="auto">
      <path d="M8,1 L0,4 L8,7 z" fill="${theme.dimension}"/>
    </marker>
  </defs>
  <rect width="${SHEET_WIDTH}" height="${SHEET_HEIGHT}" fill="${theme.background}"/>
  <rect width="${SHEET_WIDTH}" height="${SHEET_HEIGHT}" fill="url(#coarse)"/>
  <rect x="${pyNumber(MARGIN / 2, true)}" y="${pyNumber(MARGIN / 2, true)}" \
width="${SHEET_WIDTH - MARGIN}" height="${SHEET_HEIGHT - MARGIN}" fill="none" \
stroke="${theme.frame}" stroke-width="2"/>
  ${panels}
  ${titleBlock(name, stats, spec, scaleLabel, theme)}
  <text x="${MARGIN + 6}" y="${SHEET_HEIGHT - MARGIN - 10}" fill="${theme.text}" \
font-size="11" font-family="Inter,system-ui,sans-serif" opacity="0.8">
    Design output - not a certified manufacturing drawing. No GD&amp;T or material certification.
  </text>
</svg>`
}
