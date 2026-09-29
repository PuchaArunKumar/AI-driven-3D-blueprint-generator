// Technical line drawings projected from the voxel field.
//
// Same three edge families and the same view frames as the backend's
// blueprint.py, extracted from the occupancy field the mesh came from rather
// than from the mesh itself (a 64^3 marching-cubes surface is too rippled for
// dihedral-angle creases to mean much):
//
// outline  closed silhouette loops: marching squares on the maximum projection
//          of the field along the view direction, at the surface's iso-level.
//          Holes through the object come out as their own loops.
// inline   visible feature lines inside the silhouette, found in the depth map
//          of the first surface met from the camera: depth discontinuities
//          (one part in front of another) and creases (surface normals turning
//          sharply between neighbouring pixels).
// hidden   the same features in the depth map seen from the opposite side -
//          edges of the far side, which the near side occludes from this view.
//          Features that coincide with a visible one are not repeated. Edges
//          buried in the middle of the object (neither nearest the camera nor
//          nearest the opposite side) are not drawn.
//
// Frames mirror VIEW_AXES in backend/app/pipeline/blueprint.py: each view has a
// horizontal, a vertical and a depth axis, depth increasing towards the camera.
// Paths are polylines in metres, after the same placement as the mesh.

import type { BlueprintData, ModelStats, TechnicalViewData } from '../types.ts'
import type { Placement, ShapeField } from './mesher.ts'

type ViewName = 'front' | 'side' | 'top'

/** view -> [horizontal axis, vertical axis, depth axis] (x = 0, y = 1, z = 2). */
export const VIEW_AXES: Record<ViewName, [number, number, number]> = {
  front: [0, 1, 2], // look along -Z from +Z: X right, Y up
  side: [2, 1, 0], // look along -X from +X: Z right, Y up
  top: [0, 2, 1], // look down -Y from +Y: X right, Z up
}

/** A depth step larger than this (in voxels) is a discontinuity... */
const JUMP_VOXELS = 2.0
/** ...provided it is also this much larger than the local slope predicts. */
const JUMP_OVER_SLOPE = 1.5
/** Normals turning more than this between neighbouring pixels form a crease. */
const CREASE_DEGREES = 40
/** Simplification tolerance for every path, in voxels. */
const SIMPLIFY_VOXELS = 0.3
/** Silhouette loops enclosing less than this (voxels^2) are specks, not outline. */
const MIN_LOOP_AREA = 0.5
/** Feature paths shorter than this fraction of the view's diagonal are noise (as blueprint.py). */
const MIN_SEGMENT_FRACTION = 0.012
/** Loops kept per view, largest first. */
const MAX_LOOPS = 64

type Point = [number, number]

// --------------------------------------------------------------------------
// Polyline helpers
// --------------------------------------------------------------------------

function perpendicularDistance(p: Point, a: Point, b: Point): number {
  const dx = b[0] - a[0]
  const dy = b[1] - a[1]
  const length = Math.hypot(dx, dy)
  if (length < 1e-12) return Math.hypot(p[0] - a[0], p[1] - a[1])
  return Math.abs(dy * (p[0] - a[0]) - dx * (p[1] - a[1])) / length
}

/** Ramer-Douglas-Peucker on an open polyline: merges collinear runs. */
export function simplifyPolyline(points: Point[], tolerance: number): Point[] {
  if (points.length <= 2) return points.slice()
  const keep = new Uint8Array(points.length)
  keep[0] = 1
  keep[points.length - 1] = 1
  const stack: [number, number][] = [[0, points.length - 1]]
  while (stack.length) {
    const [first, last] = stack.pop()!
    let index = -1
    let distance = tolerance
    for (let i = first + 1; i < last; i += 1) {
      const d = perpendicularDistance(points[i]!, points[first]!, points[last]!)
      if (d > distance) {
        distance = d
        index = i
      }
    }
    if (index >= 0) {
      keep[index] = 1
      stack.push([first, index], [index, last])
    }
  }
  return points.filter((_, i) => keep[i])
}

/** RDP for a closed loop (no repeated end point): split at the two most distant points. */
function simplifyLoop(points: Point[], tolerance: number): Point[] {
  if (points.length <= 4) return points.slice()
  let far = 0
  let best = -1
  for (let i = 1; i < points.length; i += 1) {
    const d = Math.hypot(points[i]![0] - points[0]![0], points[i]![1] - points[0]![1])
    if (d > best) {
      best = d
      far = i
    }
  }
  const first = simplifyPolyline(points.slice(0, far + 1), tolerance)
  const second = simplifyPolyline([...points.slice(far), points[0]!], tolerance)
  return [...first.slice(0, -1), ...second.slice(0, -1)]
}

function loopArea(points: Point[]): number {
  let area = 0
  for (let i = 0; i < points.length; i += 1) {
    const [x1, y1] = points[i]!
    const [x2, y2] = points[(i + 1) % points.length]!
    area += x1 * y2 - x2 * y1
  }
  return area / 2
}

function pathLength(points: Point[]): number {
  let length = 0
  for (let i = 1; i < points.length; i += 1) {
    length += Math.hypot(points[i]![0] - points[i - 1]![0], points[i]![1] - points[i - 1]![1])
  }
  return length
}

/** Join 2-point segments into longer polylines where endpoints coincide (as blueprint._chain). */
export function chainSegments(segments: [Point, Point][]): Point[][] {
  const key = (p: Point) => `${Math.round(p[0] * 1e4)},${Math.round(p[1] * 1e4)}`
  const adjacency = new Map<string, number[]>()
  segments.forEach((segment, index) => {
    for (const end of segment) {
      const k = key(end)
      const list = adjacency.get(k)
      if (list) list.push(index)
      else adjacency.set(k, [index])
    }
  })
  const used = new Uint8Array(segments.length)
  const chains: Point[][] = []
  for (let start = 0; start < segments.length; start += 1) {
    if (used[start]) continue
    used[start] = 1
    const chain: Point[] = [segments[start]![0], segments[start]![1]]
    // Extend from both ends until nothing connects.
    for (const atEnd of [true, false]) {
      for (;;) {
        const tip = key(atEnd ? chain[chain.length - 1]! : chain[0]!)
        const candidate = (adjacency.get(tip) ?? []).find((index) => !used[index])
        if (candidate === undefined) break
        used[candidate] = 1
        const [a, b] = segments[candidate]!
        const other = key(a) === tip ? b : a
        if (atEnd) chain.push(other)
        else chain.unshift(other)
      }
    }
    chains.push(chain)
  }
  return chains
}

// --------------------------------------------------------------------------
// Field access in a view's frame
// --------------------------------------------------------------------------

interface ViewField {
  r: number
  iso: number
  /** value at (h, v, d) voxel indices, 0 outside the grid. */
  at: (h: number, v: number, d: number) => number
}

function viewField(shape: ShapeField, axes: [number, number, number]): ViewField {
  const r = shape.resolution
  const strides = [r * r, r, 1]
  const [sh, sv, sd] = [strides[axes[0]]!, strides[axes[1]]!, strides[axes[2]]!]
  const field = shape.field
  return {
    r,
    iso: shape.iso,
    at: (h, v, d) =>
      h < 0 || v < 0 || d < 0 || h >= r || v >= r || d >= r ? 0 : field[h * sh + v * sv + d * sd]!,
  }
}

// --------------------------------------------------------------------------
// Outline: marching squares on the maximum projection
// --------------------------------------------------------------------------

function silhouetteLoops(view: ViewField): Point[][] {
  const { r, iso } = view
  const n = r + 2 // padded with empty space so every loop closes
  const projection = new Float32Array(n * n)
  for (let h = 0; h < r; h += 1) {
    for (let v = 0; v < r; v += 1) {
      let maximum = 0
      for (let d = 0; d < r; d += 1) maximum = Math.max(maximum, view.at(h, v, d))
      projection[(h + 1) * n + (v + 1)] = maximum
    }
  }
  const value = (a: number, b: number) => projection[a * n + b]!

  // Crossing points keyed by grid edge: (point index) * 2 + (0 along h, 1 along v).
  const next = new Map<number, number>()
  const position = new Map<number, Point>()
  const crossing = (a: number, b: number, alongV: number): number => {
    const k = (a * n + b) * 2 + alongV
    if (!position.has(k)) {
      const a2 = alongV ? a : a + 1
      const b2 = alongV ? b + 1 : b
      const t = (iso - value(a, b)) / (value(a2, b2) - value(a, b))
      // Padded index -> voxel-index coordinates.
      position.set(k, alongV ? [a - 1, b - 1 + t] : [a - 1 + t, b - 1])
    }
    return k
  }

  for (let a = 0; a < n - 1; a += 1) {
    for (let b = 0; b < n - 1; b += 1) {
      // Corners counter-clockwise (h right, v up) and the edge leaving each.
      const corners: [number, number][] = [[a, b], [a + 1, b], [a + 1, b + 1], [a, b + 1]]
      const inside = corners.map(([p, q]) => value(p, q) >= iso)
      if (inside.every(Boolean) || !inside.some(Boolean)) continue
      // Edge k runs corner k -> corner k+1; a crossing is stored by the edge's lower end.
      const edgeKey = (k: number): number => {
        if (k === 0) return crossing(a, b, 0)
        if (k === 1) return crossing(a + 1, b, 1)
        if (k === 2) return crossing(a, b + 1, 0)
        return crossing(a, b, 1)
      }
      const list: { edge: number; exit: boolean }[] = []
      for (let k = 0; k < 4; k += 1) {
        if (inside[k] !== inside[(k + 1) % 4]) list.push({ edge: edgeKey(k), exit: inside[k]! })
      }
      // Ambiguous saddle: join the inside corners when the bilinear saddle is inside.
      let join = false
      if (list.length === 4) {
        const [w0, w1, w2, w3] = corners.map(([p, q]) => value(p, q))
        join = (w0! * w2! - w1! * w3!) / (w0! + w2! - w1! - w3!) >= iso
      }
      list.forEach((item, index) => {
        if (!item.exit) return
        const count = list.length
        const partner = count === 2 || join ? list[(index + 1) % count]! : list[(index + count - 1) % count]!
        next.set(item.edge, partner.edge)
      })
    }
  }

  const loops: Point[][] = []
  const seen = new Set<number>()
  for (const start of next.keys()) {
    if (seen.has(start)) continue
    const loop: Point[] = []
    let edge: number | undefined = start
    while (edge !== undefined && !seen.has(edge)) {
      seen.add(edge)
      loop.push(position.get(edge)!)
      edge = next.get(edge)
    }
    if (loop.length >= 3) loops.push(loop)
  }
  return loops
}

// --------------------------------------------------------------------------
// Depth maps and feature lines
// --------------------------------------------------------------------------

interface DepthMap {
  /** Depth of the first surface met, in voxel units along the depth axis; NaN = empty. */
  depth: Float64Array
  /** Unit outward normal at that point, in (h, v, d) components. */
  normal: Float64Array
}

function gradientAt(view: ViewField, h: number, v: number, d: number): [number, number, number] {
  return [
    (view.at(h + 1, v, d) - view.at(h - 1, v, d)) / 2,
    (view.at(h, v + 1, d) - view.at(h, v - 1, d)) / 2,
    (view.at(h, v, d + 1) - view.at(h, v, d - 1)) / 2,
  ]
}

/** First surface along each column, from the camera side (+d) or the far side (-d). */
function depthMap(view: ViewField, fromFront: boolean): DepthMap {
  const { r, iso } = view
  const depth = new Float64Array(r * r).fill(Number.NaN)
  const normal = new Float64Array(r * r * 3)
  const step = fromFront ? -1 : 1
  for (let h = 0; h < r; h += 1) {
    for (let v = 0; v < r; v += 1) {
      for (let k = fromFront ? r - 1 : 0; k >= 0 && k < r; k += step) {
        const inside = view.at(h, v, k)
        if (inside < iso) continue
        const outside = view.at(h, v, k - step) // the neighbour nearer this side's camera
        const t = (inside - iso) / (inside - outside)
        const pixel = h * r + v
        depth[pixel] = k - step * t
        const g0 = gradientAt(view, h, v, k)
        const g1 = gradientAt(view, h, v, k - step)
        const g = [0, 1, 2].map((c) => (1 - t) * g0[c]! + t * g1[c]!)
        const length = Math.hypot(g[0]!, g[1]!, g[2]!) || 1
        for (let c = 0; c < 3; c += 1) normal[pixel * 3 + c] = -g[c]! / length
        break
      }
    }
  }
  return { depth, normal }
}

/**
 * Feature flags per pixel pair: bit 0 for (h, v)-(h+1, v), bit 1 for (h, v)-(h, v+1).
 * Pairs touching an empty pixel are silhouette, which the outline already draws.
 */
function featurePairs(map: DepthMap, r: number): Uint8Array {
  const flags = new Uint8Array(r * r)
  const cosCrease = Math.cos((CREASE_DEGREES * Math.PI) / 180)
  const valid = (h: number, v: number) => h >= 0 && v >= 0 && h < r && v < r && !Number.isNaN(map.depth[h * r + v]!)
  const cosine = (p: number, q: number) =>
    map.normal[p * 3]! * map.normal[q * 3]! + map.normal[p * 3 + 1]! * map.normal[q * 3 + 1]! +
    map.normal[p * 3 + 2]! * map.normal[q * 3 + 2]!
  // |d depth / d axis| a smooth surface would show, from its normal.
  const slope = (p: number, axis: number) =>
    Math.min(Math.abs(map.normal[p * 3 + axis]!) / Math.max(Math.abs(map.normal[p * 3 + 2]!), 1e-3), 50)

  for (let h = 0; h < r; h += 1) {
    for (let v = 0; v < r; v += 1) {
      if (!valid(h, v)) continue
      const p = h * r + v
      for (const axis of [0, 1]) {
        const [dh, dv] = axis === 0 ? [1, 0] : [0, 1]
        if (!valid(h + dh, v + dv)) continue
        const q = (h + dh) * r + (v + dv)
        const jump = Math.abs(map.depth[p]! - map.depth[q]!)
        const predicted = (slope(p, axis) + slope(q, axis)) / 2
        let feature = jump > JUMP_VOXELS && jump > JUMP_OVER_SLOPE * predicted
        if (!feature) {
          const turn = cosine(p, q)
          if (turn < cosCrease) {
            // Non-maximum suppression along the pair's direction, so a rounded
            // corner spread over two pixels gives one line, not two.
            const before = valid(h - dh, v - dv) ? cosine((h - dh) * r + (v - dv), p) : 1
            const after = valid(h + 2 * dh, v + 2 * dv) ? cosine(q, (h + 2 * dh) * r + (v + 2 * dv)) : 1
            feature = turn <= before && turn < after
          }
        }
        if (feature) flags[p] |= 1 << axis
      }
    }
  }
  return flags
}

/**
 * Feature pairs -> polylines. Each flagged pair is a crossing at the midpoint
 * between the two pixel centres; within each 2x2 block of pixels the crossings
 * are joined (two: to each other; one or three-plus: to the block centre), so
 * lines follow diagonals smoothly instead of stair-stepping.
 */
function featureLines(flags: Uint8Array, r: number): Point[][] {
  const along = (h: number, v: number, axis: number) =>
    h >= 0 && v >= 0 && h < r && v < r && (flags[h * r + v]! >> axis) & 1
  const segments: [Point, Point][] = []
  for (let h = -1; h < r; h += 1) {
    for (let v = -1; v < r; v += 1) {
      const mids: Point[] = []
      if (along(h, v, 0)) mids.push([h + 0.5, v])
      if (along(h, v + 1, 0)) mids.push([h + 0.5, v + 1])
      if (along(h, v, 1)) mids.push([h, v + 0.5])
      if (along(h + 1, v, 1)) mids.push([h + 1, v + 0.5])
      if (mids.length === 0) continue
      if (mids.length === 2) {
        segments.push([mids[0]!, mids[1]!])
      } else {
        const centre: Point = [h + 0.5, v + 0.5]
        for (const mid of mids) segments.push([mid, centre])
      }
    }
  }
  return chainSegments(segments).map((line) => simplifyPolyline(line, SIMPLIFY_VOXELS))
}

// --------------------------------------------------------------------------
// Assembly
// --------------------------------------------------------------------------

const round6 = (value: number) => Math.round(value * 1e6) / 1e6

/** One view, with paths converted to metres by the mesh's placement. */
export function technicalView(
  shape: ShapeField, placement: Placement, view: ViewName, extents: number[],
): TechnicalViewData {
  const axes = VIEW_AXES[view]
  const field = viewField(shape, axes)
  const r = shape.resolution
  const [h, v] = axes
  const toMetres = (point: Point): number[] => [
    round6((point[0] - placement.origin[h]) * placement.metresPerIndex),
    round6((point[1] - placement.origin[v]) * placement.metresPerIndex),
  ]

  const loops = silhouetteLoops(field)
    .map((loop) => simplifyLoop(loop, SIMPLIFY_VOXELS))
    .map((loop) => ({ loop, area: Math.abs(loopArea(loop)) }))
    .filter(({ loop, area }) => loop.length >= 3 && area >= MIN_LOOP_AREA)
    .sort((a, b) => b.area - a.area)
    .slice(0, MAX_LOOPS)
    // Closed like the backend's contours: the first point repeated at the end.
    .map(({ loop }) => [...loop, loop[0]!].map(toMetres))

  const front = depthMap(field, true)
  const back = depthMap(field, false)
  const visible = featurePairs(front, r)
  const far = featurePairs(back, r)
  // Far-side features where the near side already has one would draw twice.
  for (let i = 0; i < far.length; i += 1) far[i] = far[i]! & ~visible[i]!

  const width = extents[h]!
  const height = extents[v]!
  const minimum = Math.hypot(width, height) * MIN_SEGMENT_FRACTION
  const longEnough = (paths: number[][][]) =>
    paths.filter((path) => pathLength(path as Point[]) >= minimum && path.length >= 2)

  return {
    outline: loops,
    inline: longEnough(featureLines(visible, r).map((line) => line.map(toMetres))),
    hidden: longEnough(featureLines(far, r).map((line) => line.map(toMetres))),
    width: round6(width),
    height: round6(height),
  }
}

/** The three principal views plus the stats, ready for BlueprintView and the sheet. */
export function technicalDrawing(shape: ShapeField, placement: Placement, stats: ModelStats): BlueprintData {
  const extents = [stats.width, stats.height, stats.depth]
  return {
    views: {
      front: technicalView(shape, placement, 'front', extents),
      side: technicalView(shape, placement, 'side', extents),
      top: technicalView(shape, placement, 'top', extents),
    },
    stats,
    spec: null,
    units: 'm',
  }
}
