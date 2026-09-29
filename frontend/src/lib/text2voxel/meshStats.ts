// Geometry statistics for the viewer and the blueprint title block.
//
// Mirrors mesh_statistics() in backend/app/providers/mesh/trimesh_processor.py:
// extents and bounds in metres rounded to 6 places, area always, volume only
// when the mesh is watertight (a signed volume of an open surface means
// nothing). Watertightness is measured, not assumed: every directed edge must
// occur exactly once and be matched by its reverse.

import type { ModelStats } from '../types.ts'
import type { MeshData } from './mesher.ts'
import { pyRound } from './pyformat.ts'

export interface Topology {
  /** Every edge shared by exactly two faces, in opposite directions. */
  watertight: boolean
  /** Directed edges that appear more than once (inconsistent winding or non-manifold). */
  duplicateEdges: number
  /** Directed edges without their reverse (holes). */
  boundaryEdges: number
  /** Euler characteristic V - E + F (2 per closed sphere-like component). */
  euler: number
}

export function meshTopology(mesh: MeshData): Topology {
  const indices = mesh.indices
  const vertexCount = mesh.positions.length / 3
  const keys = new Float64Array(indices.length)
  for (let i = 0; i < indices.length; i += 3) {
    for (let k = 0; k < 3; k += 1) {
      const a = indices[i + k]!
      const b = indices[i + ((k + 1) % 3)]!
      keys[i + k] = a * vertexCount + b
    }
  }
  const sorted = keys.slice().sort()
  let duplicateEdges = 0
  for (let i = 1; i < sorted.length; i += 1) if (sorted[i] === sorted[i - 1]) duplicateEdges += 1

  const has = (key: number) => {
    let lo = 0
    let hi = sorted.length - 1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      const value = sorted[mid]!
      if (value === key) return true
      if (value < key) lo = mid + 1
      else hi = mid - 1
    }
    return false
  }
  let boundaryEdges = 0
  for (let i = 0; i < keys.length; i += 1) {
    const key = keys[i]!
    const a = Math.floor(key / vertexCount)
    const b = key - a * vertexCount
    if (!has(b * vertexCount + a)) boundaryEdges += 1
  }

  const faces = indices.length / 3
  // With every edge paired, E = half the directed edges.
  const edges = boundaryEdges === 0 ? keys.length / 2 : new Set(Array.from(keys, (key) => {
    const a = Math.floor(key / vertexCount)
    const b = key - a * vertexCount
    return Math.min(a, b) * vertexCount + Math.max(a, b)
  })).size
  return {
    watertight: faces > 0 && duplicateEdges === 0 && boundaryEdges === 0,
    duplicateEdges,
    boundaryEdges,
    euler: vertexCount - edges + faces,
  }
}

/** Surface area (m^2) and signed volume (m^3, positive for outward winding). */
export function areaAndVolume(mesh: MeshData): { area: number; signedVolume: number } {
  const p = mesh.positions
  const indices = mesh.indices
  let area = 0
  let volume = 0
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i]! * 3
    const b = indices[i + 1]! * 3
    const c = indices[i + 2]! * 3
    const ux = p[b]! - p[a]!
    const uy = p[b + 1]! - p[a + 1]!
    const uz = p[b + 2]! - p[a + 2]!
    const vx = p[c]! - p[a]!
    const vy = p[c + 1]! - p[a + 1]!
    const vz = p[c + 2]! - p[a + 2]!
    const nx = uy * vz - uz * vy
    const ny = uz * vx - ux * vz
    const nz = ux * vy - uy * vx
    area += 0.5 * Math.hypot(nx, ny, nz)
    // Divergence theorem: sum of a . (b x c) / 6 over the faces.
    volume += (p[a]! * (p[b + 1]! * p[c + 2]! - p[b + 2]! * p[c + 1]!)
      - p[a + 1]! * (p[b]! * p[c + 2]! - p[b + 2]! * p[c]!)
      + p[a + 2]! * (p[b]! * p[c + 1]! - p[b + 1]! * p[c]!)) / 6
  }
  return { area, signedVolume: volume }
}

export function meshBounds(mesh: MeshData): { min: number[]; max: number[] } {
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  const p = mesh.positions
  for (let i = 0; i < p.length; i += 3) {
    for (let k = 0; k < 3; k += 1) {
      const value = p[i + k]!
      if (value < min[k]!) min[k] = value
      if (value > max[k]!) max[k] = value
    }
  }
  if (p.length === 0) return { min: [0, 0, 0], max: [0, 0, 0] }
  return { min, max }
}

export function meshStatistics(
  mesh: MeshData,
  fileSizeBytes: number,
  generationTimeS: number | null,
  topology: Topology = meshTopology(mesh),
): ModelStats {
  const { min, max } = meshBounds(mesh)
  const { area, signedVolume } = areaAndVolume(mesh)
  const faces = mesh.indices.length / 3
  return {
    vertices: mesh.positions.length / 3,
    faces,
    triangles: faces,
    bbox_min: min.map((value) => pyRound(value, 6)),
    bbox_max: max.map((value) => pyRound(value, 6)),
    width: pyRound(max[0]! - min[0]!, 6),
    height: pyRound(max[1]! - min[1]!, 6),
    depth: pyRound(max[2]! - min[2]!, 6),
    volume: topology.watertight ? pyRound(Math.abs(signedVolume), 9) : null,
    surface_area: pyRound(area, 9),
    is_watertight: topology.watertight,
    file_size_bytes: fileSizeBytes,
    file_format: 'glb',
    generation_time_s: generationTimeS,
  }
}
