// Voxel occupancy -> closed, consistently wound triangle mesh.
//
// Surface extraction is marching cubes, but the case table is not the classic
// 256-entry one: that table resolves ambiguous cube faces inconsistently
// between neighbouring cells and leaves cracks. Here every cell's surface is
// built from per-face segments, and an ambiguous face (two diagonal corners
// inside) is decided by the asymptotic decider - a function of that face's
// four values only - so both cells sharing the face agree. The segments chain
// into loops that are triangulated without adding any edge a neighbour could
// also produce. The result is a closed 2-manifold whenever the field is padded
// with empty space, which it always is here: every edge has exactly two
// triangles, wound in opposite directions (meshStats.ts checks this for real
// rather than assuming it).
//
// Frame (meta.json "frame"): occupancy index [x][y][z], x right, y up, z to the
// viewer; voxel i spans [i, i+1)/R of the unit cube, so its centre - where the
// sample lives - is at (i + 0.5)/R.

export interface VoxelGrid {
  resolution: number
  /** [x][y][z] C-order, probabilities 0..1. */
  occupancy: Float32Array
  /** [c][x][y][z] C-order, sRGB 0..1. */
  rgb: Float32Array
}

/** The occupancy field the surface and the drawing are both extracted from. */
export interface ShapeField {
  resolution: number
  /** Occupancy with dropped floaters zeroed. */
  field: Float32Array
  iso: number
  occupiedVoxels: number
  components: number
  droppedComponents: number
  droppedVoxels: number
  warnings: string[]
}

/** Mesh in continuous voxel-index coordinates (voxel centre k sits at k). */
export interface RawMesh {
  positions: Float32Array
  normals: Float32Array
  /** Linear RGB (glTF COLOR_0 / three.js working space). */
  colors: Float32Array
  indices: Uint32Array
}

/** Mesh in metres, placed per the scale contract. */
export interface MeshData extends RawMesh {}

/**
 * Connected blobs smaller than this many voxels are dropped as floaters
 * (26-connectivity, so parts touching only at a corner still count as joined).
 * The largest component is always kept, however small.
 */
export const MIN_COMPONENT_VOXELS = 8

/**
 * If fewer voxels than this reach the model's threshold the output is
 * effectively empty (the untrained placeholder model does this). The surface
 * is then taken at a lower iso-level that encloses FALLBACK_VOXELS voxels, and
 * a warning says so.
 */
export const MIN_OCCUPIED_VOXELS = 64
export const FALLBACK_VOXELS = 4096

/** Keeps interpolated vertices off the grid points, so no triangle collapses to zero area. */
const T_EPSILON = 1e-3

// --------------------------------------------------------------------------
// Occupancy preparation
// --------------------------------------------------------------------------

/** Choose the iso-level and drop floaters. */
export function prepareField(grid: VoxelGrid, threshold: number): ShapeField {
  const r = grid.resolution
  const total = r * r * r
  const source = grid.occupancy
  const warnings: string[] = []

  let occupied = 0
  let maximum = 0
  for (let i = 0; i < total; i += 1) {
    const value = source[i]!
    if (value >= threshold) occupied += 1
    if (value > maximum) maximum = value
  }

  let iso = threshold
  if (occupied < MIN_OCCUPIED_VOXELS) {
    const sorted = Float32Array.from(source).sort()
    const fallback = sorted[Math.max(0, total - FALLBACK_VOXELS)]!
    if (!(fallback > 0)) {
      throw new Error('The model produced an empty shape (no occupied voxels). Try another seed or prompt.')
    }
    iso = Math.min(fallback, threshold)
    warnings.push(
      `Only ${occupied} voxel${occupied === 1 ? '' : 's'} reached the model's ${threshold} occupancy threshold ` +
      `(peak ${maximum.toFixed(3)}), so the surface was extracted at ${iso.toFixed(3)} ` +
      'instead. This happens with an untrained or placeholder model; the shape is not meaningful.',
    )
  }

  // Label 26-connected components of the inside voxels.
  const labels = new Int32Array(total).fill(-1)
  const sizes: number[] = []
  const queue = new Int32Array(total)
  const rr = r * r
  for (let seed = 0; seed < total; seed += 1) {
    if (labels[seed] !== -1 || source[seed]! < iso) continue
    const label = sizes.length
    let head = 0
    let tail = 0
    queue[tail++] = seed
    labels[seed] = label
    while (head < tail) {
      const index = queue[head++]!
      const x = Math.floor(index / rr)
      const y = Math.floor(index / r) % r
      const z = index % r
      for (let dx = -1; dx <= 1; dx += 1) {
        const nx = x + dx
        if (nx < 0 || nx >= r) continue
        for (let dy = -1; dy <= 1; dy += 1) {
          const ny = y + dy
          if (ny < 0 || ny >= r) continue
          for (let dz = -1; dz <= 1; dz += 1) {
            const nz = z + dz
            if (nz < 0 || nz >= r) continue
            const neighbour = (nx * r + ny) * r + nz
            if (labels[neighbour] !== -1 || source[neighbour]! < iso) continue
            labels[neighbour] = label
            queue[tail++] = neighbour
          }
        }
      }
    }
    sizes.push(tail)
  }
  if (sizes.length === 0) {
    throw new Error('The model produced an empty shape (no occupied voxels). Try another seed or prompt.')
  }

  const largest = sizes.indexOf(Math.max(...sizes))
  const keep = sizes.map((size, label) => label === largest || size >= MIN_COMPONENT_VOXELS)
  const field = Float32Array.from(source)
  let droppedVoxels = 0
  let kept = 0
  for (let i = 0; i < total; i += 1) {
    const label = labels[i]!
    if (label < 0) continue
    if (keep[label]) {
      kept += 1
    } else {
      field[i] = 0
      droppedVoxels += 1
    }
  }
  const droppedComponents = keep.filter((value) => !value).length

  return {
    resolution: r,
    field,
    iso,
    occupiedVoxels: kept,
    components: sizes.length - droppedComponents,
    droppedComponents,
    droppedVoxels,
    warnings,
  }
}

// --------------------------------------------------------------------------
// Cell topology: loops of crossing edges, built once per configuration
// --------------------------------------------------------------------------

// Corner c = dx | dy << 1 | dz << 2.
const CORNER_OFFSET = Array.from({ length: 8 }, (_, c) => [c & 1, (c >> 1) & 1, (c >> 2) & 1])

/** 12 cube edges as [corner, corner | axisBit, axis]. */
const EDGES: [number, number, number][] = []
for (let c = 0; c < 8; c += 1) {
  for (let axis = 0; axis < 3; axis += 1) {
    if (!(c & (1 << axis))) EDGES.push([c, c | (1 << axis), axis])
  }
}
const EDGE_OF = new Map(EDGES.map(([a, b], index) => [a * 8 + b, index]))
const edgeBetween = (a: number, b: number) => EDGE_OF.get(Math.min(a, b) * 8 + Math.max(a, b))!

/**
 * 6 faces, each with its corners in counter-clockwise order seen from outside
 * the cube. Face f = axis * 2 + side (side 1 = the +axis face).
 */
const FACES: number[][] = []
for (let axis = 0; axis < 3; axis += 1) {
  const u = (axis + 1) % 3
  const v = (axis + 2) % 3
  for (let side = 0; side < 2; side += 1) {
    // (u, v) order (0,0) (1,0) (1,1) (0,1) is CCW about +axis, as u x v = axis.
    const square = [[0, 0], [1, 0], [1, 1], [0, 1]]
    if (side === 0) square.reverse()
    FACES.push(square.map(([bu, bv]) => (side << axis) | (bu! << u) | (bv! << v)))
  }
}
const EDGE_FACES: number[][] = EDGES.map(() => [])
FACES.forEach((corners, face) => {
  for (let k = 0; k < 4; k += 1) EDGE_FACES[edgeBetween(corners[k]!, corners[(k + 1) % 4]!)]!.push(face)
})
const shareFace = (a: number, b: number) => EDGE_FACES[a]!.some((face) => EDGE_FACES[b]!.includes(face))

interface Loop {
  /** Cube edge indices, wound so the triangle normals point out of the shape. */
  edges: number[]
  /** Fan apex index into `edges`, or -1 to triangulate around a centroid vertex. */
  apex: number
}

/** Bitmask of the faces whose four corners alternate inside/outside. */
const AMBIGUOUS_FACES = new Uint8Array(256)
for (let config = 0; config < 256; config += 1) {
  FACES.forEach((corners, face) => {
    const inside = corners.map((c) => (config >> c) & 1)
    if (inside[0] === inside[2] && inside[1] === inside[3] && inside[0] !== inside[1]) {
      AMBIGUOUS_FACES[config] |= 1 << face
    }
  })
}

function buildLoops(config: number, joinMask: number): Loop[] {
  const inside = (c: number) => (config >> c) & 1
  const next = new Int8Array(12).fill(-1)

  FACES.forEach((corners, face) => {
    // Crossings in CCW order. Walking CCW, an "exit" leaves the inside region.
    const crossings: { edge: number; exit: boolean }[] = []
    for (let k = 0; k < 4; k += 1) {
      const a = corners[k]!
      const b = corners[(k + 1) % 4]!
      if (inside(a) !== inside(b)) crossings.push({ edge: edgeBetween(a, b), exit: inside(a) === 1 })
    }
    // Each segment runs exit -> entry, which keeps the inside on its left. On
    // an ambiguous face "join" pairs each exit with the next entry (the two
    // inside corners connect), otherwise with the previous one.
    const join = (joinMask >> face) & 1
    crossings.forEach((crossing, index) => {
      if (!crossing.exit) return
      const count = crossings.length
      const partner = count === 2 || join ? crossings[(index + 1) % count]! : crossings[(index + count - 1) % count]!
      next[crossing.edge] = partner.edge
    })
  })

  const loops: Loop[] = []
  const seen = new Uint8Array(12)
  for (let start = 0; start < 12; start += 1) {
    if (next[start] === -1 || seen[start]) continue
    const edges: number[] = []
    let edge = start
    while (!seen[edge]) {
      seen[edge] = 1
      edges.push(edge)
      edge = next[edge]!
    }
    // Chained this way the loop winds clockwise seen from outside the shape;
    // reverse it so front faces are counter-clockwise (glTF convention).
    edges.reverse()

    // A fan diagonal between two vertices on one cube face could coincide
    // with a diagonal the neighbouring cell draws, making a non-manifold edge.
    // Pick an apex whose diagonals all cross the cell; failing that, use a
    // centroid vertex (whose edges are unique to this loop by construction).
    let apex = edges.length === 3 ? 0 : -1
    for (let candidate = 0; apex < 0 && candidate < edges.length; candidate += 1) {
      let ok = true
      for (let k = 2; k < edges.length - 1 && ok; k += 1) {
        if (shareFace(edges[candidate]!, edges[(candidate + k) % edges.length]!)) ok = false
      }
      if (ok) apex = candidate
    }
    loops.push({ edges, apex })
  }
  return loops
}

const LOOP_CACHE = new Map<number, Loop[]>()
function cellLoops(config: number, joinMask: number): Loop[] {
  const key = config | (joinMask << 8)
  let loops = LOOP_CACHE.get(key)
  if (!loops) {
    loops = buildLoops(config, joinMask)
    LOOP_CACHE.set(key, loops)
  }
  return loops
}

// --------------------------------------------------------------------------
// Extraction
// --------------------------------------------------------------------------

function srgbToLinear(value: number): number {
  const v = value < 0 ? 0 : value > 1 ? 1 : value
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)
}

/**
 * Colour at a point in voxel-index coordinates: trilinear over the eight
 * surrounding voxel centres, each weighted by its occupancy too. The decoder's
 * colour is only trained where the shape is, so empty voxels must not bleed
 * their (meaningless) colour into the surface.
 */
function sampleColour(grid: VoxelGrid, px: number, py: number, pz: number, out: number[]): void {
  const r = grid.resolution
  const plane = r * r * r
  const x0 = Math.floor(px)
  const y0 = Math.floor(py)
  const z0 = Math.floor(pz)
  let red = 0
  let green = 0
  let blue = 0
  let weightSum = 0
  let best = -1
  let bestWeight = -1
  for (let corner = 0; corner < 8; corner += 1) {
    const x = Math.min(r - 1, Math.max(0, x0 + (corner & 1)))
    const y = Math.min(r - 1, Math.max(0, y0 + ((corner >> 1) & 1)))
    const z = Math.min(r - 1, Math.max(0, z0 + ((corner >> 2) & 1)))
    const wx = corner & 1 ? px - x0 : 1 - (px - x0)
    const wy = (corner >> 1) & 1 ? py - y0 : 1 - (py - y0)
    const wz = (corner >> 2) & 1 ? pz - z0 : 1 - (pz - z0)
    const index = (x * r + y) * r + z
    const trilinear = Math.max(0, wx * wy * wz)
    const weight = trilinear * grid.occupancy[index]!
    if (trilinear > bestWeight) {
      bestWeight = trilinear
      best = index
    }
    red += weight * grid.rgb[index]!
    green += weight * grid.rgb[plane + index]!
    blue += weight * grid.rgb[2 * plane + index]!
    weightSum += weight
  }
  if (weightSum > 1e-6) {
    out[0] = red / weightSum
    out[1] = green / weightSum
    out[2] = blue / weightSum
  } else {
    out[0] = grid.rgb[best]!
    out[1] = grid.rgb[plane + best]!
    out[2] = grid.rgb[2 * plane + best]!
  }
}

/**
 * Extract the iso-surface of `shape.field`, colouring it from `grid.rgb`.
 * Positions are in voxel-index coordinates; see placeMesh for metres.
 */
export function extractSurface(shape: ShapeField, grid: VoxelGrid): RawMesh {
  const r = shape.resolution
  const iso = shape.iso
  // Pad by one empty sample on every side so the surface always closes.
  const p = r + 2
  const pp = p * p
  const values = new Float32Array(p * pp)
  for (let x = 0; x < r; x += 1) {
    for (let y = 0; y < r; y += 1) {
      values.set(shape.field.subarray((x * r + y) * r, (x * r + y) * r + r), ((x + 1) * p + (y + 1)) * p + 1)
    }
  }

  const positions: number[] = []
  const normals: number[] = []
  const colors: number[] = []
  const indices: number[] = []
  const flatNormal: boolean[] = []
  const vertexOfEdge = new Int32Array(p * pp * 3).fill(-1)
  const colour = [0, 0, 0]
  const strides = [pp, p, 1]

  const gradient = (point: number, axis: number): number => {
    const stride = strides[axis]!
    const coordinate = axis === 0 ? Math.floor(point / pp) : axis === 1 ? Math.floor(point / p) % p : point % p
    const lo = coordinate > 0 ? values[point - stride]! : values[point]!
    const hi = coordinate < p - 1 ? values[point + stride]! : values[point]!
    const span = (coordinate > 0 ? 1 : 0) + (coordinate < p - 1 ? 1 : 0)
    return (hi - lo) / span
  }

  const addVertex = (px: number, py: number, pz: number, nx: number, ny: number, nz: number): number => {
    const index = positions.length / 3
    positions.push(px, py, pz)
    const length = Math.hypot(nx, ny, nz)
    if (length > 1e-9) {
      normals.push(nx / length, ny / length, nz / length)
      flatNormal.push(false)
    } else {
      normals.push(0, 0, 0)
      flatNormal.push(true)
    }
    sampleColour(grid, px, py, pz, colour)
    colors.push(srgbToLinear(colour[0]!), srgbToLinear(colour[1]!), srgbToLinear(colour[2]!))
    return index
  }

  const edgeVertex = (point: number, axis: number): number => {
    const key = point * 3 + axis
    const existing = vertexOfEdge[key]!
    if (existing >= 0) return existing
    const other = point + strides[axis]!
    const a = values[point]!
    const b = values[other]!
    let t = (iso - a) / (b - a)
    t = t < T_EPSILON ? T_EPSILON : t > 1 - T_EPSILON ? 1 - T_EPSILON : t
    // Padded point -> voxel-index coordinates (subtract the pad).
    const coords = [Math.floor(point / pp) - 1, (Math.floor(point / p) % p) - 1, (point % p) - 1]
    coords[axis] += t
    // Occupancy falls outwards, so the outward normal is minus its gradient.
    const g = [0, 1, 2].map((k) => (1 - t) * gradient(point, k) + t * gradient(other, k))
    const vertex = addVertex(coords[0]!, coords[1]!, coords[2]!, -g[0]!, -g[1]!, -g[2]!)
    vertexOfEdge[key] = vertex
    return vertex
  }

  const cornerPoint = CORNER_OFFSET.map(([dx, dy, dz]) => dx! * pp + dy! * p + dz!)
  const cornerValue = new Float64Array(8)
  const loopVertices: number[] = []

  for (let x = 0; x < p - 1; x += 1) {
    for (let y = 0; y < p - 1; y += 1) {
      for (let z = 0; z < p - 1; z += 1) {
        const base = x * pp + y * p + z
        let config = 0
        for (let c = 0; c < 8; c += 1) {
          const value = values[base + cornerPoint[c]!]!
          cornerValue[c] = value
          if (value >= iso) config |= 1 << c
        }
        if (config === 0 || config === 255) continue

        // Asymptotic decider on each ambiguous face: the inside corners are
        // joined when the bilinear saddle value is itself inside.
        let joinMask = 0
        const ambiguous = AMBIGUOUS_FACES[config]!
        if (ambiguous) {
          for (let face = 0; face < 6; face += 1) {
            if (!(ambiguous & (1 << face))) continue
            const [c0, c1, c2, c3] = FACES[face]!
            const w0 = cornerValue[c0!]!
            const w1 = cornerValue[c1!]!
            const w2 = cornerValue[c2!]!
            const w3 = cornerValue[c3!]!
            const saddle = (w0 * w2 - w1 * w3) / (w0 + w2 - w1 - w3)
            if (saddle >= iso) joinMask |= 1 << face
          }
        }

        for (const loop of cellLoops(config, joinMask)) {
          loopVertices.length = 0
          for (const edge of loop.edges) {
            const [corner, , axis] = EDGES[edge]!
            loopVertices.push(edgeVertex(base + cornerPoint[corner]!, axis))
          }
          const count = loopVertices.length
          if (loop.apex >= 0) {
            const apex = loop.apex
            for (let k = 1; k < count - 1; k += 1) {
              indices.push(loopVertices[apex]!, loopVertices[(apex + k) % count]!,
                loopVertices[(apex + k + 1) % count]!)
            }
          } else {
            // Centroid vertex, attributes averaged from the loop.
            let cx = 0
            let cy = 0
            let cz = 0
            let nx = 0
            let ny = 0
            let nz = 0
            for (const vertex of loopVertices) {
              cx += positions[vertex * 3]!
              cy += positions[vertex * 3 + 1]!
              cz += positions[vertex * 3 + 2]!
              nx += normals[vertex * 3]!
              ny += normals[vertex * 3 + 1]!
              nz += normals[vertex * 3 + 2]!
            }
            const centre = addVertex(cx / count, cy / count, cz / count, nx, ny, nz)
            for (let k = 0; k < count; k += 1) {
              indices.push(centre, loopVertices[k]!, loopVertices[(k + 1) % count]!)
            }
          }
        }
      }
    }
  }

  // Rare flat spots in the field have no gradient: use the surrounding faces.
  if (flatNormal.some(Boolean)) {
    for (let i = 0; i < indices.length; i += 3) {
      const [a, b, c] = [indices[i]!, indices[i + 1]!, indices[i + 2]!]
      const ux = positions[b * 3]! - positions[a * 3]!
      const uy = positions[b * 3 + 1]! - positions[a * 3 + 1]!
      const uz = positions[b * 3 + 2]! - positions[a * 3 + 2]!
      const vx = positions[c * 3]! - positions[a * 3]!
      const vy = positions[c * 3 + 1]! - positions[a * 3 + 1]!
      const vz = positions[c * 3 + 2]! - positions[a * 3 + 2]!
      const face = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
      for (const vertex of [a, b, c]) {
        if (!flatNormal[vertex]) continue
        for (let k = 0; k < 3; k += 1) normals[vertex * 3 + k] += face[k]!
      }
    }
    for (let vertex = 0; vertex < flatNormal.length; vertex += 1) {
      if (!flatNormal[vertex]) continue
      const length = Math.hypot(normals[vertex * 3]!, normals[vertex * 3 + 1]!, normals[vertex * 3 + 2]!)
      for (let k = 0; k < 3; k += 1) normals[vertex * 3 + k] = length > 0 ? normals[vertex * 3 + k]! / length : k === 1 ? 1 : 0
    }
  }

  return {
    positions: Float32Array.from(positions),
    normals: Float32Array.from(normals),
    colors: Float32Array.from(colors),
    indices: Uint32Array.from(indices),
  }
}

// --------------------------------------------------------------------------
// Placement: centre on x/z, base at y = 0, uniform scale to metres
// --------------------------------------------------------------------------

export interface TargetDimensionsMm {
  length?: number | null
  width?: number | null
  height?: number | null
}

/** Maps voxel-index coordinates to metres: metres = (index - origin) * metresPerIndex. */
export interface Placement {
  origin: [number, number, number]
  metresPerIndex: number
  /** Which rule set the scale ("height", "footprint" or "default"). */
  rule: 'height' | 'footprint' | 'default'
}

const positive = (value: number | null | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value > 0

/**
 * The scale contract, never distorting the shape: a target height scales the
 * y-extent to it; otherwise a target length/width scales the larger of the
 * x/z extents to the larger of the two; otherwise the largest extent is 1 m.
 */
export function computePlacement(mesh: RawMesh, target: TargetDimensionsMm | null | undefined): Placement {
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  const positions = mesh.positions
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k += 1) {
      const value = positions[i + k]!
      if (value < min[k]!) min[k] = value
      if (value > max[k]!) max[k] = value
    }
  }
  const extent = [0, 1, 2].map((k) => Math.max(max[k]! - min[k]!, 1e-9))
  const origin: [number, number, number] = [(min[0]! + max[0]!) / 2, min[1]!, (min[2]! + max[2]!) / 2]

  let rule: Placement['rule'] = 'default'
  let metresPerIndex = 1 / Math.max(...extent)
  if (positive(target?.height)) {
    rule = 'height'
    metresPerIndex = target.height / 1000 / extent[1]!
  } else if (positive(target?.length) || positive(target?.width)) {
    rule = 'footprint'
    const footprint = Math.max(positive(target?.length) ? target.length : 0, positive(target?.width) ? target.width : 0)
    metresPerIndex = footprint / 1000 / Math.max(extent[0]!, extent[2]!)
  }
  return { origin, metresPerIndex, rule }
}

/** Apply a placement to a copy of the mesh (normals are unchanged by a uniform scale). */
export function placeMesh(mesh: RawMesh, placement: Placement): MeshData {
  const positions = new Float32Array(mesh.positions.length)
  const { origin, metresPerIndex } = placement
  for (let i = 0; i < positions.length; i += 3) {
    positions[i] = (mesh.positions[i]! - origin[0]) * metresPerIndex
    positions[i + 1] = (mesh.positions[i + 1]! - origin[1]) * metresPerIndex
    positions[i + 2] = (mesh.positions[i + 2]! - origin[2]) * metresPerIndex
  }
  return { positions, normals: mesh.normals, colors: mesh.colors, indices: mesh.indices }
}
