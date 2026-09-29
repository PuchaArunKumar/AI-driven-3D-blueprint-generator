// Download formats for a generated mesh, via the three.js exporters.
//
// GLB is produced once, in the worker, as part of generation; STL, OBJ and PLY
// are made on demand here. Coordinates are metres throughout. STL carries no
// colour; PLY and OBJ carry per-vertex sRGB colour (OBJ as the widely read
// "v x y z r g b" extension, added here because three's OBJExporter only
// writes colours for point clouds).

import { Mesh, MeshStandardMaterial } from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OBJExporter } from 'three/examples/jsm/exporters/OBJExporter.js'
import { PLYExporter } from 'three/examples/jsm/exporters/PLYExporter.js'
import { STLExporter } from 'three/examples/jsm/exporters/STLExporter.js'
import { buildMesh } from './glb.ts'
import type { MeshData } from './mesher.ts'

export type MeshFormat = 'glb' | 'stl' | 'obj' | 'ply'

export const MIME_TYPES: Record<MeshFormat, string> = {
  glb: 'model/gltf-binary',
  stl: 'model/stl',
  obj: 'model/obj',
  ply: 'application/octet-stream',
}

function linearToSrgb(value: number): number {
  const v = value < 0 ? 0 : value > 1 ? 1 : value
  return v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055
}

/** OBJExporter output with each vertex line's sRGB colour appended, in vertex order. */
function objWithColours(mesh: Mesh, colors: Float32Array): string {
  let vertex = 0
  return new OBJExporter()
    .parse(mesh)
    .split('\n')
    .map((line) => {
      if (!line.startsWith('v ')) return line
      const i = vertex * 3
      vertex += 1
      return `${line} ${[0, 1, 2].map((k) => linearToSrgb(colors[i + k]!).toFixed(4)).join(' ')}`
    })
    .join('\n')
}

function plyBinary(mesh: Mesh): ArrayBuffer {
  // PLYExporter also returns the result synchronously. Its callback goes
  // through requestAnimationFrame, which never fires in a background tab (and
  // does not exist in Node), so it is deliberately not passed.
  const noCallback = undefined as unknown as (result: ArrayBuffer) => void
  const result = new PLYExporter().parse(mesh, noCallback, { binary: true, littleEndian: true })
  if (!(result instanceof ArrayBuffer)) throw new Error('PLYExporter produced no output')
  return result
}

/** Rebuild mesh arrays from a GLB made by meshToGlb (used when the arrays are gone). */
export async function meshDataFromGlb(glb: ArrayBuffer): Promise<MeshData> {
  const gltf = await new GLTFLoader().parseAsync(glb, '')
  const meshes: Mesh[] = []
  gltf.scene.traverse((child) => {
    if ((child as Mesh).isMesh) meshes.push(child as Mesh)
  })
  if (meshes.length === 0) throw new Error('The GLB contains no mesh')
  const geometry = meshes[0]!.geometry
  const colour = geometry.getAttribute('color')
  const index = geometry.getIndex()
  return {
    positions: Float32Array.from(geometry.getAttribute('position').array as ArrayLike<number>),
    normals: Float32Array.from(geometry.getAttribute('normal').array as ArrayLike<number>),
    colors: colour
      ? Float32Array.from(colour.array as ArrayLike<number>)
      : new Float32Array(geometry.getAttribute('position').count * 3).fill(0.5),
    indices: index
      ? Uint32Array.from(index.array as ArrayLike<number>)
      : Uint32Array.from({ length: geometry.getAttribute('position').count }, (_, i) => i),
  }
}

/** Encode `data` as STL (binary), OBJ, or PLY (binary, with colours). */
export async function exportMeshData(data: MeshData, format: Exclude<MeshFormat, 'glb'>): Promise<Blob> {
  const mesh = buildMesh(data)
  try {
    switch (format) {
      case 'stl': {
        const view = new STLExporter().parse(mesh, { binary: true }) as DataView
        const bytes = view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength) as ArrayBuffer
        return new Blob([bytes], { type: MIME_TYPES.stl })
      }
      case 'obj':
        return new Blob([objWithColours(mesh, data.colors)], { type: MIME_TYPES.obj })
      case 'ply':
        return new Blob([plyBinary(mesh)], { type: MIME_TYPES.ply })
      default:
        throw new Error(`Unsupported mesh format: ${String(format)}`)
    }
  } finally {
    mesh.geometry.dispose()
    ;(mesh.material as MeshStandardMaterial).dispose()
  }
}
