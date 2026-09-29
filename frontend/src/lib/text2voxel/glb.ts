// Mesh arrays -> three.js mesh -> binary glTF.
//
// Vertex colours are stored linear (three.js's working space and glTF's
// COLOR_0 convention), so any glTF viewer shows the model's sRGB colours
// correctly. Kept apart from exporters.ts so the worker bundle only pulls in
// GLTFExporter.

import { BufferAttribute, BufferGeometry, Mesh, MeshStandardMaterial, Scene } from 'three'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import type { MeshData } from './mesher.ts'

export function buildMesh(data: MeshData, name = 'Text2Voxel', userData: Record<string, unknown> = {}): Mesh {
  const geometry = new BufferGeometry()
  geometry.setAttribute('position', new BufferAttribute(data.positions, 3))
  geometry.setAttribute('normal', new BufferAttribute(data.normals, 3))
  geometry.setAttribute('color', new BufferAttribute(data.colors, 3))
  const vertexCount = data.positions.length / 3
  // 16-bit indices when they fit: a third smaller download for typical shapes.
  const index = vertexCount <= 65535 ? Uint16Array.from(data.indices) : data.indices
  geometry.setIndex(new BufferAttribute(index, 1))
  geometry.computeBoundingBox()
  geometry.computeBoundingSphere()
  const material = new MeshStandardMaterial({ vertexColors: true, roughness: 0.85, metalness: 0, name })
  const mesh = new Mesh(geometry, material)
  mesh.name = name
  mesh.userData = userData
  return mesh
}

/** Binary glTF (GLB) of the mesh. `userData` is written to the node's glTF `extras`. */
export async function meshToGlb(data: MeshData, userData: Record<string, unknown> = {}): Promise<ArrayBuffer> {
  const scene = new Scene()
  const mesh = buildMesh(data, 'Text2Voxel', userData)
  scene.add(mesh)
  try {
    const result = await new GLTFExporter().parseAsync(scene, { binary: true })
    if (!(result instanceof ArrayBuffer)) throw new Error('GLTFExporter did not return binary output')
    return result
  } finally {
    mesh.geometry.dispose()
    ;(mesh.material as MeshStandardMaterial).dispose()
  }
}
