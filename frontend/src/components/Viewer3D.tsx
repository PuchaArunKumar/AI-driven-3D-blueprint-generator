import { Component, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import {
  Bounds,
  Center,
  Environment,
  GizmoHelper,
  GizmoViewport,
  Grid,
  Lightformer,
  OrbitControls,
  useGLTF,
} from '@react-three/drei'
import * as THREE from 'three'
import {
  Box,
  Boxes,
  Grid3x3,
  Lightbulb,
  Maximize2,
  Minimize2,
  Move3d,
  RotateCcw,
  Ruler,
} from 'lucide-react'
import type { ModelStats } from '../lib/types'

interface Props {
  url: string | null
  stats?: ModelStats | null
  /** Shown when there is no model yet. */
  placeholder?: React.ReactNode
  className?: string
}

type Mode = 'solid' | 'wireframe' | 'normals'

export type Theme = 'dark' | 'studio' | 'noir' | 'blueprint'

/**
 * Presentation themes for the viewer.
 *
 * `studio`, `noir` and `blueprint` reproduce the conventional CAD/mesh
 * presentation styles: dark lines on white, light lines on black, and white
 * lines over blueprint graph paper. The canvas is transparent so the backdrop
 * can be drawn in CSS, which keeps the graph paper screen-aligned no matter
 * where the camera is.
 */
export const THEMES: Record<Theme, {
  label: string
  backdrop: string
  line: string
  surface: string
  surfaceOpacity: number
  gridCell: string
  gridSection: string
  showGroundGrid: boolean
}> = {
  dark: {
    label: 'Dark',
    backdrop: '#06080f',
    line: '#4cc9f0',
    surface: '#141b2e',
    surfaceOpacity: 0.6,
    gridCell: '#1d2740',
    gridSection: '#2a3a58',
    showGroundGrid: true,
  },
  studio: {
    label: 'Studio white',
    backdrop: '#f2f3f5',
    line: '#12151c',
    surface: '#ffffff',
    surfaceOpacity: 0.72,
    gridCell: '#d7dae0',
    gridSection: '#bcc1ca',
    showGroundGrid: true,
  },
  noir: {
    label: 'Noir',
    backdrop: '#050506',
    line: '#f5f7fa',
    surface: '#0b0c10',
    surfaceOpacity: 0.66,
    gridCell: '#1a1c22',
    gridSection: '#2a2d36',
    showGroundGrid: true,
  },
  blueprint: {
    label: 'Blueprint',
    backdrop: '#0e4b8f',
    line: '#eaf2ff',
    surface: '#0b3f7a',
    surfaceOpacity: 0.5,
    gridCell: '#2f6fb5',
    gridSection: '#4a8ad0',
    showGroundGrid: false,
  },
}

const ENVIRONMENTS = ['studio', 'soft', 'contrast'] as const
type EnvPreset = (typeof ENVIRONMENTS)[number]

/**
 * Environment lighting built from in-scene Lightformers.
 *
 * drei's `preset` shorthand downloads an HDRI from a CDN, which would break
 * the offline workflow this app is built around, so the light rig is
 * generated locally instead.
 */
function StudioEnvironment({ preset }: { preset: EnvPreset }) {
  const rigs: Record<EnvPreset, { key: number; pos: [number, number, number]; scale: [number, number, number]; intensity: number; color: string }[]> = {
    studio: [
      { key: 0, pos: [0, 5, 2], scale: [8, 4, 1], intensity: 3, color: '#ffffff' },
      { key: 1, pos: [-4, 2, 1], scale: [4, 6, 1], intensity: 1.6, color: '#cfe8ff' },
      { key: 2, pos: [4, 1, -2], scale: [4, 6, 1], intensity: 1.2, color: '#ffe9cf' },
    ],
    soft: [
      { key: 0, pos: [0, 6, 0], scale: [12, 12, 1], intensity: 1.8, color: '#ffffff' },
      { key: 1, pos: [0, -3, 3], scale: [8, 4, 1], intensity: 0.8, color: '#b9d4ff' },
    ],
    contrast: [
      { key: 0, pos: [3, 5, 3], scale: [3, 3, 1], intensity: 6, color: '#ffffff' },
      { key: 1, pos: [-5, 0, -3], scale: [5, 5, 1], intensity: 0.8, color: '#4cc9f0' },
    ],
  }

  return (
    <Environment resolution={256} frames={1}>
      {rigs[preset].map((light) => (
        <Lightformer
          key={light.key}
          form="rect"
          position={light.pos}
          scale={light.scale}
          intensity={light.intensity}
          color={light.color}
          target={[0, 0, 0]}
        />
      ))}
    </Environment>
  )
}

/** Marks the wireframe overlays this component attaches, so they can be undone. */
const WIRE_CHILD = "__viewerWireframe"

/**
 * Loads the GLB and renders it in the current display mode.
 *
 * Wireframe mode draws real `WireframeGeometry` line segments rather than
 * flipping `material.wireframe`. The line segments are attached as children of
 * each mesh so they inherit its transform, and a semi-opaque surface is kept
 * underneath with a polygon offset: that surface hides most of the far-side
 * lines, which is what makes a dense mesh read as a solid object instead of a
 * flat tangle.
 */
function Model({
  url,
  mode,
  theme,
  showBBox,
}: {
  url: string
  mode: Mode
  theme: Theme
  showBBox: boolean
}) {
  const { scene } = useGLTF(url)

  // Clone so display-mode edits never mutate the cached original.
  const cloned = useMemo(() => scene.clone(true), [scene])
  const box = useMemo(() => new THREE.Box3().setFromObject(cloned), [cloned])
  const palette = THEMES[theme]

  // Keep the untouched materials so solid mode can be restored exactly.
  const originalMaterials = useMemo(() => {
    const map = new Map<string, THREE.Material | THREE.Material[]>()
    cloned.traverse((child) => {
      const mesh = child as THREE.Mesh
      if (mesh.isMesh) map.set(mesh.uuid, mesh.material)
    })
    return map
  }, [cloned])

  useEffect(() => {
    const created: THREE.Object3D[] = []

    cloned.traverse((child) => {
      const mesh = child as THREE.Mesh
      if (!mesh.isMesh) return

      // Remove any overlay from a previous mode before re-deciding.
      for (const existing of [...mesh.children]) {
        if (existing.userData[WIRE_CHILD]) {
          mesh.remove(existing)
          ;(existing as THREE.LineSegments).geometry?.dispose()
          ;((existing as THREE.LineSegments).material as THREE.Material)?.dispose()
        }
      }

      mesh.castShadow = mode === 'solid'
      mesh.receiveShadow = mode === 'solid'

      if (mode === 'normals') {
        mesh.material = new THREE.MeshNormalMaterial({ flatShading: false })
        return
      }

      if (mode === 'solid') {
        const original = originalMaterials.get(mesh.uuid)
        if (original) mesh.material = original
        return
      }

      // --- wireframe -------------------------------------------------------
      mesh.material = new THREE.MeshBasicMaterial({
        color: new THREE.Color(palette.surface),
        transparent: true,
        opacity: palette.surfaceOpacity,
        // Push the surface back so the lines sit cleanly on top of it.
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
        depthWrite: true,
      })

      const lines = new THREE.LineSegments(
        new THREE.WireframeGeometry(mesh.geometry),
        new THREE.LineBasicMaterial({
          color: new THREE.Color(palette.line),
          transparent: true,
          opacity: 0.9,
        }),
      )
      lines.userData[WIRE_CHILD] = true
      mesh.add(lines)
      created.push(lines)
    })

    return () => {
      for (const line of created) {
        line.parent?.remove(line)
        ;(line as THREE.LineSegments).geometry?.dispose()
        ;((line as THREE.LineSegments).material as THREE.Material)?.dispose()
      }
    }
  }, [cloned, mode, palette, originalMaterials])

  return (
    <group>
      <primitive object={cloned} />
      {showBBox && <box3Helper args={[box, new THREE.Color(palette.line)]} />}
    </group>
  )
}

/** Exposes a camera reset that re-frames the model. */
function CameraRig({ resetToken }: { resetToken: number }) {
  const { camera, controls } = useThree()
  useEffect(() => {
    if (resetToken === 0) return
    camera.position.set(1.6, 1.1, 1.9)
    camera.lookAt(0, 0, 0)
    // @ts-expect-error - drei attaches OrbitControls here at runtime
    controls?.reset?.()
  }, [resetToken, camera, controls])
  return null
}

/** Catches GLB parse/network failures so one bad model cannot blank the page. */
class ModelErrorBoundary extends Component<
  { onError: (message: string) => void; children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('3D model failed to load', error, info)
    this.props.onError(error.message || 'The GLB file could not be parsed.')
  }

  render() {
    return this.state.failed ? null : this.props.children
  }
}

function LoadingFallback() {
  return (
    <mesh>
      <boxGeometry args={[0.6, 0.6, 0.6]} />
      <meshStandardMaterial color="#1d2740" wireframe />
    </mesh>
  )
}

function formatBytes(bytes: number): string {
  if (!bytes) return '-'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

/** Metres -> a readable mm/cm/m string. */
export function formatLength(metres: number): string {
  const mm = metres * 1000
  if (mm < 10) return `${mm.toFixed(1)} mm`
  if (mm < 1000) return `${(mm / 10).toFixed(1)} cm`
  return `${metres.toFixed(3)} m`
}

export default function Viewer3D({ url, stats, placeholder, className = '' }: Props) {
  const [mode, setMode] = useState<Mode>('solid')
  const [showGrid, setShowGrid] = useState(true)
  const [showAxes, setShowAxes] = useState(true)
  const [showBBox, setShowBBox] = useState(false)
  const [autoRotate, setAutoRotate] = useState(false)
  const [intensity, setIntensity] = useState(1.1)
  const [preset, setPreset] = useState<EnvPreset>('studio')
  const [theme, setTheme] = useState<Theme>('dark')
  const [resetToken, setResetToken] = useState(0)
  const [fullscreen, setFullscreen] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setLoadError(null)
  }, [url])

  useEffect(() => {
    const onChange = () => setFullscreen(Boolean(document.fullscreenElement))
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  const toggleFullscreen = async () => {
    if (!containerRef.current) return
    try {
      if (document.fullscreenElement) await document.exitFullscreen()
      else await containerRef.current.requestFullscreen()
    } catch {
      // Browser refused (permissions/iframe) - the inline view still works.
    }
  }

  const toolButton = (active: boolean) =>
    `p-2 rounded-md border transition-colors ${
      active
        ? 'bg-blueprint-500/20 border-blueprint-500/60 text-blueprint-400'
        : 'bg-ink-850/80 border-ink-700 text-slate-400 hover:text-white hover:border-ink-600'
    }`

  return (
    <div
      ref={containerRef}
      className={`relative rounded-xl overflow-hidden border border-ink-700/70 ${className}`}
      style={
        theme === 'blueprint'
          ? {
              // Screen-aligned graph paper, so it stays put as the camera orbits.
              backgroundColor: THEMES.blueprint.backdrop,
              backgroundImage:
                'linear-gradient(rgba(255,255,255,.16) 1px, transparent 1px),' +
                'linear-gradient(90deg, rgba(255,255,255,.16) 1px, transparent 1px),' +
                'linear-gradient(rgba(255,255,255,.07) 1px, transparent 1px),' +
                'linear-gradient(90deg, rgba(255,255,255,.07) 1px, transparent 1px)',
              backgroundSize: '96px 96px, 96px 96px, 16px 16px, 16px 16px',
            }
          : { backgroundColor: THEMES[theme].backdrop }
      }
    >
      {url && !loadError ? (
        <Canvas
          shadows
          dpr={[1, 2]}
          camera={{ position: [1.6, 1.1, 1.9], fov: 45, near: 0.01, far: 100 }}
          gl={{ antialias: true, preserveDrawingBuffer: true, alpha: true }}
          onCreated={({ gl }) => {
            gl.toneMapping = THREE.ACESFilmicToneMapping
          }}
        >
          <hemisphereLight intensity={intensity * 0.4} groundColor="#0a0e1a" />
          <directionalLight
            position={[4, 6, 3]}
            intensity={intensity}
            castShadow
            shadow-mapSize={[1024, 1024]}
          />
          <directionalLight position={[-4, 2, -3]} intensity={intensity * 0.45} />

          <ModelErrorBoundary onError={setLoadError}>
            <Suspense fallback={<LoadingFallback />}>
              <Bounds fit clip observe margin={1.15}>
                <Center>
                  <Model url={url} mode={mode} theme={theme} showBBox={showBBox} />
                </Center>
              </Bounds>
              {mode !== 'normals' && <StudioEnvironment preset={preset} />}
            </Suspense>
          </ModelErrorBoundary>

          {showGrid && THEMES[theme].showGroundGrid && (
            <Grid
              position={[0, -0.001, 0]}
              args={[20, 20]}
              cellSize={0.1}
              cellThickness={0.6}
              cellColor="#1d2740"
              sectionSize={1}
              sectionThickness={1.1}
              sectionColor="#2a3a58"
              fadeDistance={18}
              fadeStrength={1.4}
              infiniteGrid
              followCamera={false}
            />
          )}

          {showAxes && (
            <GizmoHelper alignment="bottom-right" margin={[70, 70]}>
              <GizmoViewport
                axisColors={['#f87171', '#34d399', '#60a5fa']}
                labelColor="#e2e8f0"
              />
            </GizmoHelper>
          )}

          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.08}
            autoRotate={autoRotate}
            autoRotateSpeed={1.2}
            minDistance={0.05}
            maxDistance={40}
          />
          <CameraRig resetToken={resetToken} />
        </Canvas>
      ) : (
        <div className="absolute inset-0 grid-bg flex items-center justify-center p-8 text-center">
          {loadError ? (
            <div className="max-w-sm">
              <p className="text-signal-err font-semibold mb-1">The model could not be displayed.</p>
              <p className="text-sm text-slate-400">{loadError}</p>
            </div>
          ) : (
            placeholder ?? (
              <div className="max-w-sm">
                <Boxes className="w-10 h-10 mx-auto mb-3 text-ink-600" strokeWidth={1.5} />
                <p className="text-slate-400 text-sm">
                  No 3D model yet. Generate the views, then run 3D reconstruction.
                </p>
              </div>
            )
          )}
        </div>
      )}

      {/* ------------------------------------------------------ toolbar */}
      {url && !loadError && (
        <>
          {/* Bounded on phones so it wraps instead of running under the right-hand buttons. */}
          <div className="absolute top-3 left-3 right-[8.5rem] sm:right-auto flex flex-wrap gap-1.5">
            <button
              className={toolButton(mode === 'solid')}
              onClick={() => setMode('solid')}
              title="Solid shading"
              aria-label="Solid shading"
            >
              <Box className="w-4 h-4" />
            </button>
            <button
              className={toolButton(mode === 'wireframe')}
              onClick={() => setMode('wireframe')}
              title="Wireframe"
              aria-label="Wireframe"
            >
              <Grid3x3 className="w-4 h-4" />
            </button>
            <button
              className={toolButton(mode === 'normals')}
              onClick={() => setMode('normals')}
              title="Normals / material preview"
              aria-label="Normals preview"
            >
              <Lightbulb className="w-4 h-4" />
            </button>
            <span className="w-px bg-ink-700 mx-0.5" />
            <button
              className={toolButton(showGrid)}
              onClick={() => setShowGrid((value) => !value)}
              title="Toggle ground grid"
              aria-label="Toggle grid"
            >
              <Grid3x3 className="w-4 h-4 opacity-70" />
            </button>
            <button
              className={toolButton(showAxes)}
              onClick={() => setShowAxes((value) => !value)}
              title="Toggle axis gizmo"
              aria-label="Toggle axes"
            >
              <Move3d className="w-4 h-4" />
            </button>
            <button
              className={toolButton(showBBox)}
              onClick={() => setShowBBox((value) => !value)}
              title="Toggle bounding box"
              aria-label="Toggle bounding box"
            >
              <Ruler className="w-4 h-4" />
            </button>
          </div>

          <div className="absolute top-3 right-3 flex gap-1.5">
            <button
              className={toolButton(autoRotate)}
              onClick={() => setAutoRotate((value) => !value)}
              title="Auto-rotate"
              aria-label="Auto-rotate"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <button
              className={toolButton(false)}
              onClick={() => setResetToken((token) => token + 1)}
              title="Reset camera"
              aria-label="Reset camera"
            >
              <Maximize2 className="w-4 h-4 rotate-45" />
            </button>
            <button
              className={toolButton(fullscreen)}
              onClick={toggleFullscreen}
              title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
              aria-label="Toggle fullscreen"
            >
              {fullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            </button>
          </div>

          {/* lighting + environment */}
          <div className="absolute bottom-3 left-3 right-3 sm:right-auto flex flex-wrap items-center gap-x-3 gap-y-1.5 bg-ink-900/85 backdrop-blur border border-ink-700 rounded-lg px-3 py-2">
            <label className="flex items-center gap-2 text-[11px] text-slate-400">
              <Lightbulb className="w-3.5 h-3.5" />
              <input
                type="range"
                min={0.2}
                max={2.5}
                step={0.05}
                value={intensity}
                onChange={(event) => setIntensity(Number(event.target.value))}
                className="w-20 accent-blueprint-500"
                aria-label="Light intensity"
              />
            </label>
            <select
              value={preset}
              onChange={(event) => setPreset(event.target.value as EnvPreset)}
              className="bg-ink-850 border border-ink-700 rounded px-1.5 py-1 text-[11px] text-slate-300"
              aria-label="Environment preset"
            >
              {ENVIRONMENTS.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <span className="w-px h-4 bg-ink-700" />
            <select
              value={theme}
              onChange={(event) => setTheme(event.target.value as Theme)}
              className="bg-ink-850 border border-ink-700 rounded px-1.5 py-1 text-[11px] text-slate-300"
              aria-label="Presentation theme"
              title="Presentation theme - pair with wireframe for a CAD-style render"
            >
              {(Object.keys(THEMES) as Theme[]).map((name) => (
                <option key={name} value={name}>
                  {THEMES[name].label}
                </option>
              ))}
            </select>
          </div>

          {/* live model statistics */}
          {/* Hidden on phones, where it would cover the lighting bar. */}
          {stats && (
            <div className="hidden sm:block absolute bottom-3 right-3 bg-ink-900/85 backdrop-blur border border-ink-700 rounded-lg px-3 py-2 font-mono text-[11px] text-slate-400 space-y-0.5 max-w-[45%]">
              <div className="flex justify-between gap-4">
                <span>tris</span>
                <span className="text-slate-200">{stats.triangles.toLocaleString()}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span>verts</span>
                <span className="text-slate-200">{stats.vertices.toLocaleString()}</span>
              </div>
              <div className="flex justify-between gap-4">
                <span>size</span>
                <span className="text-slate-200">
                  {formatLength(stats.width)} x {formatLength(stats.height)} x{' '}
                  {formatLength(stats.depth)}
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>file</span>
                <span className="text-slate-200">
                  {formatBytes(stats.file_size_bytes)} {stats.file_format.toUpperCase()}
                </span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
