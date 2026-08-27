import { Suspense, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Canvas, useFrame } from '@react-three/fiber'
import { Icosahedron, OrbitControls, TorusKnot } from '@react-three/drei'
import * as THREE from 'three'
import {
  ArrowRight,
  Boxes,
  Cpu,
  Download,
  FileBox,
  Layers,
  Ruler,
  Sparkles,
  Wand2,
} from 'lucide-react'
import { api } from '../lib/api'
import type { ProjectSummary } from '../lib/types'

/** Decorative hero geometry - a rotating wireframe, not a generated result. */
function HeroObject() {
  const group = useRef<THREE.Group>(null)

  useFrame((state, delta) => {
    if (!group.current) return
    group.current.rotation.y += delta * 0.25
    group.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.2) * 0.12
  })

  return (
    <group ref={group}>
      <TorusKnot args={[1, 0.32, 180, 24]}>
        <meshStandardMaterial
          color="#4cc9f0"
          wireframe
          transparent
          opacity={0.55}
          emissive="#22b8e6"
          emissiveIntensity={0.35}
        />
      </TorusKnot>
      <Icosahedron args={[1.85, 1]}>
        <meshBasicMaterial color="#2d3a58" wireframe transparent opacity={0.3} />
      </Icosahedron>
    </group>
  )
}

const PIPELINE = [
  { icon: Wand2, title: 'Prompt understanding', body: 'Your description becomes an editable design specification: object, dimensions, materials, constraints.' },
  { icon: Layers, title: 'Multi-view images', body: 'Concept views are rendered from a shared identity clause so proportion and colour stay consistent.' },
  { icon: Boxes, title: '3D reconstruction', body: 'Silhouettes from every view are carved into a voxel volume and surfaced with marching cubes.' },
  { icon: Ruler, title: 'Blueprint', body: 'Orthographic projections traced from the real mesh, with measured dimensions.' },
  { icon: FileBox, title: 'CAD conversion', body: 'Blender imports, repairs and saves a native .blend; FreeCAD writes STEP where installed.' },
  { icon: Download, title: 'Export', body: 'GLB, glTF, OBJ, STL and PLY - every format produced from the actual geometry.' },
]

const APPLICATIONS = [
  'Assistive and accessibility devices',
  'Custom automotive and bike design',
  'Furniture and home equipment',
  'Wearables and consumer products',
  'AR/VR and game assets',
  'Rapid prototyping for makers',
]

export default function Home() {
  const [demos, setDemos] = useState<ProjectSummary[]>([])

  useEffect(() => {
    api
      .listProjects(true)
      .then((projects) => setDemos(projects.filter((project) => project.is_demo).slice(0, 4)))
      .catch(() => setDemos([]))
  }, [])

  return (
    <div>
      {/* ------------------------------------------------------------ hero */}
      <section className="relative overflow-hidden border-b border-ink-800">
        <div className="absolute inset-0 grid-bg opacity-70" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-ink-950" />

        <div className="relative mx-auto max-w-[1600px] px-4 sm:px-6 py-16 lg:py-24 grid lg:grid-cols-2 gap-12 items-center">
          <div className="animate-fade-up">
            <span className="chip border-blueprint-500/40 text-blueprint-400 bg-blueprint-500/10 mb-5">
              <Sparkles className="w-3 h-3" />
              Research prototype
            </span>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-white leading-[1.05] tracking-tight">
              Turn Ideas Into
              <span className="block bg-gradient-to-r from-blueprint-400 to-blueprint-600 bg-clip-text text-transparent">
                3D Products
              </span>
            </h1>

            <p className="mt-5 text-lg text-slate-400 max-w-xl leading-relaxed">
              Describe your product. Generate the design. Explore it in 3D. Export your
              blueprint.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/studio" className="btn-primary !px-5 !py-2.5">
                Start Designing
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link to="/gallery" className="btn-ghost !px-5 !py-2.5">
                Explore Demo
              </Link>
            </div>

            <p className="mt-6 font-mono text-[11px] text-slate-600 tracking-wide">
              TEXT → IMAGE → 3D → GLB → BLEND
            </p>
          </div>

          <div className="h-[300px] sm:h-[420px] relative">
            <Canvas camera={{ position: [0, 0, 5.2], fov: 45 }} dpr={[1, 2]}>
              <ambientLight intensity={0.6} />
              <pointLight position={[5, 5, 5]} intensity={60} color="#4cc9f0" />
              <pointLight position={[-5, -3, 2]} intensity={30} color="#8b5cf6" />
              <Suspense fallback={null}>
                <HeroObject />
              </Suspense>
              <OrbitControls
                enableZoom={false}
                enablePan={false}
                autoRotate
                autoRotateSpeed={0.4}
              />
            </Canvas>
            <p className="absolute bottom-0 right-0 text-[10px] text-slate-600 font-mono">
              drag to rotate
            </p>
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------- pipeline */}
      <section className="mx-auto max-w-[1600px] px-4 sm:px-6 py-16">
        <div className="max-w-2xl">
          <h2 className="text-2xl sm:text-3xl font-bold text-white">
            A pipeline that actually runs
          </h2>
          <p className="mt-3 text-slate-400">
            Every stage below is executed by the backend on real data. Nothing is mocked, and
            where a component is unavailable on your machine the interface says so instead of
            pretending.
          </p>
        </div>

        <div className="mt-10 grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {PIPELINE.map((step, index) => (
            <article key={step.title} className="panel p-5 hover:border-ink-600 transition-colors">
              <div className="flex items-start gap-3">
                <span className="p-2 rounded-lg bg-blueprint-500/10 border border-blueprint-500/25 shrink-0">
                  <step.icon className="w-4 h-4 text-blueprint-400" />
                </span>
                <div>
                  <p className="font-mono text-[10px] text-slate-600 mb-0.5">
                    {String(index + 1).padStart(2, '0')}
                  </p>
                  <h3 className="font-semibold text-white text-sm">{step.title}</h3>
                  <p className="mt-1.5 text-[13px] text-slate-400 leading-relaxed">{step.body}</p>
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------ demo */}
      {demos.length > 0 && (
        <section className="border-y border-ink-800 bg-ink-900/40">
          <div className="mx-auto max-w-[1600px] px-4 sm:px-6 py-16">
            <div className="flex items-end justify-between gap-4 mb-8">
              <div>
                <h2 className="text-2xl sm:text-3xl font-bold text-white">Demo projects</h2>
                <p className="mt-2 text-slate-400 text-sm">
                  Real assets generated by this pipeline - open, rotate and download them
                  without configuring any provider.
                </p>
              </div>
              <Link to="/gallery" className="btn-ghost !py-1.5 shrink-0">
                All projects
                <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>

            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {demos.map((project) => (
                <Link
                  key={project.id}
                  to={`/projects/${project.id}`}
                  className="panel overflow-hidden group hover:border-blueprint-500/50 transition-colors"
                >
                  <div className="aspect-video bg-ink-950 relative overflow-hidden">
                    {project.thumbnail_url ? (
                      <img
                        src={project.thumbnail_url}
                        alt={project.name}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                        loading="lazy"
                      />
                    ) : (
                      <div className="w-full h-full grid-bg" />
                    )}
                    <span className="absolute top-2 left-2 chip border-signal-warn/40 text-signal-warn bg-ink-950/80">
                      Demo Asset
                    </span>
                  </div>
                  <div className="p-3">
                    <h3 className="font-semibold text-sm text-white truncate">{project.name}</h3>
                    <p className="mt-0.5 text-[11px] text-slate-500 line-clamp-2">
                      {project.prompt}
                    </p>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* ---------------------------------------------------- applications */}
      <section className="mx-auto max-w-[1600px] px-4 sm:px-6 py-16">
        <div className="grid lg:grid-cols-2 gap-10 items-start">
          <div>
            <h2 className="text-2xl sm:text-3xl font-bold text-white">
              Built around accessible design
            </h2>
            <p className="mt-3 text-slate-400 leading-relaxed">
              The research motivation for this project is reducing the barrier between an idea
              and a manufacturable concept - particularly for personalised assistive equipment,
              where conventional CAD expertise is the bottleneck rather than the design itself.
            </p>
            <div className="mt-6 flex items-center gap-2 text-[13px] text-slate-500">
              <Cpu className="w-4 h-4 text-blueprint-400 shrink-0" />
              Runs locally on your own GPU, or against a hosted provider you configure.
            </div>
          </div>

          <ul className="grid sm:grid-cols-2 gap-2">
            {APPLICATIONS.map((application) => (
              <li
                key={application}
                className="flex items-center gap-2.5 panel px-3.5 py-2.5 text-[13px] text-slate-300"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-blueprint-400 shrink-0" />
                {application}
              </li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  )
}
