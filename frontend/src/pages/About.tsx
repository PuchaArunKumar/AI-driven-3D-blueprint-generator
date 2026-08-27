import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { AlertTriangle, ArrowRight, CheckCircle2, FlaskConical } from 'lucide-react'
import { api } from '../lib/api'
import type { SystemStatus } from '../lib/types'

/**
 * Implementation status for each technology the research names.
 *
 * `implemented` means this repository contains code that runs it - not that
 * the paper mentions it.
 */
const TECHNOLOGY: {
  name: string
  role: string
  implemented: boolean
  note: string
}[] = [
  {
    name: 'Stable Diffusion (diffusers)',
    role: 'Multi-view concept image generation, running locally',
    implemented: true,
    note: 'Loaded through the diffusers AutoPipeline; defaults to sd-turbo for low-VRAM GPUs.',
  },
  {
    name: 'DALL-E (OpenAI)',
    role: 'Hosted alternative for image generation',
    implemented: true,
    note: 'Fully wired; requires OPENAI_API_KEY to become available.',
  },
  {
    name: 'Stability AI API',
    role: 'Hosted Stable Diffusion',
    implemented: true,
    note: 'Fully wired; requires STABILITY_API_KEY.',
  },
  {
    name: 'TripoSR (Stability AI / Tripo AI)',
    role: 'Learned single-image 3D reconstruction - the sharpest local backend',
    implemented: true,
    note: 'Selectable as the triposr 3D provider after running scripts/install_triposr.py. Ships two compatibility shims: scikit-image replaces its CUDA marching-cubes extension, and the checkpoint’s ViT keys are remapped to the current transformers layout.',
  },
  {
    name: 'Shap-E (OpenAI)',
    role: 'Learned single-image 3D reconstruction, running locally',
    implemented: true,
    note: 'Runs through the diffusers ShapEImg2ImgPipeline. Selectable as the shap_e 3D provider; recovers object structure a silhouette hull cannot.',
  },
  {
    name: 'Trimesh',
    role: 'Mesh repair, statistics and format conversion',
    implemented: true,
    note: 'Drives welding, hole filling, smoothing, decimation and GLB/OBJ/STL/PLY export.',
  },
  {
    name: 'scikit-image / SciPy',
    role: 'Silhouette segmentation, marching cubes, contour tracing',
    implemented: true,
    note: 'The reconstruction and blueprint stages are built on these.',
  },
  {
    name: 'rembg (U^2-Net matting)',
    role: 'Separating the product from the background before carving',
    implemented: true,
    note: 'Optional but recommended - reconstruction quality depends on it most. A self-check rejects a missing or corrupt model and falls back to classical segmentation.',
  },
  {
    name: 'Blender',
    role: 'Scene import, repair, materials, native .blend output',
    implemented: true,
    note: 'Driven headlessly via subprocess against the Blender 4.x/5.x operator API.',
  },
  {
    name: 'PyTorch',
    role: 'Tensor runtime and CUDA acceleration for diffusion',
    implemented: true,
    note: 'Optional: the app runs without it, with local image generation disabled.',
  },
  {
    name: 'FreeCAD',
    role: 'Mesh to faceted B-rep, STEP / IGES export',
    implemented: true,
    note: 'Implemented but inactive unless FreeCAD is installed and FREECAD_PATH is set.',
  },
  {
    name: 'SLAT / TRELLIS',
    role: 'Learned image-to-3D reconstruction',
    implemented: false,
    note: 'Not bundled. A provider interface exists so a hosted TRELLIS-compatible endpoint can be plugged in via TRELLIS_ENDPOINT.',
  },
  {
    name: 'NeRF / Gaussian Splatting',
    role: 'Radiance-field reconstruction',
    implemented: false,
    note: 'Not implemented. Named in the research as a future direction; no code in this repository performs it.',
  },
  {
    name: 'BERT (MiniLM sentence encoder)',
    role: 'Prompt understanding - restructures the object phrase, matches vocabularies by meaning',
    implemented: true,
    note: 'A 6-layer distilled BERT refines the rule-based parse: it repairs typos and split words ("a electric wheel cchair" becomes "electric wheelchair") and recognises paraphrases. Optional - set SEMANTIC_PROMPT=false and the rules run alone.',
  },
  {
    name: 'Open3D',
    role: 'Point cloud and voxel utilities',
    implemented: false,
    note: 'Not a dependency. Its role is covered by trimesh, SciPy and scikit-image.',
  },
]

const LIMITATIONS = [
  'A visual hull cannot recover concavities that no silhouette reveals - a cup interior or an enclosed cavity will read as solid.',
  'Multi-view consistency is approximate. The views come from independent diffusion samples sharing one identity clause and seed family, not from a multi-view diffusion model.',
  'Exported STEP geometry is a faceted B-rep, not a parametric solid: no feature tree, no sketches, no editable parameters.',
  'Nothing here is a certified manufacturing drawing. Outputs are design-intent prototypes and need engineering review before fabrication.',
  'Physical dimensions come from your specification, not from the images. The reconstruction recovers shape and proportion, not absolute scale.',
  'Surface colour is projected from the source views, so faces no camera observed are approximated from the nearest-facing view.',
  'Learned reconstruction infers depth from shading, so matte, evenly-lit subjects reconstruct well while dark glossy ones tend to flatten. Regenerating the views in a lighter colour is usually the fastest fix.',
  'Reconstruction quality depends on background separation. Text-to-image models often ignore "plain white background"; without the matting model the classical fallback degrades on busy backdrops.',
]

const FUTURE = [
  'Swap the rule-based prompt parser for a fine-tuned transformer or an LLM-backed structured extractor.',
  'Integrate a learned image-to-3D model (TRELLIS/SLAT) behind the existing provider interface for concave detail.',
  'Add photometric refinement so surface normals and materials come from shading, not just silhouettes.',
  'Parametric reconstruction: fit primitives and constraints to the mesh to produce genuinely editable CAD.',
  'Multi-view diffusion for stronger cross-view identity than a shared prompt and seed can give.',
]

export default function About() {
  const [status, setStatus] = useState<SystemStatus | null>(null)

  useEffect(() => {
    api.systemStatus().then(setStatus).catch(() => undefined)
  }, [])

  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 py-10 space-y-12">
      <header>
        <h1 className="text-3xl font-bold text-white tracking-tight">About this project</h1>
        <p className="mt-3 text-slate-400 leading-relaxed">
          An AI-driven 3D blueprint generator for personalised product design: it turns a
          natural-language description into a design specification, multi-view concept images, a
          3D model, an engineering blueprint and downloadable CAD assets.
        </p>
      </header>

      <section>
        <h2 className="text-lg font-bold text-white mb-3">Research motivation</h2>
        <div className="space-y-3 text-[15px] text-slate-400 leading-relaxed">
          <p>
            Conventional CAD is a high-skill bottleneck. Someone who needs a bespoke assistive
            device - a wheelchair attachment shaped to their chair, a grip sized to their hand -
            usually cannot produce the geometry themselves, and commissioning it is slow and
            expensive. The cost falls hardest on exactly the people whose needs are least
            served by mass-produced products.
          </p>
          <p>
            This project asks how far a generative pipeline can shorten that path: from a
            sentence describing the need, to an inspectable 3D model and an exportable asset a
            maker or engineer can take forward. The target is not to replace CAD, but to remove
            the blank-page problem and produce a concrete artefact to iterate on.
          </p>
          <p>
            The documented pipeline is{' '}
            <span className="font-mono text-blueprint-400">Text → Image → 3D → GLB → BLEND</span>,
            and that is the workflow the application demonstrates end to end.
          </p>
        </div>
      </section>

      <section>
        <h2 className="text-lg font-bold text-white mb-1">Architecture</h2>
        <p className="text-sm text-slate-500 mb-4">
          Every stage sits behind a provider interface, so a component can be replaced without
          touching the orchestrator or the UI.
        </p>
        <pre className="panel p-4 overflow-x-auto font-mono text-[11px] leading-relaxed text-slate-400">
{`React + TypeScript + Three.js  (frontend)
        │  REST + WebSocket job stream
        ▼
FastAPI  (backend)
        │
        ├── prompt_parser      text  → DesignSpec
        ├── ImageGenerator     spec  → multi-view PNGs
        │     ├── diffusers (local)   ├── DALL-E   └── Stability
        ├── ThreeDGenerator    images → GLB
        │     ├── visual_hull (local) └── hosted image-to-3D API
        ├── MeshProcessor      GLB   → cleaned GLB
        │     └── trimesh
        └── CADExporter        GLB   → BLEND / STEP / OBJ / STL / PLY
              ├── Blender          └── FreeCAD
        │
        ▼
SQLite (SQLAlchemy - PostgreSQL-ready)  +  filesystem asset store`}
        </pre>
      </section>

      <section>
        <h2 className="text-lg font-bold text-white mb-1">How the 3D step works</h2>
        <p className="text-sm text-slate-500 mb-4">
          The default local reconstruction is multi-view silhouette carving - a visual hull.
        </p>
        <ol className="space-y-2.5">
          {[
            'Each generated view is segmented into a foreground silhouette using luminance, saturation and edge cues, then cleaned and hole-filled.',
            'Silhouettes are rescaled so the object’s extent along each world axis agrees between the views that observe it - without this the intersection erodes to nothing.',
            'Each silhouette is back-projected through a voxel grid and intersected with the others, carving away everything outside every view.',
            'Marching cubes extracts the surface, and each vertex takes its colour from the view whose camera most directly faces it.',
            'The mesh is centred and scaled to the dimensions in your specification, then written as GLB.',
          ].map((step, index) => (
            <li key={index} className="flex gap-3 text-[15px] text-slate-400 leading-relaxed">
              <span className="font-mono text-xs text-blueprint-400 mt-1 shrink-0">
                {String(index + 1).padStart(2, '0')}
              </span>
              {step}
            </li>
          ))}
        </ol>
        <p className="mt-4 text-[13px] text-slate-500 leading-relaxed border-l-2 border-blueprint-500/40 pl-3">
          This is a classical computer-vision method, not a learned image-to-3D model. It runs
          on CPU with no weights and genuinely consumes the generated views - and it is
          honest about what it cannot do, which is recover geometry no silhouette reveals.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-bold text-white mb-1">Technology status</h2>
        <p className="text-sm text-slate-500 mb-4">
          The research names a broader technology set than this repository implements. A
          component is marked implemented only when code here actually runs it.
        </p>
        <ul className="space-y-2">
          {TECHNOLOGY.map((item) => (
            <li key={item.name} className="panel p-3.5 flex items-start gap-3">
              <span className="mt-0.5 shrink-0">
                {item.implemented ? (
                  <CheckCircle2 className="w-4 h-4 text-signal-ok" />
                ) : (
                  <FlaskConical className="w-4 h-4 text-signal-warn" />
                )}
              </span>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="font-semibold text-sm text-slate-200">{item.name}</p>
                  {!item.implemented && (
                    <span className="chip border-signal-warn/40 text-signal-warn !text-[9px]">
                      Research Component - Experimental / Optional
                    </span>
                  )}
                </div>
                <p className="text-[12px] text-slate-500 mt-0.5">{item.role}</p>
                <p className="text-[12px] text-slate-400 mt-1 leading-relaxed">{item.note}</p>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-signal-warn" />
          Limitations
        </h2>
        <ul className="space-y-2">
          {LIMITATIONS.map((limitation) => (
            <li
              key={limitation}
              className="flex gap-2.5 text-[14px] text-slate-400 leading-relaxed"
            >
              <span className="w-1 h-1 rounded-full bg-signal-warn mt-2 shrink-0" />
              {limitation}
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="text-lg font-bold text-white mb-3">Future work</h2>
        <ul className="space-y-2">
          {FUTURE.map((item) => (
            <li key={item} className="flex gap-2.5 text-[14px] text-slate-400 leading-relaxed">
              <ArrowRight className="w-3.5 h-3.5 text-blueprint-400 mt-1 shrink-0" />
              {item}
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h2 className="text-lg font-bold text-white mb-3">Applications</h2>
        <div className="grid sm:grid-cols-2 gap-2">
          {[
            'Accessibility and assistive technology',
            'Custom automotive and bike design',
            'Furniture and home equipment prototyping',
            'Wearables and consumer products',
            'AR/VR and game asset blockouts',
            'Rapid prototyping for engineers and makers',
          ].map((application) => (
            <div key={application} className="panel px-3.5 py-2.5 text-[13px] text-slate-300">
              {application}
            </div>
          ))}
        </div>
      </section>

      {status && (
        <section className="panel p-4">
          <p className="text-[11px] text-slate-500">
            This installation is running version{' '}
            <span className="font-mono text-slate-300">{status.version}</span> with{' '}
            <span className="font-mono text-slate-300">
              {status.providers.filter((provider) => provider.available).length}
            </span>{' '}
            of {status.providers.length} providers available.{' '}
            <Link to="/settings" className="text-blueprint-400 hover:underline">
              See Settings
            </Link>{' '}
            for the full breakdown.
          </p>
        </section>
      )}
    </div>
  )
}
