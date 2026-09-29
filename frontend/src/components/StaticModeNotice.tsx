import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, BookOpen, Boxes, Globe, Server } from 'lucide-react'
import { README_URL } from '../lib/api'

interface Props {
  /** Page heading, e.g. "Design Studio". */
  title: string
  /** One sentence on what this page does when a backend is running. */
  purpose: string
  /** Extra panels that do work without a backend, shown last. */
  children?: ReactNode
}

/** What the Python backend provides - and so what the hosted demo cannot do. */
const BACKEND_ONLY = [
  'Multi-view concept images (local Stable Diffusion, DALL-E or Stability AI)',
  '3D reconstruction from those views (visual hull, TripoSR, Shap-E)',
  'Mesh repair and decimation, Blender .blend and FreeCAD STEP conversion',
  'Saving projects, editing specifications and exporting new formats',
]

const WORKS_HERE = [
  'Text → 3D: the Text2Voxel-64 model runs in your browser, prompt to mesh to blueprint',
  'The demo projects: 3D viewer, blueprint views and their pre-generated downloads',
]

/**
 * Shown in place of a backend-driven page on the hosted (GitHub Pages) build.
 *
 * The page is not broken there - it has nothing to talk to - so it says that
 * plainly and points at what does work, rather than surfacing network errors.
 */
export default function StaticModeNotice({ title, purpose, children }: Props) {
  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 py-10 space-y-6">
      <header>
        <span className="chip border-blueprint-500/40 text-blueprint-400 bg-blueprint-500/10 mb-3">
          <Globe className="w-3 h-3" />
          Hosted demo - no backend
        </span>
        <h1 className="text-2xl font-bold text-white tracking-tight">{title}</h1>
        <p className="mt-2 text-sm text-slate-400 leading-relaxed">
          {purpose} That needs the Python (FastAPI) backend, and this copy of the app is a
          static site with no server behind it. Nothing here is simulated to hide that.
        </p>
      </header>

      <div className="grid md:grid-cols-2 gap-4">
        <section className="panel p-4">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2 mb-3">
            <Server className="w-4 h-4 text-signal-warn" />
            Needs the local backend
          </h2>
          <ul className="space-y-2">
            {BACKEND_ONLY.map((item) => (
              <li key={item} className="flex gap-2.5 text-[13px] text-slate-400 leading-relaxed">
                <span className="w-1 h-1 rounded-full bg-signal-warn mt-2 shrink-0" />
                {item}
              </li>
            ))}
          </ul>
        </section>

        <section className="panel p-4">
          <h2 className="text-sm font-semibold text-white flex items-center gap-2 mb-3">
            <Boxes className="w-4 h-4 text-signal-ok" />
            Works on this site
          </h2>
          <ul className="space-y-2">
            {WORKS_HERE.map((item) => (
              <li key={item} className="flex gap-2.5 text-[13px] text-slate-400 leading-relaxed">
                <span className="w-1 h-1 rounded-full bg-signal-ok mt-2 shrink-0" />
                {item}
              </li>
            ))}
          </ul>
        </section>
      </div>

      <div className="flex flex-wrap gap-3">
        <Link to="/generate" className="btn-primary">
          Open Text → 3D
          <ArrowRight className="w-4 h-4" />
        </Link>
        <Link to="/gallery" className="btn-ghost">
          Browse the demos
        </Link>
        <a href={README_URL} target="_blank" rel="noreferrer noopener" className="btn-ghost">
          <BookOpen className="w-4 h-4" />
          Run it locally
        </a>
      </div>

      <section className="panel">
        <div className="panel-head">
          <h2 className="text-sm font-semibold text-white">Running the full app</h2>
          <span className="text-[11px] text-slate-500">Python 3.11+, Node 18+</span>
        </div>
        <pre className="p-4 overflow-x-auto font-mono text-[11px] leading-relaxed text-slate-400">
{`git clone https://github.com/PuchaArunKumar/AI-driven-3D-blueprint-generator.git
cd AI-driven-3D-blueprint-generator
pip install -r backend/requirements.txt

# terminal 1 - backend
cd backend && uvicorn app.main:app --reload --port 8000

# terminal 2 - frontend
cd frontend && npm install && npm run dev`}
        </pre>
        <p className="px-4 pb-4 text-[11px] text-slate-500">
          The README covers PyTorch, GPU setup, providers, Blender and FreeCAD.
        </p>
      </section>

      {children}
    </div>
  )
}
