import { Check, Circle, Download, Loader2, Lock } from 'lucide-react'
import { api } from '../lib/api'
import type { CapabilityInfo, ExportFormat, Project } from '../lib/types'

interface Props {
  project: Project
  capabilities: CapabilityInfo | null
  busy?: boolean
  onExport: (formats: ExportFormat[]) => void
}

const FORMATS: { id: ExportFormat; label: string; note: string }[] = [
  { id: 'glb', label: 'GLB', note: 'Binary glTF - web, Blender, Unity' },
  { id: 'gltf', label: 'glTF', note: 'JSON glTF with external buffers' },
  { id: 'obj', label: 'OBJ', note: 'Wavefront, with MTL materials' },
  { id: 'stl', label: 'STL', note: 'Triangle soup for 3D printing' },
  { id: 'ply', label: 'PLY', note: 'Keeps per-vertex colour' },
  { id: 'blend', label: 'BLEND', note: 'Native Blender scene' },
  { id: 'step', label: 'STEP', note: 'CAD interchange (faceted B-rep)' },
]

export default function ExportPanel({ project, capabilities, busy, onExport }: Props) {
  const available = new Set(capabilities?.export_formats ?? [])
  const unavailable = capabilities?.unavailable_export_formats ?? {}
  const exported = project.exports ?? {}
  const hasModel = Boolean(project.model_url)

  const ready = FORMATS.filter((format) => available.has(format.id) && !exported[format.id])
    .map((format) => format.id)

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <h3 className="text-sm font-semibold text-white">Generated Assets</h3>
          <p className="text-[11px] text-slate-500">
            Only formats this installation can actually produce are offered
          </p>
        </div>
        {hasModel && ready.length > 0 && (
          <button
            className="btn-primary !py-1 !px-2 !text-xs"
            onClick={() => onExport(ready)}
            disabled={busy}
          >
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Download className="w-3.5 h-3.5" />}
            Export all
          </button>
        )}
      </div>

      <ul className="p-2 divide-y divide-ink-800">
        {FORMATS.map((format) => {
          const isAvailable = available.has(format.id)
          const isExported = Boolean(exported[format.id])
          const reason = unavailable[format.id]

          return (
            <li key={format.id} className="flex items-center gap-3 px-2 py-2">
              <span className="shrink-0">
                {isExported ? (
                  <Check className="w-4 h-4 text-signal-ok" strokeWidth={3} />
                ) : isAvailable ? (
                  <Circle className="w-4 h-4 text-ink-600" />
                ) : (
                  <Lock className="w-4 h-4 text-ink-600" />
                )}
              </span>

              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm font-semibold ${
                    isAvailable ? 'text-slate-200' : 'text-slate-600'
                  }`}
                >
                  {format.label}
                </p>
                <p className="text-[11px] text-slate-500 truncate">
                  {isAvailable ? format.note : reason || 'Not available on this machine.'}
                </p>
              </div>

              {isExported ? (
                <a
                  href={api.downloadUrl(project.id, format.id)}
                  className="btn-subtle !py-1 !px-2 !text-xs shrink-0"
                  download
                >
                  <Download className="w-3.5 h-3.5" />
                  Download
                </a>
              ) : isAvailable ? (
                <button
                  className="btn-ghost !py-1 !px-2 !text-xs shrink-0"
                  onClick={() => onExport([format.id])}
                  disabled={busy || !hasModel}
                  title={hasModel ? undefined : 'Generate a 3D model first'}
                >
                  Export
                </button>
              ) : (
                <span className="chip border-ink-700 text-slate-600 shrink-0">unavailable</span>
              )}
            </li>
          )
        })}
      </ul>

      {/* The honest distinction between a cleaned mesh and a parametric model. */}
      <div className="px-4 py-3 border-t border-ink-800 text-[11px] text-slate-500 leading-relaxed">
        <p className="font-mono text-slate-400 mb-1">
          AI mesh → cleanup → CAD-compatible geometry
        </p>
        <p>
          STEP export writes a <strong className="text-slate-400">faceted B-rep</strong> built
          from the mesh triangles. It opens and measures in CAD software, but it is not a
          parametric model: there is no feature tree, no sketches and no editable parameters.
        </p>
      </div>
    </div>
  )
}
