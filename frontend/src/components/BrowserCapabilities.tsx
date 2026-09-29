import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Check, Globe, Loader2, X } from 'lucide-react'

interface FeatureCheck {
  label: string
  ok: boolean
  detail: string
}

/** Feature checks for the in-browser model, read from this browser - nothing assumed. */
function readChecks(): { checks: FeatureCheck[]; cores: number | null; memoryGb: number | null } {
  const hasWindow = typeof window !== 'undefined'
  const wasm = typeof WebAssembly === 'object' && typeof WebAssembly.instantiate === 'function'
  const worker = typeof Worker !== 'undefined'
  const cache = hasWindow && 'caches' in window
  const isolated = hasWindow && window.crossOriginIsolated === true
  const nav = typeof navigator !== 'undefined' ? navigator : null

  return {
    checks: [
      {
        label: 'WebAssembly',
        ok: wasm,
        detail: wasm ? 'Runs the ONNX models' : 'Required - the model cannot run without it',
      },
      {
        label: 'Web Workers',
        ok: worker,
        detail: worker ? 'Keeps the page responsive while sampling' : 'Required',
      },
      {
        label: 'Cache Storage',
        ok: cache,
        detail: cache
          ? 'Model files are downloaded once, then reused'
          : 'Unavailable (private window or insecure origin) - files re-download each visit',
      },
      {
        label: 'Cross-origin isolation',
        ok: isolated,
        detail: isolated
          ? 'SharedArrayBuffer is available for multi-threaded WASM'
          : 'Not isolated (static hosts like GitHub Pages cannot send COOP/COEP) - WASM runs single-threaded',
      },
    ],
    cores: nav?.hardwareConcurrency ?? null,
    // Chromium-only hint, rounded down by the browser for privacy.
    memoryGb: (nav as (Navigator & { deviceMemory?: number }) | null)?.deviceMemory ?? null,
  }
}

/** "This browser" panel for Settings - the environment Text → 3D actually runs in. */
export default function BrowserCapabilities() {
  const { checks, cores, memoryGb } = readChecks()
  // The engine's own check (it also needs WASM SIMD and BigInt64Array). Loaded
  // lazily so Settings does not pull the model runtime into the main bundle.
  const [runtime, setRuntime] = useState<boolean | null>(null)

  useEffect(() => {
    let cancelled = false
    import('../lib/text2voxel')
      .then((engine) => {
        if (!cancelled) setRuntime(engine.isText2VoxelSupported())
      })
      .catch(() => {
        if (!cancelled) setRuntime(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="text-sm font-semibold text-white flex items-center gap-2">
          <Globe className="w-4 h-4 text-blueprint-400" />
          This browser
        </h2>
        <Link to="/generate" className="text-[11px] text-blueprint-400 hover:underline">
          Text → 3D runs here
        </Link>
      </div>
      <ul className="divide-y divide-ink-800">
        <li className="flex items-start gap-3 px-4 py-2.5">
          <span className="mt-0.5 shrink-0">
            {runtime === null ? (
              <Loader2 className="w-4 h-4 text-slate-500 animate-spin" aria-label="checking" />
            ) : runtime ? (
              <Check className="w-4 h-4 text-signal-ok" strokeWidth={3} aria-label="available" />
            ) : (
              <X className="w-4 h-4 text-signal-err" aria-label="unavailable" />
            )}
          </span>
          <div className="min-w-0">
            <p className="text-sm font-semibold text-slate-200">Text → 3D can run here</p>
            <p className="text-[12px] text-slate-500 leading-relaxed">
              {runtime === false
                ? 'No - it needs WebAssembly with SIMD, module Web Workers and BigInt64Array. A current Chrome, Edge, Firefox or Safari has all three.'
                : 'Needs WebAssembly with SIMD, module Web Workers and BigInt64Array'}
            </p>
          </div>
        </li>
        {checks.map((check) => (
          <li key={check.label} className="flex items-start gap-3 px-4 py-2.5">
            <span className="mt-0.5 shrink-0">
              {check.ok ? (
                <Check className="w-4 h-4 text-signal-ok" strokeWidth={3} aria-label="available" />
              ) : (
                <X className="w-4 h-4 text-slate-600" aria-label="unavailable" />
              )}
            </span>
            <div className="min-w-0">
              <p className={`text-sm font-semibold ${check.ok ? 'text-slate-200' : 'text-slate-500'}`}>
                {check.label}
              </p>
              <p className="text-[12px] text-slate-500 leading-relaxed">{check.detail}</p>
            </div>
          </li>
        ))}
      </ul>
      <div className="px-4 py-3 border-t border-ink-800 grid grid-cols-2 gap-4">
        <div>
          <p className="stat-key">Logical cores</p>
          <p className="stat-val">{cores ?? 'unknown'}</p>
        </div>
        <div>
          <p className="stat-key">Device memory</p>
          <p className="stat-val">{memoryGb ? `≥ ${memoryGb} GB` : 'not reported'}</p>
        </div>
      </div>
    </section>
  )
}
