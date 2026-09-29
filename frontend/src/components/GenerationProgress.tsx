import { Check, Circle, CloudDownload, Loader2 } from 'lucide-react'
import type { T2VProgress, T2VStage } from '../lib/text2voxel'

interface Props {
  progress: T2VProgress | null
  busy: boolean
  /** True once the model has been loaded in this tab (it is cached after that). */
  modelReady: boolean
  /** Total size of the model files, when meta.json reported it. */
  totalBytes: number | null
  onPreload?: () => void
  preloadDisabled?: boolean
}

const STAGES: { key: Exclude<T2VStage, 'done'>; label: string }[] = [
  { key: 'loading', label: 'Load model' },
  { key: 'encoding', label: 'Encode prompt' },
  { key: 'sampling', label: 'Sample latent' },
  { key: 'decoding', label: 'Decode voxels' },
  { key: 'meshing', label: 'Mesh + blueprint' },
]

export function formatMB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

/**
 * Staged progress for one in-browser generation.
 *
 * The first run spends most of its time downloading the model, so the byte
 * count is shown explicitly rather than folded into a percentage.
 */
export default function GenerationProgress({
  progress,
  busy,
  modelReady,
  totalBytes,
  onPreload,
  preloadDisabled,
}: Props) {
  const current = progress ? STAGES.findIndex((stage) => stage.key === progress.stage) : -1
  const finished = progress?.stage === 'done'
  const percent = Math.round(Math.min(Math.max(progress?.fraction ?? 0, 0), 1) * 100)
  const size = totalBytes && totalBytes > 0 ? formatMB(totalBytes) : 'about 70 MB'
  const downloading =
    progress?.stage === 'loading' && typeof progress.loadedBytes === 'number' && progress.loadedBytes > 0
  // Announce stage changes only - per-step messages would flood a screen reader.
  const announcement = busy && progress ? (finished ? 'Finishing' : STAGES[current]?.label ?? '') : ''

  return (
    <section className="panel" aria-labelledby="t2v-progress-title">
      <div className="panel-head">
        <h2 id="t2v-progress-title" className="text-sm font-semibold text-white">
          Progress
        </h2>
        {busy && progress && (
          <span className="chip bg-signal-run/15 text-signal-run border-signal-run/30">
            <Loader2 className="w-3 h-3 animate-spin" />
            {percent}%
          </span>
        )}
      </div>

      <div className="p-4 space-y-3">
        <p className="sr-only" aria-live="polite">
          {announcement}
        </p>
        {busy && progress ? (
          <>
            <div
              className="h-1.5 rounded-full bg-ink-800 overflow-hidden"
              role="progressbar"
              aria-label="Generation progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={percent}
            >
              <div
                className="h-full bg-blueprint-500 transition-[width] duration-200"
                style={{ width: `${percent}%` }}
              />
            </div>

            <ol className="flex flex-wrap gap-x-3 gap-y-1.5">
              {STAGES.map((stage, index) => {
                const done = finished || (current >= 0 && index < current)
                const running = !finished && index === current
                return (
                  <li
                    key={stage.key}
                    className={`flex items-center gap-1.5 text-[11px] ${
                      running ? 'text-white' : done ? 'text-slate-400' : 'text-slate-600'
                    }`}
                    aria-current={running ? 'step' : undefined}
                  >
                    {done ? (
                      <Check className="w-3 h-3 text-signal-ok" strokeWidth={3} />
                    ) : running ? (
                      <Loader2 className="w-3 h-3 text-signal-run animate-spin" />
                    ) : (
                      <Circle className="w-3 h-3 text-ink-600" />
                    )}
                    {stage.label}
                  </li>
                )
              })}
            </ol>

            <p className="text-[12px] text-slate-300">{progress.message}</p>
            {downloading && (
              <p className="text-[11px] text-slate-400">
                {/* The engine's message usually carries the byte count already. */}
                {!/MB/.test(progress.message) && (
                  <span className="font-mono">
                    {formatMB(progress.loadedBytes ?? 0)}
                    {progress.totalBytes ? ` / ${formatMB(progress.totalBytes)}` : ''} downloaded ·{' '}
                  </span>
                )}
                First run only - your browser caches the files for next time.
              </p>
            )}
          </>
        ) : modelReady ? (
          <p className="text-[12px] text-slate-400 flex items-center gap-2">
            <Check className="w-3.5 h-3.5 text-signal-ok" strokeWidth={3} />
            Model loaded in this tab. New prompts reuse it.
          </p>
        ) : (
          <div className="space-y-2.5">
            <p className="text-[12px] text-slate-400 leading-relaxed">
              The first run downloads the model files and WebAssembly runtime ({size}) from
              this site. Your browser caches them, so later visits skip the download. Nothing
              you type is uploaded - the model runs on this device.
            </p>
            {onPreload && (
              <button
                type="button"
                className="btn-ghost !py-1.5 !text-xs"
                onClick={onPreload}
                disabled={preloadDisabled}
              >
                <CloudDownload className="w-3.5 h-3.5" />
                Download model now
              </button>
            )}
          </div>
        )}
      </div>
    </section>
  )
}
