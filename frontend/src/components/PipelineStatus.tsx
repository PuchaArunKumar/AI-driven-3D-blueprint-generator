import { AlertCircle, Ban, Check, Circle, Loader2, MinusCircle } from 'lucide-react'
import type { Job, StageStatus } from '../lib/types'

interface Props {
  job: Job | null
  onCancel?: () => void
}

/** All seven stages, so pending ones are visible before a job starts. */
const ALL_STAGES: { key: string; label: string }[] = [
  { key: 'prompt_analysis', label: 'Prompt Analysis' },
  { key: 'design_spec', label: 'Design Specification' },
  { key: 'images', label: 'Multi-view Images' },
  { key: 'reconstruction', label: '3D Reconstruction' },
  { key: 'mesh', label: 'Mesh Processing' },
  { key: 'cad', label: 'CAD Conversion' },
  { key: 'export', label: 'Export' },
]

function StageIcon({ status }: { status: StageStatus }) {
  switch (status) {
    case 'completed':
      return <Check className="w-3.5 h-3.5 text-signal-ok" strokeWidth={3} />
    case 'running':
      return <Loader2 className="w-3.5 h-3.5 text-signal-run animate-spin" />
    case 'failed':
      return <AlertCircle className="w-3.5 h-3.5 text-signal-err" />
    case 'skipped':
      return <MinusCircle className="w-3.5 h-3.5 text-slate-600" />
    default:
      return <Circle className="w-3.5 h-3.5 text-ink-600" />
  }
}

const STATUS_STYLES: Record<Job['status'], string> = {
  queued: 'bg-slate-500/15 text-slate-300 border-slate-500/30',
  running: 'bg-signal-run/15 text-signal-run border-signal-run/30',
  completed: 'bg-signal-ok/15 text-signal-ok border-signal-ok/30',
  failed: 'bg-signal-err/15 text-signal-err border-signal-err/30',
  cancelled: 'bg-slate-500/15 text-slate-400 border-slate-500/30',
}

export default function PipelineStatus({ job, onCancel }: Props) {
  const byKey = new Map((job?.stages ?? []).map((stage) => [stage.key, stage]))
  const running = job?.status === 'running' || job?.status === 'queued'
  const progress = Math.round((job?.progress ?? 0) * 100)

  return (
    <div className="panel">
      <div className="panel-head">
        <h3 className="text-sm font-semibold text-white">Generation Pipeline</h3>
        <div className="flex items-center gap-2">
          {job && (
            <span className={`chip ${STATUS_STYLES[job.status]}`}>
              {job.status === 'running' && <Loader2 className="w-3 h-3 animate-spin" />}
              {job.status}
            </span>
          )}
          {running && onCancel && (
            <button onClick={onCancel} className="btn-ghost !py-1 !px-2 !text-xs">
              <Ban className="w-3.5 h-3.5" />
              Cancel
            </button>
          )}
        </div>
      </div>

      {/* Real progress, driven by the worker - never a synthetic timer. */}
      <div className="px-4 pt-3">
        <div className="h-1.5 rounded-full bg-ink-800 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              job?.status === 'failed'
                ? 'bg-signal-err'
                : job?.status === 'completed'
                  ? 'bg-signal-ok'
                  : 'bg-blueprint-500'
            }`}
            style={{ width: `${job ? Math.max(progress, running ? 3 : 0) : 0}%` }}
            role="progressbar"
            aria-valuenow={progress}
            aria-valuemin={0}
            aria-valuemax={100}
          />
        </div>
        <p className="mt-2 text-xs text-slate-400 min-h-[1rem] truncate">
          {job?.error ?? job?.message ?? 'Idle - submit a prompt to begin.'}
        </p>
      </div>

      <ol className="p-3 pt-2 space-y-0.5">
        {ALL_STAGES.map((definition, index) => {
          const stage = byKey.get(definition.key)
          const status: StageStatus = stage?.status ?? 'pending'
          const active = status === 'running'
          return (
            <li
              key={definition.key}
              className={`flex items-center gap-3 rounded-lg px-2.5 py-1.5 transition-colors ${
                active ? 'bg-signal-run/10' : ''
              }`}
            >
              <span className="font-mono text-[10px] text-slate-600 w-5 shrink-0">
                {String(index + 1).padStart(2, '0')}
              </span>
              <StageIcon status={status} />
              <span
                className={`text-xs flex-1 truncate ${
                  status === 'completed'
                    ? 'text-slate-300'
                    : active
                      ? 'text-white font-medium'
                      : status === 'failed'
                        ? 'text-signal-err'
                        : 'text-slate-500'
                }`}
              >
                {definition.label}
              </span>
              {stage?.detail && (
                <span className="text-[10px] font-mono text-slate-500 truncate max-w-[45%]">
                  {stage.detail}
                </span>
              )}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
