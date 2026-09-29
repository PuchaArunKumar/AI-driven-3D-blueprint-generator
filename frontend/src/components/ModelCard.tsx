import { useState } from 'react'
import { AlertTriangle, BookOpen, ChevronDown, FlaskConical, Scale } from 'lucide-react'
import type { T2VModelInfo } from '../lib/text2voxel'

interface Props {
  info: T2VModelInfo | null
  /** Why meta.json could not be read, if it could not. */
  error?: string | null
  /** Called with a starter prompt when a category chip is clicked. */
  onPickCategory?: (prompt: string) => void
}

const MODEL_CARD_URL =
  'https://github.com/PuchaArunKumar/AI-driven-3D-blueprint-generator/blob/main/docs/MODEL_CARD.md'

/** What each training source is, beyond the numbers meta.json carries. */
const DATA_LABEL: Record<string, { title: string; body: string }> = {
  text2shape: {
    title: 'Text2Shape',
    body: 'ShapeNet chairs and tables with ~75k human-written captions, coloured from their textures.',
  },
  modelnet40: {
    title: 'ModelNet40',
    body: '40 CAD categories. Captions come from templates ("a red car") and colours are synthetic and uniform.',
  },
}

/** Readable names for the metrics k2_train.py writes; unknown keys fall back to the raw key. */
const METRIC_LABEL: Record<string, string> = {
  val_iou: 'VAE reconstruction IoU',
  val_iou_t2s: 'VAE IoU - Text2Shape',
  val_iou_modelnet: 'VAE IoU - ModelNet40',
  val_colour_mae: 'VAE colour error (MAE)',
  t2s_heldout_iou: 'Held-out caption IoU',
  t2s_shuffled_iou: 'Shuffled-caption baseline IoU',
  modelnet_class_accuracy: 'ModelNet class-prompt accuracy',
}

const LIMITATIONS = [
  'Detailed only for chairs and tables - the classes with human captions. Colour and material words ("oak", "glass", "leather") are understood there.',
  'Coarse for the other ModelNet40 categories: it knows the category and a plain colour, not styles or parts.',
  '64³ voxels. On a 1 m object a voxel is about 16 mm, so thin parts (chair legs, lamp stems) may thicken, break or vanish.',
  'It does not read sizes. Dimensions you enter are applied afterwards as a uniform scale.',
  'One object per prompt, in a canonical pose. Scenes, text, logos and exact counts are beyond it.',
  'Design output, not a manufacturing drawing: the mesh and blueprint communicate intent and proportion only.',
]

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null
}

function formatMetric(key: string, value: unknown): string {
  if (value === null || value === undefined) return '-'
  if (typeof value !== 'number') return String(value)
  if (key.includes('accuracy')) return `${(value * 100).toFixed(1)}%`
  return value.toFixed(3)
}

/** Flatten `{vae: {...}, prior: {...}}` into rows, skipping nested breakdowns. */
function metricRows(metrics: Record<string, unknown>): { key: string; group: string; value: unknown }[] {
  const rows: { key: string; group: string; value: unknown }[] = []
  for (const [group, block] of Object.entries(metrics)) {
    const record = asRecord(block)
    if (!record) {
      rows.push({ key: group, group: '', value: block })
      continue
    }
    for (const [key, value] of Object.entries(record)) {
      if (asRecord(value) || Array.isArray(value)) continue
      rows.push({ key, group, value })
    }
  }
  return rows
}

function readable(className: string): string {
  return className.replace(/_/g, ' ')
}

function article(word: string): string {
  return /^[aeiou]/i.test(word) ? 'an' : 'a'
}

/**
 * The model card, read from the deployed meta.json - so the numbers shown are
 * the ones for the files this page will actually run, placeholder or not.
 */
export default function ModelCard({ info, error, onPickCategory }: Props) {
  const [showCategories, setShowCategories] = useState(false)

  if (!info) {
    return (
      <section className="panel p-4 text-sm text-slate-400" aria-label="Model card">
        {error ? (
          <p className="flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-signal-warn shrink-0 mt-0.5" />
            The model card could not be read: {error}
          </p>
        ) : (
          'Reading the model card...'
        )}
      </section>
    )
  }

  const data = asRecord(asRecord(info.training)?.data) ?? {}
  const training = asRecord(info.training) ?? {}
  const rows = metricRows(info.metrics ?? {})

  return (
    <section className="panel" aria-labelledby="model-card-title">
      <div className="panel-head flex-wrap">
        <div>
          <h2 id="model-card-title" className="text-sm font-semibold text-white">
            Model card - {info.name} v{info.version}
          </h2>
          <p className="text-[11px] text-slate-500">
            Read from the deployed meta.json{info.created ? `, built ${info.created}` : ''}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {info.smoke && (
            <span className="chip border-signal-warn/40 text-signal-warn bg-signal-warn/10">
              <FlaskConical className="w-3 h-3" />
              placeholder - untrained
            </span>
          )}
          <span className="chip border-ink-600 text-slate-400 font-mono">
            {info.resolution}³ voxels
          </span>
          <a
            href={MODEL_CARD_URL}
            target="_blank"
            rel="noreferrer noopener"
            className="chip border-ink-600 text-slate-400 hover:text-white"
          >
            <BookOpen className="w-3 h-3" />
            Full model card
          </a>
        </div>
      </div>

      <div className="p-4 grid md:grid-cols-3 gap-6">
        {/* ------------------------------------------------ how + data */}
        <div className="space-y-4 min-w-0">
          <div>
            <p className="stat-key mb-1">Architecture</p>
            <p className="font-mono text-[11px] text-slate-400 leading-relaxed">
              text → MiniLM embedding (384-d) → latent diffusion prior (DDIM, {info.sampleSteps}{' '}
              steps, guidance {info.guidance}) → VAE decoder → {info.resolution}³ occupancy +
              colour → surface mesh
            </p>
          </div>

          <div>
            <p className="stat-key mb-1.5">Training data</p>
            <ul className="space-y-2.5">
              {Object.entries(data).map(([key, raw]) => {
                const entry = asRecord(raw) ?? {}
                const label = DATA_LABEL[key]
                const shapes = typeof entry.shapes === 'number' ? entry.shapes : null
                const source = typeof entry.source === 'string' ? entry.source : null
                return (
                  <li key={key} className="text-[12px] leading-relaxed">
                    <p className="text-slate-200 font-semibold">
                      {label?.title ?? key}
                      {shapes !== null && (
                        <span className="font-mono font-normal text-slate-400">
                          {' '}
                          · {shapes.toLocaleString()} shapes
                        </span>
                      )}
                    </p>
                    {label && <p className="text-slate-500">{label.body}</p>}
                    {typeof entry.licence === 'string' && (
                      <p className="text-slate-400">Licence: {entry.licence}</p>
                    )}
                    {source && (
                      <a
                        href={source}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="text-blueprint-400 hover:underline break-all"
                      >
                        {source}
                      </a>
                    )}
                  </li>
                )
              })}
              <li className="text-[12px] leading-relaxed">
                <p className="text-slate-200 font-semibold">all-MiniLM-L6-v2 text encoder</p>
                <p className="text-slate-500">
                  sentence-transformers model, int8 ONNX export by Xenova. Licence: Apache-2.0.
                </p>
              </li>
            </ul>
          </div>

          {(typeof training.vae_steps === 'number' || typeof training.hardware === 'string') && (
            <p className="text-[11px] text-slate-500 font-mono">
              {typeof training.vae_steps === 'number' && `VAE ${training.vae_steps.toLocaleString()} steps · `}
              {typeof training.prior_steps === 'number' &&
                `prior ${training.prior_steps.toLocaleString()} steps · `}
              {typeof training.hardware === 'string' && training.hardware}
            </p>
          )}
        </div>

        {/* --------------------------------------------------- metrics */}
        <div className="min-w-0">
          <p className="stat-key mb-1.5">Evaluation</p>
          {rows.length === 0 ? (
            <p className="text-[12px] text-slate-500">No metrics recorded in meta.json.</p>
          ) : (
            <dl className="divide-y divide-ink-800">
              {rows.map((row) => (
                <div key={`${row.group}.${row.key}`} className="flex justify-between gap-3 py-1.5">
                  <dt className="text-[12px] text-slate-400">
                    {METRIC_LABEL[row.key] ?? row.key}
                  </dt>
                  <dd className="stat-val !text-[12px] shrink-0">{formatMetric(row.key, row.value)}</dd>
                </div>
              ))}
            </dl>
          )}
          <p className="mt-2 text-[11px] text-slate-500 leading-relaxed">
            IoU is voxel overlap (1 = identical). A held-out caption IoU above the
            shuffled-caption baseline means the text is steering the shape.
            {info.smoke && (
              <strong className="text-signal-warn font-semibold">
                {' '}
                These numbers come from a seconds-long smoke run and mean nothing.
              </strong>
            )}
          </p>
        </div>

        {/* ----------------------------------------------- limitations */}
        <div className="min-w-0">
          <p className="stat-key mb-1.5">Limitations</p>
          <ul className="space-y-1.5">
            {LIMITATIONS.map((item) => (
              <li key={item} className="flex gap-2 text-[12px] text-slate-400 leading-relaxed">
                <span className="w-1 h-1 rounded-full bg-signal-warn mt-2 shrink-0" />
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* ------------------------------------------------ categories */}
      {info.classes.length > 0 && (
        <div className="px-4 pb-4">
          <button
            type="button"
            className="flex items-center gap-1.5 text-[12px] text-slate-400 hover:text-white"
            aria-expanded={showCategories}
            aria-controls="model-card-categories"
            onClick={() => setShowCategories((open) => !open)}
          >
            <ChevronDown
              className={`w-3.5 h-3.5 transition-transform ${showCategories ? '' : '-rotate-90'}`}
            />
            Categories it was trained on ({info.classes.length})
          </button>
          {showCategories && (
            <div id="model-card-categories" className="mt-2 flex flex-wrap gap-1.5">
              {info.classes.map((name) => {
                const label = readable(name)
                return (
                  <button
                    key={name}
                    type="button"
                    className="chip border-ink-700 text-slate-400 hover:text-blueprint-400 hover:border-blueprint-500/50 transition-colors"
                    onClick={() => onPickCategory?.(`${article(label)} ${label}`)}
                    title={`Use "${article(label)} ${label}" as the prompt`}
                  >
                    {label}
                  </button>
                )
              })}
            </div>
          )}
        </div>
      )}

      <div className="px-4 py-3 border-t border-ink-800 text-[11px] text-slate-500 leading-relaxed flex gap-2">
        <Scale className="w-3.5 h-3.5 shrink-0 mt-0.5 text-slate-400" />
        <p>
          <strong className="text-slate-400">Research use only.</strong> The training data is
          licensed for non-commercial research (ShapeNet terms of use; ModelNet for academic
          use), so treat the weights and anything they generate the same way. Attribution:
          Text2Shape - Chen et al., 2018; ShapeNet - Chang et al., 2015; ModelNet - Wu et al.,
          2015; MiniLM - Wang et al., 2020. Outputs are design concepts, not manufacturing
          drawings.
        </p>
      </div>
    </section>
  )
}
