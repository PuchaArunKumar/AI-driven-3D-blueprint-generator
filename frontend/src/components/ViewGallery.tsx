import { useRef, useState } from 'react'
import { ImageUp, Loader2, RefreshCw, Trash2, Upload, X } from 'lucide-react'
import { ALL_VIEWS } from '../lib/types'
import type { GeneratedImage, ViewName } from '../lib/types'

interface Props {
  images: GeneratedImage[]
  busy?: boolean
  onRegenerate: (views: ViewName[]) => void
  onDelete: (view: ViewName) => void
  onUpload: (view: ViewName, file: File) => Promise<unknown>
  selected: ViewName | null
  onSelect: (view: ViewName | null) => void
}

const VIEW_LABEL: Record<ViewName, string> = {
  front: 'Front',
  rear: 'Rear',
  left: 'Left',
  right: 'Right',
  top: 'Top',
  bottom: 'Bottom',
  perspective: 'Perspective',
}

export default function ViewGallery({
  images,
  busy,
  onRegenerate,
  onDelete,
  onUpload,
  selected,
  onSelect,
}: Props) {
  const [lightbox, setLightbox] = useState<GeneratedImage | null>(null)
  const [batch, setBatch] = useState<{ done: number; total: number } | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const uploadTarget = useRef<ViewName | null>(null)
  const fileInput = useRef<HTMLInputElement>(null)

  const byView = new Map(images.map((image) => [image.view, image]))
  const present = ALL_VIEWS.filter((view) => byView.has(view))

  const pickFile = (view: ViewName | null) => {
    uploadTarget.current = view
    fileInput.current?.click()
  }

  /**
   * Decide which view each dropped file belongs to.
   *
   * A filename that names a view wins ("car-left.png" -> left), because that is
   * explicit intent. Anything left over fills the still-empty views in the
   * canonical order, so a plain drop of seven files lands front..perspective.
   */
  const assignViews = (files: File[]) => {
    const claimed = new Set<ViewName>()
    const pairs: { view: ViewName; file: File }[] = []
    const leftovers: File[] = []

    for (const file of files) {
      const stem = file.name.toLowerCase()
      const named = ALL_VIEWS.find(
        (view) => !claimed.has(view) && new RegExp(`(^|[^a-z])${view}([^a-z]|$)`).test(stem),
      )
      if (named) {
        claimed.add(named)
        pairs.push({ view: named, file })
      } else {
        leftovers.push(file)
      }
    }

    const free = ALL_VIEWS.filter((view) => !claimed.has(view) && !byView.has(view))
    const spare = ALL_VIEWS.filter((view) => !claimed.has(view) && byView.has(view))
    const slots = [...free, ...spare]
    for (const file of leftovers) {
      const view = slots.shift()
      if (!view) break
      pairs.push({ view, file })
    }
    return { pairs, ignored: Math.max(0, leftovers.length - (pairs.length - claimed.size)) }
  }

  const uploadAll = async (files: File[]) => {
    const { pairs, ignored } = assignViews(files)
    if (pairs.length === 0) {
      setNote('No files could be matched to a view.')
      return
    }
    setNote(null)
    setBatch({ done: 0, total: pairs.length })
    let failed = 0
    // Sequential: each upload returns the whole updated project, so parallel
    // requests would race and the last response would win.
    for (const [index, pair] of pairs.entries()) {
      try {
        await onUpload(pair.view, pair.file)
      } catch {
        failed += 1
      }
      setBatch({ done: index + 1, total: pairs.length })
    }
    setBatch(null)
    const assigned = pairs.map((pair) => VIEW_LABEL[pair.view]).join(', ')
    const problems = [
      failed ? `${failed} failed` : '',
      ignored ? `${ignored} ignored (no free view)` : '',
    ].filter(Boolean)
    setNote(
      `Assigned ${pairs.length - failed} image(s): ${assigned}.` +
        (problems.length ? ` ${problems.join(', ')}.` : ''),
    )
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <h3 className="text-sm font-semibold text-white">Generated Views</h3>
          <p className="text-[11px] text-slate-500">
            {present.length
              ? `${present.length} view${present.length === 1 ? '' : 's'} - click one to use it as the blueprint's isometric reference`
              : 'No views yet'}
          </p>
        </div>
        {present.length > 0 && (
          <div className="flex gap-2">
            <button
              className="btn-ghost !py-1 !px-2 !text-xs"
              onClick={() => pickFile(null)}
              disabled={batch !== null}
              title="Upload several images at once; they fill the empty views in order"
            >
              {batch ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Upload className="w-3.5 h-3.5" />
              )}
              {batch ? `${batch.done}/${batch.total}` : 'Upload images'}
            </button>
            <button
              className="btn-ghost !py-1 !px-2 !text-xs"
              onClick={() => onRegenerate(present)}
              disabled={busy}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${busy ? 'animate-spin' : ''}`} />
              Regenerate all
            </button>
          </div>
        )}
      </div>

      <input
        ref={fileInput}
        type="file"
        multiple
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={(event) => {
          const files = Array.from(event.target.files ?? [])
          const view = uploadTarget.current
          if (files.length === 0) {
            event.target.value = ''
            return
          }
          if (view) {
            void onUpload(view, files[0])
          } else {
            void uploadAll(files)
          }
          event.target.value = ''
        }}
      />

      {present.length === 0 ? (
        <div className="p-8 text-center">
          <ImageUp className="w-8 h-8 mx-auto mb-3 text-ink-600" strokeWidth={1.5} />
          <p className="text-sm text-slate-400 mb-1">
            Generate multi-view concept images, or upload your own references.
          </p>
          <p className="text-[11px] text-slate-500 mb-4">
            Pick several files at once - they fill Front, Rear, Left, Right, Top,
            Bottom and Perspective in order. A filename that names a view
            (&ldquo;chair-left.png&rdquo;) is honoured instead.
          </p>
          <div className="flex flex-wrap gap-2 justify-center">
            <button
              className="btn-primary !py-1.5 !px-3 !text-xs"
              onClick={() => onRegenerate([...ALL_VIEWS])}
              disabled={busy || batch !== null}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${busy ? 'animate-spin' : ''}`} />
              Generate all views
            </button>
            <button
              className="btn-subtle !py-1.5 !px-3 !text-xs"
              onClick={() => pickFile(null)}
              disabled={batch !== null}
            >
              {batch ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Upload className="w-3.5 h-3.5" />
              )}
              {batch ? `Uploading ${batch.done}/${batch.total}` : 'Upload images'}
            </button>
          </div>
          {note && <p className="mt-3 text-[11px] text-slate-400">{note}</p>}
        </div>
      ) : (
        <div className="p-3 grid grid-cols-2 sm:grid-cols-3 gap-2.5">
          {present.map((view) => {
            const image = byView.get(view)!
            const isSelected = selected === view
            return (
              <figure
                key={view}
                className={`group relative rounded-lg overflow-hidden border transition-colors cursor-pointer ${
                  isSelected
                    ? 'border-blueprint-500 ring-1 ring-blueprint-500/50'
                    : 'border-ink-700 hover:border-ink-600'
                }`}
                onClick={() => onSelect(isSelected ? null : view)}
              >
                <img
                  src={image.url}
                  alt={`${VIEW_LABEL[view]} view`}
                  className="w-full aspect-square object-cover bg-ink-950"
                  loading="lazy"
                />

                <figcaption className="absolute bottom-0 inset-x-0 flex items-center justify-between px-2 py-1 bg-ink-950/85 backdrop-blur-sm">
                  <span className="text-[11px] font-medium text-slate-200">
                    {VIEW_LABEL[view]}
                  </span>
                  {image.is_reference && (
                    <span className="chip !px-1.5 !py-0 !text-[9px] border-signal-warn/40 text-signal-warn">
                      reference
                    </span>
                  )}
                </figcaption>

                {/* Per-view actions */}
                <div className="absolute top-1 right-1 flex gap-1 opacity-0 group-hover:opacity-100 focus-within:opacity-100 transition-opacity">
                  <button
                    className="p-1 rounded bg-ink-950/85 text-slate-300 hover:text-white"
                    title={`Regenerate ${VIEW_LABEL[view]}`}
                    aria-label={`Regenerate ${VIEW_LABEL[view]} view`}
                    disabled={busy}
                    onClick={(event) => {
                      event.stopPropagation()
                      onRegenerate([view])
                    }}
                  >
                    <RefreshCw className="w-3 h-3" />
                  </button>
                  <button
                    className="p-1 rounded bg-ink-950/85 text-slate-300 hover:text-white"
                    title={`Upload a reference for ${VIEW_LABEL[view]}`}
                    aria-label={`Upload reference for ${VIEW_LABEL[view]} view`}
                    onClick={(event) => {
                      event.stopPropagation()
                      pickFile(view)
                    }}
                  >
                    <Upload className="w-3 h-3" />
                  </button>
                  <button
                    className="p-1 rounded bg-ink-950/85 text-slate-300 hover:text-signal-err"
                    title={`Remove ${VIEW_LABEL[view]}`}
                    aria-label={`Remove ${VIEW_LABEL[view]} view`}
                    onClick={(event) => {
                      event.stopPropagation()
                      onDelete(view)
                    }}
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>

                <button
                  className="absolute inset-0"
                  aria-label={`Open ${VIEW_LABEL[view]} view full size`}
                  onDoubleClick={(event) => {
                    event.stopPropagation()
                    setLightbox(image)
                  }}
                  tabIndex={-1}
                />
              </figure>
            )
          })}

          {/* Slots for views that have not been generated */}
          {ALL_VIEWS.filter((view) => !byView.has(view)).map((view) => (
            <button
              key={view}
              className="rounded-lg border border-dashed border-ink-700 aspect-square flex flex-col items-center justify-center gap-1 text-slate-600 hover:text-slate-400 hover:border-ink-600 transition-colors"
              onClick={() => onRegenerate([view])}
              disabled={busy}
            >
              <RefreshCw className="w-4 h-4" />
              <span className="text-[11px]">{VIEW_LABEL[view]}</span>
            </button>
          ))}
        </div>
      )}

      {lightbox && (
        <div
          className="fixed inset-0 z-50 bg-ink-950/95 flex items-center justify-center p-6"
          onClick={() => setLightbox(null)}
          role="dialog"
          aria-modal="true"
        >
          <button
            className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white"
            aria-label="Close preview"
            onClick={() => setLightbox(null)}
          >
            <X className="w-6 h-6" />
          </button>
          <figure className="max-w-4xl w-full" onClick={(event) => event.stopPropagation()}>
            <img
              src={lightbox.url}
              alt={`${VIEW_LABEL[lightbox.view]} view, full size`}
              className="w-full rounded-lg"
            />
            <figcaption className="mt-3 flex flex-wrap gap-4 font-mono text-[11px] text-slate-400">
              <span>{VIEW_LABEL[lightbox.view]}</span>
              <span>
                {lightbox.width} x {lightbox.height}
              </span>
              <span>{lightbox.provider}</span>
              {lightbox.seed !== null && <span>seed {lightbox.seed}</span>}
            </figcaption>
          </figure>
        </div>
      )}
    </div>
  )
}
