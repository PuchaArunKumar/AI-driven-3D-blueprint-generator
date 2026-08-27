import { useEffect, useState } from 'react'
import { RotateCcw, Save } from 'lucide-react'
import type { DesignSpec } from '../lib/types'

interface Props {
  spec: DesignSpec | null
  onSave: (spec: DesignSpec) => Promise<void> | void
  onReanalyse?: () => Promise<void> | void
  disabled?: boolean
}

/** Comma-separated text <-> string[] for the list-valued spec fields. */
function ListField({
  label,
  value,
  onChange,
  placeholder,
  disabled,
}: {
  label: string
  value: string[]
  onChange: (next: string[]) => void
  placeholder?: string
  disabled?: boolean
}) {
  const [text, setText] = useState(value.join(', '))

  useEffect(() => {
    setText(value.join(', '))
  }, [value])

  return (
    <div>
      <label className="label">{label}</label>
      <input
        className="field"
        value={text}
        placeholder={placeholder}
        disabled={disabled}
        onChange={(event) => setText(event.target.value)}
        onBlur={() =>
          onChange(
            text
              .split(',')
              .map((item) => item.trim())
              .filter(Boolean),
          )
        }
      />
    </div>
  )
}

function NumberField({
  label,
  value,
  onChange,
  suffix,
  disabled,
  step = 1,
}: {
  label: string
  value: number | null
  onChange: (next: number | null) => void
  suffix: string
  disabled?: boolean
  step?: number
}) {
  return (
    <div>
      <label className="label">
        {label} <span className="text-slate-600 normal-case">({suffix})</span>
      </label>
      <input
        type="number"
        min={0}
        step={step}
        className="field font-mono"
        value={value ?? ''}
        disabled={disabled}
        placeholder="-"
        onChange={(event) => {
          const raw = event.target.value
          onChange(raw === '' ? null : Math.max(0, Number(raw)) || null)
        }}
      />
    </div>
  )
}

export default function SpecEditor({ spec, onSave, onReanalyse, disabled }: Props) {
  const [draft, setDraft] = useState<DesignSpec | null>(spec)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    setDraft(spec)
    setDirty(false)
  }, [spec])

  if (!draft) {
    return (
      <div className="panel p-5 text-sm text-slate-400">
        The interpreted specification appears here once you submit a prompt.
      </div>
    )
  }

  const patch = (changes: Partial<DesignSpec>) => {
    setDraft({ ...draft, ...changes })
    setDirty(true)
  }

  const patchDimensions = (changes: Partial<DesignSpec['dimensions']>) => {
    setDraft({ ...draft, dimensions: { ...draft.dimensions, ...changes } })
    setDirty(true)
  }

  const save = async () => {
    setSaving(true)
    try {
      await onSave(draft)
      setDirty(false)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <h3 className="text-sm font-semibold text-white">Design Specification</h3>
          <p className="text-[11px] text-slate-500">
            Interpreted from your prompt - edit before generating
          </p>
        </div>
        <div className="flex gap-2">
          {onReanalyse && (
            <button
              className="btn-ghost !py-1 !px-2 !text-xs"
              onClick={() => void onReanalyse()}
              disabled={disabled}
              title="Re-derive from the prompt, discarding edits"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Re-analyse
            </button>
          )}
          <button
            className="btn-primary !py-1 !px-2 !text-xs"
            onClick={() => void save()}
            disabled={!dirty || saving || disabled}
          >
            <Save className="w-3.5 h-3.5" />
            {saving ? 'Saving' : dirty ? 'Save' : 'Saved'}
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4 max-h-[46vh] overflow-y-auto">
        <div>
          <label className="label">Object</label>
          <input
            className="field"
            value={draft.object}
            disabled={disabled}
            onChange={(event) => patch({ object: event.target.value })}
          />
        </div>

        <div>
          <label className="label">Purpose / intended use</label>
          <textarea
            className="field resize-none"
            rows={2}
            value={draft.purpose}
            disabled={disabled}
            placeholder="What is it for?"
            onChange={(event) => patch({ purpose: event.target.value, intended_use: event.target.value })}
          />
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="label">Style</label>
            <input
              className="field"
              value={draft.style}
              disabled={disabled}
              placeholder="e.g. minimalist"
              onChange={(event) => patch({ style: event.target.value })}
            />
          </div>
          <div>
            <label className="label">Colour</label>
            <input
              className="field"
              value={draft.color}
              disabled={disabled}
              placeholder="e.g. matte black"
              onChange={(event) => patch({ color: event.target.value })}
            />
          </div>
        </div>

        <div>
          <p className="label !mb-2">Dimensions</p>
          <div className="grid grid-cols-2 gap-3">
            <NumberField
              label="Length / depth" suffix="mm" disabled={disabled}
              value={draft.dimensions.length_mm}
              onChange={(value) => patchDimensions({ length_mm: value })}
            />
            <NumberField
              label="Width" suffix="mm" disabled={disabled}
              value={draft.dimensions.width_mm}
              onChange={(value) => patchDimensions({ width_mm: value })}
            />
            <NumberField
              label="Height" suffix="mm" disabled={disabled}
              value={draft.dimensions.height_mm}
              onChange={(value) => patchDimensions({ height_mm: value })}
            />
            <NumberField
              label="Weight" suffix="g" disabled={disabled}
              value={draft.dimensions.weight_g}
              onChange={(value) => patchDimensions({ weight_g: value })}
            />
          </div>
          <p className="mt-1.5 text-[10px] text-slate-500">
            Dimensions scale the exported model. Leave blank to normalise to a 1 m bounding box.
          </p>
        </div>

        <ListField
          label="Materials" value={draft.materials} disabled={disabled}
          placeholder="aluminium, ABS plastic"
          onChange={(materials) => patch({ materials })}
        />
        <ListField
          label="Constraints" value={draft.constraints} disabled={disabled}
          placeholder="waterproof, foldable"
          onChange={(constraints) => patch({ constraints })}
        />
        <ListField
          label="Manufacturing method" value={draft.manufacturing_requirements} disabled={disabled}
          placeholder="CNC machining, 3D printing"
          onChange={(manufacturing_requirements) => patch({ manufacturing_requirements })}
        />
        <ListField
          label="Accessibility requirements" value={draft.accessibility_requirements}
          disabled={disabled} placeholder="wheelchair compatible, single-handed operation"
          onChange={(accessibility_requirements) => patch({ accessibility_requirements })}
        />

        <NumberField
          label="Tolerance" suffix="mm" step={0.05} disabled={disabled}
          value={draft.tolerances_mm}
          onChange={(tolerances_mm) => patch({ tolerances_mm })}
        />
      </div>
    </div>
  )
}
