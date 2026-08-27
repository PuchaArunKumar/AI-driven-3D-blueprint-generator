import { useEffect, useState } from 'react'
import {
  AlertTriangle,
  Check,
  Cpu,
  HardDrive,
  Loader2,
  MemoryStick,
  Server,
  X,
} from 'lucide-react'
import { ApiError, api } from '../lib/api'
import type { ProviderInfo, StorageUsage, SystemStatus } from '../lib/types'

function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`
}

function ProviderRow({ provider }: { provider: ProviderInfo }) {
  return (
    <li className="flex items-start gap-3 px-4 py-3">
      <span className="mt-0.5 shrink-0">
        {provider.available ? (
          <Check className="w-4 h-4 text-signal-ok" strokeWidth={3} />
        ) : (
          <X className="w-4 h-4 text-slate-600" />
        )}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <p
            className={`text-sm font-semibold ${
              provider.available ? 'text-slate-200' : 'text-slate-500'
            }`}
          >
            {provider.label}
          </p>
          <code className="font-mono text-[10px] text-slate-600">{provider.name}</code>
          {provider.experimental && (
            <span className="chip border-signal-warn/40 text-signal-warn !text-[9px]">
              Research Component - Experimental / Optional
            </span>
          )}
        </div>
        {provider.reason && (
          <p className="mt-0.5 text-[12px] text-slate-500 leading-relaxed">{provider.reason}</p>
        )}
        {provider.requires.length > 0 && !provider.available && (
          <p className="mt-1 font-mono text-[10px] text-slate-600">
            requires: {provider.requires.join(', ')}
          </p>
        )}
      </div>
    </li>
  )
}

export default function Settings() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [storage, setStorage] = useState<StorageUsage | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.systemStatus().then(setStatus).catch((cause: ApiError) => setError(cause.message))
    api.storageUsage().then(setStorage).catch(() => undefined)
  }, [])

  if (error) {
    return (
      <div className="mx-auto max-w-2xl py-24 text-center">
        <AlertTriangle className="w-8 h-8 mx-auto mb-3 text-signal-err" />
        <p className="text-slate-300">{error}</p>
      </div>
    )
  }

  if (!status) {
    return (
      <div className="py-24 text-center text-slate-500">
        <Loader2 className="w-6 h-6 mx-auto mb-3 animate-spin" />
        Reading system status
      </div>
    )
  }

  const { hardware, capabilities, providers, settings } = status
  const grouped = {
    image: providers.filter((provider) => provider.kind === 'image'),
    threed: providers.filter((provider) => provider.kind === 'threed'),
    mesh: providers.filter((provider) => provider.kind === 'mesh'),
    cad: providers.filter((provider) => provider.kind === 'cad'),
  }

  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 py-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Settings</h1>
        <p className="text-sm text-slate-500 mt-1">
          Read from the running backend. Values are configured in{' '}
          <code className="font-mono text-slate-400">.env</code> and applied on restart -
          secrets are never sent to the browser.
        </p>
      </div>

      {status.status === 'degraded' && (
        <div className="panel border-signal-warn/40 bg-signal-warn/5 p-4 flex items-start gap-3">
          <AlertTriangle className="w-4 h-4 text-signal-warn shrink-0 mt-0.5" />
          <p className="text-sm text-slate-300">
            Some pipeline stages have no available provider. The demo projects still work; see
            the provider list below for what is missing.
          </p>
        </div>
      )}

      {/* ------------------------------------------------------- providers */}
      {(
        [
          ['Image Generation', grouped.image],
          ['3D Generation', grouped.threed],
          ['Mesh Processing', grouped.mesh],
          ['CAD Tools', grouped.cad],
        ] as [string, ProviderInfo[]][]
      ).map(([title, list]) => (
        <section key={title} className="panel">
          <div className="panel-head">
            <h2 className="text-sm font-semibold text-white">{title}</h2>
            <span className="text-[11px] text-slate-500">
              {list.filter((provider) => provider.available).length}/{list.length} available
            </span>
          </div>
          <ul className="divide-y divide-ink-800">
            {list.map((provider) => (
              <ProviderRow key={`${provider.kind}-${provider.name}`} provider={provider} />
            ))}
          </ul>
        </section>
      ))}

      {/* -------------------------------------------------------- hardware */}
      <section className="panel">
        <div className="panel-head">
          <h2 className="text-sm font-semibold text-white">Hardware</h2>
        </div>
        <div className="p-4 grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[
            [<Cpu className="w-3.5 h-3.5" key="c" />, 'CPU', hardware.cpu],
            [<Server className="w-3.5 h-3.5" key="s" />, 'Cores', String(hardware.cpu_cores)],
            [
              <MemoryStick className="w-3.5 h-3.5" key="m" />,
              'RAM',
              hardware.ram_total_gb
                ? `${hardware.ram_available_gb?.toFixed(1)} / ${hardware.ram_total_gb} GB free`
                : 'unavailable',
            ],
            [
              <Cpu className="w-3.5 h-3.5" key="g" />,
              'GPU',
              hardware.gpu ?? 'none detected',
            ],
            [
              <MemoryStick className="w-3.5 h-3.5" key="v" />,
              'VRAM',
              hardware.vram_total_gb ? `${hardware.vram_total_gb} GB` : '-',
            ],
            [
              <Server className="w-3.5 h-3.5" key="t" />,
              'PyTorch',
              hardware.torch_version
                ? `${hardware.torch_version}${
                    hardware.cuda_available ? ` (CUDA ${hardware.cuda_version})` : ' (CPU only)'
                  }`
                : 'not installed',
            ],
          ].map(([icon, key, value]) => (
            <div key={key as string}>
              <p className="stat-key flex items-center gap-1.5">
                {icon}
                {key as string}
              </p>
              <p className="stat-val break-words">{value as string}</p>
            </div>
          ))}
        </div>
        <div className="px-4 pb-4">
          <span
            className={`chip ${
              hardware.cuda_available
                ? 'border-signal-ok/40 text-signal-ok bg-signal-ok/10'
                : 'border-signal-warn/40 text-signal-warn bg-signal-warn/10'
            }`}
          >
            {hardware.cuda_available
              ? 'CUDA acceleration active'
              : 'No CUDA - image generation will be slow on CPU'}
          </span>
        </div>
      </section>

      {/* ---------------------------------------------------- capabilities */}
      <section className="panel">
        <div className="panel-head">
          <h2 className="text-sm font-semibold text-white">Export capabilities</h2>
        </div>
        <div className="p-4">
          <div className="flex flex-wrap gap-1.5 mb-3">
            {capabilities.export_formats.map((format) => (
              <span
                key={format}
                className="chip border-signal-ok/40 text-signal-ok bg-signal-ok/10 uppercase"
              >
                <Check className="w-3 h-3" />
                {format}
              </span>
            ))}
            {Object.keys(capabilities.unavailable_export_formats).map((format) => (
              <span key={format} className="chip border-ink-700 text-slate-600 uppercase">
                <X className="w-3 h-3" />
                {format}
              </span>
            ))}
          </div>
          {Object.entries(capabilities.unavailable_export_formats).map(([format, reason]) => (
            <p key={format} className="text-[11px] text-slate-500">
              <span className="uppercase font-mono text-slate-400">{format}</span> - {reason}
            </p>
          ))}
          {capabilities.blender_version && (
            <p className="mt-3 text-[12px] text-slate-400">
              Blender {capabilities.blender_version} detected at{' '}
              <code className="font-mono text-[11px] text-slate-500">
                {settings.blender_path}
              </code>
            </p>
          )}
        </div>
      </section>

      {/* --------------------------------------------------------- storage */}
      {storage && (
        <section className="panel">
          <div className="panel-head">
            <h2 className="text-sm font-semibold text-white">Storage</h2>
            <span className="text-[11px] text-slate-500">
              {storage.disk_free_gb} GB free of {storage.disk_total_gb} GB
            </span>
          </div>
          <div className="p-4 space-y-3">
            <p className="font-mono text-[11px] text-slate-500 break-all flex items-start gap-2">
              <HardDrive className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              {storage.storage_dir}
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                ['Images', storage.images_bytes],
                ['Models', storage.models_bytes],
                ['Exports', storage.exports_bytes],
                ['Demo assets', storage.demo_bytes],
              ].map(([key, value]) => (
                <div key={key as string}>
                  <p className="stat-key">{key as string}</p>
                  <p className="stat-val">{formatBytes(value as number)}</p>
                </div>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* -------------------------------------------------------- settings */}
      <section className="panel">
        <div className="panel-head">
          <h2 className="text-sm font-semibold text-white">Active configuration</h2>
        </div>
        <dl className="p-4 grid sm:grid-cols-2 gap-x-6 gap-y-2.5">
          {Object.entries(settings).map(([key, value]) => (
            <div key={key} className="flex justify-between gap-3 border-b border-ink-850 pb-1.5">
              <dt className="font-mono text-[11px] text-slate-500">{key}</dt>
              <dd className="font-mono text-[11px] text-slate-300 text-right break-all">
                {typeof value === 'boolean' ? (
                  <span className={value ? 'text-signal-ok' : 'text-slate-600'}>
                    {value ? 'set' : 'not set'}
                  </span>
                ) : (
                  String(value)
                )}
              </dd>
            </div>
          ))}
        </dl>
        <p className="px-4 pb-4 text-[11px] text-slate-500">
          API keys are reported as <span className="font-mono">set</span> /{' '}
          <span className="font-mono">not set</span> only. Their values never leave the server.
        </p>
      </section>
    </div>
  )
}
