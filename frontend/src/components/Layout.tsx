import { useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { Cpu, Github, Globe, Menu, X } from 'lucide-react'
import { STATIC_MODE, api } from '../lib/api'
import type { SystemStatus } from '../lib/types'

const NAV = [
  { to: '/generate', label: 'Text → 3D' },
  { to: '/studio', label: 'Design Studio' },
  { to: '/gallery', label: 'Gallery' },
  { to: '/settings', label: 'Settings' },
  { to: '/about', label: 'About' },
]

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="w-7 h-7 shrink-0" aria-hidden="true">
      <path
        d="M16 5 27 11.5v13L16 31 5 24.5v-13z"
        fill="none"
        stroke="#4cc9f0"
        strokeWidth="2"
        strokeLinejoin="round"
      />
      <path
        d="M5 11.5 16 18l11-6.5M16 18v13"
        fill="none"
        stroke="#4cc9f0"
        strokeWidth="1.5"
        opacity=".6"
      />
    </svg>
  )
}

export default function Layout() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()

  useEffect(() => {
    // The hosted demo has no backend to ask; its badge is static (below).
    if (STATIC_MODE) return
    api.systemStatus().then(setStatus).catch(() => setStatus(null))
  }, [])

  useEffect(() => {
    setMenuOpen(false)
  }, [location.pathname])

  const gpu = status?.hardware.cuda_available
  const link = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
      isActive ? 'bg-ink-800 text-white' : 'text-slate-400 hover:text-white hover:bg-ink-850'
    }`

  return (
    <div className="min-h-full flex flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-3 focus:py-2 focus:bg-blueprint-500 focus:text-ink-950 focus:rounded-lg"
      >
        Skip to content
      </a>

      <header className="sticky top-0 z-40 border-b border-ink-800 bg-ink-950/85 backdrop-blur">
        <div className="mx-auto max-w-[1600px] px-4 sm:px-6 h-14 flex items-center gap-3">
          <Link to="/" className="flex items-center gap-2.5 mr-2 shrink-0">
            <Logo />
            <span className="font-bold text-sm text-white leading-tight hidden sm:block">
              AI 3D Blueprint
              <span className="block text-[10px] font-mono font-normal text-slate-500 tracking-wider">
                GENERATOR
              </span>
            </span>
          </Link>

          <nav className="hidden md:flex items-center gap-1 flex-1">
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} className={link}>
                {item.label}
              </NavLink>
            ))}
          </nav>

          <div className="flex-1 md:hidden" />

          {/* Live capability badge - reflects the real backend state. */}
          {STATIC_MODE && (
            <Link
              to="/settings"
              className="hidden sm:flex chip border-blueprint-500/40 text-blueprint-400 bg-blueprint-500/10"
              title="Hosted demo with no backend - Text → 3D runs in your browser (WebAssembly)"
            >
              <Globe className="w-3 h-3" />
              In-browser
            </Link>
          )}
          {status && (
            <Link
              to="/settings"
              className={`hidden sm:flex chip ${
                gpu
                  ? 'border-signal-ok/40 text-signal-ok bg-signal-ok/10'
                  : 'border-signal-warn/40 text-signal-warn bg-signal-warn/10'
              }`}
              title={
                gpu
                  ? `CUDA on ${status.hardware.gpu}`
                  : 'No CUDA device detected - generation runs on CPU'
              }
            >
              <Cpu className="w-3 h-3" />
              {gpu ? 'GPU' : 'CPU'}
            </Link>
          )}

          <a
            href="https://github.com/PuchaArunKumar/AI-driven-3D-blueprint-generator"
            target="_blank"
            rel="noreferrer noopener"
            className="p-2 text-slate-500 hover:text-white transition-colors"
            aria-label="Source on GitHub"
          >
            <Github className="w-4 h-4" />
          </a>

          <button
            className="md:hidden p-2 text-slate-400 hover:text-white"
            onClick={() => setMenuOpen((open) => !open)}
            aria-label={menuOpen ? 'Close menu' : 'Open menu'}
            aria-expanded={menuOpen}
          >
            {menuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {menuOpen && (
          <nav className="md:hidden border-t border-ink-800 px-4 py-2 space-y-1">
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} className={link} style={{ display: 'block' }}>
                {item.label}
              </NavLink>
            ))}
          </nav>
        )}
      </header>

      <main id="main" className="flex-1">
        <Outlet />
      </main>

      <footer className="border-t border-ink-800 mt-auto">
        <div className="mx-auto max-w-[1600px] px-4 sm:px-6 py-5 flex flex-col sm:flex-row items-center justify-between gap-3 text-[11px] text-slate-500">
          <p>
            AI-Driven 3D Blueprint Generator
            {status && <span className="font-mono"> v{status.version}</span>}
            {STATIC_MODE && <span className="font-mono"> (hosted static demo)</span>} - research
            prototype for accessible, personalised product design.
          </p>
          <p className="font-mono">Text → Image → 3D → GLB → BLEND · Text → 3D in-browser</p>
        </div>
      </footer>
    </div>
  )
}
