import { Link } from 'react-router-dom'
import { Compass } from 'lucide-react'

export default function NotFound() {
  return (
    <div className="mx-auto max-w-lg px-6 py-28 text-center">
      <Compass className="w-10 h-10 mx-auto mb-4 text-ink-600" strokeWidth={1.5} />
      <h1 className="text-2xl font-bold text-white">Page not found</h1>
      <p className="mt-2 text-sm text-slate-400">
        That route does not exist in the studio.
      </p>
      <Link to="/" className="btn-primary mt-6">
        Back to home
      </Link>
    </div>
  )
}
