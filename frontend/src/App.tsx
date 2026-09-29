import { Suspense, lazy } from 'react'
import { Route, Routes } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import Layout from './components/Layout'
import About from './pages/About'
import Gallery from './pages/Gallery'
import Home from './pages/Home'
import NotFound from './pages/NotFound'
import ProjectDetail from './pages/ProjectDetail'
import Settings from './pages/Settings'
import Studio from './pages/Studio'

// Split out so the in-browser model runtime only loads when someone opens it.
const Generate = lazy(() => import('./pages/Generate'))

function PageLoading() {
  return (
    <div className="py-24 text-center text-slate-500" role="status">
      <Loader2 className="w-6 h-6 mx-auto mb-3 animate-spin" />
      Loading
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Home />} />
        <Route
          path="/generate"
          element={
            <Suspense fallback={<PageLoading />}>
              <Generate />
            </Suspense>
          }
        />
        <Route path="/studio" element={<Studio />} />
        <Route path="/studio/:projectId" element={<Studio />} />
        <Route path="/gallery" element={<Gallery />} />
        <Route path="/projects/:projectId" element={<ProjectDetail />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/about" element={<About />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
