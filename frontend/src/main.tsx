import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import './index.css'

// The router lives under Vite's base path, so the same build works at the
// domain root (local dev) and under /<repo-name>/ on GitHub Pages.
const basename = import.meta.env.BASE_URL.replace(/\/+$/, '') || undefined

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter basename={basename}>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)
