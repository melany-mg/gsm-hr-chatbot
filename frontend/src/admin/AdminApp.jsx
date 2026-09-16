import { useState } from 'react'
import Login from './Login'
import Analytics from './Analytics'
import LogViewer from './LogViewer'
import Documents from './Documents'
import logo from '../assets/gsm-logo.png'
import './admin.css'

const TOKEN_KEY = 'gsm_admin_token'

const TABS = {
  analytics: { label: 'Analytics' },
  log: { label: 'Chat History' },
  documents: { label: 'HR Documents' },
}

const TODAY = new Date().toLocaleDateString(undefined, {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
})

export default function AdminApp() {
  const [token, setToken] = useState(() => {
    try { return localStorage.getItem(TOKEN_KEY) || '' } catch { return '' }
  })
  const [tab, setTab] = useState('analytics')

  function handleLogin(t) {
    try { localStorage.setItem(TOKEN_KEY, t) } catch {}
    setToken(t)
  }

  function handleLogout() {
    try { localStorage.removeItem(TOKEN_KEY) } catch {}
    setToken('')
  }

  function handleUnauthorized() {
    handleLogout()
  }

  if (!token) return (
    <div className="admin-root">
      <Login onSuccess={handleLogin} />
    </div>
  )

  const tabs = {
    analytics: <Analytics token={token} onUnauthorized={handleUnauthorized} />,
    log: <LogViewer token={token} onUnauthorized={handleUnauthorized} />,
    documents: <Documents token={token} onUnauthorized={handleUnauthorized} />,
  }

  return (
    <div className="admin-root">
      <div className="a-topbar">
        <div className="a-topbar-left">
          <span className="a-breadcrumb">HR Admin</span>
          <span className="a-breadcrumb-sep">/</span>
          <span className="a-page-label">{TABS[tab].label}</span>
        </div>
        <div className="a-topbar-right">
          <span className="a-topbar-date">{TODAY}</span>
        </div>
      </div>

      <aside className="a-sidebar">
        <div className="a-sidebar-logo">
          <img src={logo} alt="GSM" />
          <div className="a-sidebar-logo-wordmark">
            <span className="a-sidebar-logo-name">General Stamping &amp; Metalworks</span>
            <span className="a-sidebar-logo-sub">HR Admin Portal</span>
          </div>
        </div>

        <nav className="a-nav">
          <div className="a-nav-group">
            <div className="a-nav-group-label">Overview</div>
            <button
              type="button"
              className={`a-nav-item ${tab === 'analytics' ? 'active' : ''}`}
              onClick={() => setTab('analytics')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                <rect x="3" y="3" width="7" height="7" rx="1.5" />
                <rect x="14" y="3" width="7" height="7" rx="1.5" />
                <rect x="3" y="14" width="7" height="7" rx="1.5" />
                <rect x="14" y="14" width="7" height="7" rx="1.5" />
              </svg>
              Analytics
            </button>
            <button
              type="button"
              className={`a-nav-item ${tab === 'log' ? 'active' : ''}`}
              onClick={() => setTab('log')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
              Chat History
            </button>
          </div>
          <div className="a-nav-group">
            <div className="a-nav-group-label">Content</div>
            <button
              type="button"
              className={`a-nav-item ${tab === 'documents' ? 'active' : ''}`}
              onClick={() => setTab('documents')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="12" y1="18" x2="12" y2="12" />
                <line x1="9" y1="15" x2="15" y2="15" />
              </svg>
              HR Documents
            </button>
          </div>
        </nav>

        <div className="a-sidebar-footer">
          <button type="button" className="a-signout" onClick={handleLogout}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
            Sign out
          </button>
        </div>
      </aside>

      <main className="a-main">
        {tabs[tab]}
      </main>
    </div>
  )
}
