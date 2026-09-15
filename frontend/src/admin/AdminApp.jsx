import { useState, useEffect } from 'react'
import Login from './Login'
import Analytics from './Analytics'
import LogViewer from './LogViewer'
import Documents from './Documents'
import './admin.css'

const TOKEN_KEY = 'gsm_admin_token'

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
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet" />
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
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet" />

      <div className="a-topbar">
        <div className="a-topbar-left">
          <span className="a-topbar-org">GSM</span>
          <span className="a-topbar-sep">/</span>
          <span className="a-topbar-title">HR Admin</span>
        </div>
        <button className="a-btn-logout" onClick={handleLogout}>Sign out</button>
      </div>

      <div className="a-layout">
        <nav className="a-sidebar">
          <div className="a-nav-section">Overview</div>
          <div className={`a-nav-item ${tab === 'analytics' ? 'active' : ''}`} onClick={() => setTab('analytics')}>
            <span className="a-nav-icon">📊</span> Analytics
          </div>
          <div className="a-nav-section">Data</div>
          <div className={`a-nav-item ${tab === 'log' ? 'active' : ''}`} onClick={() => setTab('log')}>
            <span className="a-nav-icon">📋</span> Chat History
          </div>
          <div className="a-nav-section">Content</div>
          <div className={`a-nav-item ${tab === 'documents' ? 'active' : ''}`} onClick={() => setTab('documents')}>
            <span className="a-nav-icon">📁</span> Documents
          </div>
        </nav>

        <main className="a-main">
          {tabs[tab]}
        </main>
      </div>
    </div>
  )
}
