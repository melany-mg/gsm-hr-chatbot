import { useState } from 'react'
import { adminLogin } from './adminApi'

export default function Login({ onSuccess }) {
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const { token } = await adminLogin(password)
      onSuccess(token)
    } catch {
      setError('Incorrect password. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="a-login">
      <div className="a-login-card">
        <div className="a-login-org">General Stamping &amp; Metalworks</div>
        <div className="a-login-heading">HR Admin Portal</div>
        <div className="a-login-sub">
          Sign in to manage documents, view chat history, and review analytics.
        </div>
        <form onSubmit={handleSubmit}>
          <div className="a-label">Password</div>
          <input
            className="a-input"
            type="password"
            placeholder="Enter admin password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            autoFocus
          />
          {error && <div className="a-error">{error}</div>}
          <button className="a-btn-primary" type="submit" disabled={loading || !password}>
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  )
}
