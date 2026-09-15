import { useState, useEffect } from 'react'
import { fetchLogs } from './adminApi'

export default function LogViewer({ token, onUnauthorized }) {
  const [rows, setRows] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchLogs(token)
      .then(data => setRows([...data].reverse()))
      .catch(err => {
        if (err.message === 'unauthorized') onUnauthorized()
        else setError('Failed to load chat history.')
      })
  }, [token])

  if (error) return <div style={{ color: 'var(--a-text-2)', padding: 24 }}>{error}</div>
  if (!rows) return <div style={{ color: 'var(--a-text-3)', padding: 24 }}>Loading…</div>

  return (
    <>
      <div className="a-page-heading">Chat History</div>
      <div className="a-page-sub">Every question asked, newest first</div>

      <div className="a-card">
        {rows.length === 0 ? (
          <div style={{ padding: 24, color: 'var(--a-text-3)' }}>No questions logged yet.</div>
        ) : (
          <div className="a-table-wrap">
            <table className="a-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Session</th>
                  <th>Lang</th>
                  <th>Question</th>
                  <th>Answer</th>
                  <th>Outcome</th>
                  <th>Citations</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i}>
                    <td style={{ whiteSpace: 'nowrap', color: 'var(--a-text-2)' }}>{r['Timestamp']}</td>
                    <td className="a-meta">{r['Session ID'] ? r['Session ID'].slice(0, 8) + '…' : '—'}</td>
                    <td style={{ textTransform: 'uppercase' }}>{r['Language']}</td>
                    <td className="a-td-truncate" title={r['Question']}>{r['Question']}</td>
                    <td className="a-td-truncate" title={r['Answer']}>{r['Answer']}</td>
                    <td>
                      <span className={`a-badge ${r['Outcome'] === 'answered' ? 'a-badge-green' : 'a-badge-amber'}`}>
                        {r['Outcome']}
                      </span>
                    </td>
                    <td style={{ textAlign: 'center', fontVariantNumeric: 'tabular-nums' }}>{r['Citations']}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  )
}
