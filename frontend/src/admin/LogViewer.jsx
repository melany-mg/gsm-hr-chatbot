import { useEffect, useState } from 'react'
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

  if (error) return <div style={{ color: 'var(--text-mid)', padding: 24 }}>{error}</div>
  if (!rows) return <div style={{ color: 'var(--text-light)', padding: 24 }}>Loading…</div>

  return (
    <>
      <div className="a-section-head">
        <div>
          <div className="a-section-title">Chat History</div>
          <div className="a-section-sub">Every question asked, newest first</div>
        </div>
      </div>

      <div className="a-card">
        {rows.length === 0 ? (
          <div style={{ color: 'var(--text-light)' }}>No questions logged yet.</div>
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
                    <td style={{ whiteSpace: 'nowrap', color: 'var(--text-mid)' }}>{r['Timestamp']}</td>
                    <td className="a-meta">{r['Session ID'] ? r['Session ID'].slice(0, 8) + '…' : '—'}</td>
                    <td style={{ textTransform: 'uppercase' }}>{r['Language']}</td>
                    <td className="a-td-truncate" title={r['Question']}>{r['Question']}</td>
                    <td className="a-td-truncate" title={r['Answer']}>{r['Answer']}</td>
                    <td>
                      <span className={`pill ${r['Outcome'] === 'answered' ? 'answered' : 'redirected'}`}>
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
