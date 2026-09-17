import { useEffect, useState } from 'react'
import { fetchLogs } from './adminApi'

function DetailModal({ row, onClose }) {
  return (
    <div className="a-modal-overlay" onClick={onClose}>
      <div className="a-modal" onClick={e => e.stopPropagation()}>
        <div className="a-modal-head">
          <div className="a-modal-title">Entry Detail</div>
          <button className="a-modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="a-modal-meta-row">
          <span className="a-modal-meta-item"><span className="a-modal-meta-label">Timestamp</span>{row['Timestamp'] || '—'}</span>
          <span className="a-modal-meta-item"><span className="a-modal-meta-label">Session</span>{row['Session ID'] ? row['Session ID'].slice(0, 8) + '…' : '—'}</span>
          <span className="a-modal-meta-item"><span className="a-modal-meta-label">Language</span><span style={{ textTransform: 'uppercase' }}>{row['Language'] || 'en'}</span></span>
          <span className="a-modal-meta-item"><span className="a-modal-meta-label">Citations</span>{row['Citations'] || 0}</span>
          <span className="a-modal-meta-item">
            <span className={`pill ${row['Outcome'] === 'answered' ? 'answered' : 'redirected'}`}>{row['Outcome']}</span>
          </span>
        </div>
        <div className="a-modal-section">
          <div className="a-modal-section-label">Question</div>
          <div className="a-modal-body-text">{row['Question'] || '—'}</div>
        </div>
        <div className="a-modal-section">
          <div className="a-modal-section-label">Answer</div>
          <div className="a-modal-body-text">{row['Answer'] || '—'}</div>
        </div>
      </div>
    </div>
  )
}

export default function LogViewer({ token, onUnauthorized }) {
  const [rows, setRows] = useState(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(null)

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
      {selected && <DetailModal row={selected} onClose={() => setSelected(null)} />}
      <div className="a-section-head">
        <div>
          <div className="a-section-title">Chat History</div>
          <div className="a-section-sub">Every question asked, newest first — click a row to see full detail</div>
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
                  <tr key={i} className="a-table-row-clickable" onClick={() => setSelected(r)}>
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
