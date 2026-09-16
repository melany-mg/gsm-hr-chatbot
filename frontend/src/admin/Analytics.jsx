import { useEffect, useState } from 'react'
import { fetchAnalytics } from './adminApi'

export default function Analytics({ token, onUnauthorized }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchAnalytics(token)
      .then(setData)
      .catch(err => {
        if (err.message === 'unauthorized') onUnauthorized()
        else setError('Failed to load analytics.')
      })
  }, [token])

  if (error) return <div style={{ color: 'var(--text-mid)', padding: 24 }}>{error}</div>
  if (!data) return <div style={{ color: 'var(--text-light)', padding: 24 }}>Loading…</div>

  const { totals, daily, topics } = data
  const answeredPct = totals.total ? Math.round((totals.answered / totals.total) * 100) : 0
  const redirectedPct = totals.total ? Math.round((totals.redirected / totals.total) * 100) : 0
  const maxDay = Math.max(...daily.map(d => d.count), 1)
  const maxTopic = Math.max(...topics.map(t => t.count), 1)

  return (
    <>
      <div className="a-section-head">
        <div>
          <div className="a-section-title">Analytics</div>
          <div className="a-section-sub">Employee usage data for the GSM HR Assistant</div>
        </div>
      </div>

      <div className="a-stats">
        <div className="a-stat hl">
          <div className="a-stat-label">Total Questions</div>
          <div className="a-stat-value">{totals.total}</div>
          <div className="a-stat-sub">All time</div>
        </div>
        <div className="a-stat">
          <div className="a-stat-label">Unique Sessions</div>
          <div className="a-stat-value">{totals.sessions}</div>
          <div className="a-stat-sub">
            {totals.sessions > 0 ? `Avg ${(totals.total / totals.sessions).toFixed(1)} questions/session` : '—'}
          </div>
        </div>
        <div className="a-stat">
          <div className="a-stat-label">Answered</div>
          <div className="a-stat-value">{answeredPct}%</div>
          <div className="a-stat-sub">{totals.answered} of {totals.total}</div>
        </div>
        <div className="a-stat">
          <div className="a-stat-label">Redirected to HR</div>
          <div className="a-stat-value">{redirectedPct}%</div>
          <div className="a-stat-sub">{totals.redirected} of {totals.total}</div>
        </div>
      </div>

      <div className="a-content-row">
        <div className="a-card">
          <div className="a-card-head">
            <span className="a-card-title">Questions per day</span>
            <span className="a-card-meta">Last 30 days</span>
          </div>
          <div className="chart-area">
            <div className="chart-bars">
              {daily.map(d => {
                const pct = d.count / maxDay
                return (
                  <div className="bar-col" key={d.date}>
                    <div
                      className={`bar ${d.count === maxDay && maxDay > 0 ? 'peak' : pct > 0.7 ? 'hi' : ''}`}
                      style={{ height: `${Math.max(pct * 100, d.count > 0 ? 3 : 0)}%` }}
                      title={`${d.date}: ${d.count} questions`}
                    />
                  </div>
                )
              })}
            </div>
            <div className="chart-x-labels">
              {daily.filter((_, i) => i === 0 || i % 7 === 0 || i === daily.length - 1).map(d => (
                <span key={d.date}>{d.date.slice(5)}</span>
              ))}
            </div>
          </div>
        </div>

        <div className="a-card">
          <div className="a-card-head">
            <span className="a-card-title">Topics</span>
            <span className="a-card-meta">All time</span>
          </div>
          <div className="topic-list">
            {topics.map(t => (
              <div className="topic-row" key={t.topic}>
                <div className="topic-meta">
                  <span className="topic-name">{t.topic}</span>
                  <span className="topic-count">{t.count}</span>
                </div>
                <div className="topic-track">
                  <div className="topic-fill" style={{ width: `${(t.count / maxTopic) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
