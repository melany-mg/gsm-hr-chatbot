import { useState, useEffect, useRef } from 'react'
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

  if (error) return <div style={{ color: 'var(--a-text-2)', padding: 24 }}>{error}</div>
  if (!data) return <div style={{ color: 'var(--a-text-3)', padding: 24 }}>Loading…</div>

  const { totals, daily, topics } = data
  const answeredPct = totals.total ? Math.round((totals.answered / totals.total) * 100) : 0
  const redirectedPct = totals.total ? Math.round((totals.redirected / totals.total) * 100) : 0
  const maxDay = Math.max(...daily.map(d => d.count), 1)
  const maxTopic = Math.max(...topics.map(t => t.count), 1)
  const totalTopics = topics.reduce((s, t) => s + t.count, 0)

  return (
    <>
      <div className="a-page-heading">Analytics</div>
      <div className="a-page-sub">Usage summary for the GSM HR Chatbot</div>

      <div className="a-stats-row">
        <div className="a-stat-card">
          <div className="a-stat-label">Total Questions</div>
          <div className="a-stat-value">{totals.total}</div>
          <div className="a-stat-sub">All time</div>
        </div>
        <div className="a-stat-card">
          <div className="a-stat-label">Unique Sessions</div>
          <div className="a-stat-value">{totals.sessions}</div>
          <div className="a-stat-sub">
            {totals.sessions > 0 ? `Avg ${(totals.total / totals.sessions).toFixed(1)} questions/session` : '—'}
          </div>
        </div>
        <div className="a-stat-card">
          <div className="a-stat-label">Answered</div>
          <div className="a-stat-value" style={{ color: 'var(--a-green)' }}>{answeredPct}%</div>
          <div className="a-stat-sub">{totals.answered} of {totals.total}</div>
        </div>
        <div className="a-stat-card">
          <div className="a-stat-label">Redirected to HR</div>
          <div className="a-stat-value" style={{ color: 'var(--a-amber)' }}>{redirectedPct}%</div>
          <div className="a-stat-sub">{totals.redirected} of {totals.total}</div>
        </div>
      </div>

      <div className="a-two-col">
        <div className="a-card">
          <div className="a-card-header">
            <div className="a-card-title">Questions per Day</div>
            <div className="a-card-sub">Last 30 days</div>
          </div>
          <div className="a-card-body">
            <div className="a-chart-bars">
              {daily.map(d => (
                <div className="a-chart-bar-wrap" key={d.date}>
                  <div
                    className="a-chart-bar"
                    style={{ height: `${Math.max((d.count / maxDay) * 100, d.count > 0 ? 3 : 0)}%` }}
                    title={`${d.date}: ${d.count} questions`}
                  />
                  <div className="a-chart-bar-val">{d.count || ''}</div>
                </div>
              ))}
            </div>
            <div className="a-chart-x-labels">
              {daily.filter((_, i) => i % 5 === 0).map(d => (
                <div key={d.date} style={{ flex: '5', fontSize: 9, color: 'var(--a-text-3)', textAlign: 'center' }}>
                  {d.date.slice(5)}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="a-card">
          <div className="a-card-header">
            <div className="a-card-title">Topics Asked About</div>
            <div className="a-card-sub">By question volume</div>
          </div>
          <div className="a-card-body">
            {topics.map(t => (
              <div className="a-topic-row" key={t.topic}>
                <div className="a-topic-name">{t.topic}</div>
                <div className="a-topic-track">
                  <div className="a-topic-fill" style={{ width: `${totalTopics ? (t.count / maxTopic) * 100 : 0}%` }} />
                </div>
                <div className="a-topic-pct">
                  {totalTopics ? `${Math.round((t.count / totalTopics) * 100)}%` : '0%'}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
