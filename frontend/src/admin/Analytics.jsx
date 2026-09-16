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

  const { totals, daily, topics, languages = [], recent = [] } = data
  const answeredPct = totals.total ? Math.round((totals.answered / totals.total) * 100) : 0
  const redirectedPct = totals.total ? Math.round((totals.redirected / totals.total) * 100) : 0
  const maxDay = Math.max(...daily.map(d => d.count), 1)
  const maxTopic = Math.max(...topics.map(t => t.count), 1)
  const maxLang = Math.max(...languages.map(l => l.count), 1)
  const topTopic = topics.reduce((a, b) => b.count > a.count ? b : a, { topic: '—', count: 0 })

  return (
    <>
      <div className="a-section-head">
        <div>
          <div className="a-section-title">Analytics</div>
          <div className="a-section-sub">Employee usage data for the GSM HR Assistant</div>
        </div>
      </div>

      <div className="a-stats a-stats-5">
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
        <div className="a-stat">
          <div className="a-stat-label">Most Asked Topic</div>
          <div className="a-stat-value a-stat-value-sm">{topTopic.topic}</div>
          <div className="a-stat-sub">{topTopic.count} questions</div>
        </div>
      </div>

      <div className="a-content-row">
        <div className="a-card">
          <div className="a-card-head">
            <span className="a-card-title">Questions per day</span>
            <span className="a-card-meta">Last 30 days</span>
          </div>
          <div className="chart-area">
            <div className="chart-with-yaxis">
              <div className="chart-yaxis">
                <span>{maxDay}</span>
                <span>{Math.round(maxDay / 2)}</span>
                <span>0</span>
              </div>
              <div className="chart-bars-wrap">
                <div className="chart-bars">
                  {daily.map(d => {
                    const pct = d.count / maxDay
                    return (
                      <div
                        className="bar-col"
                        key={d.date}
                        data-tooltip={`${d.date}: ${d.count} question${d.count !== 1 ? 's' : ''}`}
                      >
                        <div
                          className={`bar ${d.count === maxDay && maxDay > 0 ? 'peak' : pct > 0.7 ? 'hi' : ''}`}
                          style={{ height: `${Math.max(pct * 100, d.count > 0 ? 3 : 0)}%` }}
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

      <div className="a-bottom-row">
        <div className="a-card">
          <div className="a-card-head">
            <span className="a-card-title">Recent Questions</span>
            <span className="a-card-meta">Last 5</span>
          </div>
          <div className="a-table-wrap">
            <table className="a-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Question</th>
                  <th>Lang</th>
                  <th>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {recent.length === 0 ? (
                  <tr><td colSpan={4} style={{ color: 'var(--text-light)' }}>No questions yet.</td></tr>
                ) : recent.map((r, i) => (
                  <tr key={i}>
                    <td className="a-meta" style={{ whiteSpace: 'nowrap' }}>{r.timestamp}</td>
                    <td className="a-td-truncate" style={{ maxWidth: 340 }}>{r.question}</td>
                    <td><span className="pill answered" style={{ textTransform: 'uppercase', fontSize: 10 }}>{r.language || 'en'}</span></td>
                    <td><span className={`pill ${r.outcome}`}>{r.outcome}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="a-card">
          <div className="a-card-head">
            <span className="a-card-title">Language Breakdown</span>
            <span className="a-card-meta">All time</span>
          </div>
          <div className="topic-list">
            {languages.map(l => (
              <div className="topic-row" key={l.code}>
                <div className="topic-meta">
                  <span className="topic-name">{l.name}</span>
                  <span className="topic-count">{l.count}</span>
                </div>
                <div className="topic-track">
                  <div className="topic-fill" style={{ width: `${(l.count / maxLang) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
