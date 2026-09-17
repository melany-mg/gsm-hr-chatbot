import { useEffect, useState } from 'react'
import { fetchAnalytics, fetchLogs } from './adminApi'

const TOPIC_KEYWORDS = [
  ["PTO / Vacation", ["pto", "vacation", "time off", "days off", "accrual", "accrued"]],
  ["Benefits / Insurance", ["benefit", "insurance", "health", "dental", "vision", "medical", "coverage"]],
  ["FMLA / Leave", ["fmla", "leave", "maternity", "paternity", "family leave", "medical leave"]],
  ["Bereavement", ["bereavement", "funeral", "death", "passing"]],
  ["Holidays", ["holiday", "christmas", "thanksgiving", "labor day", "memorial day", "new year"]],
  ["Pay / Payroll", ["pay", "payroll", "salary", "wage", "overtime", "direct deposit"]],
  ["Conduct / Policy", ["conduct", "policy", "disciplinary", "harassment", "code of conduct"]],
]

function classifyTopic(question) {
  const q = (question || '').toLowerCase()
  for (const [topic, keywords] of TOPIC_KEYWORDS) {
    if (keywords.some(kw => q.includes(kw))) return topic
  }
  return "Other"
}

function TopicModal({ topic, logs, onClose }) {
  const filtered = logs.filter(r => classifyTopic(r['Question']) === topic)
  return (
    <div className="a-modal-overlay" onClick={onClose}>
      <div className="a-modal a-modal-wide" onClick={e => e.stopPropagation()}>
        <div className="a-modal-head">
          <div className="a-modal-title">{topic} <span style={{ color: 'var(--text-light)', fontWeight: 400, fontSize: 14 }}>— {filtered.length} question{filtered.length !== 1 ? 's' : ''}</span></div>
          <button className="a-modal-close" onClick={onClose}>✕</button>
        </div>
        {filtered.length === 0 ? (
          <div style={{ color: 'var(--text-light)', padding: '12px 0' }}>No questions for this topic.</div>
        ) : (
          <div className="a-modal-list">
            {filtered.map((r, i) => (
              <div key={i} className="a-modal-list-item">
                <div className="a-modal-list-meta">
                  <span style={{ color: 'var(--text-light)', fontSize: 12 }}>{r['Timestamp']}</span>
                  <span className={`pill ${r['Outcome'] === 'answered' ? 'answered' : 'redirected'}`}>{r['Outcome']}</span>
                </div>
                <div className="a-modal-section-label" style={{ marginTop: 8 }}>Question</div>
                <div className="a-modal-body-text">{r['Question'] || '—'}</div>
                <div className="a-modal-section-label" style={{ marginTop: 10 }}>Answer</div>
                <div className="a-modal-body-text">{r['Answer'] || '—'}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default function Analytics({ token, onUnauthorized }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [selectedTopic, setSelectedTopic] = useState(null)
  const [allLogs, setAllLogs] = useState(null)

  useEffect(() => {
    fetchAnalytics(token)
      .then(setData)
      .catch(err => {
        if (err.message === 'unauthorized') onUnauthorized()
        else setError('Failed to load analytics.')
      })
  }, [token])

  function handleTopicClick(topicName) {
    setSelectedTopic(topicName)
    if (!allLogs) {
      fetchLogs(token)
        .then(setAllLogs)
        .catch(() => setAllLogs([]))
    }
  }

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
      {selectedTopic && allLogs && (
        <TopicModal topic={selectedTopic} logs={allLogs} onClose={() => setSelectedTopic(null)} />
      )}
      {selectedTopic && !allLogs && (
        <div className="a-modal-overlay" onClick={() => setSelectedTopic(null)}>
          <div className="a-modal" onClick={e => e.stopPropagation()}>
            <div className="a-modal-head">
              <div className="a-modal-title">{selectedTopic}</div>
              <button className="a-modal-close" onClick={() => setSelectedTopic(null)}>✕</button>
            </div>
            <div style={{ color: 'var(--text-light)', padding: '12px 0' }}>Loading…</div>
          </div>
        </div>
      )}
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

      <div className="a-analytics-grid">
        {/* Row 1: Chart + Topics */}
        <div className="a-card a-card-chart">
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
                      <div className="bar-col" key={d.date}>
                        <div
                          className={`bar ${d.count === maxDay && maxDay > 0 ? 'peak' : pct > 0.7 ? 'hi' : ''}`}
                          style={{ height: `${Math.max(pct * 100, d.count > 0 ? 3 : 0)}%` }}
                          data-tooltip={`${d.date}: ${d.count} question${d.count !== 1 ? 's' : ''}`}
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
          {topTopic.count > 0 && (
            <div className="a-top-topic">
              <span className="a-top-topic-label">Top topic</span>
              <span className="a-top-topic-value">{topTopic.topic}</span>
              <span className="a-top-topic-count">{topTopic.count} questions</span>
            </div>
          )}
          <div className="topic-list">
            {topics.map(t => (
              <div
                className={`topic-row topic-row-clickable${t.count === 0 ? ' topic-row-empty' : ''}`}
                key={t.topic}
                onClick={() => t.count > 0 && handleTopicClick(t.topic)}
                title={t.count > 0 ? `View ${t.count} question${t.count !== 1 ? 's' : ''} in ${t.topic}` : undefined}
              >
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

        {/* Row 2: Recent Questions + Language */}
        <div className="a-card">
          <div className="a-card-head">
            <span className="a-card-title">Recent Questions</span>
            <span className="a-card-meta">Last 3</span>
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
                    <td className="a-td-truncate" style={{ maxWidth: 280 }}>{r.question}</td>
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
