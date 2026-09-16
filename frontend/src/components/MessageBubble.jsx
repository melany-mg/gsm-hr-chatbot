import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function MessageBubble({ role, content, citations }) {
  return (
    <div className={`message ${role}`}>
      <span className="message-label">{role === 'user' ? 'You' : 'HR Assistant'}</span>
      <div className="bubble">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
      {citations && citations.length > 0 && (
        <div className="citations">
          {citations.map((c, i) => (
            <div className="citation" key={i}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              <span>
                <strong>{c.source}</strong> &middot; p. {c.page}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
