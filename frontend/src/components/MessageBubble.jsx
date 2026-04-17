import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function MessageBubble({ role, content, citations }) {
  return (
    <div className={`message ${role}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      {citations && citations.length > 0 && (
        <p className="citations">
          {'Sources: '}
          {citations.map((c, i) => (
            <span key={i}>
              {c.source} (p. {c.page}){i < citations.length - 1 ? ', ' : ''}
            </span>
          ))}
        </p>
      )}
    </div>
  )
}
