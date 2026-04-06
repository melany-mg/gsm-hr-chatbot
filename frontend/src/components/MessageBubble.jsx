export default function MessageBubble({ role, content, citations }) {
  return (
    <div className={`message ${role}`}>
      <p>{content}</p>
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
