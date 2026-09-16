import { useState } from 'react'

export default function InputBar({ onSend, disabled }) {
  const [input, setInput] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    if (!input.trim()) return
    onSend(input.trim())
    setInput('')
  }

  return (
    <div className="input-bar-wrap">
      <form onSubmit={handleSubmit} className="input-bar">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          disabled={disabled}
          placeholder="Ask a question about GSM HR policies…"
        />
        <button type="submit" disabled={disabled || !input.trim()}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
          Send
        </button>
      </form>
      <p className="chat-disclaimer">
        Answers are grounded in official GSM HR documents. For complex situations, contact HR directly.
      </p>
    </div>
  )
}
