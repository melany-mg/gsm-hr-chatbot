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
    <form onSubmit={handleSubmit} className="input-bar">
      <input
        type="text"
        value={input}
        onChange={e => setInput(e.target.value)}
        disabled={disabled}
        placeholder="Type your question..."
      />
      <button type="submit" disabled={disabled || !input.trim()}>
        Send
      </button>
    </form>
  )
}
