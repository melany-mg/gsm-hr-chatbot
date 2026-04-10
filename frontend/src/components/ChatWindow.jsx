import { useEffect, useRef } from 'react'
import MessageBubble from './MessageBubble'

export default function ChatWindow({ messages, loading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return (
    <div className="chat-window">
      {messages.map((msg, i) => (
        <MessageBubble
          key={i}
          role={msg.role}
          content={msg.content}
          citations={msg.citations}
        />
      ))}
      {loading && (
        <div className="message assistant typing-indicator">
          <span /><span /><span />
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
