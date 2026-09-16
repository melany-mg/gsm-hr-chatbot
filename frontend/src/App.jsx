import { useState, useRef } from 'react'
import LanguageDropdown from './components/LanguageDropdown'
import ChatWindow from './components/ChatWindow'
import InputBar from './components/InputBar'
import { sendMessage } from './api'
import logo from './assets/gsm-logo.png'
import './App.css'

export default function App() {
  const [language, setLanguage] = useState('en')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const sessionId = useRef(
    typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2) + Date.now().toString(36)
  )

  async function handleSend(text) {
    const newUserMsg = { role: 'user', content: text, citations: [] }
    const updatedMessages = [...messages, newUserMsg]
    setMessages(updatedMessages)
    setLoading(true)

    try {
      const history = updatedMessages.slice(0, -1).map(m => ({
        role: m.role === 'user' ? 'user' : 'assistant',
        content: m.content,
      }))
      const result = await sendMessage({ message: text, language, history, sessionId: sessionId.current })
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: result.answer, citations: result.citations },
      ])
    } catch {
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'Something went wrong. Please try again.',
          citations: [],
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="app-card">
        <header>
          <div className="header-logo">
            <img className="header-logo-icon" src={logo} alt="GSM" />
            <div className="header-logo-divider" />
            <div className="header-logo-wordmark">
              <span className="header-logo-name">General Stamping &amp; Metalworks</span>
              <span className="header-logo-sub">HR Assistant</span>
            </div>
          </div>
          <LanguageDropdown value={language} onChange={setLanguage} />
        </header>

        <div className="subheader">
          <span className="status-dot" />
          <span className="status-text">
            Online &middot; Ask me about vacation, benefits, payroll, leave, and more
          </span>
        </div>

        <ChatWindow messages={messages} loading={loading} />
        <InputBar onSend={handleSend} disabled={loading} />
      </div>
    </div>
  )
}
