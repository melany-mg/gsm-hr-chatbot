import { useState } from 'react'
import LanguageDropdown from './components/LanguageDropdown'
import ChatWindow from './components/ChatWindow'
import InputBar from './components/InputBar'
import { sendMessage } from './api'
import './App.css'

export default function App() {
  const [language, setLanguage] = useState('en')
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)

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
      const result = await sendMessage({ message: text, language, history })
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
      <header>
        <h1>GSM HR Assistant</h1>
        <LanguageDropdown value={language} onChange={setLanguage} />
      </header>
      <ChatWindow messages={messages} loading={loading} />
      <InputBar onSend={handleSend} disabled={loading} />
    </div>
  )
}
