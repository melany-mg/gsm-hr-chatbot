import { useState, useEffect, useRef } from 'react'
import { fetchDocuments, uploadDocument, deleteDocument, triggerIngest, fetchIngestStatus } from './adminApi'

function formatBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1048576) return `${(b / 1024).toFixed(1)} KB`
  return `${(b / 1048576).toFixed(1)} MB`
}

export default function Documents({ token, onUnauthorized }) {
  const [docs, setDocs] = useState(null)
  const [ingestStatus, setIngestStatus] = useState({ state: 'idle', last_run: null })
  const [toast, setToast] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef()
  const pollRef = useRef()

  function showToast(msg, type = 'success') {
    setToast({ msg, type })
    setTimeout(() => setToast(null), 4000)
  }

  async function loadDocs() {
    try {
      setDocs(await fetchDocuments(token))
    } catch (err) {
      if (err.message === 'unauthorized') onUnauthorized()
    }
  }

  async function loadIngestStatus() {
    try {
      const s = await fetchIngestStatus(token)
      setIngestStatus(s)
      return s
    } catch { return null }
  }

  useEffect(() => {
    loadDocs()
    loadIngestStatus()
  }, [token])

  async function handleFiles(files) {
    for (const file of Array.from(files)) {
      if (file.size > 20 * 1024 * 1024) { showToast(`${file.name} exceeds 20 MB.`, 'error'); continue }
      const allowed = ['.pdf', '.docx', '.txt']
      const ext = '.' + file.name.split('.').pop().toLowerCase()
      if (!allowed.includes(ext)) { showToast(`${file.name}: only PDF, DOCX, TXT allowed.`, 'error'); continue }
      try {
        await uploadDocument(token, file)
        showToast(`${file.name} uploaded.`)
      } catch (err) {
        showToast(err.message, 'error')
      }
    }
    loadDocs()
  }

  async function handleDelete(name) {
    if (!window.confirm(`Remove ${name}? This cannot be undone.`)) return
    try {
      await deleteDocument(token, name)
      showToast(`${name} removed.`)
      loadDocs()
    } catch (err) {
      showToast(err.message, 'error')
    }
  }

  async function handleIngest() {
    try {
      await triggerIngest(token)
      setIngestStatus({ state: 'running', last_run: null })
      pollRef.current = setInterval(async () => {
        const s = await loadIngestStatus()
        if (s && s.state !== 'running') {
          clearInterval(pollRef.current)
          if (s.state === 'done') showToast('Chatbot updated successfully.')
          else showToast('Ingest failed. Check server logs.', 'error')
        }
      }, 3000)
    } catch (err) {
      showToast(err.message, 'error')
    }
  }

  useEffect(() => () => clearInterval(pollRef.current), [])

  const isRunning = ingestStatus.state === 'running'
  const docIcon = name => name.endsWith('.txt') ? '📝' : '📄'

  return (
    <>
      <div className="a-page-heading">Documents</div>
      <div className="a-page-sub">Manage the HR documents the chatbot learns from</div>

      <div className="a-card">
        <div className="a-card-header">
          <div className="a-card-title">Current Documents</div>
          <div className="a-card-sub">{docs ? `${docs.length} file${docs.length !== 1 ? 's' : ''}` : '…'}</div>
        </div>

        {docs && docs.map(f => (
          <div className="a-doc-row" key={f.name}>
            <div className="a-doc-icon">{docIcon(f.name)}</div>
            <div className="a-doc-info">
              <div className="a-doc-name">{f.name}</div>
              <div className="a-doc-meta">{formatBytes(f.size)} · Modified {f.modified}</div>
            </div>
            <button className="a-btn-remove" onClick={() => handleDelete(f.name)}>Remove</button>
          </div>
        ))}

        <div
          className={`a-upload-zone ${dragOver ? 'drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={e => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files) }}
          onClick={() => fileInputRef.current.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.docx,.txt"
            multiple
            style={{ display: 'none' }}
            onChange={e => handleFiles(e.target.files)}
          />
          <div className="a-upload-icon">⬆️</div>
          <div className="a-upload-text">Drop files here, or click to browse</div>
          <div className="a-upload-hint">PDF, DOCX, TXT · Max 20 MB each</div>
        </div>

        <div className="a-ingest-bar">
          <div className="a-ingest-info">
            Last chatbot update:{' '}
            <strong>{ingestStatus.last_run || 'Never'}</strong>
          </div>
          <button className="a-btn-ingest" onClick={handleIngest} disabled={isRunning}>
            {isRunning && <span className="a-spinner" />}
            {isRunning ? 'Updating…' : 'Update Chatbot'}
          </button>
        </div>
      </div>

      {toast && (
        <div className={`a-toast ${toast.type}`}>{toast.msg}</div>
      )}
    </>
  )
}
