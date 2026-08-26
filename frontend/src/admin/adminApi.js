const BASE = '/api/admin'

function authHeaders(token) {
  return { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }
}

async function handleAuth(res) {
  if (res.status === 401) throw new Error('unauthorized')
  return res
}

export async function adminLogin(password) {
  const res = await fetch(`${BASE}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ password }),
  })
  if (!res.ok) throw new Error('Invalid password')
  return res.json()
}

export async function fetchLogs(token) {
  const res = await handleAuth(await fetch(`${BASE}/logs`, { headers: authHeaders(token) }))
  return res.json()
}

export async function fetchAnalytics(token) {
  const res = await handleAuth(await fetch(`${BASE}/analytics`, { headers: authHeaders(token) }))
  return res.json()
}

export async function fetchDocuments(token) {
  const res = await handleAuth(await fetch(`${BASE}/documents`, { headers: authHeaders(token) }))
  return res.json()
}

export async function uploadDocument(token, file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Upload failed')
  }
  return res.json()
}

export async function deleteDocument(token, filename) {
  const res = await fetch(`${BASE}/documents/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    headers: authHeaders(token),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Delete failed')
  }
  return res.json()
}

export async function triggerIngest(token) {
  const res = await fetch(`${BASE}/ingest`, {
    method: 'POST',
    headers: authHeaders(token),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Ingest failed to start')
  }
  return res.json()
}

export async function fetchIngestStatus(token) {
  const res = await handleAuth(await fetch(`${BASE}/ingest/status`, { headers: authHeaders(token) }))
  return res.json()
}
