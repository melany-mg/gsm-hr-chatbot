export async function sendMessage({ message, language, history }) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, language, history }),
  })
  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`)
  }
  return response.json()
}
