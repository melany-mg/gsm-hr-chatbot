import { sendMessage } from '../api'

afterEach(() => {
  vi.restoreAllMocks()
})

test('sendMessage posts to /api/chat with correct body', async () => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ answer: 'Ten days.', citations: [] }),
  })

  const result = await sendMessage({ message: 'Vacation?', language: 'en', history: [] })

  expect(global.fetch).toHaveBeenCalledWith('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: 'Vacation?', language: 'en', history: [] }),
  })
  expect(result.answer).toBe('Ten days.')
  expect(result.citations).toEqual([])
})

test('sendMessage throws on non-ok response', async () => {
  global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 500 })

  await expect(
    sendMessage({ message: 'Vacation?', language: 'en', history: [] })
  ).rejects.toThrow('Server error: 500')
})
