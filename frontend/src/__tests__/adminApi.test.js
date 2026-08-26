import { adminLogin, fetchLogs, uploadDocument, deleteDocument } from '../admin/adminApi'

afterEach(() => {
  vi.restoreAllMocks()
})

test('adminLogin posts password and returns token', async () => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ token: 'abc123' }),
  })
  const result = await adminLogin('mypassword')
  expect(global.fetch).toHaveBeenCalledWith('/api/admin/login', expect.objectContaining({ method: 'POST' }))
  expect(result.token).toBe('abc123')
})

test('adminLogin throws on wrong password', async () => {
  global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 401 })
  await expect(adminLogin('wrong')).rejects.toThrow('Invalid password')
})

test('fetchLogs includes auth header', async () => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    status: 200,
    json: async () => [],
  })
  await fetchLogs('mytoken')
  expect(global.fetch).toHaveBeenCalledWith('/api/admin/logs', expect.objectContaining({
    headers: expect.objectContaining({ Authorization: 'Bearer mytoken' }),
  }))
})

test('fetchLogs throws unauthorized on 401', async () => {
  global.fetch = vi.fn().mockResolvedValue({ ok: false, status: 401 })
  await expect(fetchLogs('badtoken')).rejects.toThrow('unauthorized')
})

test('deleteDocument sends DELETE request', async () => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ deleted: 'file.pdf' }),
  })
  await deleteDocument('tok', 'file.pdf')
  expect(global.fetch).toHaveBeenCalledWith(
    '/api/admin/documents/file.pdf',
    expect.objectContaining({ method: 'DELETE' })
  )
})
