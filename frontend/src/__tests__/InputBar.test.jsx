import { render, screen, fireEvent } from '@testing-library/react'
import InputBar from '../components/InputBar'

test('renders input and send button', () => {
  render(<InputBar onSend={() => {}} disabled={false} />)
  expect(screen.getByRole('textbox')).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument()
})

test('calls onSend with trimmed message on submit', () => {
  const onSend = vi.fn()
  render(<InputBar onSend={onSend} disabled={false} />)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '  Hello  ' } })
  fireEvent.click(screen.getByRole('button', { name: /send/i }))
  expect(onSend).toHaveBeenCalledWith('Hello')
})

test('clears input after submit', () => {
  render(<InputBar onSend={() => {}} disabled={false} />)
  const input = screen.getByRole('textbox')
  fireEvent.change(input, { target: { value: 'Hello' } })
  fireEvent.click(screen.getByRole('button', { name: /send/i }))
  expect(input).toHaveValue('')
})

test('does not call onSend for empty or whitespace input', () => {
  const onSend = vi.fn()
  render(<InputBar onSend={onSend} disabled={false} />)
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '   ' } })
  fireEvent.click(screen.getByRole('button', { name: /send/i }))
  expect(onSend).not.toHaveBeenCalled()
})

test('disables input and button when disabled prop is true', () => {
  render(<InputBar onSend={() => {}} disabled={true} />)
  expect(screen.getByRole('textbox')).toBeDisabled()
  expect(screen.getByRole('button', { name: /send/i })).toBeDisabled()
})
