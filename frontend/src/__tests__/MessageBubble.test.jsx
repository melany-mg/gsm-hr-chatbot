import { render, screen } from '@testing-library/react'
import MessageBubble from '../components/MessageBubble'

test('renders message content', () => {
  render(<MessageBubble role="user" content="Hello there" citations={[]} />)
  expect(screen.getByText('Hello there')).toBeInTheDocument()
})

test('applies user class for user messages', () => {
  const { container } = render(
    <MessageBubble role="user" content="Hi" citations={[]} />
  )
  expect(container.firstChild).toHaveClass('message', 'user')
})

test('applies assistant class for assistant messages', () => {
  const { container } = render(
    <MessageBubble role="assistant" content="Hi" citations={[]} />
  )
  expect(container.firstChild).toHaveClass('message', 'assistant')
})

test('renders citations when present', () => {
  const citations = [
    { source: 'handbook.pdf', page: 5 },
    { source: 'benefits.pdf', page: 2 },
  ]
  render(<MessageBubble role="assistant" content="Here is the info." citations={citations} />)
  expect(screen.getByText(/handbook.pdf/)).toBeInTheDocument()
  expect(screen.getByText(/p\. 5/)).toBeInTheDocument()
  expect(screen.getByText(/benefits.pdf/)).toBeInTheDocument()
})

test('does not render citations section when empty', () => {
  render(<MessageBubble role="assistant" content="Contact HR." citations={[]} />)
  expect(screen.queryByText(/Sources:/)).not.toBeInTheDocument()
})
