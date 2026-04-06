import { render, screen, fireEvent } from '@testing-library/react'
import LanguageDropdown from '../components/LanguageDropdown'
import { vi } from 'vitest'

test('renders all five language options', () => {
  render(<LanguageDropdown value="en" onChange={() => {}} />)
  expect(screen.getByText('English')).toBeInTheDocument()
  expect(screen.getByText('Español')).toBeInTheDocument()
  expect(screen.getByText('پښتو')).toBeInTheDocument()
  expect(screen.getByText('دری')).toBeInTheDocument()
  expect(screen.getByText('Bosanski')).toBeInTheDocument()
})

test('calls onChange with selected language code', () => {
  const onChange = vi.fn()
  render(<LanguageDropdown value="en" onChange={onChange} />)
  fireEvent.change(screen.getByRole('combobox'), { target: { value: 'es' } })
  expect(onChange).toHaveBeenCalledWith('es')
})

test('shows current value as selected', () => {
  render(<LanguageDropdown value="bs" onChange={() => {}} />)
  expect(screen.getByRole('combobox')).toHaveValue('bs')
})
