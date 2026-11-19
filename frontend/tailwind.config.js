/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: { extend: { colors: { studio: { bg: 'var(--bg)', panel: 'var(--panel)', line: 'var(--line)', text: 'var(--text)', muted: 'var(--muted)', accent: 'var(--green)' } } } },
  plugins: [],
}
