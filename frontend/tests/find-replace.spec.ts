import { readFileSync } from 'node:fs'
import { expect, test, type Page } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

const transcript = [
  { start: 0, end: 2, text: 'hello world' },
  { start: 2, end: 4, text: 'Hello again' },
  { start: 4, end: 6, text: 'nothing to see' },
]

async function openCompletedTranscript(page: Page) {
  await mockHealthAndModels(page)
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-replace-1', status: 'completed', segments: transcript, info: { language: 'en', duration: 6 } } })
  })
  await page.goto('/')
  await page.locator('input[type="file"][accept^="audio"]:not([multiple])').setInputFiles({ name: 'replace.wav', mimeType: 'audio/wav', buffer: silentWav() })
  await expect(page.getByText('replace.wav').first()).toBeVisible()
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
}

function silentWav(): Buffer {
  const sampleRate = 8_000
  const data = Buffer.alloc(sampleRate * 2)
  const header = Buffer.alloc(44)
  header.write('RIFF', 0)
  header.writeUInt32LE(36 + data.length, 4)
  header.write('WAVE', 8)
  header.write('fmt ', 12)
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(1, 22)
  header.writeUInt32LE(sampleRate, 24)
  header.writeUInt32LE(sampleRate * 2, 28)
  header.writeUInt16LE(2, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36)
  header.writeUInt32LE(data.length, 40)
  return Buffer.concat([header, data])
}

test('replace all updates the segments and the exported TXT, ignoring case by default', async ({ page }) => {
  await openCompletedTranscript(page)

  await page.getByRole('textbox', { name: 'Search transcript' }).fill('hello')
  await page.getByRole('textbox', { name: 'Replace with' }).fill('hi')
  await page.getByRole('button', { name: 'Replace all' }).click()

  await expect(page.getByRole('status').filter({ hasText: '2 replaced' })).toBeVisible()
  // Clearing the search shows every row again. The count resets once the search changes.
  await page.getByRole('textbox', { name: 'Search transcript' }).fill('')
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hi world')
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:02' })).toHaveValue('hi again')
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:04' })).toHaveValue('nothing to see')

  await page.getByRole('button', { name: 'Export' }).click()
  const downloaded = page.waitForEvent('download')
  await page.getByRole('menuitem', { name: /Plain text/ }).click()
  const content = readFileSync(await (await downloaded).path(), 'utf8')
  expect(content).toBe('hi world\nhi again\nnothing to see\n')
})

test('Match case limits replacements to the exact casing', async ({ page }) => {
  await openCompletedTranscript(page)

  await page.getByRole('textbox', { name: 'Search transcript' }).fill('hello')
  await page.getByRole('textbox', { name: 'Replace with' }).fill('hi')
  await page.getByRole('checkbox', { name: 'Match case' }).check()
  await page.getByRole('button', { name: 'Replace all' }).click()

  await expect(page.getByRole('status').filter({ hasText: '1 replaced' })).toBeVisible()
  await page.getByRole('textbox', { name: 'Search transcript' }).fill('')
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hi world')
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:02' })).toHaveValue('Hello again')
})

test('Replace all is disabled without a search term and reports zero matches', async ({ page }) => {
  await openCompletedTranscript(page)
  const replaceAll = page.getByRole('button', { name: 'Replace all' })
  await expect(replaceAll).toBeDisabled()

  await page.getByRole('textbox', { name: 'Search transcript' }).fill('absent phrase')
  await expect(replaceAll).toBeEnabled()
  await replaceAll.click()
  await expect(page.getByRole('status').filter({ hasText: '0 replaced' })).toBeVisible()
  await page.getByRole('textbox', { name: 'Search transcript' }).fill('')
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
})
