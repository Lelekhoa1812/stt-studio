import { expect, test, type Page } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

const transcript = [
  { start: 0, end: 2, text: 'hello world' },
  { start: 2, end: 4, text: 'second line' },
]

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

async function transcribeAndEdit(page: Page) {
  await mockHealthAndModels(page)
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-draft-1', status: 'completed', segments: transcript, info: { language: 'en', duration: 4 } } })
  })
  await page.goto('/')
  await page.locator('input[type="file"][accept^="audio"]:not([multiple])').setInputFiles({ name: 'draft.wav', mimeType: 'audio/wav', buffer: silentWav() })
  await expect(page.getByText('draft.wav').first()).toBeVisible()
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  const first = page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })
  await expect(first).toHaveValue('hello world')
  await first.fill('hello brave world')
  // Wait past the 500 ms debounce so the draft is written before the reload.
  await expect.poll(() => page.evaluate(() => window.localStorage.getItem('sttStudio.draft') ?? '')).toContain('hello brave world')
}

test('an unsaved edit survives a reload and can be restored', async ({ page }) => {
  await transcribeAndEdit(page)

  await page.reload()
  await expect(page.getByText('Recovered unsaved edits')).toBeVisible()
  await page.getByRole('button', { name: 'Restore', exact: true }).click()

  await expect(page.getByText('Recovered unsaved edits')).toHaveCount(0)
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello brave world')
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:02' })).toHaveValue('second line')
  await expect(page.getByText('Unsaved edits · Session only until you save')).toBeVisible()
})

test('Discard removes the draft so the notice does not come back after a reload', async ({ page }) => {
  await transcribeAndEdit(page)

  await page.reload()
  await expect(page.getByText('Recovered unsaved edits')).toBeVisible()
  await page.getByRole('button', { name: 'Discard', exact: true }).click()
  await expect(page.getByText('Recovered unsaved edits')).toHaveCount(0)
  expect(await page.evaluate(() => window.localStorage.getItem('sttStudio.draft'))).toBeNull()

  await page.reload()
  await expect(page.getByRole('heading', { name: 'Choose an audio file' })).toBeVisible()
  await expect(page.getByText('Recovered unsaved edits')).toHaveCount(0)
})

test('a reverted edit leaves no draft to recover', async ({ page }) => {
  await transcribeAndEdit(page)
  await page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' }).fill('hello world')
  // The baseline copy in the draft still reads "hello world", so wait until the edited copy is gone.
  await expect.poll(() => page.evaluate(() => window.localStorage.getItem('sttStudio.draft') ?? '')).not.toContain('hello brave world')

  await page.reload()
  await expect(page.getByText('Recovered unsaved edits')).toHaveCount(0)
})

test('starting a new run discards the recoverable draft', async ({ page }) => {
  await transcribeAndEdit(page)

  await page.reload()
  await expect(page.getByText('Recovered unsaved edits')).toBeVisible()
  await page.locator('input[type="file"][accept^="audio"]:not([multiple])').setInputFiles({ name: 'second.wav', mimeType: 'audio/wav', buffer: silentWav() })
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  await expect(page.getByText('Recovered unsaved edits')).toHaveCount(0)
  expect(await page.evaluate(() => window.localStorage.getItem('sttStudio.draft') ?? '')).not.toContain('hello brave world')
})
