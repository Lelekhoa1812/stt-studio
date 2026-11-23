import { expect, test, type Page } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

// A generated, silent 10 second WAV. The segments below sit inside it so seeking has room to move.
function silentWav(seconds: number): Buffer {
  const sampleRate = 8_000
  const data = Buffer.alloc(sampleRate * seconds * 2)
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

const segments = [
  { start: 0, end: 2, text: 'first line' },
  { start: 2, end: 4, text: 'second line' },
  { start: 4, end: 6, text: 'third line' },
]

async function openCompletedTranscript(page: Page) {
  await mockHealthAndModels(page)
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-keys-1', status: 'completed', segments, info: { language: 'en', duration: 10 } } })
  })
  await page.goto('/')
  await page.locator('input[type="file"][accept^="audio"]:not([multiple])').setInputFiles({ name: 'keys.wav', mimeType: 'audio/wav', buffer: silentWav(10) })
  await expect(page.getByText('keys.wav').first()).toBeVisible()
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('first line')
  // Clicking the heading moves focus to the page body, where the shortcuts apply.
  await page.getByRole('heading', { name: 'Transcript', exact: true }).click()
}

const audioState = (page: Page) => page.locator('audio').evaluate(el => { const audio = el as HTMLAudioElement; return { paused: audio.paused, currentTime: audio.currentTime } })

test('Space toggles play and pause when focus is not in a text field', async ({ page }) => {
  await openCompletedTranscript(page)
  expect((await audioState(page)).paused).toBe(true)

  await page.keyboard.press('Space')
  await expect.poll(async () => (await audioState(page)).paused).toBe(false)

  await page.keyboard.press('Space')
  await expect.poll(async () => (await audioState(page)).paused).toBe(true)
})

test('Space types a space inside a segment editor instead of toggling playback', async ({ page }) => {
  await openCompletedTranscript(page)
  const editor = page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })
  await editor.click()
  await page.keyboard.press('End')
  await page.keyboard.press('Space')
  expect((await audioState(page)).paused).toBe(true)
  await expect(editor).toHaveValue('first line ')
})

test('ArrowDown and ArrowUp move the active segment and seek the audio to it', async ({ page }) => {
  await openCompletedTranscript(page)
  const active = page.locator('.segment-row[aria-current="true"] textarea')

  await page.keyboard.press('ArrowDown')
  await expect(active).toHaveValue('first line')
  await expect.poll(async () => (await audioState(page)).currentTime).toBeCloseTo(0, 1)

  await page.keyboard.press('ArrowDown')
  await expect(active).toHaveValue('second line')
  await expect.poll(async () => (await audioState(page)).currentTime).toBeCloseTo(2, 1)

  await page.keyboard.press('ArrowUp')
  await expect(active).toHaveValue('first line')
  await expect.poll(async () => (await audioState(page)).currentTime).toBeCloseTo(0, 1)

  // The active segment stops at the ends of the transcript instead of wrapping around.
  await page.keyboard.press('ArrowUp')
  await expect(active).toHaveValue('first line')
  for (let i = 0; i < 5; i++) await page.keyboard.press('ArrowDown')
  await expect(active).toHaveValue('third line')
  await expect.poll(async () => (await audioState(page)).currentTime).toBeCloseTo(4, 1)
})

test('Cmd or Ctrl+F focuses the transcript search box', async ({ page }) => {
  await openCompletedTranscript(page)
  await page.keyboard.press('ControlOrMeta+f')
  await expect(page.getByRole('textbox', { name: 'Search transcript' })).toBeFocused()
})

test('Escape clears the transcript search, from the search box or from the page', async ({ page }) => {
  await openCompletedTranscript(page)
  const search = page.getByRole('textbox', { name: 'Search transcript' })

  await search.fill('second')
  await expect(page.locator('.segments-list textarea')).toHaveCount(1)
  await search.press('Escape')
  await expect(search).toHaveValue('')
  await expect(page.locator('.segments-list textarea')).toHaveCount(3)

  await search.fill('third')
  await page.getByRole('heading', { name: 'Transcript', exact: true }).click()
  await page.keyboard.press('Escape')
  await expect(search).toHaveValue('')
})
