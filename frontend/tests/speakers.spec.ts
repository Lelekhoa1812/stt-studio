import { expect, test, type Page } from '@playwright/test'
import { chooseSyntheticAudio, mockHealthAndModels, syntheticWav } from './helpers'

const unavailableReason = 'HF_TOKEN is not set; add it to .env (the account must have accepted the model terms).'
const available = { available: true, model: 'pyannote/speaker-diarization-3.1', license: 'MIT', reason: null }
const unavailable = { ...available, available: false, reason: unavailableReason }
const transcript = [
  { start: 0.25, end: 1.5, text: 'hello world' },
  { start: 1.75, end: 3, text: 'local speech recognition' },
]

async function mockCapabilities(page: Page, diarization: Record<string, unknown>) {
  await page.route('**/api/capabilities', route => route.fulfill({ json: { service: 'stt-studio', diarization } }))
}

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('disables speaker labels in both panels and shows the reason when diarization is unavailable', async ({ page }) => {
  await mockCapabilities(page, unavailable)
  await page.goto('/')
  const single = page.locator('.config-panel .speaker-option')
  await expect(single.getByRole('checkbox', { name: 'Label speakers' })).toBeDisabled()
  await expect(single.getByText(unavailableReason)).toBeVisible()
  const batch = page.locator('.batch-panel .speaker-option')
  await expect(batch.getByRole('checkbox', { name: 'Label speakers' })).toBeDisabled()
  await expect(batch.getByText(unavailableReason)).toBeVisible()
})

test('sends diarize=true with a single job only when labelling is enabled', async ({ page }) => {
  await mockCapabilities(page, available)
  const bodies: string[] = []
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    bodies.push(route.request().postData() || '')
    await route.fulfill({ status: 202, json: { id: 'job-diarize-1', status: 'completed', segments: transcript } })
  })

  await page.goto('/')
  await chooseSyntheticAudio(page)
  const checkbox = page.locator('.config-panel .speaker-option').getByRole('checkbox', { name: 'Label speakers' })
  await expect(checkbox).toBeEnabled()
  await expect(checkbox).not.toBeChecked()
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  expect(bodies[0]).toMatch(/name="diarize"\r\n\r\nfalse\r\n/)

  await page.reload()
  await chooseSyntheticAudio(page)
  await page.locator('.config-panel .speaker-option').getByRole('checkbox', { name: 'Label speakers' }).check()
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  expect(bodies).toHaveLength(2)
  expect(bodies[1]).toMatch(/name="diarize"\r\n\r\ntrue\r\n/)
})

test('sends diarize=true with a batch when labelling is enabled', async ({ page }) => {
  await mockCapabilities(page, available)
  let batchBody = ''
  await page.route('**/api/batches', route => {
    batchBody = route.request().postData() || ''
    return route.fulfill({ status: 202, json: { id: 'batch-diarize-1', total: 1, done: true,
      counts: { queued: 0, running: 0, completed: 1, failed: 0, cancelled: 0 },
      jobs: [{ id: 'job-b1', filename: 'one.wav', status: 'completed', progress: null, error: null, segments: transcript, info: null }] } })
  })

  await page.goto('/')
  await page.locator('.batch-panel').getByRole('checkbox', { name: 'Label speakers' }).check()
  await page.getByLabel('Batch audio files').setInputFiles([{ name: 'one.wav', mimeType: 'audio/wav', buffer: syntheticWav() }])
  await page.getByRole('button', { name: 'Start batch (1)' }).click()
  await expect(page.getByText('1 of 1 complete')).toBeVisible()
  expect(batchBody).toMatch(/name="diarize"\r\n\r\ntrue\r\n/)
})
