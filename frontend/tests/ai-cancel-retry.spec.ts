import { expect, test, type Page } from '@playwright/test'
import { chooseSyntheticAudio, mockHealthAndModels } from './helpers'

const transcript = [
  { start: 0.25, end: 1.5, text: 'We agreed to ship the beta on Friday.' },
  { start: 2, end: 3, text: 'Priya will write the release notes.' },
]
const model = 'nvidia/nemotron-3-ultra-550b-a55b'
const available = { available: true, model, modes: ['summary', 'action_items'], reason: null }
const summary = { mode: 'summary', text: 'Overview: the team agreed on a Friday beta.', model, created_at: 1 }

async function mockCapabilities(page: Page) {
  await page.route('**/api/capabilities', route => route.fulfill({ json: { service: 'stt-studio', ai: available } }))
}

async function mockJob(page: Page) {
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-ai-cancel-1', status: 'completed', segments: transcript, info: { language: 'en', duration: 3 } } })
  })
}

async function transcribe(page: Page) {
  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('We agreed to ship the beta on Friday.')
}

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('cancelling a pending summary returns to Off with no result and no error', async ({ page }) => {
  await mockCapabilities(page)
  await mockJob(page)
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  let requests = 0
  let served = false
  // The route stays pending until the test releases it, so Cancel is clicked while the summary is still running.
  await page.route('**/api/jobs/*/ai', async route => {
    requests += 1
    await gate
    try { await route.fulfill({ json: summary }); served = true } catch { /* the page already aborted this request */ }
  })
  await transcribe(page)

  await page.getByRole('radio', { name: 'Summary' }).click()
  await expect(page.getByRole('status').filter({ hasText: 'Generating summary' })).toBeVisible()
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()

  await expect(page.getByRole('radio', { name: 'Off' })).toHaveAttribute('aria-checked', 'true')
  await expect(page.getByRole('status').filter({ hasText: 'Generating' })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toHaveCount(0)
  await expect(page.getByRole('alert')).toHaveCount(0)

  release()
  await expect.poll(() => served).toBe(true)
  await expect(page.getByText('Overview: the team agreed on a Friday beta.')).toHaveCount(0)
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(requests).toBe(1)
})

test('a failed summary shows the error, and Retry runs the same mode again until it succeeds', async ({ page }) => {
  await mockCapabilities(page)
  await mockJob(page)
  const requests: Array<{ mode: string }> = []
  let calls = 0
  await page.route('**/api/jobs/*/ai', async route => {
    requests.push(route.request().postDataJSON())
    calls += 1
    if (calls === 1) return route.fulfill({ status: 500, json: { detail: 'NVIDIA NIM is busy. Try again in a moment.' } })
    await route.fulfill({ json: summary })
  })
  await transcribe(page)

  await page.getByRole('radio', { name: 'Summary' }).click()
  const alert = page.getByRole('alert').filter({ hasText: 'Summary could not be generated' })
  await expect(alert).toContainText('NVIDIA NIM is busy. Try again in a moment.')
  await alert.getByRole('button', { name: 'Retry' }).click()

  await expect(page.getByText('Overview: the team agreed on a Friday beta.')).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
  expect(requests.map(request => request.mode)).toEqual(['summary', 'summary'])
})
