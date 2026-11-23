import { expect, test, type Page } from '@playwright/test'
import { chooseSyntheticAudio, mockHealthAndModels } from './helpers'

const transcript = [
  { start: 0.25, end: 1.5, text: 'We agreed to ship the beta on Friday.' },
  { start: 2, end: 3, text: 'Priya will write the release notes.' },
]
const model = 'nvidia/nemotron-3-ultra-550b-a55b'
const available = { available: true, model, modes: ['summary', 'action_items'], reason: null }
const summaryText = 'Overview: the team agreed on a Friday beta.'
const summary = { mode: 'summary', text: summaryText, model, created_at: 1 }

async function mockCapabilities(page: Page) {
  await page.route('**/api/capabilities', route => route.fulfill({ json: { service: 'stt-studio', ai: available } }))
}

async function mockJob(page: Page) {
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-ai-cancel-2', status: 'completed', segments: transcript, info: { language: 'en', duration: 3 } } })
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

// Holds the AI request open until the test releases it, then answers with `response`.
// Returns the release function plus the aborted-request tracking the browser reports.
async function holdAiRequest(page: Page, response: { status?: number; json: unknown }) {
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  const requestErrors: string[] = []
  let requestCount = 0
  let routeSettled = false
  page.on('requestfailed', request => {
    if (/\/api\/jobs\/[^/]+\/ai$/.test(new URL(request.url()).pathname)) requestErrors.push(request.failure()?.errorText ?? 'unknown')
  })
  await page.route('**/api/jobs/*/ai', async route => {
    requestCount += 1
    await gate
    try { await route.fulfill({ status: response.status ?? 200, json: response.json }) } catch { /* the page already aborted this request */ }
    routeSettled = true
  })
  return {
    release,
    requestErrors,
    requestCount: () => requestCount,
    routeSettled: () => routeSettled,
  }
}

test('cancelling a pending summary aborts the request and a late result is never shown or saved', async ({ page }) => {
  await mockCapabilities(page)
  await mockJob(page)
  const held = await holdAiRequest(page, { json: summary })
  await transcribe(page)

  await page.getByRole('radio', { name: 'Summary' }).click()
  await expect(page.getByRole('status').filter({ hasText: 'Generating summary' })).toBeVisible()
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()

  // The browser must report the AI request as aborted, not just hide it.
  await expect.poll(() => held.requestErrors).toEqual(['net::ERR_ABORTED'])
  await expect(page.getByRole('radio', { name: 'Off' })).toHaveAttribute('aria-checked', 'true')
  await expect(page.getByRole('status').filter({ hasText: 'Generating' })).toHaveCount(0)
  await expect(page.getByRole('alert')).toHaveCount(0)

  // The route would now complete with a summary. The cancelled request must not change the UI.
  held.release()
  await expect.poll(() => held.routeSettled()).toBe(true)
  await expect(page.getByText(summaryText)).toHaveCount(0)
  await expect(page.getByRole('region', { name: 'Summary result' })).toHaveCount(0)
  await expect(page.getByRole('alert')).toHaveCount(0)

  // Nothing was saved, so picking Summary again sends a fresh request and shows its result.
  await page.getByRole('radio', { name: 'Summary' }).click()
  await expect(page.getByText(summaryText)).toBeVisible()
  expect(held.requestCount()).toBe(2)
  expect(held.requestErrors).toEqual(['net::ERR_ABORTED'])
})

test('cancelling a pending summary shows no error even when the late response would fail', async ({ page }) => {
  await mockCapabilities(page)
  await mockJob(page)
  const held = await holdAiRequest(page, { status: 500, json: { detail: 'NVIDIA NIM is busy. Try again in a moment.' } })
  await transcribe(page)

  await page.getByRole('radio', { name: 'Summary' }).click()
  await expect(page.getByRole('status').filter({ hasText: 'Generating summary' })).toBeVisible()
  await page.getByRole('button', { name: 'Cancel', exact: true }).click()
  await expect.poll(() => held.requestErrors).toEqual(['net::ERR_ABORTED'])

  held.release()
  await expect.poll(() => held.routeSettled()).toBe(true)
  await expect(page.getByRole('alert')).toHaveCount(0)
  await expect(page.getByText('NVIDIA NIM is busy. Try again in a moment.')).toHaveCount(0)
  await expect(page.getByRole('radio', { name: 'Off' })).toHaveAttribute('aria-checked', 'true')
})
