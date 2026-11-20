import { expect, test, type Page } from '@playwright/test'
import { chooseSyntheticAudio, mockHealthAndModels } from './helpers'

const transcript = [
  { start: 0.25, end: 1.5, text: 'We agreed to ship the beta on Friday.' },
  { start: 2, end: 3, text: 'Priya will write the release notes.' },
]
const model = 'nvidia/nemotron-3-ultra-550b-a55b'
const available = { available: true, model, modes: ['summary', 'action_items'], reason: null }
const unavailableReason = 'AI modes need an NVIDIA NIM key. Add NVIDIA_NIM_API_KEY_1 to .env or the environment to turn them on.'
const notice = 'Selecting a mode sends the transcript text, never the audio, to NVIDIA NIM.'
const summary = { mode: 'summary', text: 'Overview: the team agreed on a Friday beta.\n- Beta ships on Friday\n- Priya writes the release notes', model, created_at: 1 }
const revised = { ...summary, text: 'Revised overview: the beta ships on Friday.' }

type Reply = { status?: number; json: unknown }

async function mockCapabilities(page: Page, ai: Record<string, unknown>) {
  await page.route('**/api/capabilities', route => route.fulfill({ json: { service: 'stt-studio', ai } }))
}

async function mockJob(page: Page) {
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-ai-1', status: 'completed', segments: transcript, info: { language: 'en', duration: 3 } } })
  })
}

async function mockAi(page: Page, replies: Reply[]) {
  const requests: Array<{ mode: string; segments: unknown }> = []
  await page.route('**/api/jobs/*/ai', async route => {
    requests.push(route.request().postDataJSON())
    const reply = replies[Math.min(requests.length - 1, replies.length - 1)]
    await route.fulfill({ status: reply.status ?? 200, json: reply.json })
  })
  return requests
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

test('disables the AI switcher and shows the reason when no NVIDIA key is configured', async ({ page }) => {
  await mockCapabilities(page, { ...available, available: false, reason: unavailableReason })
  await mockJob(page)
  await transcribe(page)
  const group = page.getByRole('radiogroup', { name: 'AI mode' })
  await expect(group.getByRole('radio', { name: 'Summary' })).toBeDisabled()
  await expect(group.getByRole('radio', { name: 'Action items' })).toBeDisabled()
  await expect(page.getByText(unavailableReason)).toBeVisible()
})

test('keeps the AI switcher disabled until a transcript is complete', async ({ page }) => {
  await mockCapabilities(page, available)
  await page.goto('/')
  const group = page.getByRole('radiogroup', { name: 'AI mode' })
  await expect(group.getByRole('radio', { name: 'Summary' })).toBeDisabled()
  await expect(page.getByText('AI modes are available once the transcript is complete.')).toBeVisible()
})

test('shows the transcript notice next to the switcher', async ({ page }) => {
  await mockCapabilities(page, available)
  await mockJob(page)
  await transcribe(page)
  await expect(page.getByText(notice)).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Off' })).toHaveAttribute('aria-checked', 'true')
})

test('selecting a mode sends exactly one request with the mode and the transcript as shown', async ({ page }) => {
  await mockCapabilities(page, available)
  await mockJob(page)
  const requests = await mockAi(page, [{ json: summary }])
  await transcribe(page)
  await page.getByRole('radio', { name: 'Summary' }).click()
  await expect(page.getByText('Overview: the team agreed on a Friday beta.')).toBeVisible()
  expect(requests).toEqual([{ mode: 'summary', segments: [{ text: 'We agreed to ship the beta on Friday.' }, { text: 'Priya will write the release notes.' }] }])
  await expect(page.getByRole('button', { name: 'Copy AI result' })).toBeVisible()
})

test('Off sends no request, hides the result and keeps the stored result for the next selection', async ({ page }) => {
  await mockCapabilities(page, available)
  await mockJob(page)
  const requests = await mockAi(page, [{ json: summary }])
  await transcribe(page)
  await page.getByRole('radio', { name: 'Off' }).click()
  await expect(page.getByRole('radio', { name: 'Off' })).toHaveAttribute('aria-checked', 'true')
  await expect(page.getByText('Overview: the team agreed on a Friday beta.')).toHaveCount(0)

  await page.getByRole('radio', { name: 'Summary' }).click()
  await expect(page.getByText('Overview: the team agreed on a Friday beta.')).toBeVisible()
  await page.getByRole('radio', { name: 'Off' }).click()
  await page.getByRole('radio', { name: 'Summary' }).click()
  await expect(page.getByText('Overview: the team agreed on a Friday beta.')).toBeVisible()
  expect(requests).toHaveLength(1)
})

test('Regenerate replaces the stored result for the mode', async ({ page }) => {
  await mockCapabilities(page, available)
  await mockJob(page)
  const requests = await mockAi(page, [{ json: summary }, { json: revised }])
  await transcribe(page)
  await page.getByRole('radio', { name: 'Summary' }).click()
  await expect(page.getByText('Overview: the team agreed on a Friday beta.')).toBeVisible()
  await page.getByRole('button', { name: 'Regenerate' }).click()
  await expect(page.getByText('Revised overview: the beta ships on Friday.')).toBeVisible()
  expect(requests).toHaveLength(2)
})

test('shows a readable error and the switcher can be retried', async ({ page }) => {
  await mockCapabilities(page, available)
  await mockJob(page)
  const detail = 'NVIDIA NIM did not accept any configured key (last HTTP 429). Try again later, or check the keys in .env.'
  const requests = await mockAi(page, [{ status: 502, json: { detail } }, { json: { ...summary, mode: 'action_items', text: '- Priya writes the release notes. Owner named: Priya.' } }])
  await transcribe(page)
  await page.getByRole('radio', { name: 'Action items' }).click()
  const alert = page.getByRole('alert').filter({ hasText: 'could not be generated' })
  await expect(alert).toContainText(detail)
  await alert.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByText('- Priya writes the release notes. Owner named: Priya.')).toBeVisible()
  expect(requests.map(request => request.mode)).toEqual(['action_items', 'action_items'])
})
