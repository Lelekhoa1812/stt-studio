import { expect, test, type Page } from '@playwright/test'
import { chooseSyntheticAudio, mockHealthAndModels } from './helpers'

const jobId = 'job-live-1'
const first = { start: 0.25, end: 1.5, text: 'hello world' }
const second = { start: 1.75, end: 3, text: 'local speech recognition' }
const completedState = { status: 'completed', progress: 100, segments: [first, second], info: { language: 'en', duration: 3 } }

// Each poll returns states[step]. The test moves `step` forward only after it has checked the held state.
async function scriptedJob(page: Page, states: Array<Record<string, unknown> | null>) {
  const script = { step: 0, polls: 0 }
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: jobId, status: 'queued' } })
  })
  await page.route(`**/api/jobs/${jobId}`, async route => {
    script.polls += 1
    const state = states[Math.min(script.step, states.length - 1)]
    if (state === null) return route.fulfill({ status: 404, json: { detail: 'Job not found' } })
    await route.fulfill({ json: { id: jobId, ...state } })
  })
  return script
}

async function startTranscription(page: Page) {
  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
}

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
  await page.route('**/api/capabilities', route => route.fulfill({ json: { service: 'stt-studio',
    diarization: { available: false, model: 'pyannote/speaker-diarization-3.1', license: 'MIT', reason: 'Not needed for these tests.' } } }))
})

test('shows partial segments as they arrive, then the final transcript', async ({ page }) => {
  const script = await scriptedJob(page, [
    { status: 'running', progress: 20, segments: [first] },
    { status: 'running', progress: 60, segments: [first, second] },
    completedState,
  ])
  await startTranscription(page)
  await expect(page.getByText('Transcribing, 1 segment so far')).toBeVisible()
  await expect(page.getByText('20%')).toBeVisible()
  await expect(page.locator('.partial-row').first()).toContainText('hello world')

  script.step = 1
  await expect(page.getByText('Transcribing, 2 segments so far')).toBeVisible()
  await expect(page.getByText('60%')).toBeVisible()
  await expect(page.locator('.partial-row')).toHaveCount(2)

  script.step = 2
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  await expect(page.locator('.progress-card')).toHaveCount(0)
  await expect(page.locator('.partial-transcript')).toHaveCount(0)
})

test('keeps partial text read-only and says when editing and renaming become available', async ({ page }) => {
  const script = await scriptedJob(page, [{ status: 'running', progress: 40, segments: [first] }, completedState])
  await startTranscription(page)
  await expect(page.getByText('read-only until the job finishes, then speaker names can be added')).toBeVisible()
  await expect(page.getByRole('textbox', { name: /Transcript segment/ })).toHaveCount(0)
  await expect(page.getByLabel(/Name for Speaker/)).toHaveCount(0)

  script.step = 1
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  await expect(page.getByText('read-only until the job finishes')).toHaveCount(0)
})

test('polls about once a second and stops when the job completes', async ({ page }) => {
  const script = await scriptedJob(page, [
    { status: 'running', progress: 20, segments: [first] },
    { status: 'running', progress: 60, segments: [first, second] },
    completedState,
  ])
  await startTranscription(page)
  await expect(page.getByText('Transcribing, 1 segment so far')).toBeVisible()
  script.step = 1
  await expect(page.getByText('Transcribing, 2 segments so far')).toBeVisible()
  script.step = 2
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')

  const settled = script.polls
  await new Promise(resolve => setTimeout(resolve, 2500))
  expect(script.polls).toBe(settled)
})

test('stops polling and shows a clear message when the job is no longer known', async ({ page }) => {
  const script = await scriptedJob(page, [{ status: 'running', progress: 20, segments: [first] }, null])
  await startTranscription(page)
  await expect(page.getByText('Transcribing, 1 segment so far')).toBeVisible()
  script.step = 1
  await expect(page.getByRole('alert')).toContainText('The local engine no longer has this job.')
  await expect(page.getByRole('heading', { name: /Transcription didn.t finish/ })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Transcribe audio' })).toBeEnabled()

  const settled = script.polls
  await new Promise(resolve => setTimeout(resolve, 2500))
  expect(script.polls).toBe(settled)
})

test('shows an indeterminate state and no percentage for an MLX job', async ({ page }) => {
  const script = await scriptedJob(page, [
    { status: 'running', runtime: 'mlx-audio', progress: null, segments: [] },
    { status: 'completed', runtime: 'mlx-audio', progress: 100, segments: [first] },
  ])
  await startTranscription(page)
  await expect(page.getByText('Transcribing audio', { exact: true })).toBeVisible()
  await expect(page.getByText('Progress details unavailable')).toBeVisible()
  await expect(page.locator('.progress-bar.indeterminate')).toBeVisible()
  await expect(page.locator('.progress-card')).not.toContainText('%')
  await expect(page.locator('.partial-transcript')).toHaveCount(0)

  script.step = 1
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
})

test('reports the speaker labelling phase once transcription reaches 100 percent', async ({ page }) => {
  const script = await scriptedJob(page, [
    { status: 'running', progress: 100, diarize: true, segments: [first] },
    completedState,
  ])
  await startTranscription(page)
  await expect(page.getByText('Labelling speakers')).toBeVisible()
  script.step = 1
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
})
