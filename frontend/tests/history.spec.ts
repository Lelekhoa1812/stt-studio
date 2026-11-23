import { expect, test, type Page, type Route } from '@playwright/test'
import { chooseSyntheticAudio, mockHealthAndModels } from './helpers'

type Job = Record<string, unknown>
type HistoryState = { jobs: Job[]; details: Record<string, Job>; storage: Record<string, unknown>; listFailure?: string; deleteStatus?: number; deleteDetail?: string; listRequests: string[]; deleted: string[] }

const FIRST = Date.UTC(2026, 10, 19, 9, 30) / 1000
const SECOND = Date.UTC(2026, 10, 18, 14, 5) / 1000
const STATUS_TEXT: Record<string, string> = { completed: 'Completed', failed: 'Failed', running: 'Running', queued: 'Queued', cancelled: 'Cancelled' }

function dayOf(seconds: number) {
  return new Date(seconds * 1000).toLocaleString('en-AU', { day: 'numeric', month: 'short', year: 'numeric' })
}

function summary(overrides: Job): Job {
  return { status: 'completed', model: 'small', runtime: 'faster-whisper', compute_type: 'int8', language: null, beam_size: 5, vad_filter: true, info: { language: 'en', duration: 3 }, error: null, progress: 100, cancel_requested: false, filename: 'meeting-notes.wav', batch_id: null, diarize: false, created_at: FIRST, ...overrides }
}

const meeting = summary({ id: 'job-h1', created_at: FIRST, filename: 'meeting-notes.wav' })
const interview = summary({ id: 'job-h2', created_at: SECOND, filename: 'interview.mp3', status: 'failed', model: 'medium', language: 'vi', info: null, error: 'Interrupted by a server restart', batch_id: 'batch-1' })

const meetingSegments = [
  { start: 0, end: 1.5, text: 'first line', avg_logprob: -0.1, no_speech_prob: 0.01 },
  { start: 2, end: 3, text: 'second line', avg_logprob: -0.2, no_speech_prob: 0.02 },
]

function freshState(jobs: Job[] = [meeting, interview]): HistoryState {
  return {
    jobs: [...jobs],
    details: { 'job-h1': { ...meeting, segments: meetingSegments }, 'job-h2': { ...interview, segments: [] } },
    storage: { data_dir: 'data', jobs: jobs.length, batches: 1, bytes: 3145728 },
    listRequests: [],
    deleted: [],
  }
}

async function mockHistoryApi(page: Page, state: HistoryState) {
  await page.route('**/api/storage', route => route.fulfill({ json: state.storage }))
  await page.route(/\/api\/jobs\?/, async (route: Route) => {
    const url = new URL(route.request().url())
    state.listRequests.push(url.search.slice(1))
    if (state.listFailure) return route.fulfill({ status: 500, json: { detail: state.listFailure } })
    const limit = Number(url.searchParams.get('limit') ?? 50)
    const offset = Number(url.searchParams.get('offset') ?? 0)
    await route.fulfill({ json: { jobs: state.jobs.slice(offset, offset + limit), total: state.jobs.length, limit, offset } })
  })
  await page.route(/\/api\/jobs\/[^/?]+$/, async (route: Route) => {
    const id = decodeURIComponent(new URL(route.request().url()).pathname.split('/').pop() ?? '')
    if (route.request().method() === 'DELETE') {
      if (state.deleteStatus) return route.fulfill({ status: state.deleteStatus, json: { detail: state.deleteDetail ?? 'Request failed' } })
      state.deleted.push(id)
      state.jobs = state.jobs.filter(job => job.id !== id)
      return route.fulfill({ json: { id, deleted: true } })
    }
    const detail = state.details[id]
    if (!detail) return route.fulfill({ status: 404, json: { detail: 'Job not found' } })
    await route.fulfill({ json: detail })
  })
}

async function openHistory(page: Page) {
  await page.getByRole('button', { name: 'History', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Saved transcripts', exact: true })).toBeVisible()
}

test.describe('History (mocked backend)', () => {
  test.beforeEach(async ({ page }) => {
    await mockHealthAndModels(page)
  })

  test('lists saved transcripts with their metadata and the storage summary', async ({ page }) => {
    const state = freshState()
    await mockHistoryApi(page, state)
    await page.goto('/')
    await openHistory(page)

    const list = page.getByRole('list', { name: 'Saved transcripts list' })
    const rows = list.getByRole('listitem')
    await expect(rows).toHaveCount(2)
    await expect(rows.nth(0)).toContainText('meeting-notes.wav')
    await expect(rows.nth(0)).toContainText('Completed')
    await expect(rows.nth(0)).toContainText(dayOf(FIRST))
    await expect(rows.nth(0)).toContainText('small')
    await expect(rows.nth(0)).toContainText('English')
    await expect(rows.nth(0)).not.toContainText('In batch')
    await expect(rows.nth(1)).toContainText('interview.mp3')
    await expect(rows.nth(1)).toContainText(STATUS_TEXT.failed)
    await expect(rows.nth(1)).toContainText('medium')
    await expect(rows.nth(1)).toContainText('Vietnamese')
    await expect(rows.nth(1)).toContainText('In batch')
    await expect(page.getByText('Showing 2 of 2')).toBeVisible()
    await expect(page.getByText('2 transcripts')).toBeVisible()
    await expect(page.getByText('1 batch', { exact: true })).toBeVisible()
    await expect(page.getByText('3.0 MB used')).toBeVisible()
    expect(state.listRequests[0]).toBe('limit=50&offset=0')
  })

  test('loads the next page with Load more and appends it', async ({ page }) => {
    const many = Array.from({ length: 60 }, (_, i) => summary({ id: `job-${i}`, filename: `talk-${i}.wav`, created_at: FIRST - i * 60 }))
    const state = freshState(many)
    await mockHistoryApi(page, state)
    await page.goto('/')
    await openHistory(page)

    await expect(page.getByText('Showing 50 of 60')).toBeVisible()
    await page.getByRole('button', { name: 'Load more' }).click()
    await expect(page.getByText('Showing 60 of 60')).toBeVisible()
    await expect(page.locator('.history-item')).toHaveCount(60)
    await expect(page.getByRole('button', { name: 'Load more' })).toHaveCount(0)
    expect(state.listRequests).toContain('limit=50&offset=50')
  })

  test('shows the empty state when nothing has been saved', async ({ page }) => {
    await mockHistoryApi(page, freshState([]))
    await page.goto('/')
    await openHistory(page)
    await expect(page.getByRole('heading', { name: 'No saved transcripts yet.' })).toBeVisible()
  })

  test('shows a readable error when the list cannot load', async ({ page }) => {
    const state = freshState()
    state.listFailure = 'Local job storage is unavailable; check disk space and file permissions'
    await mockHistoryApi(page, state)
    await page.goto('/')
    await openHistory(page)
    const alert = page.getByRole('alert')
    await expect(alert).toContainText('Could not load saved transcripts')
    await expect(alert).toContainText('Local job storage is unavailable')
    await expect(page.getByRole('button', { name: 'Try again' })).toBeVisible()
  })

  test('shows the offline state without requesting the list when the local engine is unreachable', async ({ page }) => {
    const state = freshState()
    await page.route('**/api/health', route => route.abort())
    await mockHistoryApi(page, state)
    await page.goto('/')
    await openHistory(page)
    await expect(page.getByRole('heading', { name: 'Local engine offline' })).toBeVisible()
    expect(state.listRequests).toHaveLength(0)
  })

  test('opens a saved transcript read only and exports TXT and SRT from it', async ({ page }) => {
    await mockHistoryApi(page, freshState())
    await page.goto('/')
    await openHistory(page)
    await page.getByRole('button', { name: /meeting-notes\.wav Completed/ }).click()

    const first = page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })
    await expect(first).toHaveValue('first line')
    await expect(first).toHaveAttribute('readonly', '')
    await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:02' })).toHaveValue('second line')
    await expect(page.getByText('Audio is not stored, so there is no playback', { exact: false })).toBeVisible()
    await expect(page.getByRole('button', { name: '00:00:00' })).toBeDisabled()

    await page.getByRole('button', { name: 'Export' }).click()
    const [txt] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('menuitem', { name: /Plain text/ }).click(),
    ])
    expect(txt.suggestedFilename()).toBe('meeting-notes.txt')
    const { readFile } = await import('node:fs/promises')
    expect(await readFile((await txt.path())!, 'utf8')).toBe('first line\nsecond line\n')

    await page.getByRole('button', { name: 'Export' }).click()
    const [srt] = await Promise.all([
      page.waitForEvent('download'),
      page.getByRole('menuitem', { name: /SubRip captions/ }).click(),
    ])
    expect(srt.suggestedFilename()).toBe('meeting-notes.srt')
    const srtText = await readFile((await srt.path())!, 'utf8')
    expect(srtText.startsWith('1\n00:00:00,000 --> 00:00:01,500\nfirst line\n')).toBe(true)
    expect(srtText).toContain('2\n00:00:02,000 --> 00:00:03,000\nsecond line\n')
  })

  test('asks before deleting, sends DELETE, refreshes the list and closes the open transcript', async ({ page }) => {
    const state = freshState()
    await mockHistoryApi(page, state)
    await page.goto('/')
    await openHistory(page)
    await page.getByRole('button', { name: /meeting-notes\.wav Completed/ }).click()
    await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toBeVisible()

    await page.getByRole('button', { name: 'Delete meeting-notes.wav' }).click()
    const dialog = page.getByRole('dialog', { name: 'Delete saved transcript?' })
    await expect(dialog).toBeVisible()
    expect(state.deleted).toEqual([])
    await dialog.getByRole('button', { name: 'Delete transcript' }).click()

    await expect(page.getByText('Saved transcript deleted from this Mac.')).toBeVisible()
    expect(state.deleted).toEqual(['job-h1'])
    await expect(page.getByText('Showing 1 of 1')).toBeVisible()
    await expect(page.locator('.history-item')).toHaveCount(1)
    await expect(page.getByRole('heading', { name: 'Choose a saved transcript' })).toBeVisible()
  })

  test('shows the 409 message when the job is still running and keeps the item', async ({ page }) => {
    const state = freshState()
    state.deleteStatus = 409
    state.deleteDetail = 'The job is running; cancel it and wait for it to finish before deleting it'
    await mockHistoryApi(page, state)
    await page.goto('/')
    await openHistory(page)

    await page.getByRole('button', { name: 'Delete meeting-notes.wav' }).click()
    const dialog = page.getByRole('dialog', { name: 'Delete saved transcript?' })
    await dialog.getByRole('button', { name: 'Delete transcript' }).click()
    await expect(dialog.getByRole('alert')).toContainText('cancel it and wait for it to finish before deleting it')
    await expect(dialog).toBeVisible()
    expect(state.deleted).toEqual([])
    await expect(page.locator('.history-item')).toHaveCount(2)
  })

  test('shows storage counts and no absolute path', async ({ page }) => {
    const state = freshState()
    state.storage = { data_dir: '/Users/example/Downloads/secret-data', jobs: 12, batches: 2, bytes: 524288 }
    await mockHistoryApi(page, state)
    await page.goto('/')
    await openHistory(page)
    await expect(page.getByLabel('Storage summary')).toContainText('12 transcripts')
    await expect(page.getByLabel('Storage summary')).toContainText('2 batches')
    await expect(page.getByLabel('Storage summary')).toContainText('512 KB used')
    const panelText = await page.locator('.history-panel').innerText()
    expect(panelText).not.toContain('/Users')
    expect(panelText).not.toContain('secret-data')
  })
})

test.describe('History (real backend on 127.0.0.1:8765)', () => {
  test('lists the transcripts the local backend really stores', async ({ page }) => {
    const probe = await fetch('http://127.0.0.1:8765/api/jobs?limit=1&offset=0').then(response => response.ok ? response.json() as Promise<{ jobs: Job[]; total: number }> : null).catch(() => null)
    test.skip(!probe, 'Skipped: the local backend on 127.0.0.1:8765 is not running.')
    const total = probe!.total
    await page.goto('/')
    await openHistory(page)
    if (total === 0) {
      await expect(page.getByRole('heading', { name: 'No saved transcripts yet.' })).toBeVisible()
      return
    }
    await expect(page.getByText(`Showing ${Math.min(50, total)} of ${total}`)).toBeVisible()
    const first = probe!.jobs[0]
    await expect(page.locator('.history-item').first()).toContainText(STATUS_TEXT[String(first.status)] ?? String(first.status))
    await expect(page.locator('.history-item').first()).toContainText(String(first.filename || 'Untitled audio'))
  })
})

test.describe('Transcript save status (mocked backend)', () => {
  test.beforeEach(async ({ page }) => {
    await mockHealthAndModels(page)
  })

  test('says a completed run is saved to History, and only an unsaved edit is session only', async ({ page }) => {
    await page.route('**/api/jobs', async route => {
      if (route.request().method() !== 'POST') return route.continue()
      await route.fulfill({ status: 202, json: { id: 'job-footer-1', status: 'completed', segments: [{ start: 0.25, end: 1.5, text: 'first line' }], info: { language: 'en', duration: 1.5 } } })
    })
    await page.goto('/')
    await chooseSyntheticAudio(page)
    await page.getByRole('button', { name: 'Transcribe audio' }).click()
    const first = page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })
    await expect(first).toHaveValue('first line')
    await expect(page.getByText('Saved to History on this Mac')).toBeVisible()
    await expect(page.getByText(/session only/i)).toHaveCount(0)

    await first.fill('edited line')
    await expect(page.getByText('Unsaved edits · Session only until you save')).toBeVisible()
    await expect(page.getByText('Saved to History on this Mac')).toHaveCount(0)
  })
})
