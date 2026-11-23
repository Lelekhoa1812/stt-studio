import { expect, test, type Page } from '@playwright/test'
import { mockHealthAndModels, syntheticWav } from './helpers'

const seconds = (iso: string) => Date.parse(iso) / 1000
const counts = (values: Partial<Record<'queued' | 'running' | 'completed' | 'failed' | 'cancelled', number>>) =>
  ({ queued: 0, running: 0, completed: 0, failed: 0, cancelled: 0, ...values })

const listed = {
  batches: [
    { id: 'hist-new', total: 3, done: true, counts: counts({ completed: 2, failed: 1 }), created_at: seconds('2026-11-22T09:00:00Z') },
    { id: 'hist-old', total: 1, done: true, counts: counts({ completed: 1 }), created_at: seconds('2026-11-20T09:00:00Z') },
  ],
  total: 2,
  limit: 20,
  offset: 0,
}

// GET /api/batches?limit=…&offset=… — a regex so the query string is matched without a glob.
const listUrl = /\/api\/batches\?limit=/

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

async function mockList(page: Page, body: unknown) {
  await page.route(listUrl, route => route.fulfill({ json: body }))
}

test('lists batches newest first with file count and status', async ({ page }) => {
  await mockList(page, listed)
  await page.goto('/')
  const rows = page.getByRole('list', { name: 'Batch history list' }).getByRole('listitem')
  await expect(rows).toHaveCount(2)
  await expect(rows.nth(0)).toContainText('3 files')
  await expect(rows.nth(0)).toContainText('Finished with failures')
  await expect(rows.nth(1)).toContainText('1 file')
  await expect(rows.nth(1)).toContainText('Completed')
  await expect(page.getByText('Showing 2 of 2')).toBeVisible()
})

test('shows a clear empty state when there are no batches', async ({ page }) => {
  await mockList(page, { batches: [], total: 0, limit: 20, offset: 0 })
  await page.goto('/')
  await expect(page.getByText('No batches yet.')).toBeVisible()
  await expect(page.getByRole('list', { name: 'Batch history list' })).toHaveCount(0)
})

test('asks for confirmation before deleting a batch, then removes it', async ({ page }) => {
  await mockList(page, listed)
  const deleted: string[] = []
  await page.route('**/api/batches/hist-old', route => {
    deleted.push(route.request().method())
    return route.fulfill({ json: { id: 'hist-old', deleted: true } })
  })
  await page.goto('/')
  const oldRow = page.getByRole('listitem').filter({ hasText: '1 file' })
  await oldRow.getByRole('button', { name: 'Delete', exact: true }).click()

  const dialog = page.getByRole('dialog', { name: 'Delete batch?' })
  await expect(dialog).toBeVisible()
  await dialog.getByRole('button', { name: 'Cancel' }).click()
  await expect(dialog).toHaveCount(0)
  expect(deleted).toHaveLength(0)
  await expect(oldRow).toBeVisible()

  await oldRow.getByRole('button', { name: 'Delete', exact: true }).click()
  await page.getByRole('dialog', { name: 'Delete batch?' }).getByRole('button', { name: 'Delete batch' }).click()
  await expect(page.getByText('Batch deleted from this Mac.')).toBeVisible()
  await expect(page.getByRole('listitem').filter({ hasText: '1 file' })).toHaveCount(0)
  await expect(page.getByRole('listitem')).toHaveCount(1)
  expect(deleted).toEqual(['DELETE'])
})

test('shows readable text when the batch list cannot be fetched, and retries', async ({ page }) => {
  let online = false
  await page.route(listUrl, route => online ? route.fulfill({ json: listed }) : route.abort())
  await page.goto('/')
  const error = page.locator('.batch-history-error')
  await expect(error).toContainText('Can’t reach the local transcription service')
  await expect(error).not.toContainText('Failed to fetch')
  online = true
  await error.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByRole('list', { name: 'Batch history list' }).getByRole('listitem')).toHaveCount(2)
  await expect(error).toHaveCount(0)
})

test('reopening a failed batch asks for its files again instead of submitting', async ({ page }) => {
  await mockList(page, listed)
  const opened = {
    id: 'hist-new', total: 3, done: true, counts: counts({ completed: 2, failed: 1 }),
    jobs: [
      { id: 'a', filename: 'one.wav', status: 'completed', segments: [], info: null, error: null, progress: null, batch_id: 'hist-new' },
      { id: 'b', filename: 'two.wav', status: 'failed', segments: [], info: null, error: 'Could not decode', progress: null, batch_id: 'hist-new' },
      { id: 'c', filename: 'three.wav', status: 'completed', segments: [], info: null, error: null, progress: null, batch_id: 'hist-new' },
    ],
  }
  await page.route('**/api/batches/hist-new', route => route.fulfill({ json: opened }))
  const posted: string[] = []
  await page.route('**/api/batches', async route => {
    if (route.request().method() !== 'POST') return route.fallback()
    posted.push(route.request().postData() || '')
    await route.fulfill({ status: 202, json: { ...opened, id: 'hist-retry', total: 1, counts: counts({ completed: 1 }), jobs: [{ ...opened.jobs[1], id: 'retry-b', status: 'completed', error: null }] } })
  })
  await page.goto('/')
  await page.getByRole('button', { name: 'Open · 1 failed' }).click()
  await expect(page.getByText('Select the failed files again to retry.')).toBeVisible()
  await expect(page.getByRole('button', { name: /Retry failed/ })).toHaveCount(0)
  expect(posted).toHaveLength(0)

  await page.getByLabel('Failed batch files').setInputFiles({ name: 'two.wav', mimeType: 'audio/wav', buffer: syntheticWav() })
  await page.getByRole('button', { name: 'Retry failed (1)' }).click()
  await expect(page.getByText('1 of 1 complete')).toBeVisible()
  expect(posted).toHaveLength(1)
  expect(posted[0].match(/name="files"/g)).toHaveLength(1)
  expect(posted[0]).toContain('filename="two.wav"')
})
