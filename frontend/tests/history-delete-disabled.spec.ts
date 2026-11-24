import { expect, test } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

test('delete is disabled for a queued saved transcript', async ({ page }) => {
  await mockHealthAndModels(page)
  await page.route('**/api/storage', route => route.fulfill({ json: { jobs: 1, batches: 0, bytes: 0 } }))
  await page.route(/\/api\/jobs\?/, route =>
    route.fulfill({ json: { jobs: [{ id: 'job-q1', status: 'queued', model: 'small', filename: 'wait.wav', created_at: 1, language: null, batch_id: null, info: null }], total: 1, limit: 50, offset: 0 } }))
  await page.goto('/')
  await page.getByRole('button', { name: 'History', exact: true }).click()
  const row = page.getByRole('list', { name: 'Saved transcripts list' }).getByRole('listitem').first()
  await expect(row.getByRole('button', { name: 'Delete wait.wav' })).toBeDisabled()
})