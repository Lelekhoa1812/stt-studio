import { expect, test } from '@playwright/test'
import { mockHealthAndModels, syntheticWav } from './helpers'

type Job = { id: string; filename: string; status: string; error?: string | null }

function batch(id: string, jobs: Job[]) {
  const counts = { queued: 0, running: 0, completed: 0, failed: 0, cancelled: 0 }
  for (const job of jobs) counts[job.status as keyof typeof counts] += 1
  return {
    id, total: jobs.length, done: true, counts,
    jobs: jobs.map(job => ({ progress: null, segments: [], info: null, batch_id: id, ...job })),
  }
}

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('retries only the failed file of a two-file batch', async ({ page }) => {
  const posted: string[] = []
  const first = batch('batch-retry-1', [
    { id: 'j-one', filename: 'one.wav', status: 'completed' },
    { id: 'j-two', filename: 'two.wav', status: 'failed', error: 'Audio decoder could not read this file.' },
  ])
  const retry = batch('batch-retry-2', [{ id: 'j-two-retry', filename: 'two.wav', status: 'completed' }])
  await page.route('**/api/batches', async route => {
    if (route.request().method() !== 'POST') return route.fallback()
    posted.push(route.request().postData() || '')
    await route.fulfill({ status: 202, json: posted.length === 1 ? first : retry })
  })
  await page.goto('/')
  await page.getByLabel('Batch audio files').setInputFiles([
    { name: 'one.wav', mimeType: 'audio/wav', buffer: syntheticWav() },
    { name: 'two.wav', mimeType: 'audio/wav', buffer: syntheticWav() },
  ])
  await page.getByRole('button', { name: 'Start batch (2)' }).click()
  await expect(page.getByText('1 of 2 complete')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Retry failed (1)' })).toBeVisible()

  await page.getByRole('button', { name: 'Retry failed (1)' }).click()
  await expect(page.getByText('1 of 1 complete')).toBeVisible()
  expect(posted).toHaveLength(2)
  expect(posted[1].match(/name="files"/g)).toHaveLength(1)
  expect(posted[1]).toContain('filename="two.wav"')
  expect(posted[1]).not.toContain('filename="one.wav"')
})

test('shows no retry control when every file in the batch succeeded', async ({ page }) => {
  await page.route('**/api/batches', route => route.fulfill({ status: 202, json: batch('batch-ok-1', [
    { id: 'j-ok-1', filename: 'ok.wav', status: 'completed' },
  ]) }))
  await page.goto('/')
  await page.getByLabel('Batch audio files').setInputFiles({ name: 'ok.wav', mimeType: 'audio/wav', buffer: syntheticWav() })
  await page.getByRole('button', { name: 'Start batch (1)' }).click()
  await expect(page.getByRole('button', { name: 'New batch' })).toBeVisible()
  await expect(page.getByRole('button', { name: /Retry failed/ })).toHaveCount(0)
})
