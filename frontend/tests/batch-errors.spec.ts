import { expect, test } from '@playwright/test'
import { mockHealthAndModels, syntheticWav } from './helpers'

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('an aborted batch request shows readable text, not the browser wording', async ({ page }) => {
  await page.route('**/api/batches', route => route.request().method() === 'POST' ? route.abort('aborted') : route.fallback())
  await page.goto('/')
  await page.getByLabel('Batch audio files').setInputFiles({ name: 'aborted.wav', mimeType: 'audio/wav', buffer: syntheticWav() })
  await page.getByRole('button', { name: 'Start batch (1)' }).click()

  const alert = page.locator('.batch-panel').getByRole('alert')
  await expect(alert).toContainText('Can’t reach the local transcription service')
  await expect(alert).not.toContainText('Failed to fetch')
  await expect(alert).not.toContainText('aborted')
  await expect(page.getByRole('button', { name: 'Start batch (1)' })).toBeEnabled()
})

test('an unreadable batch reply shows a readable message instead of a parser error', async ({ page }) => {
  await page.route('**/api/batches', route => route.request().method() === 'POST' ? route.fulfill({ status: 202, body: '<html>not json</html>', contentType: 'text/html' }) : route.fallback())
  await page.goto('/')
  await page.getByLabel('Batch audio files').setInputFiles({ name: 'reply.wav', mimeType: 'audio/wav', buffer: syntheticWav() })
  await page.getByRole('button', { name: 'Start batch (1)' }).click()

  const alert = page.locator('.batch-panel').getByRole('alert')
  await expect(alert).toContainText('The local service sent a reply that could not be read.')
  await expect(alert).not.toContainText('Unexpected token')
})

test('saved transcripts keep their readable network message after the shared helper change', async ({ page }) => {
  await page.route(/\/api\/jobs\?/, route => route.abort('aborted'))
  await page.route('**/api/storage', route => route.fulfill({ json: { data_dir: 'data', jobs: 0, batches: 0, bytes: 0 } }))
  await page.goto('/')
  await page.getByRole('button', { name: 'History', exact: true }).click()
  const alert = page.locator('.history-panel').getByRole('alert')
  await expect(alert).toContainText('Could not load saved transcripts')
  await expect(alert).toContainText('Can’t reach the local transcription service')
  await expect(alert).not.toContainText('Failed to fetch')
})
