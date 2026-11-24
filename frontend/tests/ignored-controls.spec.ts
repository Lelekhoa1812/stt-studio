import { expect, test } from '@playwright/test'
import { chooseSyntheticAudio, mockHealthAndModels } from './helpers'

const transcript = [{ start: 0, end: 1, text: 'hello' }]

test('shows a notice when the job reports ignored controls', async ({ page }) => {
  await mockHealthAndModels(page)
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-ignored-1', status: 'queued', ignored_controls: ['beam_size'] } })
  })
  await page.route('**/api/jobs/job-ignored-1', route =>
    route.fulfill({ json: { id: 'job-ignored-1', status: 'completed', ignored_controls: ['beam_size'], segments: transcript, info: { language: 'en' } } }))
  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByText('This model ignored beam size.')).toBeVisible()
})