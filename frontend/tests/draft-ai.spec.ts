import { expect, test, type Page } from '@playwright/test'
import { mockHealthAndModels, syntheticWav } from './helpers'

const transcript = [{ start: 0, end: 2, text: 'hello world' }, { start: 2, end: 4, text: 'second line' }]

async function seedDraft(page: Page) {
  await mockHealthAndModels(page)
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-draft-ai', status: 'completed', segments: transcript } })
  })
  await page.goto('/')
  await page.locator('input[type="file"][accept^="audio"]:not([multiple])').setInputFiles({
    name: 'draft.wav', mimeType: 'audio/wav', buffer: syntheticWav(),
  })
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' }).fill('edited')
  await expect.poll(() => page.evaluate(() => window.localStorage.getItem('sttStudio.draft') ?? '')).toContain('edited')
  await page.reload()
  await page.getByRole('button', { name: 'Restore', exact: true }).click()
}

test('disables AI after restoring a local draft', async ({ page }) => {
  await seedDraft(page)
  await expect(page.getByText('AI runs on a saved job. Open it from History.')).toBeVisible()
  await expect(page.getByRole('radio', { name: 'Summary' })).toBeDisabled()
})