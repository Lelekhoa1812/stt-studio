import { expect, test } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('accepts a .mov file and rejects a .txt file', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('heading', { name: 'Choose an audio file' }).waitFor()
  const single = page.locator('input[type="file"][accept^="audio"]:not([multiple])')

  await single.setInputFiles({ name: 'clip.mov', mimeType: 'video/quicktime', buffer: Buffer.from('not-a-real-mov') })
  await expect(page.getByText('clip.mov').first()).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)

  await single.setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('plain text') })
  await expect(page.getByRole('alert')).toContainText(/audio|video/i)
  await expect(page.getByText('notes.txt')).toHaveCount(0)
})