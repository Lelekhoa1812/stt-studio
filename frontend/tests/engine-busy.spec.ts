import { expect, test } from '@playwright/test'
import { defaultCapabilities, mockHealthAndModels, syntheticWav } from './helpers'

test('disables Transcribe when active jobs reach the limit', async ({ page }) => {
  await mockHealthAndModels(page)
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok', active_jobs: 1, loaded_models: 0 } }))
  await page.route('**/api/capabilities', route =>
    route.fulfill({ json: { ...defaultCapabilities, limits: { beam_size: [1, 20], max_batch_files: 25, max_active_jobs: 1 } } }))
  await page.goto('/')
  await page.locator('input[type="file"][accept^="audio"]:not([multiple])').setInputFiles({ name: 'busy.wav', mimeType: 'audio/wav', buffer: syntheticWav() })
  await expect(page.getByRole('button', { name: 'Engine busy' })).toBeDisabled()
})