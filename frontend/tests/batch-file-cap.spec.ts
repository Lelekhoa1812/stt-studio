import { expect, test } from '@playwright/test'
import { defaultCapabilities, mockHealthAndModels, syntheticWav } from './helpers'

const file = (name: string) => ({ name, mimeType: 'audio/wav', buffer: syntheticWav() })

test('batch queue rejects files above the capabilities limit', async ({ page }) => {
  await mockHealthAndModels(page)
  await page.route('**/api/capabilities', route =>
    route.fulfill({ json: { ...defaultCapabilities, limits: { beam_size: [1, 20], max_batch_files: 2 } } }))
  await page.goto('/')
  const input = page.locator('input[type="file"][multiple]')
  await input.setInputFiles([file('one.wav'), file('two.wav'), file('three.wav')])
  await expect(page.getByText(/at most 2 files/)).toBeVisible()
  await expect(page.getByRole('list', { name: 'Files queued for the batch' }).getByRole('listitem')).toHaveCount(2)
})