import { expect, test } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('the Float16 option and its helper text do not promise GPU speed on every Mac', async ({ page }) => {
  await page.goto('/')
  await page.locator('summary').filter({ hasText: 'Advanced options' }).click()

  const compute = page.getByLabel('COMPUTE TYPE')
  await expect(compute.locator('option', { hasText: 'Float16 · faster on NVIDIA GPUs' })).toHaveCount(1)
  await expect(page.getByText('faster on GPU', { exact: false })).toHaveCount(0)

  // The helper text only appears once Float16 is selected, so Int8 users do not see it.
  await expect(page.getByText('Float16 is faster on NVIDIA GPUs and may be slower on Apple Silicon CPU runs.')).toHaveCount(0)
  await compute.selectOption('float16')
  await expect(page.getByText('Float16 is faster on NVIDIA GPUs and may be slower on Apple Silicon CPU runs.')).toBeVisible()

  await compute.selectOption('int8')
  await expect(page.getByText('Float16 is faster on NVIDIA GPUs and may be slower on Apple Silicon CPU runs.')).toHaveCount(0)
})
