import { expect, test } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

test('beam size slider max follows capabilities limit', async ({ page }) => {
  await mockHealthAndModels(page)
  await page.goto('/')
  await page.getByText('Advanced options').click()
  await expect(page.locator('.advanced-body input[type="range"]')).toHaveAttribute('max', '20')
})