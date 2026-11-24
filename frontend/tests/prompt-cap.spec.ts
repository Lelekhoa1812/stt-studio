import { expect, test } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

test('vocabulary hint textarea is capped at 1000 characters', async ({ page }) => {
  await mockHealthAndModels(page)
  await page.goto('/')
  await page.getByText('Advanced options').click()
  await expect(page.getByPlaceholder(/Names, technical terms/)).toHaveAttribute('maxlength', '1000')
})