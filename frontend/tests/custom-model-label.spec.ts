import { expect, test } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('the custom repository input is labelled and can add a model', async ({ page }) => {
  await page.goto('/')
  await page.locator('summary').filter({ hasText: 'Advanced options' }).click()

  const input = page.getByLabel('Custom model repository')
  await expect(input).toHaveCount(1)
  await expect(input).toBeVisible()

  await input.fill('team/custom-whisper')
  await page.getByRole('button', { name: 'Add', exact: true }).click()
  await expect(page.getByRole('combobox', { name: /^MODEL/ })).toHaveValue('team/custom-whisper')
})

test('the MODEL select keeps one match next to the custom input', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('combobox', { name: /^MODEL/ })).toHaveCount(1)
  await expect(page.getByLabel('Custom model repository')).toHaveCount(1)
})
