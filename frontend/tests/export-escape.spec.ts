import { expect, test } from '@playwright/test'
import { chooseSyntheticAudio, mockHealthAndModels } from './helpers'

const transcript = [{ start: 0, end: 1.5, text: 'hello world' }]

test('Escape closes Export without clearing transcript search', async ({ page }) => {
  await mockHealthAndModels(page)
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-export-esc', status: 'completed', segments: transcript, info: { language: 'en' } } })
  })
  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  await page.getByRole('textbox', { name: 'Search transcript' }).fill('hello')
  await page.getByRole('button', { name: 'Export' }).click()
  await expect(page.getByRole('menu')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('menu')).toHaveCount(0)
  await expect(page.getByRole('textbox', { name: 'Search transcript' })).toHaveValue('hello')
})