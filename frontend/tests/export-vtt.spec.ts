import { readFile } from 'node:fs/promises'
import { expect, test, type Route } from '@playwright/test'
import { chooseSyntheticAudio, mockHealthAndModels } from './helpers'

const transcript = [
  { start: 0.25, end: 1.5, text: 'hello world', avg_logprob: -0.1, no_speech_prob: 0.01 },
  { start: 1.75, end: 3, text: 'local speech recognition', avg_logprob: -0.2, no_speech_prob: 0.02 },
]

async function completeJob(route: Route, pollCount: { value: number }) {
  pollCount.value += 1
  if (pollCount.value === 1) {
    await route.fulfill({ json: { id: 'job-vtt-1', status: 'running' } })
    return
  }
  await route.fulfill({ json: { id: 'job-vtt-1', status: 'completed', segments: transcript,
    info: { language: 'en', duration: 3 } } })
}

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('exports a WebVTT caption file from the Export menu', async ({ page }) => {
  const polls = { value: 0 }
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-vtt-1', status: 'queued' } })
  })
  await page.route('**/api/jobs/job-vtt-1', route => completeJob(route, polls))

  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')

  await page.getByRole('button', { name: 'Export' }).click()
  await expect(page.getByRole('menu')).toBeVisible()
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('menuitem', { name: /WebVTT captions \(\.vtt\)/ }).click(),
  ])
  expect(download.suggestedFilename()).toBe('synthetic-silence.vtt')
  expect(download.suggestedFilename()).toMatch(/\.vtt$/)
  const path = await download.path()
  expect(path).not.toBeNull()
  const contents = await readFile(path!, 'utf8')
  expect(contents.startsWith('WEBVTT')).toBe(true)
  expect(contents).toMatch(/^\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}$/m)
  expect(contents).toContain('00:00:00.250 --> 00:00:01.500\nhello world\n')
})
