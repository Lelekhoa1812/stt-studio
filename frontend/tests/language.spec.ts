import { expect, test, type Page } from '@playwright/test'
import { mockHealthAndModels, syntheticWav } from './helpers'

const DEFAULT_KEY = 'sttStudio.defaultLanguage'
const segments = [{ start: 0.25, end: 1.5, text: 'xin chào' }]

function projectFile(language: string): Buffer {
  const audio = syntheticWav()
  const manifest = {
    format: 'stt-studio-project', version: 1, name: 'Language project', createdAt: 1_700_000_000_000, savedAt: 1_700_000_100_000,
    exportedAt: 1_700_000_200_000, duration: 0.25,
    audio: { fileName: 'synthetic-silence.wav', mimeType: 'audio/wav', bytes: audio.length },
    settings: { model: 'small', compute: 'int8', language, beam: 5, vad: true, prompt: '' },
    segments: [],
  }
  const json = Buffer.from(JSON.stringify(manifest))
  const header = Buffer.alloc(12)
  header.write('STTPRJ01', 0)
  header.writeUInt32BE(json.length, 8)
  return Buffer.concat([header, json, audio])
}

function languageSelect(page: Page) {
  return page.locator('.config-panel label.field').filter({ has: page.locator('.field-label', { hasText: /^LANGUAGE$/ }) }).locator('select')
}

async function savedDefault(page: Page) {
  return page.evaluate(key => window.localStorage.getItem(key), DEFAULT_KEY)
}

async function setSavedDefault(page: Page, code: string) {
  await page.goto('/')
  await page.evaluate(([key, value]) => window.localStorage.setItem(key, value), [DEFAULT_KEY, code])
  await page.reload()
}

async function mockJobs(page: Page, detected: string[]) {
  const bodies: string[] = []
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    bodies.push(route.request().postData() || '')
    const language = detected[Math.min(bodies.length - 1, detected.length - 1)]
    await route.fulfill({ status: 202, json: { id: `job-language-${bodies.length}`, status: 'completed', segments, info: { language, duration: 0.25 } } })
  })
  return bodies
}

async function importProject(page: Page, language: string) {
  await page.locator('input[aria-label="Project file to import"]').setInputFiles({
    name: 'language.sttproj', mimeType: 'application/octet-stream', buffer: projectFile(language),
  })
  await expect(page.getByText('Imported “Language project” from a project file.')).toBeVisible()
}

async function transcribe(page: Page) {
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('xin chào')
}

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('starts the language select at the saved default, or auto detect when none is saved', async ({ page }) => {
  await page.goto('/')
  await expect(languageSelect(page)).toHaveValue('auto')
  await setSavedDefault(page, 'vi')
  await expect(languageSelect(page)).toHaveValue('vi')
  await expect(page.getByText('Saved as your default')).toHaveCount(0)
})

test('saves a changed language as the default for later visits', async ({ page }) => {
  await page.goto('/')
  await languageSelect(page).selectOption('vi')
  await expect(page.getByText('Saved as your default for later visits.')).toBeVisible()
  expect(await savedDefault(page)).toBe('vi')

  await page.reload()
  await expect(languageSelect(page)).toHaveValue('vi')
  await languageSelect(page).selectOption('auto')
  await expect(page.getByText('Saved as your default. Auto-detect can pick the wrong language.')).toBeVisible()
  expect(await savedDefault(page)).toBe('auto')
})

test('warns when auto detect picks another language and re-runs the same file in the default language', async ({ page }) => {
  await setSavedDefault(page, 'vi')
  const bodies = await mockJobs(page, ['en', 'vi'])
  await importProject(page, 'auto')
  await expect(languageSelect(page)).toHaveValue('auto')
  await transcribe(page)

  await expect(page.getByText('Auto-detect chose English, but your default is Vietnamese. Check the transcript, or re-run in Vietnamese.')).toBeVisible()
  expect(bodies[0]).not.toContain('name="language"')

  await page.getByRole('button', { name: 'Re-run in Vietnamese' }).click()
  await page.getByRole('button', { name: 'Discard edits and run again' }).click()
  await expect(page.getByText('Auto-detect chose')).toHaveCount(0)
  expect(bodies).toHaveLength(2)
  expect(bodies[1]).toMatch(/name="language"\r\n\r\nvi\r\n/)
  await expect(languageSelect(page)).toHaveValue('vi')
})

test('shows no warning when the run used an explicit language', async ({ page }) => {
  await setSavedDefault(page, 'vi')
  const bodies = await mockJobs(page, ['fr'])
  await importProject(page, 'fr')
  await transcribe(page)
  expect(bodies[0]).toMatch(/name="language"\r\n\r\nfr\r\n/)
  await expect(page.getByText('Auto-detect chose')).toHaveCount(0)
  await expect(page.getByRole('button', { name: /Re-run in/ })).toHaveCount(0)
})

test('shows no warning when the detected language matches the default', async ({ page }) => {
  await setSavedDefault(page, 'vi')
  await mockJobs(page, ['vi'])
  await importProject(page, 'auto')
  await transcribe(page)
  await expect(page.getByText('Auto-detect chose')).toHaveCount(0)
  await expect(page.getByRole('button', { name: /Re-run in/ })).toHaveCount(0)
})

test('shows the detected language after an auto-detect run that matches the default', async ({ page }) => {
  await setSavedDefault(page, 'vi')
  await mockJobs(page, ['vi'])
  await importProject(page, 'auto')
  await transcribe(page)
  await expect(page.getByText('Detected: Vietnamese')).toBeVisible()
  await expect(page.getByText('Auto-detect chose')).toHaveCount(0)
})

test('shows no warning when the default is auto detect', async ({ page }) => {
  await page.goto('/')
  await mockJobs(page, ['vi'])
  await importProject(page, 'auto')
  await transcribe(page)
  await expect(page.getByText('Auto-detect chose')).toHaveCount(0)
  await expect(page.getByRole('button', { name: /Re-run in/ })).toHaveCount(0)
})
