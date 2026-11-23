import { expect, test, type Page } from '@playwright/test'
import { chooseSyntheticAudio, mockHealthAndModels } from './helpers'

const capabilities = {
  service: 'stt-studio',
  diarization: { available: false, model: 'pyannote/speaker-diarization-3.1', license: 'MIT', reason: 'Not needed for these tests.' },
}
const transcript = [{ start: 0.25, end: 1.5, text: 'hello world' }]

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
  await page.route('**/api/capabilities', route => route.fulfill({ json: capabilities }))
})

// Resolves the rendered colour of each visible element against the effective background,
// compositing semi-transparent ancestors and defaulting to white. Runs inside the page.
async function contrastSamples(page: Page, selector: string, placeholder = false) {
  return page.evaluate(({ selector, placeholder }) => {
    type RGBA = { r: number; g: number; b: number; a: number }
    const parse = (value: string): RGBA | null => {
      const m = value.match(/rgba?\(([^)]+)\)/)
      if (!m) return null
      const parts = m[1].split(/[\s,/]+/).filter(Boolean).map(Number)
      return { r: parts[0], g: parts[1], b: parts[2], a: parts.length > 3 ? parts[3] : 1 }
    }
    const over = (top: RGBA, below: RGBA): RGBA => ({
      r: top.r * top.a + below.r * (1 - top.a),
      g: top.g * top.a + below.g * (1 - top.a),
      b: top.b * top.a + below.b * (1 - top.a),
      a: 1,
    })
    const background = (el: Element): RGBA => {
      const layers: RGBA[] = []
      for (let node: Element | null = el; node; node = node.parentElement) {
        const color = parse(getComputedStyle(node).backgroundColor)
        if (color && color.a > 0) {
          layers.push(color)
          if (color.a >= 1) break
        }
      }
      let base: RGBA = { r: 255, g: 255, b: 255, a: 1 }
      for (const layer of layers.reverse()) base = over(layer, base)
      return base
    }
    const luminance = ({ r, g, b }: RGBA) => {
      const channel = (c: number) => {
        const s = c / 255
        return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
      }
      return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    }
    const hex = ({ r, g, b }: RGBA) => `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`
    return Array.from(document.querySelectorAll<HTMLElement>(selector))
      .filter(el => el.getClientRects().length > 0)
      .map(el => {
        const bg = background(el)
        const textSource = getComputedStyle(el, placeholder ? '::placeholder' : undefined).color
        const fgRaw = parse(textSource)!
        const fg = fgRaw.a < 1 ? over(fgRaw, bg) : fgRaw
        const [l1, l2] = [luminance(fg), luminance(bg)].sort((a, b) => b - a)
        return {
          selector,
          text: (placeholder ? el.getAttribute('placeholder') : el.textContent?.trim()) || selector,
          color: hex(fg),
          background: hex(bg),
          ratio: Number(((l1 + 0.05) / (l2 + 0.05)).toFixed(2)),
        }
      })
  }, { selector, placeholder })
}

async function expectAa(page: Page, selector: string, options: { placeholder?: boolean } = {}) {
  const samples = await contrastSamples(page, selector, options.placeholder)
  expect(samples.length, `${selector} should be visible`).toBeGreaterThan(0)
  for (const sample of samples) {
    expect(sample.ratio, `${sample.selector} "${sample.text}" ${sample.color} on ${sample.background}`).toBeGreaterThanOrEqual(4.5)
  }
}

async function mockCompletedJob(page: Page) {
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-a11y-1', status: 'completed', segments: transcript, info: { language: 'en', duration: 3 } } })
  })
}

test('muted labels, field hints and placeholders meet 4.5:1 contrast', async ({ page }) => {
  await page.goto('/')
  await expectAa(page, '.side-label')
  await expectAa(page, '.field-label')
  await page.locator('summary').filter({ hasText: 'Advanced options' }).click()
  await expectAa(page, '.field-label em')
  await expectAa(page, '.prompt-field textarea', { placeholder: true })
  await expectAa(page, '.custom-model-input input', { placeholder: true })

  await mockCompletedJob(page)
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  await expectAa(page, '.edit-label')
  await expectAa(page, '.search-box input', { placeholder: true })
})

test('progress details meet 4.5:1 contrast while a job runs', async ({ page }) => {
  const jobId = 'job-a11y-running'
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: jobId, status: 'queued' } })
  })
  await page.route(`**/api/jobs/${jobId}`, route => route.fulfill({ json: { id: jobId, status: 'running', progress: 40, segments: [] } }))
  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.locator('.progress-card')).toBeVisible()
  await expectAa(page, '.progress-bottom span')
})

test('each dialog labels itself with a unique title id', async ({ page }) => {
  await page.goto('/')
  // The two "Replace the current workspace?" dialogs are both mounted, so their titles used to share one id.
  const replaceDialogs = page.getByRole('dialog', { name: 'Replace the current workspace?', includeHidden: true })
  await expect(replaceDialogs).toHaveCount(2)

  const labelling = await page.evaluate(() => Array.from(document.querySelectorAll('dialog')).map(dialog => {
    const labelId = dialog.getAttribute('aria-labelledby') || ''
    const matches = Array.from(document.querySelectorAll(`[id="${CSS.escape(labelId)}"]`))
    const ids = Array.from(document.querySelectorAll('[id]')).map(el => el.id)
    return {
      title: dialog.querySelector('h2')?.textContent ?? '',
      labelId,
      resolvesTo: matches.length,
      resolvedText: matches[0]?.textContent ?? '',
      idIsUnique: ids.filter(id => id === labelId).length === 1,
    }
  }))
  expect(labelling.length).toBeGreaterThan(0)
  for (const entry of labelling) {
    expect(entry.labelId, 'dialog must have aria-labelledby').not.toBe('')
    expect(entry.resolvesTo, `${entry.labelId} should resolve to exactly one element`).toBe(1)
    expect(entry.resolvedText).toBe(entry.title)
    expect(entry.idIsUnique).toBe(true)
  }
  const allIds = await page.evaluate(() => Array.from(document.querySelectorAll('[id]')).map(el => el.id))
  expect(new Set(allIds).size).toBe(allIds.length)
})

test('a dialog traps Tab, closes on Escape and returns focus to its trigger', async ({ page }) => {
  await page.goto('/')
  await chooseSyntheticAudio(page)
  const trigger = page.getByRole('button', { name: 'Remove selected audio' })
  await trigger.focus()
  await trigger.click()

  const dialog = page.getByRole('dialog', { name: 'Remove this audio?' })
  await expect(dialog).toBeVisible()
  const focusInside = () => page.evaluate(() => Boolean(document.activeElement?.closest('dialog[open]')))

  expect(await focusInside()).toBe(true)
  for (let i = 0; i < 25; i++) {
    await page.keyboard.press('Tab')
    expect(await focusInside(), `focus escaped the dialog after Tab ${i + 1}`).toBe(true)
  }
  for (let i = 0; i < 5; i++) {
    await page.keyboard.press('Shift+Tab')
    expect(await focusInside(), `focus escaped the dialog after Shift+Tab ${i + 1}`).toBe(true)
  }

  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
  await expect(trigger).toBeFocused()
})
