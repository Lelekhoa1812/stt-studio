import { expect, test, type Page } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

// A generated, silent WAV of the given length. Silence keeps the peaks flat, which is enough to prove the canvas draws.
function silentWav(seconds: number): Buffer {
  const sampleRate = 8_000
  const data = Buffer.alloc(sampleRate * seconds * 2)
  const header = Buffer.alloc(44)
  header.write('RIFF', 0)
  header.writeUInt32LE(36 + data.length, 4)
  header.write('WAVE', 8)
  header.write('fmt ', 12)
  header.writeUInt32LE(16, 16)
  header.writeUInt16LE(1, 20)
  header.writeUInt16LE(1, 22)
  header.writeUInt32LE(sampleRate, 24)
  header.writeUInt32LE(sampleRate * 2, 28)
  header.writeUInt16LE(2, 32)
  header.writeUInt16LE(16, 34)
  header.write('data', 36)
  header.writeUInt32LE(data.length, 40)
  return Buffer.concat([header, data])
}

async function loadAudio(page: Page, name: string, buffer: Buffer) {
  await mockHealthAndModels(page)
  await page.goto('/')
  await page.locator('input[type="file"][accept^="audio"]:not([multiple])').setInputFiles({ name, mimeType: 'audio/wav', buffer })
  await expect(page.getByText(name).first()).toBeVisible()
}

const audioTime = (page: Page) => page.locator('audio').evaluate(el => (el as HTMLAudioElement).currentTime)

test('the waveform canvas draws the decoded audio and a click seeks the audio element', async ({ page }) => {
  await loadAudio(page, 'ten-seconds.wav', silentWav(10))

  const timeline = page.getByRole('img', { name: /Waveform timeline/ })
  await expect(page.locator('.waveform-timeline')).toHaveAttribute('data-state', 'ready')
  await expect(timeline).toBeVisible()

  // The canvas holds drawn pixels, not just its blank backing store.
  const drawn = await page.locator('canvas').first().evaluate(canvas => {
    const element = canvas as HTMLCanvasElement
    const data = element.getContext('2d')!.getImageData(0, 0, element.width, element.height).data
    let painted = 0
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) painted++
    return { width: element.width, height: element.height, painted }
  })
  expect(drawn.width).toBeGreaterThan(0)
  expect(drawn.painted).toBeGreaterThan(0)

  const box = await timeline.boundingBox()
  expect(box).not.toBeNull()
  await page.mouse.click(box!.x + box!.width * 0.5, box!.y + box!.height / 2)
  await expect.poll(() => audioTime(page)).toBeCloseTo(5, 0)
})

test('dragging across the waveform seeks along the drag', async ({ page }) => {
  await loadAudio(page, 'drag-seconds.wav', silentWav(10))
  const timeline = page.getByRole('img', { name: /Waveform timeline/ })
  await expect(timeline).toBeVisible()

  const box = await timeline.boundingBox()
  expect(box).not.toBeNull()
  const y = box!.y + box!.height / 2
  await page.mouse.move(box!.x + box!.width * 0.2, y)
  await page.mouse.down()
  await page.mouse.move(box!.x + box!.width * 0.8, y, { steps: 5 })
  await page.mouse.up()
  await expect.poll(() => audioTime(page)).toBeCloseTo(8, 0)
})

test('an undecodable file shows Waveform unavailable and the rest of the page still transcribes', async ({ page }) => {
  await mockHealthAndModels(page)
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-wave-1', status: 'completed', segments: [{ start: 0, end: 1, text: 'still works' }], info: { language: 'en', duration: 1 } } })
  })
  await page.goto('/')
  await page.locator('input[type="file"][accept^="audio"]:not([multiple])').setInputFiles({ name: 'broken.wav', mimeType: 'audio/wav', buffer: Buffer.from('this is not audio data') })
  await expect(page.getByText('Waveform unavailable')).toBeVisible()
  await expect(page.locator('.waveform-timeline')).toHaveAttribute('data-state', 'failed')

  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('still works')
})
