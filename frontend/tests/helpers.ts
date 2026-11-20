import { expect, type Page } from '@playwright/test'

export function syntheticWav(): Buffer {
  // A short, generated mono PCM WAV. It contains silence and no user audio.
  const sampleRate = 8_000
  const sampleCount = sampleRate / 4
  const data = Buffer.alloc(sampleCount * 2)
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

export async function mockHealthAndModels(page: Page) {
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok', active_jobs: 0, loaded_models: 0 } }))
  await page.route('**/api/models', route => route.fulfill({
    json: { default_model: 'small', compute_types: ['int8', 'float16', 'float32'], cached_models: [], loaded_models: [] },
  }))
}

export async function chooseSyntheticAudio(page: Page) {
  await page.getByRole('heading', { name: 'Choose an audio file' }).waitFor()
  await page.locator('input[type="file"][accept^="audio"]:not([multiple])').setInputFiles({
    name: 'synthetic-silence.wav',
    mimeType: 'audio/wav',
    buffer: syntheticWav(),
  })
  await expect(page.getByText('synthetic-silence.wav').first()).toBeVisible()
}
