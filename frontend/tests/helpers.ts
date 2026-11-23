import { expect, type Page } from '@playwright/test'

const LOCAL_HOSTS = new Set(['127.0.0.1', 'localhost'])

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

// Shape of GET /api/capabilities as backend/app.py returns it. The App reads diarization and ai from it on mount.
export const defaultCapabilities = {
  service: 'stt-studio',
  api_version: 'test',
  default_model: 'small',
  compute_types: ['int8', 'float16', 'float32'],
  diarization: {
    available: false,
    model: 'pyannote/speaker-diarization-3.1',
    license: 'MIT',
    reason: 'HF_TOKEN is not set; add it to .env (the account must have accepted the model terms).',
  },
  ai: {
    available: false,
    model: 'nvidia/nemotron-3-ultra-550b-a55b',
    modes: ['summary', 'action_items'],
    reason: 'AI modes need an NVIDIA NIM key. Add NVIDIA_NIM_API_KEY_1 to .env or the environment to turn them on.',
  },
}

// Mocks the calls the App makes on mount, and fails any other local API call. The catch-all is registered first,
// so routes a spec registers later take precedence; it only answers requests that nothing else mocked. Specs must
// not reach a real backend on 127.0.0.1:8765: a missing mock fails the test with the request it could not answer.
export async function mockHealthAndModels(page: Page) {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    // Other hosts (for example avatar lookups) keep their normal behaviour.
    if (!LOCAL_HOSTS.has(url.hostname)) return route.fallback()
    const request = `${route.request().method()} ${url.pathname}`
    await route.fulfill({ status: 599, json: { detail: `Unmocked API request: ${request}` } })
    throw new Error(`Unmocked API request: ${request}. Add a page.route mock for it in the spec.`)
  })
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok', active_jobs: 0, loaded_models: 0 } }))
  await page.route('**/api/models', route => route.fulfill({
    json: { default_model: 'small', compute_types: ['int8', 'float16', 'float32'], cached_models: [], loaded_models: [] },
  }))
  await page.route('**/api/capabilities', route => route.fulfill({ json: defaultCapabilities }))
  // The batch panel lists batch history on mount. Specs that need rows register their own route after this one.
  await page.route(/\/api\/batches\?/, route => route.fulfill({ json: { batches: [], total: 0, limit: 20, offset: 0 } }))
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
