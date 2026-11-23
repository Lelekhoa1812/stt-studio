import { expect, test, type Page } from '@playwright/test'
import { readFile } from 'node:fs/promises'
import { chooseSyntheticAudio, mockHealthAndModels, syntheticWav } from './helpers'

const unavailableReason = 'HF_TOKEN is not set; add it to .env (the account must have accepted the model terms).'
const available = { available: true, model: 'pyannote/speaker-diarization-3.1', license: 'MIT', reason: null }
const unavailable = { ...available, available: false, reason: unavailableReason }
const transcript = [
  { start: 0.25, end: 1.5, text: 'hello world' },
  { start: 1.75, end: 3, text: 'local speech recognition' },
]
const labelled = [
  { start: 0.25, end: 1.5, text: 'hello world', speaker: 'Speaker 1' },
  { start: 1.75, end: 3, text: 'local speech recognition', speaker: 'Speaker 2' },
]
const diarizationCompleted = { status: 'completed', speakers: 2, model: 'pyannote/speaker-diarization-3.1', license: 'MIT' }

async function mockCapabilities(page: Page, diarization: Record<string, unknown>) {
  await page.route('**/api/capabilities', route => route.fulfill({ json: { service: 'stt-studio', diarization } }))
}

async function transcribeLabelled(page: Page) {
  await mockCapabilities(page, available)
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-speakers-1', status: 'completed', segments: labelled,
      info: { language: 'en', duration: 3, diarization: diarizationCompleted } } })
  })
  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
}

async function exportText(page: Page, item: RegExp) {
  await page.getByRole('button', { name: 'Export' }).click()
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('menuitem', { name: item }).click(),
  ])
  return readFile((await download.path())!, 'utf8')
}

function oldProjectFile(): Buffer {
  // A version 1 project written before speaker names existed: no speakerNames and no segment speaker.
  const audio = syntheticWav()
  const manifest = {
    format: 'stt-studio-project', version: 1, name: 'Old project', createdAt: 1_700_000_000_000, savedAt: 1_700_000_100_000,
    exportedAt: 1_700_000_200_000, duration: 0.25,
    audio: { fileName: 'synthetic-silence.wav', mimeType: 'audio/wav', bytes: audio.length },
    settings: { model: 'small', compute: 'int8', language: 'auto', beam: 5, vad: true, prompt: '' },
    segments: [{ start: 0.25, end: 1.5, text: 'hello world' }],
  }
  const json = Buffer.from(JSON.stringify(manifest))
  const header = Buffer.alloc(12)
  header.write('STTPRJ01', 0)
  header.writeUInt32BE(json.length, 8)
  return Buffer.concat([header, json, audio])
}

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('disables speaker labels in both panels and shows the reason when diarization is unavailable', async ({ page }) => {
  await mockCapabilities(page, unavailable)
  await page.goto('/')
  const single = page.locator('.config-panel .speaker-option')
  await expect(single.getByRole('checkbox', { name: 'Label speakers' })).toBeDisabled()
  await expect(single.getByText(unavailableReason)).toBeVisible()
  const batch = page.locator('.batch-panel .speaker-option')
  await expect(batch.getByRole('checkbox', { name: 'Label speakers' })).toBeDisabled()
  await expect(batch.getByText(unavailableReason)).toBeVisible()
})

test('sends diarize=true with a single job only when labelling is enabled', async ({ page }) => {
  await mockCapabilities(page, available)
  const bodies: string[] = []
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    bodies.push(route.request().postData() || '')
    await route.fulfill({ status: 202, json: { id: 'job-diarize-1', status: 'completed', segments: transcript } })
  })

  await page.goto('/')
  await chooseSyntheticAudio(page)
  const checkbox = page.locator('.config-panel .speaker-option').getByRole('checkbox', { name: 'Label speakers' })
  await expect(checkbox).toBeEnabled()
  await expect(checkbox).not.toBeChecked()
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  expect(bodies[0]).toMatch(/name="diarize"\r\n\r\nfalse\r\n/)

  await page.reload()
  await chooseSyntheticAudio(page)
  await page.locator('.config-panel .speaker-option').getByRole('checkbox', { name: 'Label speakers' }).check()
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  expect(bodies).toHaveLength(2)
  expect(bodies[1]).toMatch(/name="diarize"\r\n\r\ntrue\r\n/)
})

test('sends diarize=true with a batch when labelling is enabled', async ({ page }) => {
  await mockCapabilities(page, available)
  let batchBody = ''
  await page.route('**/api/batches', route => {
    batchBody = route.request().postData() || ''
    return route.fulfill({ status: 202, json: { id: 'batch-diarize-1', total: 1, done: true,
      counts: { queued: 0, running: 0, completed: 1, failed: 0, cancelled: 0 },
      jobs: [{ id: 'job-b1', filename: 'one.wav', status: 'completed', progress: null, error: null, segments: transcript, info: null }] } })
  })

  await page.goto('/')
  await page.locator('.batch-panel').getByRole('checkbox', { name: 'Label speakers' }).check()
  await page.getByLabel('Batch audio files').setInputFiles([{ name: 'one.wav', mimeType: 'audio/wav', buffer: syntheticWav() }])
  await page.getByRole('button', { name: 'Start batch (1)' }).click()
  await expect(page.getByText('1 of 1 complete')).toBeVisible()
  expect(batchBody).toMatch(/name="diarize"\r\n\r\ntrue\r\n/)
})

test('shows a speaker label on each segment and a rename field per speaker', async ({ page }) => {
  await transcribeLabelled(page)
  await expect(page.locator('.segment-row').nth(0).locator('.speaker-tag')).toHaveText('Speaker 1')
  await expect(page.locator('.segment-row').nth(1).locator('.speaker-tag')).toHaveText('Speaker 2')
  await expect(page.getByLabel('Name for Speaker 1')).toHaveValue('')
  await expect(page.getByLabel('Name for Speaker 2')).toBeVisible()
})

test('renames a speaker, keeps the original label, and keeps the name after save and reload', async ({ page }) => {
  await transcribeLabelled(page)
  await page.getByLabel('Name for Speaker 1').fill('Alice')
  await expect(page.locator('.segment-row').nth(0).locator('.speaker-tag')).toHaveText('Alice')
  await expect(page.locator('.segment-row').nth(0).locator('.speaker-tag')).toHaveAttribute('title', 'Speaker 1')
  await expect(page.locator('.segment-row').nth(1).locator('.speaker-tag')).toHaveText('Speaker 2')

  await page.getByRole('button', { name: 'Save project', exact: true }).click()
  const saveDialog = page.getByRole('dialog', { name: 'Save a local project' })
  await saveDialog.getByLabel('Project name').fill('Speaker project')
  await saveDialog.getByRole('button', { name: 'Save project', exact: true }).click()
  await expect(page.getByRole('status')).toContainText('Project saved in this browser on this Mac.')

  await page.reload()
  await page.getByRole('button', { name: 'Open project' }).click()
  const library = page.getByRole('dialog', { name: 'Local projects' })
  await library.getByRole('button', { name: 'Open', exact: true }).click()
  await expect(page.locator('.segment-row').nth(0).locator('.speaker-tag')).toHaveText('Alice')
  await expect(page.getByLabel('Name for Speaker 1')).toHaveValue('Alice')
  await expect(page.locator('.segment-row').nth(1).locator('.speaker-tag')).toHaveText('Speaker 2')
})

test('exports unrenamed JSON in the backend shape and prefixes names in TXT, SRT and JSON', async ({ page }) => {
  await transcribeLabelled(page)
  expect(await exportText(page, /Segment data/)).toBe(
    '[\n  {\n    "start": 0.25,\n    "end": 1.5,\n    "text": "hello world",\n    "speaker": "Speaker 1"\n  },\n  {\n    "start": 1.75,\n    "end": 3.0,\n    "text": "local speech recognition",\n    "speaker": "Speaker 2"\n  }\n]\n',
  )

  await page.getByLabel('Name for Speaker 1').fill('Alice')
  expect(await exportText(page, /Plain text/)).toBe('Alice: hello world\nSpeaker 2: local speech recognition\n')
  expect(await exportText(page, /SubRip/)).toBe(
    '1\n00:00:00,250 --> 00:00:01,500\nAlice: hello world\n\n2\n00:00:01,750 --> 00:00:03,000\nSpeaker 2: local speech recognition\n',
  )
  expect(await exportText(page, /Segment data/)).toContain('"speaker": "Alice"')
})

test('shows speaker labels in batch results', async ({ page }) => {
  await mockCapabilities(page, available)
  await page.route('**/api/batches', route => route.fulfill({ status: 202, json: { id: 'batch-labels-1', total: 1, done: true,
    counts: { queued: 0, running: 0, completed: 1, failed: 0, cancelled: 0 },
    jobs: [{ id: 'job-l1', filename: 'call.wav', status: 'completed', progress: null, error: null, segments: labelled, info: { diarization: diarizationCompleted } }] } }))

  await page.goto('/')
  await page.getByLabel('Batch audio files').setInputFiles([{ name: 'call.wav', mimeType: 'audio/wav', buffer: syntheticWav() }])
  await page.getByRole('button', { name: 'Start batch (1)' }).click()
  await page.getByRole('button', { name: /show text/ }).click()
  const text = page.locator('.batch-text')
  await expect(text.locator('.speaker-tag').nth(0)).toHaveText('Speaker 1')
  await expect(text.locator('.speaker-tag').nth(1)).toHaveText('Speaker 2')
})

test('imports a version 1 project saved before speaker names existed', async ({ page }) => {
  await page.goto('/')
  await page.locator('input[aria-label="Project file to import"]').setInputFiles({
    name: 'old-project.sttproj', mimeType: 'application/octet-stream', buffer: oldProjectFile(),
  })
  await expect(page.getByText('Imported “Old project” from a project file.')).toBeVisible()
  await expect(page.getByLabel('Transcript segment at 00:00:00')).toHaveValue('hello world')
  await expect(page.locator('.speaker-tag')).toHaveCount(0)
  await expect(page.getByLabel('Name for Speaker 1')).toHaveCount(0)
})

test('shows a note when speaker labelling fails and keeps the transcript', async ({ page }) => {
  const failed = { status: 'failed', speakers: 0, model: 'pyannote/speaker-diarization-3.1', license: 'MIT', reason: 'The diarizer environment is missing; run scripts/setup-diarizer.sh.' }
  await mockCapabilities(page, available)
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-failed-diarize', status: 'completed', segments: transcript, info: { language: 'en', duration: 3, diarization: failed } } })
  })
  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByText('Speaker labels could not be produced.')).toBeVisible()
  await expect(page.getByText('The diarizer environment is missing; run scripts/setup-diarizer.sh.')).toBeVisible()
  await expect(page.getByText('The transcript is still usable.')).toBeVisible()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  await expect(page.locator('.speaker-tag')).toHaveCount(0)
})

test('shows the same note on a batch file whose speaker labelling failed', async ({ page }) => {
  const failed = { status: 'failed', speakers: 0, model: 'pyannote/speaker-diarization-3.1', license: 'MIT', reason: 'Preparing audio for speaker labels timed out' }
  await mockCapabilities(page, available)
  await page.route('**/api/batches', route => route.fulfill({ status: 202, json: { id: 'batch-failed-1', total: 1, done: true,
    counts: { queued: 0, running: 0, completed: 1, failed: 0, cancelled: 0 },
    jobs: [{ id: 'job-f1', filename: 'one.wav', status: 'completed', progress: null, error: null, segments: transcript, info: { diarization: failed } }] } }))
  await page.goto('/')
  await page.getByLabel('Batch audio files').setInputFiles([{ name: 'one.wav', mimeType: 'audio/wav', buffer: syntheticWav() }])
  await page.getByRole('button', { name: 'Start batch (1)' }).click()
  const item = page.locator('.batch-item')
  await expect(item.getByText('Speaker labels could not be produced.')).toBeVisible()
  await expect(item.getByText('Preparing audio for speaker labels timed out')).toBeVisible()
  await page.getByRole('button', { name: /show text/ }).click()
  await expect(page.locator('.batch-text').getByText('local speech recognition')).toBeVisible()
})

test('disables speaker labels for an untimed MLX model, whose single segment cannot be matched to speakers', async ({ page }) => {
  await mockCapabilities(page, available)
  await page.route('**/api/models', route => route.fulfill({
    json: { default_model: 'small', compute_types: ['int8'], loaded_models: [], models: [
      { id: 'small', name: 'Small', family: 'Whisper', runtime: 'faster-whisper',
        capabilities: { computeTypes: true, beamSize: true, vad: true, initialPrompt: true, languageHint: 'codes', timedSegments: true } },
      { id: 'mlx-community/Qwen3-ASR-0.6B-8bit', name: 'Qwen3-ASR (MLX conversion)', family: 'Qwen3-ASR (MLX conversion)', runtime: 'mlx-audio',
        publisher: 'mlx-community', publisher_url: 'https://huggingface.co/mlx-community', license: 'Apache-2.0',
        capabilities: { computeTypes: false, beamSize: false, vad: false, initialPrompt: false, languageHint: 'names', timedSegments: false } },
    ] },
  }))
  await page.goto('/')
  await page.getByRole('combobox', { name: /^MODEL/ }).selectOption('mlx-community/Qwen3-ASR-0.6B-8bit')
  const single = page.locator('.config-panel .speaker-option')
  await expect(single.getByRole('checkbox', { name: 'Label speakers' })).toBeDisabled()
  await expect(single.getByText('Speaker labels need timed segments. The MLX runtime returns one untimed segment, so choose a faster-whisper model.')).toBeVisible()
})
