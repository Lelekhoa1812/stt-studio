import { expect, test, type Page, type Route } from '@playwright/test'

const transcript = [
  { start: 0.25, end: 1.5, text: 'hello world', avg_logprob: -0.1, no_speech_prob: 0.01 },
  { start: 1.75, end: 3, text: 'local speech recognition', avg_logprob: -0.2, no_speech_prob: 0.02 },
]

function syntheticWav(): Buffer {
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

async function mockHealthAndModels(page: Page) {
  await page.route('**/api/health', route => route.fulfill({ json: { status: 'ok', active_jobs: 0, loaded_models: 0 } }))
  await page.route('**/api/models', route => route.fulfill({
    json: { default_model: 'small', compute_types: ['int8', 'float16', 'float32'], cached_models: [], loaded_models: [] },
  }))
}

async function chooseSyntheticAudio(page: Page) {
  await page.getByRole('heading', { name: 'Choose an audio file' }).waitFor()
  await page.locator('input[type="file"]').setInputFiles({
    name: 'synthetic-silence.wav',
    mimeType: 'audio/wav',
    buffer: syntheticWav(),
  })
  await expect(page.getByText('synthetic-silence.wav').first()).toBeVisible()
}

async function completeJob(route: Route, pollCount: { value: number }) {
  pollCount.value += 1
  if (pollCount.value === 1) {
    await route.fulfill({ json: { id: 'job-test-1', status: 'running' } })
    return
  }
  await route.fulfill({ json: { id: 'job-test-1', status: 'completed', segments: transcript,
    info: { language: 'en', duration: 3 } } })
}

test.beforeEach(async ({ page }) => {
  await mockHealthAndModels(page)
})

test('uploads synthetic audio, omits automatic language, edits a segment and exports SRT', async ({ page }) => {
  const submittedBodies: string[] = []
  const polls = { value: 0 }
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    submittedBodies.push(route.request().postData() || '')
    await route.fulfill({ status: 202, json: { id: 'job-test-1', status: 'queued' } })
  })
  await page.route('**/api/jobs/job-test-1', route => completeJob(route, polls))

  await page.goto('/')
  await chooseSyntheticAudio(page)
  await expect(page.getByLabel('LANGUAGE')).toHaveValue('auto')
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  expect(submittedBodies).toHaveLength(1)
  expect(submittedBodies[0]).not.toContain('name="language"')

  await page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' }).fill('hello edited world')
  await page.getByRole('button', { name: 'Export' }).click()
  await expect(page.getByRole('menu')).toBeVisible()
  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('menuitem', { name: /SubRip captions/ }).click(),
  ])
  expect(download.suggestedFilename()).toBe('synthetic-silence.srt')
  const path = await download.path()
  expect(path).not.toBeNull()
  const { readFile } = await import('node:fs/promises')
  const contents = await readFile(path!, 'utf8')
  expect(contents).toContain('00:00:00,250 --> 00:00:01,500')
  expect(contents).toContain('hello edited world')
})

test('shows a submit error and allows retry with the selected file', async ({ page }) => {
  let submissions = 0
  const polls = { value: 0 }
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    submissions += 1
    if (submissions === 1) {
      await route.fulfill({ status: 503, json: { detail: 'Model is still loading. Please retry.' } })
      return
    }
    await route.fulfill({ status: 202, json: { id: 'job-test-1', status: 'queued' } })
  })
  await page.route('**/api/jobs/job-test-1', route => completeJob(route, polls))

  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('alert')).toContainText('Model is still loading. Please retry.')
  await expect(page.getByText('synthetic-silence.wav').first()).toBeVisible()

  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  expect(submissions).toBe(2)
})

test('keyboard shortcut submits and Escape closes the export menu', async ({ page }) => {
  let submitted = false
  const polls = { value: 0 }
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    submitted = true
    await route.fulfill({ status: 202, json: { id: 'job-test-1', status: 'queued' } })
  })
  await page.route('**/api/jobs/job-test-1', route => completeJob(route, polls))

  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Open project' }).focus()
  await page.keyboard.press('Control+Enter')
  await expect.poll(() => submitted).toBe(true)
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  await page.getByRole('button', { name: 'Export' }).click()
  await expect(page.getByRole('menu')).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByRole('menu')).toBeHidden()
})

test('keeps the workspace usable at a narrow mobile viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Transcription workspace' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Transcribe audio' })).toBeVisible()
  const dimensions = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
  }))
  expect(dimensions.content).toBeLessThanOrEqual(dimensions.viewport + 1)
})


test('saves, restores after reload, and deletes a local project', async ({ page }) => {
  const polls = { value: 0 }
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    await route.fulfill({ status: 202, json: { id: 'job-test-1', status: 'queued' } })
  })
  await page.route('**/api/jobs/job-test-1', route => completeJob(route, polls))

  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByLabel('LANGUAGE').selectOption('vi')
  await page.getByLabel('MODEL').selectOption('medium')
  await page.locator('summary').filter({ hasText: 'Advanced options' }).click()
  await page.getByLabel('COMPUTE TYPE').selectOption('float16')
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  const segment = page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })
  await expect(segment).toHaveValue('hello world')
  await segment.fill('persisted synthetic transcript')

  await page.getByRole('button', { name: 'Save project', exact: true }).click()
  const saveDialog = page.getByRole('dialog', { name: 'Save a local project' })
  await saveDialog.getByLabel('Project name').fill('Synthetic browser project')
  await saveDialog.getByRole('button', { name: 'Save project', exact: true }).click()
  await expect(page.getByRole('status')).toContainText('Project saved in this browser on this Mac.')

  // Reload this isolated Playwright context to prove the record survives a page restart.
  await page.reload()
  await page.getByRole('button', { name: 'Open project' }).click()
  const library = page.getByRole('dialog', { name: 'Local projects' })
  await expect(library.getByText('Synthetic browser project')).toBeVisible()
  await library.getByRole('button', { name: 'Open', exact: true }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' }))
    .toHaveValue('persisted synthetic transcript')
  await expect(page.getByText('synthetic-silence.wav').first()).toBeVisible()
  await expect(page.getByLabel('LANGUAGE')).toHaveValue('vi')
  await expect(page.getByLabel('MODEL')).toHaveValue('medium')
  await page.locator('summary').filter({ hasText: 'Advanced options' }).click()
  await expect(page.getByLabel('COMPUTE TYPE')).toHaveValue('float16')

  await page.getByRole('button', { name: 'Open project' }).click()
  const reopenedLibrary = page.getByRole('dialog', { name: 'Local projects' })
  await reopenedLibrary.getByRole('button', { name: 'Delete saved project Synthetic browser project' }).click()
  const deleteDialog = page.getByRole('dialog', { name: 'Delete saved project?' })
  await deleteDialog.getByRole('button', { name: 'Delete project', exact: true }).click()
  await expect(reopenedLibrary.getByRole('heading', { name: 'No saved projects yet' })).toBeVisible()

  await page.reload()
  await page.getByRole('button', { name: 'Open project' }).click()
  await expect(page.getByRole('dialog', { name: 'Local projects' }).getByRole('heading', { name: 'No saved projects yet' }))
    .toBeVisible()
})


test('protects unsaved transcript edits before removing audio', async ({ page }) => {
  let submissions = 0
  const polls = { value: 0 }
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    submissions += 1
    await route.fulfill({ status: 202, json: { id: 'job-test-1', status: 'queued' } })
  })
  await page.route('**/api/jobs/job-test-1', route => completeJob(route, polls))

  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  const segment = page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })
  await expect(segment).toHaveValue('hello world')
  await segment.fill('unsaved words to keep')

  await page.getByRole('button', { name: 'Remove selected audio' }).click()
  const dialog = page.getByRole('dialog', { name: 'Remove this audio?' })
  await expect(dialog).toBeVisible()
  await page.keyboard.press('Control+Enter')
  await expect(dialog).toBeVisible()
  expect(submissions).toBe(1)
  await dialog.getByRole('button', { name: 'Keep working' }).click()
  await expect(segment).toHaveValue('unsaved words to keep')
  await expect(page.getByText('synthetic-silence.wav').first()).toBeVisible()

  await page.getByRole('button', { name: 'Remove selected audio' }).click()
  await page.getByRole('dialog', { name: 'Remove this audio?' })
    .getByRole('button', { name: 'Discard changes and remove audio' }).click()
  await expect(page.getByText('Your transcript will appear here')).toBeVisible()
  await expect(page.getByText('synthetic-silence.wav')).toHaveCount(0)
  expect(submissions).toBe(1)
})

test('shows project controls with accessible names on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await chooseSyntheticAudio(page)

  await expect(page.getByRole('button', { name: 'Open project' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Save project' })).toBeVisible()
  await page.getByRole('button', { name: 'Open project' }).click()
  await expect(page.getByRole('dialog', { name: 'Local projects' })).toBeVisible()
})

test('tracks loaded state for the selected model and compute pair', async ({ page }) => {
  let loadedModels = [{ model: 'small', compute_type: 'int8' }]
  await page.route('**/api/models', route => route.fulfill({
    json: { default_model: 'small', compute_types: ['int8', 'float16', 'float32'], cached_models: [], loaded_models: loadedModels },
  }))
  await page.route('**/api/models/load', async route => {
    const selection = route.request().postDataJSON() as { model: string; compute_type: string }
    loadedModels = [...loadedModels.filter(item => item.model !== selection.model || item.compute_type !== selection.compute_type), selection]
    await route.fulfill({ json: { model: selection.model, compute_type: selection.compute_type, loaded: true, warmup: 'complete' } })
  })

  await page.goto('/')
  await expect(page.getByText('Model loaded and ready')).toBeVisible()
  await page.locator('summary').filter({ hasText: 'Advanced options' }).click()
  await page.getByLabel('COMPUTE TYPE').selectOption('float16')
  await expect(page.getByText('Not loaded for float16')).toBeVisible()
  await page.getByRole('button', { name: /Load and warm model/ }).click()
  await expect(page.getByText('Model loaded and ready')).toBeVisible()

  await page.getByLabel('MODEL').selectOption('medium')
  await expect(page.getByText('Not loaded for float16')).toBeVisible()
})

test('shows the curated model family catalog and warms a custom CTranslate2 model on demand', async ({ page }) => {
  let selected = ''
  await page.route('**/api/models', route => route.fulfill({
    json: { default_model: 'small', compute_types: ['int8'], loaded_models: [], models: [
      { id: 'small', name: 'Small', family: 'whisper', publisher: 'SYSTRAN', family_publisher: 'OpenAI', publisher_url: 'https://huggingface.co/Systran', source_url: 'https://huggingface.co/Systran/faster-whisper-small', license: 'MIT', runtime: 'faster-whisper' },
      { id: 'Systran/faster-distil-whisper-large-v3', name: 'Distil-Whisper Large v3', family: 'distil-whisper', publisher: 'Systran', publisher_url: 'https://huggingface.co/Systran', source_url: 'https://huggingface.co/Systran/faster-distil-whisper-large-v3', license: 'MIT', runtime: 'faster-whisper' },
    ] },
  }))
  await page.route('**/api/models/load', async route => {
    selected = route.request().postDataJSON().model
    await route.fulfill({ json: { model: selected, compute_type: 'int8', loaded: true, warmup: 'complete' } })
  })
  await page.goto('/')
  await expect(page.getByRole('link', { name: 'Open SYSTRAN profile on Hugging Face' })).toHaveAttribute('href', 'https://huggingface.co/Systran')
  await expect(page.getByText('Upstream: OpenAI')).toBeVisible()
  await expect(page.getByLabel('MODEL')).toContainText('Distil-Whisper Large v3')
  await page.getByLabel('MODEL').selectOption('Systran/faster-distil-whisper-large-v3')
  await expect(page.getByText('distil-whisper', { exact: true })).toBeVisible()
  await page.locator('summary').filter({ hasText: 'Advanced options' }).click()
  await page.getByPlaceholder('owner/model (CTranslate2 Whisper)').fill('team/custom-whisper')
  await page.getByRole('button', { name: 'Add', exact: true }).click()
  await page.getByRole('button', { name: /Load and warm model/ }).click()
  await expect(page.getByText('Model is ready on this Mac.')).toBeVisible()
  expect(selected).toBe('team/custom-whisper')
})

test('ignores transcription shortcut while save and rerun dialogs are open', async ({ page }) => {
  let submissions = 0
  const polls = { value: 0 }
  await page.route('**/api/jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue()
    submissions += 1
    await route.fulfill({ status: 202, json: { id: 'job-test-1', status: 'queued' } })
  })
  await page.route('**/api/jobs/job-test-1', route => completeJob(route, polls))

  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: 'Save project' }).click()
  const saveDialog = page.getByRole('dialog', { name: 'Save a local project' })
  await expect(saveDialog).toBeVisible()
  await page.keyboard.press('Control+Enter')
  await expect(saveDialog).toBeVisible()
  expect(submissions).toBe(0)
  await page.keyboard.press('Escape')
  await expect(saveDialog).toBeHidden()

  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
  await page.getByRole('button', { name: 'Transcribe audio' }).click()
  const rerunDialog = page.getByRole('dialog', { name: 'Run transcription again?' })
  await expect(rerunDialog).toBeVisible()
  await page.keyboard.press('Control+Enter')
  await expect(rerunDialog).toBeVisible()
  expect(submissions).toBe(1)
  await rerunDialog.getByRole('button', { name: 'Keep current transcript' }).click()
  await expect(page.getByRole('textbox', { name: 'Transcript segment at 00:00:00' })).toHaveValue('hello world')
})

test('hides Whisper-only controls for an MLX runtime model and loads it with the native compute sentinel', async ({ page }) => {
  let loadBody: { compute_type?: string; runtime?: string } = {}
  await page.route('**/api/models', route => route.fulfill({
    json: { default_model: 'small', compute_types: ['int8'], loaded_models: [],
      runtime_capabilities: {}, models: [
        { id: 'small', name: 'Small', family: 'Whisper', runtime: 'faster-whisper', capabilities: { computeTypes: true, beamSize: true, vad: true, initialPrompt: true, languageHint: 'codes' } },
        { id: 'mlx-community/parakeet-tdt-0.6b-v3', name: 'Parakeet v3 (MLX)', family: 'Parakeet (MLX conversion)', runtime: 'mlx-audio',
          publisher: 'mlx-community', publisher_url: 'https://huggingface.co/mlx-community', license: 'CC-BY-4.0',
          capabilities: { computeTypes: false, beamSize: false, vad: false, initialPrompt: false, languageHint: 'auto' } },
      ] },
  }))
  await page.route('**/api/models/load', async route => {
    loadBody = route.request().postDataJSON()
    await route.fulfill({ json: { model: loadBody, loaded: true, warmup: 'complete' } })
  })
  await page.goto('/')
  await page.getByLabel('MODEL').selectOption('mlx-community/parakeet-tdt-0.6b-v3')
  await expect(page.getByLabel('LANGUAGE')).toBeDisabled()
  await expect(page.getByText('This model detects the language on its own.')).toBeVisible()
  await page.locator('summary').filter({ hasText: 'Advanced options' }).click()
  await expect(page.getByLabel('COMPUTE TYPE')).toHaveCount(0)
  await expect(page.getByText('BEAM SIZE')).toHaveCount(0)
  await expect(page.getByText('Voice activity detection')).toHaveCount(0)
  await expect(page.getByText('VOCABULARY HINT')).toHaveCount(0)
  await page.getByRole('button', { name: /Load and warm model/ }).click()
  await expect(page.getByText('Model is ready on this Mac.')).toBeVisible()
  expect(loadBody.compute_type).toBe('native')
  expect(loadBody.runtime).toBe('mlx-audio')
})

test('filters the model list by family and keeps the selected model available', async ({ page }) => {
  await page.route('**/api/models', route => route.fulfill({
    json: { default_model: 'small', compute_types: ['int8'], loaded_models: [], models: [
      { id: 'small', name: 'Small', family: 'Whisper', runtime: 'faster-whisper', capabilities: { computeTypes: true, beamSize: true, vad: true, initialPrompt: true, languageHint: 'codes' } },
      { id: 'Systran/faster-distil-whisper-large-v3', name: 'Distil-Whisper Large v3', family: 'Distil-Whisper', runtime: 'faster-whisper' },
    ] },
  }))
  await page.goto('/')
  await page.getByLabel('Search list').fill('distil')
  // The selected model (small) stays listed so the select never loses its current value.
  await expect(page.getByLabel('MODEL').locator('option', { hasText: 'Distil-Whisper Large v3' })).toHaveCount(1)
  await expect(page.getByLabel('MODEL').locator('option')).toHaveCount(2)
  await page.getByLabel('Search list').fill('zzz-nothing')
  await expect(page.getByLabel('MODEL').locator('option')).toHaveCount(1)
  await expect(page.getByLabel('MODEL')).toHaveValue('small')
})

test('cancels a running job, shows measured progress and keeps the partial transcript', async ({ page }) => {
  const partial = [transcript[0]]
  let cancelRequested = false
  await page.route('**/api/jobs', route => route.fulfill({ status: 202, json: { id: 'job-cancel-1', status: 'queued' } }))
  await page.route('**/api/jobs/job-cancel-1/cancel', route => {
    cancelRequested = true
    return route.fulfill({ json: { id: 'job-cancel-1', status: 'running', cancel_requested: true, progress: 33.3, segments: partial } })
  })
  await page.route('**/api/jobs/job-cancel-1', route => route.fulfill({ json: cancelRequested
    ? { id: 'job-cancel-1', status: 'cancelled', cancel_requested: true, progress: 33.3, segments: partial }
    : { id: 'job-cancel-1', status: 'running', cancel_requested: false, progress: 33.3, segments: partial } }))
  await page.goto('/')
  await chooseSyntheticAudio(page)
  await page.getByRole('button', { name: /Transcribe audio/ }).click()
  await expect(page.getByText('33.3%')).toBeVisible()
  await expect(page.getByText('Live partial transcript')).toBeVisible()
  await expect(page.getByText('hello world')).toBeVisible()
  await page.getByRole('button', { name: 'Cancel transcription' }).click()
  await expect(page.locator('.progress-status').getByText('Cancelling…')).toBeVisible()
  await expect(page.getByText('Transcription cancelled. Kept 1 segment transcribed so far.')).toBeVisible()
  await expect(page.getByLabel('Transcript segment at 00:00:00')).toHaveValue('hello world')
  await expect(page.getByRole('button', { name: /Transcribe audio/ })).toBeEnabled()
})
