import { expect, test, type Page, type Route } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

type CacheRow = { id: string; cached: boolean; sizeBytes: number | null }
type UnloadedRow = { model: string; compute_type: string; runtime: string }

type UnloadState = {
  rows: CacheRow[]
  listRequests: number
  unloadCalls: number
  unloaded: UnloadedRow[]
  unloadStatus?: number
  unloadDetail?: string
}

const GIB = 1024 ** 3

const initialRows: CacheRow[] = [
  { id: 'large-v3', cached: true, sizeBytes: Math.round(1.6 * GIB) },
  { id: 'medium', cached: false, sizeBytes: null },
  { id: 'small', cached: true, sizeBytes: 500 * 1024 ** 2 },
  { id: 'base', cached: false, sizeBytes: null },
]

function freshState(overrides: Partial<UnloadState> = {}): UnloadState {
  return {
    rows: initialRows.map(row => ({ ...row })),
    listRequests: 0,
    unloadCalls: 0,
    unloaded: [{ model: 'small', compute_type: 'int8', runtime: 'faster-whisper' }],
    ...overrides,
  }
}

async function mockModelApis(page: Page, state: UnloadState) {
  await page.route('**/api/models/cache', async (route: Route) => {
    if (route.request().method() !== 'GET') return route.continue()
    state.listRequests += 1
    await route.fulfill({ json: { models: state.rows } })
  })
  await page.route('**/api/models/unload', async (route: Route) => {
    if (route.request().method() !== 'POST') return route.continue()
    state.unloadCalls += 1
    if (state.unloadStatus) return route.fulfill({ status: state.unloadStatus, json: { detail: state.unloadDetail ?? 'Request failed' } })
    await route.fulfill({ json: { unloaded: state.unloaded, loaded_models: [] } })
  })
}

test.describe('Model unload (mocked backend)', () => {
  test.beforeEach(async ({ page }) => {
    await mockHealthAndModels(page)
  })

  test('unloading asks first, then reports how many models were released', async ({ page }) => {
    const state = freshState({
      unloaded: [
        { model: 'small', compute_type: 'int8', runtime: 'faster-whisper' },
        { model: 'large-v3', compute_type: 'int8', runtime: 'faster-whisper' },
      ],
    })
    await mockModelApis(page, state)
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Model storage' })).toBeVisible()

    await page.getByRole('button', { name: 'Unload models' }).click()
    const dialog = page.getByRole('dialog', { name: 'Unload models?' })
    await expect(dialog).toBeVisible()
    expect(state.unloadCalls).toBe(0)

    await dialog.getByRole('button', { name: 'Unload models' }).click()
    await expect(page.getByRole('status').filter({ hasText: 'Released 2 loaded models.' })).toBeVisible()
    await expect(dialog).toBeHidden()
    expect(state.unloadCalls).toBe(1)
  })

  test('an empty unload list says no models were loaded', async ({ page }) => {
    const state = freshState({ unloaded: [] })
    await mockModelApis(page, state)
    await page.goto('/')
    await page.getByRole('button', { name: 'Unload models' }).click()
    const dialog = page.getByRole('dialog', { name: 'Unload models?' })
    await dialog.getByRole('button', { name: 'Unload models' }).click()
    await expect(page.getByRole('status').filter({ hasText: 'No models were loaded.' })).toBeVisible()
  })

  test('a 409 shows the server reason in the dialog', async ({ page }) => {
    const detail = 'A transcription is running; wait for it or cancel it before unloading'
    const state = freshState({ unloadStatus: 409, unloadDetail: detail })
    await mockModelApis(page, state)
    await page.goto('/')
    await page.getByRole('button', { name: 'Unload models' }).click()
    const dialog = page.getByRole('dialog', { name: 'Unload models?' })
    await dialog.getByRole('button', { name: 'Unload models' }).click()

    const alert = dialog.getByRole('alert')
    await expect(alert).toHaveText(detail)
    await expect(alert).not.toContainText('Request failed')
    await expect(alert).not.toContainText('409')
    await expect(dialog).toBeVisible()
    await expect(page.getByRole('status').filter({ hasText: 'Released' })).toHaveCount(0)
  })
})