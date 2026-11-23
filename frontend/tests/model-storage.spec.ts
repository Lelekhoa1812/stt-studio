import { expect, test, type Page, type Route } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

type CacheRow = { id: string; cached: boolean; sizeBytes: number | null }
type StorageState = { rows: CacheRow[]; listRequests: number; deletes: string[]; deleteStatus?: number; deleteDetail?: string }

const GIB = 1024 ** 3
// The catalog presets the App falls back to when /api/models lists no models (see mockHealthAndModels).
const initialRows: CacheRow[] = [
  { id: 'large-v3', cached: true, sizeBytes: Math.round(1.6 * GIB) },
  { id: 'medium', cached: false, sizeBytes: null },
  { id: 'small', cached: true, sizeBytes: 500 * 1024 ** 2 },
  { id: 'base', cached: false, sizeBytes: null },
]

function freshState(overrides: Partial<StorageState> = {}): StorageState {
  return { rows: initialRows.map(row => ({ ...row })), listRequests: 0, deletes: [], ...overrides }
}

async function mockCacheApi(page: Page, state: StorageState) {
  await page.route('**/api/models/cache', async (route: Route) => {
    if (route.request().method() !== 'GET') return route.continue()
    state.listRequests += 1
    await route.fulfill({ json: { models: state.rows } })
  })
  await page.route(/\/api\/models\/cache\/[^/?]+$/, async (route: Route) => {
    if (route.request().method() !== 'DELETE') return route.continue()
    const id = decodeURIComponent(new URL(route.request().url()).pathname.split('/').pop() ?? '')
    state.deletes.push(id)
    if (state.deleteStatus) return route.fulfill({ status: state.deleteStatus, json: { detail: state.deleteDetail ?? 'Request failed' } })
    const row = state.rows.find(item => item.id === id)
    if (!row) return route.fulfill({ status: 404, json: { detail: 'Model not found in the catalog' } })
    const freedBytes = row.sizeBytes ?? 0
    row.cached = false
    row.sizeBytes = null
    await route.fulfill({ json: { id, deleted: true, freedBytes } })
  })
}

function rowFor(page: Page, name: string) {
  return page.getByRole('list', { name: 'Model storage list' }).getByRole('listitem').filter({ hasText: name })
}

test.describe('Model storage (mocked backend)', () => {
  test.beforeEach(async ({ page }) => {
    await mockHealthAndModels(page)
  })

  test('lists each model with its size and cached status', async ({ page }) => {
    const state = freshState()
    await mockCacheApi(page, state)
    await page.goto('/')

    await expect(page.getByRole('heading', { name: 'Model storage' })).toBeVisible()
    const rows = page.getByRole('list', { name: 'Model storage list' }).getByRole('listitem')
    await expect(rows).toHaveCount(4)

    const large = rowFor(page, 'Large v3')
    await expect(large).toContainText('1.6 GB')
    await expect(large).toContainText('Cached')
    await expect(large.getByRole('button', { name: 'Delete Large v3 files' })).toBeVisible()

    const medium = rowFor(page, 'Medium')
    await expect(medium).toContainText('Not cached')
    await expect(medium.getByRole('button', { name: /Delete/ })).toHaveCount(0)

    await expect(page.getByText('2 of 4 cached')).toBeVisible()
  })

  test('deleting a cached model asks first, then frees its space and refreshes the list', async ({ page }) => {
    const state = freshState()
    await mockCacheApi(page, state)
    await page.goto('/')
    await expect(rowFor(page, 'Large v3')).toContainText('1.6 GB')
    const listsBefore = state.listRequests

    await rowFor(page, 'Large v3').getByRole('button', { name: 'Delete Large v3 files' }).click()
    const dialog = page.getByRole('dialog', { name: 'Delete downloaded model?' })
    await expect(dialog).toBeVisible()
    // Nothing is sent until the user confirms.
    expect(state.deletes).toEqual([])
    await dialog.getByRole('button', { name: 'Delete model files' }).click()

    await expect(page.getByRole('status').filter({ hasText: 'Freed 1.6 GB' })).toBeVisible()
    await expect(dialog).toBeHidden()
    expect(state.deletes).toEqual(['large-v3'])
    await expect.poll(() => state.listRequests).toBeGreaterThan(listsBefore)
    await expect(rowFor(page, 'Large v3')).toContainText('Not cached')
    await expect(rowFor(page, 'Large v3').getByRole('button', { name: /Delete/ })).toHaveCount(0)
    // The other cached model is untouched.
    await expect(rowFor(page, 'Small')).toContainText('Cached')
  })

  test('a 409 shows the server reason readably and keeps the model cached', async ({ page }) => {
    const state = freshState({ deleteStatus: 409, deleteDetail: 'This model is in use' })
    await mockCacheApi(page, state)
    await page.goto('/')
    await expect(rowFor(page, 'Large v3')).toContainText('1.6 GB')

    await rowFor(page, 'Large v3').getByRole('button', { name: 'Delete Large v3 files' }).click()
    const dialog = page.getByRole('dialog', { name: 'Delete downloaded model?' })
    await dialog.getByRole('button', { name: 'Delete model files' }).click()

    const alert = dialog.getByRole('alert')
    await expect(alert).toHaveText('This model is in use')
    await expect(alert).not.toContainText('Request failed')
    await expect(alert).not.toContainText('409')
    // The dialog stays open so the user can cancel, and no success message appears.
    await expect(dialog).toBeVisible()
    await expect(page.getByRole('status').filter({ hasText: 'Freed' })).toHaveCount(0)

    await dialog.getByRole('button', { name: 'Cancel' }).click()
    await expect(dialog).toBeHidden()
    await expect(rowFor(page, 'Large v3')).toContainText('Cached')
  })
})
