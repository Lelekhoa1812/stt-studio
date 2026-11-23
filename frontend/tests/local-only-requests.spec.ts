import { expect, test, type Request } from '@playwright/test'
import { mockHealthAndModels } from './helpers'

const LOCAL_HOSTS = new Set(['127.0.0.1', 'localhost'])

// The model list names a publisher, which used to trigger avatar lookups on huggingface.co when the app opened.
const modelsWithPublisher = {
  default_model: 'small',
  compute_types: ['int8'],
  cached_models: [],
  loaded_models: [],
  models: [
    {
      id: 'small', name: 'Small', family: 'whisper', runtime: 'faster-whisper',
      publisher: 'SYSTRAN', publisher_slug: 'systran', publisher_url: 'https://huggingface.co/Systran',
      source_url: 'https://huggingface.co/Systran/faster-whisper-small', license: 'MIT',
    },
  ],
}

test('loading the app makes no requests outside 127.0.0.1 or localhost', async ({ page }) => {
  await mockHealthAndModels(page)
  await page.route('**/api/models', route => route.fulfill({ json: modelsWithPublisher }))

  const requests: Request[] = []
  page.on('request', request => requests.push(request))

  await page.goto('/')
  await expect(page.getByRole('link', { name: 'Open SYSTRAN profile on Hugging Face' })).toBeVisible()
  await page.waitForLoadState('networkidle')

  const external = requests
    .map(request => new URL(request.url()))
    .filter(url => !['data:', 'blob:'].includes(url.protocol) && !LOCAL_HOSTS.has(url.hostname))
    .map(url => url.href)
  expect(external, 'requests that left the local machine').toEqual([])
  expect(requests.some(request => request.url().includes('huggingface.co/api'))).toBe(false)
})

test('the publisher mark shows the bundled display name without loading an image', async ({ page }) => {
  await mockHealthAndModels(page)
  await page.route('**/api/models', route => route.fulfill({ json: modelsWithPublisher }))
  await page.goto('/')

  const mark = page.getByRole('link', { name: 'Open SYSTRAN profile on Hugging Face' })
  await expect(mark).toHaveAttribute('href', 'https://huggingface.co/Systran')
  await expect(mark).toHaveAttribute('title', 'SYSTRAN · model publisher')
  await expect(mark.locator('img')).toHaveCount(0)
})
