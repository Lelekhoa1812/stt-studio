import { expect, test } from '@playwright/test'
import { ProjectFileError, decodeProjectFile, encodeProjectFile, projectFileName } from '../src/lib/projectFile'

const settings = { model: 'small', compute: 'int8', language: 'auto', beam: 5, vad: true, prompt: 'Names: Ada' }
const segments = [
  { start: 0.25, end: 1.5, text: 'hello world' },
  { start: 1.75, end: 3, text: 'local speech recognition' },
]

function input(overrides: Partial<Parameters<typeof encodeProjectFile>[0]> = {}) {
  return {
    name: 'Board call', createdAt: 1_700_000_000_000, savedAt: 1_700_000_100_000, duration: 3,
    settings, segments, audio: new Blob([new Uint8Array([1, 2, 3, 4, 5])]), fileName: 'call.wav', mimeType: 'audio/wav',
    ...overrides,
  }
}

async function bytesOf(blob: Blob): Promise<Uint8Array> {
  return new Uint8Array(await blob.arrayBuffer())
}

test('round-trips settings, transcript edits and audio bytes', async () => {
  const encoded = encodeProjectFile(input(), 1_700_000_200_000)
  const { manifest, audio } = await decodeProjectFile(encoded)
  expect(manifest.format).toBe('stt-studio-project')
  expect(manifest.version).toBe(1)
  expect(manifest.name).toBe('Board call')
  expect(manifest.exportedAt).toBe(1_700_000_200_000)
  expect(manifest.settings).toEqual(settings)
  expect(manifest.segments).toEqual(segments)
  expect(manifest.audio).toEqual({ fileName: 'call.wav', mimeType: 'audio/wav', bytes: 5 })
  expect(Array.from(await bytesOf(audio))).toEqual([1, 2, 3, 4, 5])
})

test('drops internal segment ids and keeps the header magic first', async () => {
  const encoded = encodeProjectFile(input({ segments: [{ start: 0, end: 1, text: 'x' }] as never }))
  expect(new TextDecoder().decode((await bytesOf(encoded)).subarray(0, 8))).toBe('STTPRJ01')
  const { manifest } = await decodeProjectFile(encoded)
  expect(Object.keys(manifest.segments[0]).sort()).toEqual(['end', 'start', 'text'])
})

test('rejects files that are not projects, damaged, or incomplete', async () => {
  const good = await bytesOf(encodeProjectFile(input()))

  const notProject = new Blob([new TextEncoder().encode('RIFF....WAVEdata')])
  await expect(decodeProjectFile(notProject)).rejects.toThrow('The file is not an STT Studio project.')

  const tiny = new Blob([good.subarray(0, 4)])
  await expect(decodeProjectFile(tiny)).rejects.toBeInstanceOf(ProjectFileError)

  const truncated = new Blob([good.subarray(0, good.length - 2)])
  await expect(decodeProjectFile(truncated)).rejects.toThrow('does not match the recorded size')

  const badHeader = new Uint8Array(good)
  new DataView(badHeader.buffer).setUint32(8, 0xffffffff, false)
  await expect(decodeProjectFile(new Blob([badHeader]))).rejects.toThrow('header is damaged')
})

function build(manifest: Record<string, unknown>) {
  const json = new TextEncoder().encode(JSON.stringify(manifest))
  const header = new Uint8Array(12)
  header.set(new TextEncoder().encode('STTPRJ01'), 0)
  new DataView(header.buffer).setUint32(8, json.length, false)
  return new Blob([header, json, new Uint8Array([9])])
}

const base = {
  format: 'stt-studio-project', version: 1, name: 'Call', createdAt: 1, savedAt: 1, exportedAt: 1,
  audio: { fileName: 'a.wav', mimeType: 'audio/wav', bytes: 1 }, settings, segments: [],
}

test('keeps speaker labels and names through a round trip', async () => {
  const labelled = [{ start: 0, end: 1, text: 'hi', speaker: 'Speaker 1' }, { start: 1, end: 2, text: 'yo' }]
  const { manifest } = await decodeProjectFile(encodeProjectFile(input({ segments: labelled, speakerNames: { 'Speaker 1': 'Alice' } })))
  expect(manifest.segments).toEqual(labelled)
  expect(manifest.speakerNames).toEqual({ 'Speaker 1': 'Alice' })
})

test('loads a version 1 project saved before speaker names existed', async () => {
  const { manifest } = await decodeProjectFile(build({ ...base, segments }))
  expect(manifest.version).toBe(1)
  expect(manifest.speakerNames).toBeUndefined()
  expect(manifest.segments).toEqual(segments)
})

test('rejects invalid speaker names and labels', async () => {
  await expect(decodeProjectFile(build({ ...base, speakerNames: ['Alice'] }))).rejects.toThrow('invalid speaker names')
  await expect(decodeProjectFile(build({ ...base, speakerNames: { 'Speaker 1': '' } }))).rejects.toThrow('invalid speaker name')
  await expect(decodeProjectFile(build({ ...base, segments: [{ start: 0, end: 1, text: 'x', speaker: 7 }] }))).rejects.toThrow('invalid speaker label')
})

test('rejects unsupported versions and invalid manifest fields', async () => {
  await expect(decodeProjectFile(build({ ...base, version: 2 }))).rejects.toThrow('format version 2')
  await expect(decodeProjectFile(build({ ...base, name: '' }))).rejects.toThrow('invalid project name')
  await expect(decodeProjectFile(build({ ...base, segments: [{ start: 2, end: 1, text: 'bad' }] }))).rejects.toThrow('invalid segment end time')
  await expect(decodeProjectFile(build({ ...base, settings: { ...settings, beam: 99 } }))).rejects.toThrow('invalid beam size')
  await expect(decodeProjectFile(build({ ...base, settings: { ...settings, vad: 'yes' } }))).rejects.toThrow('invalid VAD setting')
})

test('builds safe download names', () => {
  expect(projectFileName('Board: call/Q3?')).toBe('Board call Q3.sttproj')
  expect(projectFileName('   ')).toBe('project.sttproj')
})
